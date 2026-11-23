"""CSVの各行を3構成で各1枚生成する。同じ行の3構成は同じseedを使う。"""

import argparse
import csv
import datetime as dt
import json
import random
import time
from pathlib import Path

import generate_images as gen


ROOT = gen.ROOT
LOG_DIR = gen.LOG_DIR
ComfyClient = gen.ComfyClient
PROFILES = (
    ("dw060_chenbin080_ip030_hires", "sdxl_dreamwash_lora_reference_hires_api.json",
     "common_positive_wai_cool_cel.txt", "common_negative_wai_cool_cel.txt",
     {"lora": "il_chenbin_try_3-000006.safetensors", "lora_strength": 0.8,
      "prepend": "dreamwash, artist: chen bin", "steps": 28, "cfg": 5.5,
      "sampler": "euler_ancestral", "scheduler": "normal", "ipadapter_weight": 0.3}, True),
    ("dw060_anmi080_hires", "sdxl_dreamwash_lora_hires_api.json",
     "common_positive_wai_cool_cel.txt", "common_negative_wai_cool_cel.txt",
     {"lora": "anmi.safetensors", "lora_strength": 0.8, "prepend": "dreamwash, anmihuafeng",
      "steps": 28, "cfg": 5.5, "sampler": "euler_ancestral", "scheduler": "normal"}, False),
    ("dw060_anmi030_original", "sdxl_dreamwash_anmi_api.json",
     "common_positive_dreamwash_anmi.txt", "common_negative_mixed.txt", {}, False),
)


def make_refiners(face_refine, larger_eyes):
    refiners = []
    if face_refine:
        from refine_faces import FaceRefiner

        refiners.append((FaceRefiner(ROOT, gen.INPUT_DIR, gen.RAW_DIR), "顔補修"))
    if larger_eyes:
        from refine_eyes import EyeRefiner

        refiners.append((EyeRefiner(ROOT, gen.INPUT_DIR, gen.RAW_DIR), "目元補修"))
    return refiners


def prepare_profiles(dry_run):
    prepared = []
    for name, filename, positive, negative, options, reference in PROFILES:
        base = gen.load_workflow(ROOT / "workflows" / filename)
        if reference and not (ROOT / "input/wai_cool_cel_reference.jpg").is_file():
            raise FileNotFoundError("画風参照画像がありません: input/wai_cool_cel_reference.jpg")
        gen.configure_style_reference(
            base, image=str(ROOT / "input/wai_cool_cel_reference.jpg") if reference else "",
            directory="", disabled=not reference, dry_run=dry_run,
        )
        # 保存するのは完成画像1枚だけ。参照プレビューは量産出力に混ぜない。
        base = {key: node for key, node in base.items()
                if key == gen.NODE_SAVE or node.get("class_type") != "SaveImage"}
        overrides = {**options, "batch": 1,
                     "common_positive": gen.load_common_prompt(ROOT / "prompts" / positive),
                     "common_negative": gen.load_common_prompt(ROOT / "prompts" / negative)}
        prepared.append((name, filename, base, overrides))
    return prepared


def iter_jobs(rows, profiles, seed, stamp, order="profile"):
    seeded_rows = [(path, index, row, seed if seed >= 0 else random.randint(0, gen.SEED_MAX))
                   for path, index, row in rows]
    if order == "row":
        schedule = ((row, profile) for row in seeded_rows for profile in profiles)
    else:
        schedule = ((row, profile) for profile in profiles for row in seeded_rows)
    for (path, index, row, row_seed), (name, filename, base, overrides) in schedule:
        workflow = gen.build_workflow(base, row, row_seed, overrides)
        workflow[gen.NODE_SAVE]["inputs"]["filename_prefix"] = (
            f"{path.stem}/three_styles/{stamp}/row_{index:05d}_{name}"
        )
        yield path, index, row, name, filename, row_seed, workflow


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, nargs="+", default=[ROOT / "prompts/prompts_26.csv"])
    parser.add_argument("--start", type=int, default=0, help="先頭から飛ばすCSV行数")
    parser.add_argument("--limit", type=int, default=0, help="処理するCSV行数（0=全部）")
    parser.add_argument("--seed", type=int, default=-1, help="-1=行ごとにランダム、0以上=全行固定")
    parser.add_argument("--order", choices=["profile", "row"], default="profile",
                        help="profile=構成ごとに全行を生成（既定・切替削減）、row=行ごとに3構成")
    parser.add_argument("--server", default=gen.DEFAULT_SERVER)
    parser.add_argument("--timeout", type=float, default=1200, help="1構成の完了待ち秒数")
    parser.add_argument("--poll", type=float, default=0.5)
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--face-refine", "--refine-face", action="store_true", help="1.5倍仕上げ後に顔を補修")
    parser.add_argument("--larger-eyes", action="store_true", help="顔補修後に目元を調整")
    parser.add_argument("--dry-run", action="store_true", help="投入・ファイル保存なしで条件をJSON表示")
    args = parser.parse_args(argv)
    if args.start < 0 or args.limit < 0:
        parser.error("--start / --limit は0以上で指定してください")
    if args.seed < -1 or args.seed > gen.SEED_MAX:
        parser.error("--seed は-1またはunsigned 64bitの範囲で指定してください")
    if args.timeout <= 0 or args.poll <= 0:
        parser.error("--timeout / --poll は0より大きい値で指定してください")
    try:
        rows = [(path, index, row) for path in args.prompts
                for index, row in enumerate(gen.load_prompts(path), 1)]
        rows = rows[args.start:]
        if args.limit:
            rows = rows[:args.limit]
        if not rows:
            print("プロンプトが0件です")
            return 0
        profiles = prepare_profiles(args.dry_run)
    except (OSError, ValueError, KeyError) as exc:
        print(f"入力の読み込みに失敗: {exc}")
        return 2

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    jobs = iter_jobs(rows, profiles, args.seed, stamp, args.order)
    print(f"CSV {len(rows)} 行 × 3構成 = {len(rows) * 3} 枚")
    print("生成順: " + ("構成ごとに全行（切り替えを削減）" if args.order == "profile" else "行ごとに3構成"))
    if args.face_refine or args.larger_eyes:
        print("後処理: " + " → ".join(name for enabled, name in
              [(args.face_refine, "顔補修"), (args.larger_eyes, "目元補修")] if enabled))
    if args.dry_run:
        for path, index, row, name, filename, seed, workflow in jobs:
            print(json.dumps({"source_csv": str(path), "row_index": index, "row_id": row.id,
                              "profile": name, "workflow": filename, "seed": seed,
                              "positive": workflow["2"]["inputs"]["text"],
                              "negative": workflow["3"]["inputs"]["text"],
                              "face_refine": args.face_refine, "larger_eyes": args.larger_eyes,
                              "settings": gen.effective_values(workflow, []),
                              "prefix": workflow["7"]["inputs"]["filename_prefix"]}, ensure_ascii=False))
        return 0

    client = ComfyClient(args.server)
    ready, message = client.check_ready()
    print(message)
    if not ready:
        return 3
    try:
        refiners = make_refiners(args.face_refine, args.larger_eyes)
    except (ImportError, OSError) as exc:
        print(f"補修の初期化に失敗: {exc}")
        return 2
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"three_styles_{stamp}.csv"
    fields = ["source_csv", "row_index", "row_id", "profile", "workflow", "seed",
              "prompt_id", "status", "seconds", "files", "error", "positive", "negative",
              "face_refine", "larger_eyes", "original_files", "refinements"]
    failures = 0
    print(f"ログ: {log_path}")
    with log_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        stream.flush()
        for path, index, row, name, filename, seed, workflow in jobs:
            record = dict(source_csv=str(path), row_index=index, row_id=row.id, profile=name,
                          workflow=filename, seed=seed, prompt_id="", status="ERROR", files="", error="",
                          positive=workflow["2"]["inputs"]["text"], negative=workflow["3"]["inputs"]["text"],
                          face_refine=args.face_refine, larger_eyes=args.larger_eyes,
                          original_files="", refinements="[]")
            started = time.monotonic()
            timed_out = False
            try:
                record["prompt_id"] = client.queue_prompt(workflow)
                entry = client.wait(record["prompt_id"], poll=args.poll, timeout=args.timeout)
                images = entry.get("outputs", {}).get(gen.NODE_SAVE, {}).get("images", [])
                if len(images) != 1:
                    raise RuntimeError(f"完成画像が1枚ではありません: {len(images)}")
                record["files"] = "|".join(client.output_files(entry))
                record["original_files"] = record["files"]
                filename = record["files"]
                stages = []
                for refiner, label in refiners:
                    stage = {"stage": label, "input": filename, "status": "ERROR", "output": ""}
                    stages.append(stage)
                    try:
                        refined = refiner.refine(client, workflow, filename, args.poll, args.timeout)
                        stage["status"] = "OK" if refined is not None else "SKIPPED"
                        if refined is None:
                            print(f"  {label}スキップ（対象が小さいか、領域を特定できない）: {filename}", flush=True)
                        else:
                            filename = refined
                            stage["output"] = refined
                            record["files"] = refined
                    finally:
                        record["refinements"] = json.dumps(stages, ensure_ascii=False)
                record["status"] = "OK"
            except Exception as exc:
                failures += 1
                timed_out = isinstance(exc, TimeoutError)
                record["error"] = repr(exc)
            record["seconds"] = round(time.monotonic() - started, 2)
            writer.writerow(record)
            stream.flush()
            print(f"row={index} id={row.id} {name} seed={seed} {record['status']} {record['files']}", flush=True)
            if record["error"]:
                print(record["error"], flush=True)
                if timed_out or args.stop_on_error:
                    return 1
    print(f"完了: 失敗 {failures} 件 / {len(rows) * 3} 件。ログ: {log_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
