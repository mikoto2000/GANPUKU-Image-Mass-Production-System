"""Compare anmi 0.15/0.30/0.50 while holding the current workflow constant."""
import argparse
import copy
import datetime as dt
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

import generate_images as gen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    stamp = dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).strftime("%Y%m%d_%H%M%S")
    destination = gen.RAW_DIR / "lora_comparison" / stamp
    base = gen.load_workflow(gen.ROOT / "workflows/sdxl_dreamwash_anmi_api.json")
    references = gen.configure_style_reference(
        base, image="", directory=str(gen.STYLE_REFERENCE_DIR), disabled=False,
        dry_run=args.dry_run,
    )
    source = gen.ROOT / "prompts/prompts_anmi_stack_test.csv"
    rows = gen.load_prompts(source)[:2]
    strengths = (0.15, 0.30, 0.50)
    jobs = []
    for index, row in enumerate(rows):
        for strength in strengths:
            workflow = gen.build_workflow(base, row, 246813579 + index, {
                "lora_strength": strength, "batch": 1,
                "steps": 28, "cfg": 5.5, "sampler": "euler_ancestral", "scheduler": "normal",
                "common_positive": gen.load_common_prompt(gen.ROOT / "prompts/common_positive_dreamwash_anmi.txt"),
                "common_negative": gen.load_common_prompt(gen.ROOT / "prompts/common_negative_mixed.txt"),
            })
            label = f"row{index + 1}_dw060_anmi{round(strength * 100):03d}"
            workflow["7"]["inputs"]["filename_prefix"] = f"lora_comparison/{stamp}/{label}"
            assert workflow["9"]["inputs"]["strength_model"] == 0.6
            assert workflow["9"]["inputs"]["strength_clip"] == 0.6
            assert workflow["8"]["inputs"]["strength_model"] == strength
            jobs.append({"label": label, "row": index + 1, "strength": strength,
                         "seed": 246813579 + index, "workflow": workflow})
    for index in range(len(rows)):
        normalized = []
        for job in jobs[index * 3:index * 3 + 3]:
            wf = copy.deepcopy(job["workflow"])
            wf["8"]["inputs"]["strength_model"] = 0
            wf["8"]["inputs"]["strength_clip"] = 0
            wf["7"]["inputs"]["filename_prefix"] = ""
            normalized.append(wf)
        assert normalized[0] == normalized[1] == normalized[2]
    if args.dry_run:
        print(json.dumps({"jobs": len(jobs), "references": references,
                          "strengths": strengths, "output": str(destination)}, ensure_ascii=False))
        return
    destination.mkdir(parents=True, exist_ok=False)
    manifest = {"source": str(source), "references": references, "jobs": jobs}
    manifest_path = destination / "manifest.json"
    def save():
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    save()
    client = gen.ComfyClient(gen.DEFAULT_SERVER)
    for job in jobs:
        workflow = job["workflow"]
        (destination / (job["label"] + ".json")).write_text(
            json.dumps(workflow, ensure_ascii=False, indent=2), encoding="utf-8")
        job["prompt_id"] = client.queue_prompt(workflow)
        save()
        print(f"Generating {job['label']} seed={job['seed']}", flush=True)
        entry = client.wait(job["prompt_id"], poll=1, timeout=1200)
        images = entry.get("outputs", {}).get("7", {}).get("images", [])
        if len(images) != 1:
            raise RuntimeError(f"Expected one finished image: {images}")
        result = images[0]
        job["file"] = str(gen.RAW_DIR / result.get("subfolder", "") / result["filename"])
        with Image.open(job["file"]) as image:
            image.verify()
        job["status"] = "OK"
        save()
        print(f"Saved {job['file']}", flush=True)
    thumb_w, thumb_h = 416, 608
    header = 55
    sheet = Image.new("RGB", (thumb_w * 3, (thumb_h + header) * len(rows)), "#f1f1f1")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 21)
    for index, job in enumerate(jobs):
        x = (index % 3) * thumb_w
        y = (index // 3) * (thumb_h + header)
        draw.text((x + 12, y + 5), f"Dreamwash 0.60 + anmi {job['strength']:.2f}", fill="black", font=font)
        draw.text((x + 12, y + 29), f"Row {job['row']} / seed {job['seed']}", fill="#555555", font=font)
        with Image.open(job["file"]) as image:
            sheet.paste(image.convert("RGB").resize((thumb_w, thumb_h), Image.Resampling.LANCZOS), (x, y + header))
    sheet.save(destination / "comparison.jpg", quality=95)
    print(f"COMPARISON={destination / 'comparison.jpg'}", flush=True)


if __name__ == "__main__":
    main()
