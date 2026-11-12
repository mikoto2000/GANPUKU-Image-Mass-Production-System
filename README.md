# F:\ai-image — ローカル画像生成環境

RTX 4070 Ti SUPER (16GB VRAM) / Windows 11 / Python 3.12 向けの
アニメキャラ絵量産構成。

## 好みの絵柄を比較する

普段の WAI v17 + Dreamwash + anmi ワークフローは、`input/style_reference` 直下の
画像を IP-Adapter の `style transfer`、強度 `0.20` で使う。
複数枚ある場合は画像の特徴量を平均する。フォルダがない場合や画像が0枚の場合は、
スタイル参照なしで生成する。別のフォルダは `--ref-dir "フォルダのパス"`、
1枚だけ使う場合は `--ref-image "画像のパス"` で指定できる。
参照を明示的に無効にするには、末尾に `--no-style-reference` を付ける。
生成画像は Lanczos で1.5倍に拡大し、同じ LoRA とスタイル参照の設定で
`denoise 0.25` の再サンプリングをかけてから保存する。
以下の量産例は、前回の比較で使った CSV と共通プロンプトで各行1枚を生成する。
少量で試すには末尾に `--limit 3`、条件確認だけなら `--limit 3 --dry-run` を付ける。

```powershell
.\venv\Scripts\python.exe scripts\generate_images.py `
  --workflow workflows\sdxl_dreamwash_anmi_api.json `
  --prompts prompts\prompts_combined_3600_v5.csv `
  --common-positive prompts\common_positive_dreamwash_anmi.txt `
  --common-negative prompts\common_negative_mixed.txt `
  --batch 1 --seed -1 --prefix "dreamwash_anmi/$(Get-Date -Format 'yyyyMMdd_HHmmss')/img"
```

`generate_images.py` は画像のファイル名にコマンド実行日時を
`YYYYMMDD_HHMMSS` 形式で付ける（例 `img_20261112_153000_00001_.png`）。
同じ実行で生成した画像と CSV ログには同じ日時が入る。

```powershell
# 参照画像・プロンプト・生成条件を確認する（生成なし）
.\compare-style.ps1 -DryRun
# ComfyUI を起動してから実行。手持ちの参照画像に差し替える
.\compare-style.ps1 -RefImage '.\input\reference.png' -Limit 3 -Seed 12345
```

WAI v17 で、同じプロンプト・シードを使って参照なし（`base`）、
IP-Adapter の絵柄参照 0.35（`ref_035`）、0.60（`ref_060`）を比較する。
既定は3プロンプトで計9枚。`-Csv` と `-Start` で比較対象を変えられる。
出力先は `output\raw\style_compare\<実行日時>\`。
同じプロンプト同士を見比べ、線・塗り・色が好みに近づいたかと、
顔や構図が参照に引っ張られすぎていないかを確認する。

`prompts/style_line_paint.csv` には、細い色付きの線、髪と服のセル塗り、
柔らかい肌の陰影を指定した比較用の2題材がある。
複数の画像は `-RefImage @('参照1.jpg', '参照2.jpg')` で渡す。
参照なしは各題材につき1枚。参照ありは画像ごとに強度2通りで生成する。
出力名の `ref_01`、`ref_02` は指定順に対応する。

参照なしでも別のモデルの絵柄が好みに近ければ、そのモデルを起点にする。
複数の題材で再現したい絵柄が決まったら、学習に使える作例を専用フォルダに集め、
`prepare_lora_dataset.py --source` で指定する。別の人物・服・構図も含め、
学習に入れなかった題材で、LoRA なしとありを同じ条件で比較する。
既存の `train_lora.py` は WAI v17 用。ほかのモデルで学習する場合は、
学習のベースも合わせて設定する。

## Animagine XL 4.0 Opt（2026-11-01追加）

```powershell
# 専用CSVの3パターンを各1枚生成（太ももまで・上半身・全身）
.\generate-quality.ps1 -Model animagine -Count 1 -Seed 12345
# 1.5倍の拡大仕上げも行う
.\generate-quality.ps1 -Model animagine -Count 1 -Seed -1 -Hires
# 自分のCSVを使用
.\generate-quality.ps1 -Model animagine -Csv '.\prompts\別のファイル.csv' -Count 1 -Seed -1
```

指定を省略したモデルは従来どおりWAI。`-Model wai`でも明示的に選択できる。
Animagineの既定CSVは `prompts/animagine.csv`、出力は `output/raw/animagine/`。
モデルファイルは `models/checkpoints/animagine-xl-4.0-opt.safetensors`。
専用ワークフローは `workflows/animagine_base_api.json` と `workflows/animagine_hires_api.json`。

公式推奨に合わせ、専用CSVはEuler a・28 steps・CFG 5を使用。
品質タグ `masterpiece, high score, great score, absurdres` はpositiveの末尾に置く。
`-Csv`で別のCSVを指定した場合、そのCSVのプロンプトと設定をそのまま使う。
現在の `-Lora` はWAIで学習したrei_style用プリセットのため、Animagine選択時は併用しない。

配布元: https://huggingface.co/cagliostrolab/animagine-xl-4.0
公式ファイルのSHA256は `logs/animagine_install_20261101.json` に記録する。

## 品質改善版（2026-10-31）

既存の `prompts.csv` と旧 LoRA は保持。新しい生成入口は以下。

```powershell
# 短い negative / WAI v17 / CFG 5.5 / Euler a / 28 steps
.\generate-quality.ps1 -Count 4 -Seed 12345
# 別のCSVを使用（省略時はスクリプト配下のprompts\quality.csv）
.\generate-quality.ps1 -Csv '.\prompts\prompts.csv' -Count 1 -Seed -1
# 同じ生成に、1.5倍の画像拡大 + denoise 0.25 の再生成を追加
.\generate-quality.ps1 -Count 1 -Seed 12345 -Hires
# 新規LoRAを試す（既定では有効にしない）
.\generate-quality.ps1 -Count 1 -Seed 12345 -Hires -Lora
# 採用した PNG を仕上げる（元画像のモデル・プロンプト・LoRAを継承）
.\venv\Scripts\python.exe scripts\quality_pipeline.py finish --image '生成PNGのパス' --face
```

`-Csv` は絶対パスと、実行時の作業フォルダーからの相対パスに対応する。
空白を含むパスは引用符で囲む。`-Count` はCSVの各行あたりの枚数。

- `workflows/sdxl_hires_api.json`: 2段階生成。顔補修は `quality_pipeline.py finish --face` で追加。
- `prompts/quality.csv`: 品質関連だけの短い negative。元の表情・衣装指定は保持。
- img2img は既定で元画像の縦横比と指定面積を保持。`--resize-mode crop` で固定サイズへ中央切抜き。
- ワークフローを直接使用する img2img も幅0・高さ1216で比率を保持。
- 選別の既定重みは美的0.75・人体0.15・ディテール0.10。左右非対称だけでは減点しない。
- 顔検出は `deepghs/anime_face_detection` の `face_detect_v1.4_n` ONNX。1顔のみ検出した場合に補修。
- NoobAI-XL 1.1 は比較用として追加。WAIより常に優れるという意味ではなく、既定モデルはWAIを継続。
- 比較画像・完全なAPIグラフ・seed・実行時間: `output/quality_20261031/`。

学習は画像別WD-v3キャプション、約1MPの縦横比別バケット、確認用データの除外、
FP32 VAE、正しいサイズ条件、gradient accumulation、勾配クリップに対応。
`--batch` は実際のミニバッチ数になり、`--accumulate` は勾配を蓄積する回数。

```powershell
.\venv\Scripts\python.exe scripts\prepare_lora_dataset.py
.\venv\Scripts\python.exe scripts\train_lora.py --epochs 1 --lr 3e-5 --batch 1 --accumulate 4
.\venv\Scripts\python.exe scripts\test_quality_regressions.py
```

新規LoRAは `rei_style_v3.safetensors` に保存。同名がある場合は上書きせず停止するため、
追加実験では `--out models/loras/別名.safetensors` を指定する。
実際に使用したキャプションは `.captions.json`、損失・時間は `.training.json` に保存。
修正前の学習スクリプトと不要な途中チェックポイントは整理済み。
以下の旧学習例・74枚での測定値は、修正前の履歴として読むこと。

## 構成

| 要素 | 採用 |
|---|---|
| UI | ComfyUI（API で自動化しやすい） |
| 推論 | PyTorch 2.11.0 + CUDA 12.8 |
| ベースモデル | SDXL 系アニメ特化 finetune（Illustrious / NoobAI 系） |
| プロンプト | Danbooru タグ |
| 追加 | 自作 LoRA / ONNX アニメ顔検出 / 1.5倍の2段階生成 |

Flux は文字入れなど必要な場面に限定する方針。

## ディレクトリ

```
F:\ai-image\
  ComfyUI\            ComfyUI 本体（git clone）
  venv\               Python 仮想環境
  models\
    checkpoints\      ベースモデル (.safetensors)
    loras\            キャラ・絵柄 LoRA
    vae\              VAE
    controlnet\       ControlNet
    upscale_models\   アップスケーラ
    embeddings\       textual inversion
  output\
    raw\              生成直後
    selected\         選別後
    upscaled\         アップスケール後
  input\              入力画像（img2img / ControlNet 用）
  workflows\          ComfyUI ワークフロー JSON
  scripts\            量産・選別スクリプト
  prompts\            タグ組み合わせ CSV/JSON
  logs\               実行ログ
```

モデルは `ComfyUI\extra_model_paths.yaml` 経由で `F:\ai-image\models` を参照する。
ComfyUI 配下に二重に置かなくて済む。

## 起動

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File F:\ai-image\start-comfyui.ps1
```

ブラウザで http://127.0.0.1:8188 を開く。

## 停止

```powershell
.\stop-comfyui.ps1
```

この環境の `venv\Scripts\python.exe` で `ComfyUI\main.py` を実行しているプロセスを終了する。
生成中の場合、その処理も中断される。停止対象の確認だけなら `.\stop-comfyui.ps1 -WhatIf` を使う。

## モデルの置き場所

| 種類 | 置く場所 |
|---|---|
| ベースモデル | `models\checkpoints\` |
| LoRA | `models\loras\` |
| VAE | `models\vae\` |
| ControlNet | `models\controlnet\` |
| アップスケーラ | `models\upscale_models\` |

配置後は ComfyUI の画面で **Refresh** を押すと認識される。

## 生成の初期設定

| 項目 | 値 |
|---|---|
| 解像度 | 832x1216（縦長）または 1024x1024 |
| サンプラー | Euler a / DPM++ 2M Karras |
| steps | 20〜30 |
| CFG | 4〜7 |
| バッチ | 4〜8 |

v-prediction モデルを使う場合はモデルカードの指定に従うこと。
設定を間違えると色が破綻する。

## 量産の考え方

ボトルネックは生成ではなく**選別**。

1. **生成** — タグの組み合わせを外部 CSV/JSON から回す。ComfyUI の `/prompt` API に投げる。
2. **選別** — 美的スコア・人体構造（DWPose）・ディテール・重複排除で自動的に絞る。上位 20〜30% だけ残す。
3. **後処理** — 採用画像を1.5倍に拡大して低denoiseで再生成。必要なら顔だけを追加補修。
4. **管理** — PNG メタデータと生成ログ（モデル・LoRA・タグ・シード）を必ず残す。

## 注意

- 夜間バッチは「生成 → 選別 → 後処理」を段ごとに分ける。一気にやると失敗時に全部やり直しになる。
- アップスケール込みだと 1 枚 20MB 超。1000 枚で 20GB 程度を見込む。
- Civitai のモデルは商用不可・再配布不可が多い。公開する場合はライセンスを確認する。
- JDK ではなく Python 環境なので、`cacerts` の話は無関係。ただしモデルのダウンロード元は HTTPS なので証明書は OS 側で管理される。


## 量産手順

生成 → 選別 → 後処理を段階ごとに実行する。各段は独立して再実行できる。

### 0. 前提（導入済み）

| 要素 | 実体 | 備考 |
|---|---|---|
| ベースモデル | `models\checkpoints\waiIllustriousSDXL_v170.safetensors` | 6616.6 MB / SHA256 検証済み |
| 人体検出 | `models\dwpose\yolox_l.onnx` | 206.7 MB |
| 姿勢推定 | `models\dwpose\dw-ll_ucoco_384.onnx` | 128.2 MB |
| 選別ライブラリ | `opencv-python` / `imagehash` / `onnxruntime-gpu` / `aesthetic-predictor-v2-5` / `controlnet-dwpose` | venv に導入済み |

Python は必ず venv のものを使う。

```powershell
F:\ai-image\venv\Scripts\python.exe --version
```

### 1. モデルの追加

`get-model.ps1` でダウンロードする。`curl.exe` によるストリーミングなので数 GB でも落ちない。

```powershell
cd F:\ai-image
.\get-model.ps1 -Url 'https://civitai.com/api/download/models/2883731' `
  -Kind checkpoints -FileName 'waiIllustriousSDXL_v170.safetensors'
```

| 引数 | 内容 |
|---|---|
| `-Url` | Civitai なら `https://civitai.com/api/download/models/<versionId>` |
| `-Kind` | `checkpoints` / `loras` / `vae` / `controlnet` / `upscale_models` / `embeddings` |
| `-FileName` | 保存名。拡張子まで含める |
| `-Sha256` | 省略可。指定すると検証し、不一致なら破棄する |
| `-Token` | 省略可。API トークンが必要なモデル用 |

同名ファイルが既にあればスキップする。配置後は ComfyUI 画面の **Refresh** を押す。

### 2. 生成

`scripts\generate_images.py` が `prompts\prompts.csv` の各行を `workflows\sdxl_base_api.json` へ流し込み、
ComfyUI の `/prompt` API に順次投入する。完了は `/history/{prompt_id}` のポーリングで待つ。

```powershell
cd F:\ai-image

# 組み立てだけ確認（投入しない）
.\venv\Scripts\python.exe scripts\generate_images.py --dry-run --limit 2

# 少数枚で構図と色を確認
.\venv\Scripts\python.exe scripts\generate_images.py --limit 4 --seed 12345

# 本番（バッチ 4 で 30 プロンプト = 120 枚）
.\venv\Scripts\python.exe scripts\generate_images.py --limit 30 --batch 4

# 衣装バリエーション（101〜200）だけを回す
.\venv\Scripts\python.exe scripts\generate_images.py --start 100 --limit 100 --batch 2
```

出力は `output\raw\gen\` 配下（`SaveImage` の `filename_prefix` が `gen/img`）。
実行内容は `logs\gen_YYYYMMDD_HHMMSS.csv` に「プロンプト・シード・出力ファイル・所要時間・成否」で残る。

#### ファイル構成

| ファイル | 役割 |
|---|---|
| `workflows\sdxl_base_api.json` | API 形式ワークフロー雛形（7 ノード） |
| `prompts\prompts.csv` | タグ組み合わせ。1 行 1 生成単位 |
| `scripts\generate_images.py` | 投入・待機・ログ出力 |

ワークフローのノード ID は固定で、スクリプト側が次のように流し込む。

| ノード | ID | 流し込む値 |
|---|---|---|
| CheckpointLoaderSimple | 1 | `ckpt_name` |
| CLIPTextEncode (positive) | 2 | `text` |
| CLIPTextEncode (negative) | 3 | `text`（CSV が空なら雛形のまま） |
| EmptyLatentImage | 4 | `width` / `height` / `batch_size` |
| KSampler | 5 | `seed` / `steps` / `cfg` / `sampler_name` / `scheduler` |
| VAEDecode | 6 | （固定） |
| SaveImage | 7 | `filename_prefix` |

ノードを増やす場合は、雛形とスクリプト冒頭の `NODE_*` 定数を両方合わせること。

#### prompts.csv の書式

| 列 | 内容 |
|---|---|
| `id` | 識別子。ログに出る |
| `positive` | 必須。空行は読み飛ばす |
| `negative` | 全行に共通の negative を入れてある。空にすると雛形の negative に戻る |
| `width` / `height` | 0 なら雛形の値 |
| `steps` / `cfg` | 0 なら雛形の値 |
| `sampler` / `scheduler` | 空なら雛形の値 |
| `batch` | 0 なら雛形の値 |
| `note` | メモ。処理には使わない |

優先順位は **引数 > CSV > 雛形** の順。

#### ポーズと表情の方針

`prompts.csv` は全 200 行について、次の条件を満たすように作ってある。

- **棒立ちにしない** — `standing` を使う行には必ず重心移動タグ（`contrapposto` / `weight on one leg`）を併記する。さらに腕の位置を変えるタグ（`hand on own hip` / `hand in hair` / `arms behind back` など）を必ず 1 つ入れる。
- **無表情にしない** — 全行にポジティブな表情タグ（`smile` / `open mouth, smile` / `grin` / `laughing` / `gentle smile` など）を入れる。
- **negative で抑える** — 全行の `negative` に `expressionless, blank stare, dead eyes, emotionless, serious, frown, scowl, straight-on, arms at sides, stiff pose, t-pose` を含める。

衣装行（101〜200）は服のシルエットが隠れないよう、腕組みを避けて `hand on own hip` / `hand in hair` / `head tilt` などを割り当てている。

新しい行を追加するときは、この 3 条件を守ること。守られているかは次のコマンドで確認できる。

```powershell
.\venv\Scripts\python.exe _probe_csv_check.py
```

`ALL_OK: True` なら条件を満たしている。`standing_without_stance` / `standing_without_arm` / `without_positive_expr` に id が出たら、その行を直す。

#### 主な引数

| 引数 | 既定 | 内容 |
|---|---|---|
| `--start` | 0 | 先頭 N 行を飛ばす（衣装だけ回すときなど） |
| `--limit` | 0 | `--start` の位置から N 行だけ処理（0=全部） |
| `--batch` | 0 | 1 プロンプトあたりの枚数（CSV より優先） |
| `--seed` | -1 | 固定シード（-1=毎回ランダム） |
| `--steps` / `--cfg` | 0 | 上書き |
| `--sampler` / `--scheduler` | 空 | 上書き |
| `--width` / `--height` | 0 | 上書き |
| `--ckpt` | 空 | チェックポイント名を上書き |
| `--prefix` | 空 | `filename_prefix` を上書き。ファイル名の接頭辞には実行日時が付く |
| `--workflow` | `workflows\sdxl_base_api.json` | 使用するワークフロー JSON |
| `--lora` | 空 | LoRA 名を上書き（`workflows\sdxl_lora_api.json` が必要） |
| `--lora-strength` | -1.0 | LoRA の `strength_model` / `strength_clip` を上書き |
| `--prepend` | 空 | positive の先頭に付ける文字列（トリガーワード等） |
| `--image` | 空 | img2img の元画像パス（`input\` へ自動コピー） |
| `--denoise` | -1.0 | denoise を上書き（img2img は 0.3〜0.6 推奨） |
| `--dry-run` | off | POST せず組み立て内容を表示 |
| `--timeout` | 900 | 1 生成あたりの待ち時間（秒） |
| `--stop-on-error` | off | 1 件失敗したら中断 |

`--seed` を固定すると全行が同じシードになる。バッチ内のばらつきを見たいときは `-1` のままにする。

生成を始める前に、必ず少数枚（4〜8 枚）で構図と色を確認する。
v-prediction モデルはサンプラー設定を間違えると色が破綻するので、モデルカードの指定に従うこと。

### 2.5 LoRA を使った生成

`models\loras\` に置いた LoRA を適用して生成する。`workflows\sdxl_lora_api.json` を使う。

```powershell
cd F:\ai-image

# 組み立て確認
.\venv\Scripts\python.exe scripts\generate_images.py `
  --workflow workflows\sdxl_lora_api.json `
  --lora rei_style_v2.safetensors --lora-strength 0.8 `
  --prepend rei_style --dry-run --limit 2

# 生成
.\venv\Scripts\python.exe scripts\generate_images.py `
  --workflow workflows\sdxl_lora_api.json `
  --lora rei_style_v2.safetensors --lora-strength 0.8 `
  --prepend rei_style --prefix lora_v2 --limit 4
```

| 引数 | 内容 |
|---|---|
| `--workflow` | `workflows\sdxl_lora_api.json` を指定する（LoRA ノード 8 を持つ） |
| `--lora` | `models\loras\` 内のファイル名。拡張子まで含める |
| `--lora-strength` | `strength_model` と `strength_clip` の両方に同じ値を入れる |
| `--prepend` | positive の先頭に付ける。学習時のトリガーワードを入れる |

`--lora` / `--lora-strength` は LoRA ノード（ID 8）を持つワークフローでのみ有効。
`workflows\sdxl_base_api.json` に対して指定すると `KeyError` で止まる。

#### 学習済み LoRA の実測結果

`favorites\`（74 枚）から学習した LoRA の効果を、同一シード・同一プロンプトで測定した。

| LoRA | epoch | favorites との cos（重心） | base との画素 MAE |
|---|---|---|---|
| なし（base） | — | 0.9106 | — |
| `rei_style_v1` | 1 | 0.8944 | 7.84 |
| `rei_style_v2` | 20 | **0.8104** | 42.84 |

**LoRA は確かに効いている**（画素 MAE 42.84、埋め込み cos 0.8148）。ただし **favorites には寄らず、むしろ離れる**。

途中チェックポイントを測ると、epoch 1 が最も近く、回すほど離れていく。

| epoch | cos（重心） |
|---|---|
| 1 | 0.8974 |
| 2 | 0.8933 |
| 3 | 0.8964 |
| 4 | 0.8813 |
| 6 | 0.7912 |
| 8 | 0.8112 |
| 10 | 0.8624 |
| 15 | 0.8281 |
| 20 | 0.8104 |

過学習や学習条件の問題が疑われるが、この類似度だけでは原因を断定できない。
favorites 同士の cos 平均 0.9291 / std 0.0205 も、学習できないことの証拠ではない。
修正前コードには全画像共通キャプションとサイズ条件の幅・高さ逆転があり、修正版で再評価する。

#### 現状の使い方

- **絵柄を少し変えたい**なら `rei_style_v2_step00074.safetensors`（epoch 1）を strength 0.8 で使う。base からの変化が小さく、破綻しにくい。
- **favorites に寄せたい**なら、この LoRA では効果が無い。キャプションの個別化（WD14 タガー等）と favorites の増量が必要。
- `rei_style_v2.safetensors`（epoch 20）は変化が大きいぶん、構図や色が崩れやすい。使うなら strength 0.4〜0.6 に下げる。

### 2.7 img2img（元画像から似た画像を作る）

`workflows\sdxl_img2img_api.json` を使うと、手元の画像を元に似た画像を生成できる。
元画像の構図を保つ手段。測定では cos 0.99 台だったが、この値は画質や絵柄の一致を保証しない。

```powershell
cd F:\ai-image

# 組み立て確認
.\venv\Scripts\python.exe scripts\generate_images.py `
  --workflow workflows\sdxl_img2img_api.json `
  --prompts prompts\img2img.csv `
  --image 'F:\ai-image\output\favorites\HQ2ob5ZbUAAIbNF.jpg' `
  --denoise 0.45 --dry-run --limit 1

# 生成
.\venv\Scripts\python.exe scripts\generate_images.py `
  --workflow workflows\sdxl_img2img_api.json `
  --prompts prompts\img2img.csv `
  --image 'F:\ai-image\output\favorites\HQ2ob5ZbUAAIbNF.jpg' `
  --denoise 0.45 --prefix img2img --limit 3
```

| 引数 | 内容 |
|---|---|
| `--image` | 元画像のパス。`input\` へ自動コピーされる（`LoadImage` は input 配下しか読めないため） |
| `--denoise` | 元画像をどこまで保つか。小さいほど元画像に近い |

#### denoise の目安（実測）

| denoise | 元画像との cos | 画素 MAE | 用途 |
|---|---|---|---|
| 0.30 | **0.9925** | 12.25 | ほぼ同じ絵。表情や小物だけ変える |
| 0.45 | 0.9876 | 16.85 | 構図を保ちつつ変化を出す（既定） |
| 0.55 | 0.9950 | 5.64 | さらに変化。ポーズは保たれる |
| 0.75 | 0.9858 | 8.92 | 構図が変わり始める |
| 0.95 | 0.9319 | 24.07 | ほぼ別画像。元画像は雰囲気だけ |

**0.30〜0.55 が実用帯**。1 枚 7〜9 秒で回せる。

#### ワークフローのノード構成

`EmptyLatentImage` の代わりに `LoadImage` → `ImageScale` → `VAEEncode` を使う。

| ノード | ID | 流し込む値 |
|---|---|---|
| LoadImage | 11 | `image` |
| ImageScale | 12 | `width` / `height`（`crop` は `disabled`） |
| VAEEncode | 13 | （固定） |
| KSampler | 5 | `denoise` を含む |

`ImageScale` で元画像を目標解像度へリサイズしてから VAEEncode する。`crop` を `disabled` にしているので、アスペクト比が違う画像は潰れる。切り抜きたい場合は `center` にする。

`--batch` は img2img では無視される（`VAEEncode` は 1 枚しか扱わない）。枚数を増やしたいときは `--limit` で行を増やす。

### 3. 選別

`start-clip-gallery.cmd` をダブルクリックすると、CLIP 画像ギャラリーが開く。
初期設定では `output\selected` を見本に、`output\raw` 以下の生成画像を検索する。
見本は画像ファイルを複数選ぶか、フォルダーを指定して切り替えられる。
「類似度順に並べる」を押すと、各候補に最も似ている見本との CLIP 類似度で並ぶ。
見本自体は検索結果から除外される。サムネイルをクリックすると選択・解除でき、ダブルクリックすると元画像が開く。
画像一覧はマウスホイールやスクロールバーで縦に移動できる。ウィンドウの幅に応じて列数が変わる。
「サムネイルサイズ」のスライダー、または Ctrl を押しながらマウスホイールを上下すると、サムネイルを拡大・縮小できる。
サムネイルや画像名の横のチェック欄を使って複数選べる。スクロールやウィンドウのサイズ変更後も選択は保持される。
「移動…」から移動先を指定する。同名ファイルがある場合は、移動する画像の名前に連番が付く。
「移動して見本に追加…」を使うと、移動できた画像を既存の見本に加え、残りの候補が自動で並び直される。
「見本に追加」を押すと、選択した画像を元の場所に置いたまま見本に加え、残りの候補を並べ直せる。
追加した見本は保存され、同じ見本フォルダーまたは同じ画像の組み合わせを指定したときに再利用される。
画像にチェックを付けて「好みではない」を押すと、登録画像に似ている候補のスコアが下がり、自動で並び直される。
登録は見本の組み合わせごとに保存される。登録済みの画像だけを選ぶと「好みではない登録を解除」で取り消せる。
「登録画像を管理…」には「追加した見本」と「好みではない」のタブがある。
サムネイルを見ながらチェックを付けて「選択した登録を解除」を押すと、画像を元の場所に残したまま登録を解除できる。
「登録を初期化…」を押すと、現在の見本設定で追加した見本、「好みではない」、見本の解除履歴をまとめてリセットできる。
検索結果が表示されている場合は、管理画面を閉じると新しい登録内容で並び直される。

既存の ComfyUI の CLIP Vision モデルを使用する。初回は画像ごとに特徴量を計算し、
次回からは `logs\clip_gallery.sqlite3` のキャッシュを再利用する。

`scripts\select_images.py` が `output\raw` を走査し、上位を `output\selected` へコピーする。

```powershell
cd F:\ai-image

# スコアだけ確認（コピーなし）
.\venv\Scripts\python.exe scripts\select_images.py --dry-run --top 0.3 --require-person

# 実行（上位 30% をコピー）
.\venv\Scripts\python.exe scripts\select_images.py --top 0.3 --require-person
```

#### 評価指標

| 指標 | 重み | 内容 |
|---|---|---|
| 美的スコア | 0.75 | SigLIP ベース（aesthetic-predictor-v2-5） |
| 人体構造 | 0.15 | DWPose のキーポイント信頼度・四肢長比。左右対称性は既定で評価しない |
| ディテール | 0.10 | Canny エッジ密度が適正帯（2〜20%）に入っているか。帯内でも密度に応じて 0.5〜1.0 で連続評価 |
| 好み | 0.0（既定） | favorites から学習した好みスコア。`--w-preference` で有効化 |

人体構造は次の 4 点で減点する。

- 体幹の必須点（首・両肩）が信頼できない
- 左右対応点の体幹中心線からの距離が偏っている
- 上腕/前腕、大腿/下腿の長さ比が 0.55〜1.80 を外れる
- 可視キーポイントの信頼度平均が低い

脚や腰の欠落は減点しない。バストアップや膝上構図を不当に落とさないため、
首・両肩のみを必須とし、6 点以上見えていれば構図として十分とみなす。

#### 主な引数

| 引数 | 既定 | 内容 |
|---|---|---|
| `--top` | 0.30 | 上位何割を残すか |
| `--require-person` | off | 人物未検出の画像を除外する |
| `--dup-threshold` | 6 | pHash のハミング距離がこれ以下なら重複 |
| `--w-aesthetic` | 0.75 | 美的スコアの重み |
| `--w-anatomy` | 0.15 | 人体構造の重み |
| `--w-detail` | 0.10 | ディテールの重み |
| `--w-preference` | 0.0 | 好みモデルの重み（0 なら使わない） |
| `--pref-model` | `models\preference\pref_model.npz` | 好みモデルのパス |
| `--device` | cuda | CUDA が使えなければ自動で CPU に落ちる |
| `--dry-run` | off | コピーせずスコアだけ出す |
| `--limit` | 0 | 先頭 N 枚だけ処理（0=全部） |
| `--raw` / `--out` | `output\raw` / `output\selected` | 入出力ディレクトリ |

全件のスコアは `logs\select_YYYYMMDD_HHMMSS.csv` に残る。
閾値はこの CSV を見て後から調整する。

### 3.5 好みモデル（favorites の学習）

`favorites\` に好みの画像を入れて学習すると、選別が「自分の好み」を加味するようになる。

```powershell
cd F:\ai-image

# 1) favorites に画像を入れる（10 枚程度から始められる）
#    F:\ai-image\favorites\ へコピーするだけ

# 2) 学習（評価だけ見たいときは --dry-run）
.\venv\Scripts\python.exe scripts\train_preference.py --dry-run

# 3) 学習してモデルを保存
.\venv\Scripts\python.exe scripts\train_preference.py

# 4) 好みを加味して選別
.\venv\Scripts\python.exe scripts\select_images.py --top 0.3 --require-person `
  --w-preference 0.5 --w-aesthetic 0.3 --w-anatomy 0.15 --w-detail 0.05
```

#### 仕組み

SigLIP の埋め込み（1152 次元）上で、favorites を正例、`output\raw` と `output\selected` を負例として学習する。
美的スコアと同じモデルを使うため、追加のモデルロードは発生しない（埋め込みは約 0.1 秒/枚）。

| 方式 | 内容 | 使われる条件 |
|---|---|---|
| centroid | 中心化埋め込みと favorites 重心のコサイン類似度 | favorites が 10 枚以下では主役 |
| LogReg | 中心化埋め込みへのロジスティック回帰 | favorites が 30 枚以上で主役 |
| ハイブリッド | 10〜30 枚の間で連続的に移行 | 常に有効 |

少数データでは LogReg が fold によって崩れる（実測で AUC 0.364 まで低下）ため、
10 枚では centroid 主体、30 枚以上で LogReg 主体になるよう自動で重みを移行する。

#### 学習時に出る指標

| 指標 | 意味 |
|---|---|
| `centroid separation` | favorites と負例の平均スコア差。大きいほど分離している |
| `LOO rank` | favorites の 1 枚を抜いて採点したときの順位。**これが実際の汎化の目安** |
| `LogReg AUC` | 交差検証の AUC。少数では参考値 |

実測例（favorites 10 枚 / negatives 174 枚）: separation 5.683、LOO 中央値 26 位 / 184、AUC 0.813。

#### 注意点

- **favorites と同一内容の画像は負例から自動で除外する。** コピーして置いた場合に同じ画像が正例と負例の両方へ入り、指標が不当に高くなるのを防ぐ。
- **好みスコアはバッチ内正規化しない。** 美的・人体はバッチ内の相対順位で正規化するが、好みは絶対値（0..1）で出す。同じ画像が「一緒に処理する画像次第でスコアが変わる」のを避けるため。
- **favorites を入れ替えたら再学習が必要。** モデルは `models\preference\pref_model.npz` に保存される。
- **10 枚では「好みの方向」しか捉えられない。** 構図や色の細かい好みまで反映したいなら 30 枚以上を目安にする。

#### 最重要の注意: ドメイン交絡

**favorites と negatives は、同じ生成元・同じ形式で揃えること。**

実測で確認された問題: favorites が Twitter 由来の JPEG（904×1600）、negatives が ComfyUI 出力の PNG（832×1216）だった場合、
埋め込み空間で両群が**完全に分離**した（favorites の重心類似度 最小 0.309 / 生成画像 最大 0.148、重なりゼロ）。
リサイズして解像度を揃えても結果は変わらなかった。

これは「好みを学習した」のではなく「**画像の出所を学習した**」状態。この状態では、
自分の生成物の中から好みを選ぶ能力はほぼ無い（生成画像184枚のうち favorites の下位10%閾値を超えたのは 0 枚）。

`train_preference.py` はこれを検出して警告を出す。

```
=== ドメイン検査 ===
favorites 形式: {'JPEG': 74}  主なサイズ: [((904, 1600), 51), ((1600, 904), 21)]
negatives 形式: {'PNG': 184}  主なサイズ: [((832, 1216), 183), ((1024, 1536), 1)]
  ※ 形式またはサイズが一致しない。埋め込みが好みではなく出所を
    分離している可能性がある。スコアの絶対値は信用せず、順位だけを見てね
```

**正しい使い方:**

1. `output\raw` に自分の生成物を十分ためる（100枚以上）
2. その中から好みのものを `favorites\` へコピーする（**同じ生成元から選ぶ**）
3. 学習する

こうすれば favorites と negatives が同じ分布になり、「好み」の差だけを学習できる。
外部画像（Twitter 等）を favorites に使いたい場合は、negatives 側にも同程度の外部画像を混ぜる必要がある。

`LogReg AUC` が 1.0 に張り付いたら、まずドメイン交絡を疑うこと。

### 4. 後処理

**2026-10-31実装済み。** `quality_pipeline.py finish --image <PNG> --face` で
元画像の生成条件を継承し、1.5倍拡大・低denoise再生成・任意の顔補修を実行する。
出力は `output\upscaled`、処理のAPIグラフは `output\quality_20261031\finishing`。
以下の4倍アップスケーラ案は未採用の旧案。

1. FaceDetailer で顔を描き直す
2. 4x アップスケーラで拡大
3. 目標サイズへ縮小して `output\upscaled` へ

アップスケール込みだと 1 枚 20MB 超になる。1000 枚で 20GB 程度を見込む。

### 5. 夜間バッチ

段ごとに分けて実行する。一気にやると失敗時に全部やり直しになる。

```powershell
cd F:\ai-image

# 1) 生成
.\venv\Scripts\python.exe scripts\generate_images.py --limit 30 --batch 4

# 2) 選別
.\venv\Scripts\python.exe scripts\select_images.py --top 0.25 --require-person

# 3) 後処理（実装後）
```

各段の成果物はディレクトリで分かれているので、途中で失敗しても
成功した段から再開できる。

### 6. トラブルシューティング

| 症状 | 原因 | 対処 |
|---|---|---|
| `get-model.ps1` が `param` でエラー | `param()` より前に文があると PowerShell が `param` をコマンド名として解釈する | `param()` をファイル先頭に置く |
| ダウンロードが 0MB のまま進まない | `Invoke-WebRequest` が数 GB をメモリに溜め込む | `curl.exe` を使う（現行実装は対応済み） |
| 選別が極端に遅い | ONNX Runtime が CPU に落ちている | `onnxruntime-gpu==1.22.0` + `nvidia-cudnn-cu12` + `nvidia-cublas-cu12` を入れる |
| 人体スコアが全件 0.0 | DWPose の戻り値の形の誤解 | `candidate` は `(N*18, 2)`、`score` は `(N, 18)`。`(N, 18, 2)` に reshape する |
| 無関係な画像が重複扱いされる | 一様な画像の pHash が退化値になる | 退化ハッシュを重複判定から除外する（対応済み） |
| ノイズ画像がディテール最上位 | ラプラシアン分散はノイズに弱い | Canny エッジ密度で評価する（対応済み） |
| ディテールが全件 1.0 で同点 | 適正帯を一律満点にしていた | 帯内を 0.5〜1.0 に線形写像する（対応済み） |
| 生成スクリプトの日本語出力が化ける | Python が CP932 で書く | `sys.stdout.reconfigure(encoding="utf-8")` を入れる（対応済み） |
| `/prompt` が 400 を返す | ノード入力の検証エラー（ckpt 名違いなど） | エラー本文の `node_errors` を読む。スクリプトは要点だけ表示する |
| 生成が `TimeoutError` | 1 枚の所要時間が `--timeout` を超えた | `--timeout` を延ばす。VRAM 不足なら他プロセスを止める |

CUDA プロバイダが有効かは次のコマンドで確認できる。

```powershell
.\venv\Scripts\python.exe -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

`CUDAExecutionProvider` が含まれていれば GPU で動く。実測では 412.7ms → 23.8ms（約 17 倍）。

### 7. 運用上の注意

- 人体スコアは「破綻の検出」であって「正しさの保証」ではない。大振りなポーズや強い遠近は誤検出しうる。
- 最初は美的寄りの重みで回し、破綻が混ざるようなら `--w-anatomy` を上げる。
- 選別結果は必ず目視で数十枚確認する。スコアだけで判断しない。
- Civitai のモデルは商用不可・再配布不可が多い。公開する場合はライセンスを確認する。
- WAI v17 は API からライセンス情報が取得できなかった。モデルページで直接確認すること。
