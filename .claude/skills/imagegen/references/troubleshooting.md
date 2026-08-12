# 画像生成の失敗切り分け

exit codeごとの原因と対処。生成コマンドが失敗したときはここを見る。

## exit code 一覧

| code | 例外 | 意味 |
| --- | --- | --- |
| 0 | - | 成功 |
| 1 | - | 想定外の内部エラー |
| 2 | `InvalidGenerationSpec` | Specが不正 |
| 3 | `ComfyUIUnavailable` | ComfyUIへ到達できない |
| 4 | `WorkflowValidationError` | Workflowテンプレートが不正 |
| 5 | `WorkflowSubmissionError` | Workflowの投入が拒否された |
| 6 | `GenerationTimeout` | 生成がタイムアウトした |
| 7 | `GenerationFailed` | ComfyUI側で実行が失敗した |
| 8 | `OutputNotFound` | 出力画像が見つからない |
| 9 | `InvalidConfiguration` | 環境変数の設定値が不正 |

## code 2: Specが不正

メッセージにフィールド名が出る。よくある原因:

- `width` / `height` が8の倍数でない
- `checkpoint` がComfyUIに存在しない。`ls ~/ComfyUI/models/checkpoints/` と突き合わせる
- `checkpoint` にPath Traversal (`..` / 絶対パス / 2階層以上のサブフォルダ) が含まれる
- preset名が見つからない。`ls presets/characters presets/scenes presets/styles` で確認する
- preset名に使えない文字が入っている。英数字始まりで `[A-Za-z0-9._-]` のみ
- `presets:` を書いたのに探索ルートが渡っていない。CLI経由なら通常起きない
- 解像度やbatch_sizeが `IMAGEGEN_MAX_*` の上限を超えている

Specを直して `validate` からやり直す。検証を緩めて通すことはしない。

## code 3: ComfyUIへ到達できない

```bash
uv run imagegen health
```

- 未起動なら起動する: `cd ~/ComfyUI && ./.venv/bin/python main.py --listen 127.0.0.1 --port 8188`
- ポートを変えている場合は `COMFYUI_BASE_URL` を合わせる
- 起動直後はモデル一覧の準備中で失敗することがある。数秒待って再試行する

## code 4 / 5: Workflowテンプレート由来

テンプレートのノード構成が想定と違う。`workflows/README.md` の手順で書き出し直す。
**テンプレートを手で編集して辻褄を合わせない。**

`workflows/txt2img.json` が期待するノード構成:

| Node ID | class_type |
| --- | --- |
| 3 | KSampler |
| 4 | CheckpointLoaderSimple |
| 5 | EmptyLatentImage |
| 6 | CLIPTextEncode (positive) |
| 7 | CLIPTextEncode (negative) |
| 8 | VAEDecode |
| 9 | SaveImage |

## code 6: タイムアウト

CPU推論では時間がかかる。まず `imagegen health` の `Devices:` を見る。

- `cpu` になっている場合、XPUが使えるなら切り替える ([docs/xpu-setup.md](../../../../docs/xpu-setup.md))
- `IMAGEGEN_TIMEOUT` を伸ばす (CPUなら1200程度)
- `steps` を下げる、解像度を512x512へ落とす
- SDXL / Illustrious系はCPUだと数十分かかる。常用しない

## code 7: ComfyUI側で実行が失敗した

ComfyUIの起動ログを見る。よくある原因:

- メモリ不足。`batch_size` を1に、解像度を512x512へ下げる。WSLの割当メモリも確認する
- checkpointのアーキテクチャが `workflows/txt2img.json` の構成と合っていない
  (DiT系など)。SD1.5 / SDXL系を使う

## code 9: 環境変数が不正

`IMAGEGEN_MAX_*` は1以上の整数、`IMAGEGEN_MAX_BATCH` は4以下、
`COMFYUI_BASE_URL` は `http://` または `https://` 始まり。
`IMAGEGEN_OUTPUT_ROOT` / `IMAGEGEN_PRESETS_ROOT` に空文字は指定できない。

## 画像は出たが意図と違う

- prompt が長すぎる可能性がある。presetを重ねすぎていないか確認する
  (重複トークンは自動で除去されるが、総量は減らない)
- `metadata.json` の `spec.prompt.positive` に展開後のpromptが記録されている。
  実際にComfyUIへ渡った内容はここで確認する
- 同じ画を再現するには `metadata.json` の `resolved_seed` をSpecの `seed` へ書き戻す
- 以前と同じSpecなのに結果が変わった場合は `metadata.json` の `workflow_hash` を
  前回の結果と比べる。値が違えばWorkflowテンプレート自体が変わっている
- `metadata.json` の `backend` で実行基盤 (ComfyUI版とデバイス) を確認できる。
  XPUとCPUではfp16とfp32の違いがあり、同じseedでも完全には一致しない
- 構図だけ変えたい場合は scene preset を差し替える。character presetは変えない
