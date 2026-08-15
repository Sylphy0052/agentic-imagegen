# 画像生成の失敗切り分け

exit codeごとの原因と対処。生成コマンドが失敗したときはここを見る。

codeと例外クラスの対応は [CLAUDE.mdの「exit code」](../../../../CLAUDE.md#exit-code) を参照。
ここには対処だけを書く。

## code 2: Specが不正

メッセージにフィールド名が出る。よくある原因:

- `width` / `height` が8の倍数でない
- `checkpoint` がComfyUIに存在しない。`uv run imagegen catalog` の `checkpoints` と突き合わせる
- `checkpoint` にPath Traversal (`..` / 絶対パス / 2階層以上のサブフォルダ) が含まれる
- preset名が見つからない。`uv run imagegen catalog` の `presets/*` で確認する
- preset名に使えない文字が入っている。英数字始まりで `[A-Za-z0-9._-]` のみ
- `presets:` を書いたのに探索ルートが渡っていない。CLI経由なら通常起きない
- 解像度やbatch_sizeが `IMAGEGEN_MAX_*` の上限を超えている

Specを直して `validate` からやり直す。検証を緩めて通すことはしない。

## code 3: ComfyUIへ到達できない

```bash
uv run imagegen health
```

- 生成は `scripts/comfyui-session.sh generate <spec>` で行う (起動と停止を含む)
- 手で起動する場合: `cd ~/ComfyUI && ./.venv/bin/python main.py --listen 127.0.0.1 --port 8188`
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

## code 5: 投入が拒否された (ControlNet / IPAdapter)

ComfyUIがWorkflowの投入を拒否した。カスタムノードとモデルの有無をまず疑う。

- IPAdapterを使う場合、[ComfyUI_IPAdapter_plus](https://github.com/cubiq/ComfyUI_IPAdapter_plus)
  が要る。未導入だと `IPAdapterModelLoader` などのノード自体が存在しない。
  `uv run python -c "..."` ではなくMCPの `list_ipadapters` で確認できる (空なら未導入)
- モデル名が実在しないと拒否される。`uv run imagegen catalog` の
  `controlnets` / `ipadapters` / `clip_visions` で確認する。ただし `Backend:` が
  `filesystem` の場合はカスタムノードの導入有無まで分からないため、
  `scripts/comfyui-session.sh catalog` で見直す
- IPAdapterモデルとCLIP Visionには対応関係がある。`ip-adapter-plus_sd15` には
  ViT-H (`CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors`) を使う
- checkpointのアーキテクチャとも対応が要る。SD1.5用のIPAdapterをSDXLのcheckpointへ
  かけると次元が合わずに失敗する

## code 6: タイムアウト

CPU推論では時間がかかる。まず `imagegen health` の `Devices:` を見る。

- `cpu` になっている場合、XPUが使えるなら切り替える ([docs/xpu-setup.md](../../../../docs/xpu-setup.md))
- `IMAGEGEN_TIMEOUT` を伸ばす (CPUなら1200程度)
- `steps` を下げる、解像度を512x512へ落とす
- SDXL / Illustrious系はCPUだと数十分かかる。常用しない

**設定を変えていないのに急に時間切れするようになった場合は、ComfyUIのプロセスを疑う。**
生成中にComfyUIがOOMで落ちても、待っている側からは応答が返らないだけに見えるため、
遅いのか死んだのかを区別できない。

```bash
ps -eo pid,rss,etime,args | grep "[C]omfyUI/main.py"
free -h
```

プロセスが消えていれば落ちている。生きていてもswapを使い切っていると生成は事実上進まない。
2026-08-13の実測では、swapが満杯の状態で512x768 / 10 stepsが900秒でも完了せず、
ComfyUIを再起動した後は同じSpecが約100秒で完了した。長時間動かし続けたComfyUIは
メモリを抱えたままになるため、再起動して作り直す。

## code 7: ComfyUI側で実行が失敗した

ComfyUIの起動ログを見る。よくある原因:

- メモリ不足。`batch_size` を1に、解像度を512x512へ下げる。WSLの割当メモリも確認する
- checkpointのアーキテクチャが `workflows/txt2img.json` の構成と合っていない
  (DiT系など)。SD1.5 / SDXL系を使う

## code 9: 環境変数が不正

`IMAGEGEN_MAX_*` は1以上の整数、`IMAGEGEN_MAX_BATCH` は4以下、
`COMFYUI_BASE_URL` は `http://` または `https://` 始まり。
`IMAGEGEN_OUTPUT_ROOT` / `IMAGEGEN_PRESETS_ROOT` / `IMAGEGEN_FONTS_ROOT` に空文字は指定できない。

## code 10: テキスト合成に失敗した

`imagegen compose`、または `generate` に `text:` を含むSpecを渡したときに発生する。
生成そのものは完了しているので、生成した画像 (`image_XXXX.png`) は残る。

- フォントが見つからない場合、メッセージに探索ルートと利用できる候補が列挙される
  - `fonts/` (または `IMAGEGEN_FONTS_ROOT`) の中身を確認する: `ls fonts/`
  - フォントを配置していないなら [docs/fonts-setup.md](../../../../docs/fonts-setup.md) の手順で置く
  - `.ttc` を使う場合は `font_index` がコレクション内の索引と合っているか確認する
- 出力先が既に存在する場合は上書きせず失敗する。`--output/-o` で別のパスを指定するか、
  既存ファイルを退避してから再実行する
- 出力先が作業ルートの外を指している場合も失敗する (`--output/-o` は作業ルート配下のみ)
- 画像が大きすぎる場合は `IMAGEGEN_MAX_PIXELS` の上限に収まるよう縮小するか、
  上限自体を緩める
- `batch_size` > 1の生成では、失敗した時点で後続は処理しない。
  それまでに成功した分は `metadata.json` の `text.outputs` に残る
  (スキーマの詳細は
  [docs/spec-reference.md](../../../../docs/spec-reference.md#metadatajson) を参照)

## 画像は出たが意図と違う

- promptが長すぎる可能性がある。presetを重ねすぎていないか確認する
  (重複トークンは自動で除去されるが、総量は減らない)
- `metadata.json` の `spec.prompt.positive` に展開後のpromptが記録されている。
  実際にComfyUIへ渡った内容はここで確認する
- 同じ画を再現するには `metadata.json` の `resolved_seed` をSpecの `seed` へ書き戻す
- 以前と同じSpecなのに結果が変わった場合は `metadata.json` の `workflow_hash` を
  前回の結果と比べる。値が違えばWorkflowテンプレート自体が変わっている
- `metadata.json` の `backend` で実行基盤 (ComfyUI版とデバイス) を確認できる。
  XPUとCPUではfp16とfp32の違いがあり、同じseedでも完全には一致しない
- 構図だけ変えたい場合はscene presetを差し替える。character presetは変えない
- 同一キャラクタのはずが別人になる場合は
  [character-consistency.md](character-consistency.md) の手順を使う。
  presetだけでは顔立ちまでは固定できない
- IPAdapterを指定したのに参照画像が効かない場合、`metadata.json` の `workflow` が
  `*_ipadapter` になっているか確認する。`reference` の書き忘れでは
  テンプレート自体が切り替わらない
