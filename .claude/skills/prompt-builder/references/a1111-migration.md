# A1111から設定を移す

A1111 (Stable Diffusion web UI) で気に入った絵ができているなら、その設定をSpecへ写せば
同じ絵柄を再現できる。実際に4枚の生成画像から設定を復元し、絵柄が一致するところまで
確認した手順を残す。

## 生成画像から設定を読む

A1111はPNGのテキストチャンクへ生成パラメータを丸ごと書き込む。

```bash
python3 -c "from PIL import Image; print(Image.open('inputs/ref.png').info['parameters'])"
```

読めるのは positive / negative / Steps / Sampler / Schedule type / CFG scale / Seed / Size /
Model hash / Model / VAE hash / VAE / Denoising strength / Clip skip / Hires系の各値、
それにA1111のVersion。ここに無い項目は既定値だったということ。

- **LoRA・ADetailer・ControlNetを使っていれば必ず記録される。** 記載が無ければ使っていない
- **`RNG` と `ENSD` が無ければ既定値。** 設定を変えた場合だけ書き込まれる
- **Hires系の値 (`Hires upscale` / `Hires steps` / `Hires upscaler`) が無ければhires fixは未使用**

## モデルの同一性をハッシュで確かめる

同じファイル名でも中身が違うことがある。A1111が記録する `Model hash` / `VAE hash` は
AutoV2 (ファイル全体のsha256の先頭10桁) のため、手元のファイルと直接照合できる。

```bash
sha256sum ~/ComfyUI/models/vae/*.safetensors | cut -c1-10
```

一致しなければ別のファイル。civitaiは
`https://civitai.com/api/v1/model-versions/by-hash/<sha256先頭10桁>` でハッシュから
元のファイルを引けるため、そこから正しいものを落とす。

実例として、手元の `kl-f8-anime2.ckpt` (`df3c506e51`) は参考画像が使っていた
`vaeKlF8Anime2_klF8Anime2VAE.safetensors` (`b8821a5d58`) とは別のファイルだった。
名前が同じ系統でも中身が違い、これが絵柄の差の主因だった。

## Specへ写す

sampler / scheduler / cfg / steps とプロンプト、clip skipとVAEはstyle presetへ入れられる。
hires fixだけはpresetに書く場所が無いため、Spec側に書く。

| A1111の項目 | Specの書き場所 |
| --- | --- |
| Sampler / Schedule type / CFG scale / Steps | style presetの `generation` (またはSpecの `generation`) |
| Clip skip | style presetの `model.clip_skip` (またはSpecの `model.clip_skip`) |
| VAE | style presetの `model.vae` (またはSpecの `model.vae`) |
| Hires upscaler / upscale / steps / Denoising strength | `generation.upscale` |

`model.clip_skip` はどちらにも無いと1相当になり、2を前提にしたアニメ系モデルでは
絵柄が変わる。
書き方をまとめた例は
[specs/examples/txt2img_a1111_compat.yaml](../../../../specs/examples/txt2img_a1111_compat.yaml) にある。

Textual Inversionのnegative embedding (`negativeXL_D` など) は
プロンプト中へ `embedding:negativeXL_D` と書く
([common.md](common.md#textual-inversion-embedding)を参照)。

## 揃うものと揃わないもの

上をすべて合わせると絵柄・塗り・線は一致する。構図とポーズは同じseedでも一致しない。

- **初期ノイズの生成器が違う。** A1111はCPU、ComfyUIはGPUでノイズを作るため、
  同じseedでもノイズのパターン自体が別物になる
- **プロンプトの重み付けの正規化方式が違う。** `(word:1.2)` の効き方が完全には一致しない
- **75トークンを超えた分の連結方法が違う。** 長いプロンプトほど差が出る

構図まで含めて同じ絵が要るなら、A1111側の出力をそのまま使う。
ComfyUI側では絵柄を合わせたうえでseedを振り直し、好みの構図を選び直す。
