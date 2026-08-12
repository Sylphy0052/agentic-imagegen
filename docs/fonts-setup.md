# フォントのセットアップ

テキスト合成で使うフォントは `fonts/` へ置く。フォントごとに再配布の可否が異なるため、
リポジトリには含めない (`.gitignore` で追跡対象から外してある)。

## 置き方

```text
fonts/
  NotoSansJP-Regular.ttf
  NotoSansJP-Bold.ttf
  noto/               # サブフォルダは1階層まで使える
    NotoSerifJP-Regular.otf
```

Spec からはファイル名で参照する。サブフォルダに置いた場合は `noto/NotoSerifJP-Regular.otf`
のように書く。

```yaml
text:
  layers:
    - content: 夜の街
      font: NotoSansJP-Bold.ttf
      size: 72
```

受け付ける拡張子は `.ttf` / `.otf` / `.ttc`。探索ルートは環境変数
`IMAGEGEN_FONTS_ROOT` で変更できる (既定は `fonts`)。

## 入手する

再配布条件が緩く、日本語の字形が揃っているものとして Noto Sans JP を推奨する。

```bash
mkdir -p fonts
curl -L -o /tmp/noto-sans-jp.zip \
  https://fonts.google.com/download?family=Noto%20Sans%20JP
unzip -j /tmp/noto-sans-jp.zip 'static/NotoSansJP-Regular.ttf' \
  'static/NotoSansJP-Bold.ttf' -d fonts/
```

配布形態は変わることがあるため、展開後に `fonts/` の中身を確認する。

## 環境にあるフォントを使う

新しく入手せず、環境にあるものを写して使ってもよい。

```bash
# WSL / Linux 側にあるフォントを探す
fc-list :lang=ja file family

# 例: IPAゴシックを使う
cp /usr/share/fonts/opentype/ipafont-gothic/ipag.ttf fonts/
```

WSL から Windows 側のフォントを使う場合は `/mnt/c/Windows/Fonts/` にある。

```bash
# 例: BIZ UDゴシック
cp /mnt/c/Windows/Fonts/BIZ-UDGothicR.ttc fonts/
```

`.ttc` は複数の書体をまとめた形式なので、どの書体を使うかを `font_index` で指定する
(既定は 0)。

```yaml
text:
  layers:
    - content: 夜の街
      font: BIZ-UDGothicR.ttc
      font_index: 0
      size: 72
```

写して使う場合は、そのフォントのライセンスが用途を許すかを確認する。とくに生成した
画像を配布する場合は、埋め込みや商用利用の条件を読むこと。

## 確認する

指定したフォントが見つからない場合、`imagegen` は別の書体へ代替せず exit code 10 で
失敗し、`fonts/` 配下にある候補を表示する。意図しない書体で出力されるより、
その場で止める方が扱いやすいため。

```bash
uv run imagegen compose inputs/base.png specs/generated/caption.yaml
# フォントが見つかりません: Missing.ttf
#   探索ルート: fonts
#   利用できるフォント: NotoSansJP-Bold.ttf / NotoSansJP-Regular.ttf
```

探索ルートは作業ルート配下なら `fonts` のように相対パスで表示する。
`IMAGEGEN_FONTS_ROOT` で作業ルートの外を指した場合だけ絶対パスのまま表示する。
