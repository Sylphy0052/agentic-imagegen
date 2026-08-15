"""diffusers (プロセス内推論) バックエンド。

ComfyUIアダプタと同じく、ここから外へは特定バックエンドの事情を出さない。
torch / diffusers は optional-dependencies の `diffusers` extra であり、
未インストールの環境でもパッケージ全体が読み込めるよう、重いimportは
実際に使う関数の中で行う。
"""

from __future__ import annotations
