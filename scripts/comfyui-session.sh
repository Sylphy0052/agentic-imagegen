#!/usr/bin/env bash
# ComfyUIを起動し、生成コマンドを1回実行し、必ず停止する。
#
# 常駐させたままモデルを何本も切り替えるとXPUのアロケータが断片化し、
# 空き容量が十分にあっても数十MiBの確保に失敗するようになる (exit 7 / XPU out of memory)。
# 1回の生成ごとにプロセスを立て直せばこれを避けられる。
#
# 使い方:
#   scripts/comfyui-session.sh generate specs/generated/foo.yaml
#   scripts/comfyui-session.sh batch specs/generated/a.yaml specs/generated/b.yaml
#   scripts/comfyui-session.sh catalog
#
# 環境変数:
#   COMFYUI_HOME      ComfyUIの場所 (既定: ~/ComfyUI)
#   COMFYUI_BASE_URL  接続先 (既定: http://127.0.0.1:8188)
#   IMAGEGEN_TIMEOUT  生成のタイムアウト秒 (既定: 2400)
#   COMFYUI_BOOT_TIMEOUT  起動待ちの上限秒 (既定: 300)
#   COMFYUI_LOG_DIR   起動ログの置き場 (既定: ~/.cache/imagegen-logs)
#   COMFYUI_EXTRA_ARGS  ComfyUIへ渡す追加の起動引数 (既定: なし)
#                       モデルのdtypeを切り替えて破綻を切り分けるときに使う
#                       (例: COMFYUI_EXTRA_ARGS="--bf16-unet")
#
# 出力を head などパイプの読み手が先に閉じるコマンドへ繋がない。
# SIGPIPEで落ちてもComfyUIを残さないようにはしてあるが、途中経過が切れて
# 実行結果を読み違える。全部見たいならファイルへリダイレクトする。
#
# 既に起動しているComfyUIがある場合はそれを使い、停止もしない
# (手動で立ち上げて作業している最中に落とさないため)。

set -euo pipefail

COMFYUI_HOME="${COMFYUI_HOME:-$HOME/ComfyUI}"
COMFYUI_BASE_URL="${COMFYUI_BASE_URL:-http://127.0.0.1:8188}"
IMAGEGEN_TIMEOUT="${IMAGEGEN_TIMEOUT:-2400}"
COMFYUI_BOOT_TIMEOUT="${COMFYUI_BOOT_TIMEOUT:-300}"
COMFYUI_LOG_DIR="${COMFYUI_LOG_DIR:-$HOME/.cache/imagegen-logs}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMFYUI_PID=""

usage() {
    echo "usage: $(basename "$0") <generate|batch|validate|health|catalog> [args...]" >&2
    exit 2
}

is_up() {
    curl -sf --max-time 3 "${COMFYUI_BASE_URL}/system_stats" > /dev/null 2>&1
}

stop_comfyui() {
    [ -n "$COMFYUI_PID" ] || return 0
    kill -0 "$COMFYUI_PID" 2> /dev/null || return 0

    # 先に止める。head などパイプの読み手が閉じている場合、
    # ここでstderrへ書くと再びSIGPIPEを受けてkillへ到達しないまま落ちる。
    kill -TERM "$COMFYUI_PID" 2> /dev/null || true
    echo "[comfyui-session] stopping ComfyUI (pid ${COMFYUI_PID})" 2> /dev/null >&2 || true
    for _ in $(seq 1 30); do
        kill -0 "$COMFYUI_PID" 2> /dev/null || return 0
        sleep 1
    done

    echo "[comfyui-session] SIGTERM did not stop it, sending SIGKILL" >&2
    kill -KILL "$COMFYUI_PID" 2> /dev/null || true
}

start_comfyui() {
    [ -x "${COMFYUI_HOME}/.venv/bin/python" ] || {
        echo "[comfyui-session] ${COMFYUI_HOME}/.venv/bin/python が無い (COMFYUI_HOME を確認する)" >&2
        exit 3
    }

    mkdir -p "$COMFYUI_LOG_DIR"
    local log="${COMFYUI_LOG_DIR}/comfyui.log"
    echo "[comfyui-session] starting ComfyUI (log: ${log})" >&2

    local extra_args=()
    if [ -n "${COMFYUI_EXTRA_ARGS:-}" ]; then
        read -r -a extra_args <<< "$COMFYUI_EXTRA_ARGS"
        echo "[comfyui-session] extra args: ${extra_args[*]}" >&2
    fi

    (cd "$COMFYUI_HOME" && exec ./.venv/bin/python main.py --listen 127.0.0.1 --port 8188 "${extra_args[@]}") \
        > "$log" 2>&1 < /dev/null &
    COMFYUI_PID=$!
    trap stop_comfyui EXIT INT TERM HUP PIPE

    for _ in $(seq 1 "$COMFYUI_BOOT_TIMEOUT"); do
        if is_up; then
            echo "[comfyui-session] ComfyUI is up (pid ${COMFYUI_PID})" >&2
            return 0
        fi
        kill -0 "$COMFYUI_PID" 2> /dev/null || {
            echo "[comfyui-session] ComfyUIが起動途中で終了した。${log} を見る" >&2
            exit 3
        }
        sleep 1
    done

    echo "[comfyui-session] ${COMFYUI_BOOT_TIMEOUT}秒待っても起動しない。${log} を見る" >&2
    exit 3
}

[ $# -ge 1 ] || usage
case "$1" in
    generate | batch | validate | health | catalog) ;;
    *) usage ;;
esac

if is_up; then
    echo "[comfyui-session] 既に起動しているComfyUIを使う (このスクリプトでは停止しない)" >&2
else
    start_comfyui
fi

cd "$PROJECT_ROOT"
IMAGEGEN_TIMEOUT="$IMAGEGEN_TIMEOUT" COMFYUI_BASE_URL="$COMFYUI_BASE_URL" \
    uv run imagegen "$@"
