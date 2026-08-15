#!/usr/bin/env python3
"""evals.jsonのcaseを1件ずつ、まっさらなセッションで回す。

    python3 evals/run_case.py inventory-check-uses-catalog
    python3 evals/run_case.py --all

`claude -p` は毎回新しいセッションとして立ち上がるため、READMEの
「1 caseにつき1セッションで回す」を機械的に満たせる。判定そのものはここでは行わない。
応答と、その過程で叩こうとしたコマンドを `evals/results/` へ落とすところまでが仕事。

既定では読み取りとskillの発火しか許さない。生成やファイル書き込みを無人で走らせないための
制約で、この条件では**叩いたコマンドは実行ログではなく応答本文の宣言として観測する**。
コマンドの実行結果が無いと最後まで判定できないcaseだけ、`shell` でそのコマンドを
名指しで許す。直前のやり取りが要るcaseは `context` でその前提を渡す。

台帳は `evals/fixtures/registry` を見せる。実運用の `registry/` は空だったり中身が
変わったりするため、caseの判定が環境に左右される。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

EVALS = Path(__file__).resolve().parent
REPOSITORY = EVALS.parent
FIXTURES = EVALS / "fixtures"

# 読み取りとskillの発火だけを許す。生成も書き込みもさせない。
ALLOWED_TOOLS = "Read,Glob,Grep,Skill"

TIMEOUT_SECONDS = 900

# queryは書き換えずに投げる (READMEの「queryをそのまま投げる」)。
# 実行だけを止めたいので、制約はsystem prompt側へ足す。
CONSTRAINTS = """\
これはskillの判断を確認するための評価実行です。画像は生成しません。

- コマンドは実行しないでください。実行するはずのコマンドは本文へそのまま書き出してください。
- Specはファイルへ保存せず、本文へYAMLとして示してください。
- 参照した文書とpresetの名前は本文へ明記してください。
- 対話はできません。要求が曖昧で質問すべきだと判断した場合は、その質問を書いて終えてください。
"""

# `shell` を持つcaseだけへ足す。許したコマンドは実際に叩かせ、結果を見て判断させる。
# 許可は前方一致で判定されるため、呼び方がずれると拒否される。形をそのまま見せる。
SHELL_CONSTRAINT = """
このcaseでは次のコマンドだけ実行を許可しています。許可されたコマンドは実行してください。
書かれている形のまま呼んでください (オプションを前へ挟むと許可から外れます)。

{patterns}

許可されていないコマンドはこれまでどおり書き出すだけにしてください。
"""


def allowed_tools(case: dict[str, Any]) -> str:
    """このcaseで許すツール。`shell` は名指しのパターンだけを受ける。"""
    patterns = list(case.get("shell", ()))
    for pattern in patterns:
        # `Bash` をそのまま許すと生成コマンドまで通る。括弧付きのパターンだけを受ける。
        if not pattern.startswith("Bash(") or not pattern.endswith(")"):
            raise ValueError(f"{case['id']}: shellは Bash(<コマンド>:*) の形で書く: {pattern}")
    return ",".join([ALLOWED_TOOLS, *patterns])


def _command_of(pattern: str) -> str:
    """`Bash(<コマンド>:*)` からコマンドの部分だけを取り出す。"""
    inner = pattern[len("Bash(") : -1]
    return inner[: -len(":*")] if inner.endswith(":*") else inner


def system_prompt(case: dict[str, Any]) -> str:
    parts = [CONSTRAINTS]
    patterns = list(case.get("shell", ()))
    if patterns:
        shown = "\n".join(f"- `{_command_of(pattern)}`" for pattern in patterns)
        parts.append(SHELL_CONSTRAINT.format(patterns=shown))
    context = case.get("context")
    if context:
        parts.append(
            f"\nこのセッションの直前に、次のことがあったものとして扱ってください。\n\n{context}\n"
        )
    return "".join(parts)


def environment() -> dict[str, str]:
    """caseの判定が実運用のデータに左右されないよう、台帳はフィクスチャを見せる。"""
    return {**os.environ, "IMAGEGEN_REGISTRY_ROOT": str(FIXTURES / "registry")}


def load_cases() -> dict[str, dict[str, Any]]:
    payload = json.loads((EVALS / "evals.json").read_text(encoding="utf-8"))
    return {case["id"]: case for case in payload["cases"]}


def build_command(case: dict[str, Any]) -> list[str]:
    """`claude -p` の引数。許可の指定が効く条件まで含めて組み立てる。

    利用者のグローバル設定は `defaultMode: bypassPermissions` のことがあり、
    そのままだと `--allowedTools` が素通りして生成コマンドまで通る。
    設定の読み込み元をプロジェクトへ限り、権限モードも明示する。
    """
    return [
        "claude",
        "-p",
        case["query"],
        "--output-format",
        "json",
        "--permission-mode",
        "default",
        "--setting-sources",
        "project",
        "--allowedTools",
        allowed_tools(case),
        "--append-system-prompt",
        system_prompt(case),
    ]


def run(case: dict[str, Any]) -> dict[str, Any]:
    command = build_command(case)
    completed = subprocess.run(  # noqa: S603
        command,
        cwd=REPOSITORY,
        env=environment(),
        capture_output=True,
        text=True,
        timeout=TIMEOUT_SECONDS,
        check=False,
    )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {"is_error": True, "result": completed.stdout + completed.stderr}
    return payload if isinstance(payload, dict) else {"is_error": True, "result": completed.stdout}


def write_result(outdir: Path, case: dict[str, Any], payload: dict[str, Any]) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{case['id']}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    cost = payload.get("total_cost_usd") or 0.0
    seconds = (payload.get("duration_ms") or 0) / 1000
    lines = [
        f"# {case['id']}",
        "",
        f"- skill: `{case['skill']}`",
        f"- query: {case['query']}",
        f"- 所要: {cost:.2f} USD / {seconds:.0f}秒",
    ]
    if case.get("shell"):
        lines.append(f"- 許可したコマンド: {', '.join(case['shell'])}")
    if case.get("context"):
        lines.append(f"- 前提: {case['context']}")
    lines += ["", "## 応答", "", str(payload.get("result", ""))]
    path = outdir / f"{case['id']}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="evalのcaseを新しいセッションで回す")
    parser.add_argument("case_id", nargs="*", help="回すcaseのid")
    parser.add_argument("--all", action="store_true", help="全caseを回す")
    parser.add_argument("--outdir", default=None, help="出力先 (既定: evals/results/latest)")
    args = parser.parse_args()

    cases = load_cases()
    targets = list(cases) if args.all else list(args.case_id)
    if not targets:
        parser.error("caseのidか --all を指定する")
    unknown = [name for name in targets if name not in cases]
    if unknown:
        parser.error(f"evals.jsonに無いcase: {', '.join(unknown)}")

    outdir = Path(args.outdir) if args.outdir else EVALS / "results" / "latest"
    for name in targets:
        print(f"--- {name}", flush=True)
        payload = run(cases[name])
        path = write_result(outdir, cases[name], payload)
        cost = payload.get("total_cost_usd") or 0.0
        print(f"    {path} ({cost:.2f} USD)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
