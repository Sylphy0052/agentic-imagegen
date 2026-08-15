#!/usr/bin/env python3
"""evals.jsonのcaseを1件ずつ、まっさらなセッションで回す。

    python3 evals/run_case.py inventory-check-uses-catalog
    python3 evals/run_case.py --all

`claude -p` は毎回新しいセッションとして起動するため、READMEの
「1 caseにつき1セッションで回す」を機械的に満たせる。判定そのものはここでは行わない。
応答と、その過程で叩こうとしたコマンドを `evals/results/` へ落とすところまでが仕事。

実行環境の制約として、生成とファイル書き込みは禁じてある (allowedToolsは読み取りのみ)。
そのため「叩いたコマンド」は実行ログではなく、応答本文の宣言として観測する。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

EVALS = Path(__file__).resolve().parent
REPOSITORY = EVALS.parent

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


def load_cases() -> dict[str, dict[str, Any]]:
    payload = json.loads((EVALS / "evals.json").read_text(encoding="utf-8"))
    return {case["id"]: case for case in payload["cases"]}


def run(case: dict[str, Any]) -> dict[str, Any]:
    command = [
        "claude",
        "-p",
        case["query"],
        "--output-format",
        "json",
        "--allowedTools",
        ALLOWED_TOOLS,
        "--append-system-prompt",
        CONSTRAINTS,
    ]
    completed = subprocess.run(  # noqa: S603
        command,
        cwd=REPOSITORY,
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
        "",
        "## 応答",
        "",
        str(payload.get("result", "")),
    ]
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
