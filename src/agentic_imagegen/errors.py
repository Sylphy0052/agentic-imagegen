"""アプリケーション例外とCLI exit codeの対応。

内部例外をそのままユーザーへ露出させず、原因を特定できる粒度で分類する。
exit codeは docs/plan/phase1.md 6.6節の表に対応する。
"""

from __future__ import annotations


class ImageGenError(Exception):
    """本アプリケーションの基底例外。"""

    exit_code: int = 1


class InvalidGenerationSpec(ImageGenError):
    """GenerationSpecの読み込みまたは検証に失敗した。"""

    exit_code = 2


class ComfyUIUnavailable(ImageGenError):
    """ComfyUIサーバへ到達できない。"""

    exit_code = 3


class WorkflowValidationError(ImageGenError):
    """Workflowテンプレートの構造が想定と一致しない。"""

    exit_code = 4


class WorkflowSubmissionError(ImageGenError):
    """ComfyUIへのWorkflow投入が拒否された。"""

    exit_code = 5


class GenerationTimeout(ImageGenError):
    """生成が制限時間内に完了しなかった。"""

    exit_code = 6


class GenerationFailed(ImageGenError):
    """ComfyUI側で実行が失敗した。"""

    exit_code = 7


class OutputNotFound(ImageGenError):
    """実行は完了したが出力画像を特定できない。"""

    exit_code = 8


class InvalidConfiguration(ImageGenError):
    """環境変数由来の設定値が不正である。"""

    exit_code = 9


__all__ = [
    "ComfyUIUnavailable",
    "GenerationFailed",
    "GenerationTimeout",
    "ImageGenError",
    "InvalidConfiguration",
    "InvalidGenerationSpec",
    "OutputNotFound",
    "WorkflowSubmissionError",
    "WorkflowValidationError",
]
