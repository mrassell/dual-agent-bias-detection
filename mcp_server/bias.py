"""Structured bias JSON from the auditor model."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class BiasDetectionResult(BaseModel):
    lexical_score: float = Field(ge=0.0, le=1.0)
    informational_score: float = Field(ge=0.0, le=1.0)
    cognitive_bias_tags: list[str] = Field(default_factory=list)
    reasoning_trace: str = Field(description="Short model id + score summary.")
    predicted_lexical_bias: bool
    prediction_confidence: float = Field(ge=0.0, le=1.0)
    auditor_model_id: str

    @field_validator("cognitive_bias_tags", mode="before")
    @classmethod
    def _strip_tags(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, str):
            return [v.strip()] if v.strip() else []
        return [str(x).strip() for x in v if str(x).strip()]


def build_detection_result(
    text: str,
    pred: dict[str, Any],
    auditor_model_id: str,
) -> BiasDetectionResult:
    lex = float(pred["lexical_probability"])
    inf = lex
    tags: list[str] = []
    if pred["predicted_lexical_bias"]:
        tags.append("lexical_bias_suspected")
    if lex >= 0.7:
        tags.append("high_lexical_score")

    summary = (
        f"{auditor_model_id} lexical={lex:.3f} "
        f"{'biased' if pred['predicted_lexical_bias'] else 'neutral'} "
        f"conf={float(pred['confidence']):.3f}"
    )
    return BiasDetectionResult(
        lexical_score=lex,
        informational_score=inf,
        cognitive_bias_tags=sorted(set(tags)),
        reasoning_trace=summary,
        predicted_lexical_bias=bool(pred["predicted_lexical_bias"]),
        prediction_confidence=float(pred["confidence"]),
        auditor_model_id=auditor_model_id,
    )
