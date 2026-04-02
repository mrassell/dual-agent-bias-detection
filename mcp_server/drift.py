"""Temporal drift metrics over dated bias summaries."""

from __future__ import annotations

import math
from typing import Any


def _normalize(dist: dict[str, float]) -> dict[str, float]:
    s = sum(dist.values()) or 1.0
    return {k: v / s for k, v in dist.items()}


def jensen_shannon_divergence(p: dict[str, float], q: dict[str, float]) -> float:
    """Symmetric divergence in [0, ln 2]; 0 = identical."""
    keys = set(p) | set(q)
    p_n = _normalize({k: p.get(k, 0.0) for k in keys})
    q_n = _normalize({k: q.get(k, 0.0) for k in keys})
    m = {k: 0.5 * (p_n.get(k, 0.0) + q_n.get(k, 0.0)) for k in keys}

    def kl(a: dict[str, float], b: dict[str, float]) -> float:
        out = 0.0
        for k in keys:
            ak = a.get(k, 0.0)
            bk = b.get(k, 0.0)
            if ak > 0 and bk > 0:
                out += ak * math.log(ak / bk)
        return out

    return 0.5 * kl(p_n, m) + 0.5 * kl(q_n, m)


def temporal_drift_analysis(
    buckets: list[dict[str, Any]],
) -> dict[str, Any]:
    """JS divergence between first and last bucket (sort key: period or date)."""
    if len(buckets) < 2:
        return {
            "error": "need_at_least_two_buckets",
            "bucket_count": len(buckets),
        }

    def key(b: dict[str, Any]) -> str:
        return str(b.get("period") or b.get("date") or "")

    sorted_b = sorted(buckets, key=key)
    first, last = sorted_b[0], sorted_b[-1]
    c0 = {str(k): float(v) for k, v in (first.get("bias_tag_counts") or {}).items()}
    c1 = {str(k): float(v) for k, v in (last.get("bias_tag_counts") or {}).items()}
    if not c0 and not c1:
        # Fall back: use mean lexical / informational if provided
        def means(b: dict[str, Any]) -> dict[str, float]:
            return {
                "lexical_mean": float(b.get("mean_lexical_score", 0.0)),
                "informational_mean": float(b.get("mean_informational_score", 0.0)),
            }

        c0, c1 = means(first), means(last)

    js = jensen_shannon_divergence(c0, c1)
    return {
        "first_bucket": key(first) or "bucket_0",
        "last_bucket": key(last) or "bucket_n",
        "jensen_shannon_divergence": round(js, 4),
        "interpretation": (
            "strong_shift" if js > 0.15 else "moderate_shift" if js > 0.07 else "stable"
        ),
        "tag_mass_first": c0,
        "tag_mass_last": c1,
    }
