#!/usr/bin/env python3
"""Golden-set eval + optional naive-RAG baseline comparison.

Usage:
    ./.venv/bin/python -m eval.eval_retrieval
    ./.venv/bin/python -m eval.eval_retrieval --baseline
    ./.venv/bin/python -m eval.eval_retrieval --route-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.runner import compare_to_baseline, load_questions, run_eval  # noqa: E402
from rag.pipeline import RAGPipeline  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=None)
    parser.add_argument("--route-only", action="store_true")
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Also run naive vector-RAG baseline and print the delta",
    )
    args = parser.parse_args(argv)

    questions = load_questions(args.questions)
    if not questions:
        print("No questions found. Edit eval/sample_questions.json", file=sys.stderr)
        return 1

    pipeline = RAGPipeline()
    ok, message = pipeline.health()
    if not ok:
        print(f"unhealthy: {message}", file=sys.stderr)
        return 1

    stats = run_eval(pipeline, questions, route_only=args.route_only)
    for row in stats["rows"]:
        mark = "OK" if row["ok"] else "MISS"
        print(
            f"[{mark}] {row['id']} route={row['route']} answered={row['answered']} "
            f"query={row['query']!r}"
        )

    print()
    print(
        f"total={stats['total']}  route_accuracy={stats['route_accuracy']:.0%}  "
        f"answered={stats['answered']}  refused={stats['refused']}  "
        f"refuse_rate={stats['refuse_rate']:.0%}  "
        f"avg_confidence={stats['avg_confidence']}"
    )
    if stats.get("avg_groundedness") is not None:
        print(f"avg_groundedness={stats['avg_groundedness']}")

    if args.baseline and not args.route_only:
        print("\n--- baseline comparison (naive vector RAG) ---")
        cmp = compare_to_baseline(pipeline, questions)
        print(f"system:   {cmp['system']}")
        print(f"baseline: {cmp['baseline']}")
        print(
            f"delta route_accuracy={cmp['delta_route_accuracy']:+.0%}  "
            f"delta refuse_rate={cmp['delta_refuse_rate']:+.0%}"
        )

    return 0 if stats["route_ok"] == stats["total"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
