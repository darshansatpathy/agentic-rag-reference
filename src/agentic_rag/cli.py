"""`arag` command line: demo, ask, search, eval, serve."""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from collections.abc import Sequence
from pathlib import Path

from agentic_rag.config import ConfigError, Settings, build_pipeline
from agentic_rag.evaluation import evaluate, load_questions, load_thresholds
from agentic_rag.pipeline import RagPipeline
from agentic_rag.retrieval import Principal

DEMO = [
    ("an employee", [], "How many days of annual leave does a permanent employee get?"),
    ("an employee", [], "What is the per diem in Belgium and who approves an expense of 2,000 EUR?"),
    ("an employee", [], "What is the salary range for band E4?"),
    ("someone in HR", ["hr"], "What is the salary range for band E4?"),
    ("an employee", [], "What is the company's stock ticker?"),
    ("an employee", [], "Do I need receipts for expense claims?"),
]


def _principal(groups: str | None, user: str) -> Principal:
    return Principal.of(user, [g.strip() for g in (groups or "").split(",") if g.strip()])


def _print_answer(data: dict[str, object], show_trace: bool) -> None:
    print(textwrap.fill(str(data["answer"]), width=100))
    for c in data["citations"]:  # type: ignore[attr-defined]
        print(f"  [{c['n']}] {c['title']} - {c['section']} ({c['doc_id']})")
    if data.get("abstained"):
        print(f"  (abstained: {data.get('reason')})")
    if show_trace:
        for step in data["trace"]:  # type: ignore[attr-defined]
            print("  trace:", json.dumps(step))


def cmd_demo(rag: RagPipeline, _: argparse.Namespace) -> int:
    print(f"Corpus: {len({c.doc_id for c in rag.chunks})} documents, {len(rag.chunks)} chunks\n")
    for who, groups, question in DEMO:
        print(f"> {who} asks: {question}")
        answer = rag.ask(question, Principal.of("demo", groups))
        _print_answer(answer.to_dict(), show_trace=False)
        quarantine = [t for t in answer.trace if t.get("step") == "quarantine"]
        if quarantine:
            chunks = ", ".join(str(c) for c in quarantine[0]["chunks"])
            print(f"  (quarantined as possible prompt injection: {chunks})")
        print()
    return 0


def cmd_ask(rag: RagPipeline, args: argparse.Namespace) -> int:
    answer = rag.ask(args.question, _principal(args.groups, args.user))
    data = answer.to_dict()
    if args.json:
        print(json.dumps(data, indent=2))
    else:
        _print_answer(data, args.trace)
    return 0


def cmd_search(rag: RagPipeline, args: argparse.Namespace) -> int:
    for h in rag.search(args.query, _principal(args.groups, args.user), args.k)[: args.k]:
        print(f"{h.score:.4f}  {h.chunk.chunk_id:32s} {h.chunk.title} - {h.chunk.section}  ranks={h.ranks}")
    return 0


def cmd_eval(rag: RagPipeline, args: argparse.Namespace) -> int:
    report = evaluate(rag, load_questions(args.questions))
    md = report.markdown()
    if args.json:
        print(
            json.dumps(
                {"summary": report.summary(), "retrieval": report.retrieval, "sweep": report.sweep}, indent=2
            )
        )
    else:
        print(md)
    if args.report:
        Path(args.report).write_text(md + "\n", encoding="utf-8")
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_file and args.check:
        with open(summary_file, "a", encoding="utf-8") as fh:
            fh.write("## RAG evaluation\n\n" + md + "\n")
    if args.check:
        failures = report.check(load_thresholds(args.thresholds))
        if failures:
            print("\nEvaluation gate FAILED:", *failures, sep="\n  - ", file=sys.stderr)
            return 1
        print("\nEvaluation gate passed.", file=sys.stderr)
    return 0


def cmd_serve(_: RagPipeline | None, args: argparse.Namespace) -> int:  # pragma: no cover - runs a server
    import uvicorn

    from agentic_rag.api import create_app

    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="warning")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="arag", description="Agentic RAG reference implementation")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("demo", help="answer a scripted set of questions as different users")

    ask = sub.add_parser("ask", help="answer one question")
    ask.add_argument("question")
    ask.add_argument("--groups", help="comma-separated groups, e.g. hr,finance")
    ask.add_argument("--user", default="cli")
    ask.add_argument("--trace", action="store_true", help="print the agent's steps")
    ask.add_argument("--json", action="store_true")

    search = sub.add_parser("search", help="show ranked passages for a query")
    search.add_argument("query")
    search.add_argument("--groups")
    search.add_argument("--user", default="cli")
    search.add_argument("-k", type=int, default=5)

    ev = sub.add_parser("eval", help="run the evaluation suite")
    ev.add_argument("--questions", help="path to questions.yaml")
    ev.add_argument("--thresholds", help="path to thresholds.yaml")
    ev.add_argument("--report", help="write the markdown report here")
    ev.add_argument("--check", action="store_true", help="exit 1 if a threshold is not met")
    ev.add_argument("--json", action="store_true")

    serve = sub.add_parser("serve", help="run the HTTP API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "serve":  # pragma: no cover
        return cmd_serve(None, args)
    try:
        rag = build_pipeline(Settings.from_env())
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    handlers = {"demo": cmd_demo, "ask": cmd_ask, "search": cmd_search, "eval": cmd_eval}
    return handlers[args.command](rag, args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
