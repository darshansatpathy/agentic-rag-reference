# Contributing

## Set up

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
make check
```

`make check` runs lint, formatting, strict type checks, the tests with a coverage gate,
and the evaluation gate. CI runs the same steps.

## Changing retrieval or answering

Retrieval changes are easy to make and hard to judge by reading a few answers. For any
change to chunking, retrieval, re-ranking, sufficiency or answering:

1. Run `arag eval --report before.md` on `main` and `arag eval --report after.md` on
   your branch.
2. Put both tables in the pull request description.
3. If a metric moves the wrong way, say why the trade is worth it.

Do not tune against individual questions in `evals/questions.yaml`. If a question is
wrong, fix the question in its own commit and explain it. New behaviour gets new
questions, including at least one the system should refuse.

## Permissions

Any change that touches how chunks are filtered needs a test in `tests/test_pipeline.py`
that runs across all retrieval modes, and a question in the eval. The leak threshold is
zero and stays zero.

## Corpus

Edit `scripts/generate_corpus.py`, then run `make corpus` and commit the regenerated
files. CI fails if the committed corpus differs from the generator's output.

## Commits

Small commits with a message that says what changed and why. Update `CHANGELOG.md`
under an "Unreleased" heading.
