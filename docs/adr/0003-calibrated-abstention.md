# ADR-0003: Abstain on insufficient evidence, with a measured threshold

**Status:** accepted

## Context

Retrieval always returns something. Without a sufficiency check, a question the corpus
can't answer gets an answer built from the nearest passage, which is the most damaging
failure in an enterprise setting because it looks authoritative.

## Decision

For each sub-question, measure IDF-weighted term coverage of the best passage against the
original sub-question (never the rewritten query, whose extra terms come from the passage
and would inflate the score). Unseen terms get the highest weight. Below `min_coverage`
the part is missing. Sentence-level checks follow in the answerer, including an
answer-type check: a when-question needs a sentence that states a time.

The threshold is chosen from the sweep in the eval report (accuracy, abstention recall,
false abstentions at 0.3 to 0.7). 0.5 is where abstention recall stops improving and
before false abstentions climb.

## Alternatives considered

- **Raw retrieval score threshold.** Scores are not comparable across queries.
- **Ask the model whether it can answer.** Useful as an extra signal with the model
  answerer, and used there; not available offline, and not a measured control on its own.

## Consequences

- All six misses on answerable questions in the eval are refusals, not wrong answers.
- Lexical coverage cannot verify paraphrases, so some answerable questions are refused
  offline. The model answerer is the intended fix.
- The threshold is a configuration value (`ARAG_MIN_COVERAGE`), and changing it should
  come with a new eval report.
