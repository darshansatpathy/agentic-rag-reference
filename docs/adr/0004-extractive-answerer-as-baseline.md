# ADR-0004: An extractive answerer as the default and the fallback

**Status:** accepted

## Context

A generative answerer reads better but can introduce claims that are not in the
sources. Demos that need an API key also don't run in CI or on a reviewer's laptop.

## Decision

The default answerer selects sentences from the evidence and cites each one. It can't
produce a claim that is not in a cited passage. The model answerer sits behind the same
interface; its citations are verified against the sources it was given, invalid ones are
removed, and an answer with no valid citation becomes an abstention.

## Consequences

- Groundedness is 100% by construction offline, and measured for the model answerer.
- The full pipeline, eval and API run in CI without secrets.
- Extractive answers are less fluent and can't combine facts into one sentence. That is
  acceptable for policy questions, where quoting the policy is often what users want.
- It doubles as the fallback when the model endpoint is down.
