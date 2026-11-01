---
id: eng-ai-assistant-guidelines
title: "Guidelines for building on the internal AI assistant"
groups: [engineering]
source: internal
---

# Guidelines for building on the internal AI assistant

## Approved models

Teams may call the internal AI gateway, which routes to approved models hosted in the EU. Calling external model APIs directly from production code is not allowed.

## Evaluation

Every AI feature needs an evaluation set of at least 100 labelled examples before launch and must report its results in the design review.

## Data

Personal data must be removed or pseudonymised before it is sent to any model, unless the data protection officer has approved the use case.
