# Security policy

## Reporting a vulnerability

Please do not open a public issue for security problems. Use GitHub's
[private vulnerability reporting](https://github.com/darshansatpathy/agentic-rag-reference/security/advisories/new)
for this repository. You should get an acknowledgement within three working days.

## Scope

This project is a reference implementation. The [threat model](docs/threat-model.md)
lists what it is designed to defend against and what is deliberately out of scope.

## Operating it safely

- Put the service behind a gateway that authenticates users and forwards their groups,
  and set `ARAG_SERVICE_TOKEN` so nothing else can assert an identity.
- Sync document ACLs from the source systems at ingest. A document with a stale ACL is
  a leak this service cannot detect.
- Review quarantined chunks (`arag.query` logs and the answer trace): they are usually a
  poisoned or mis-sourced document that should be removed from the index.
- Keep model and embedding API keys in a secret manager, never in `.env` files that are
  committed.
