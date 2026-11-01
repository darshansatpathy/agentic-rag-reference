---
id: it-oncall-runbook-tms
title: "On-call runbook: transport management system"
groups: [engineering]
source: internal
---

# On-call runbook: transport management system

## Queue backlog

If the dispatch queue backlog exceeds 5,000 messages, scale the dispatch workers to 12 replicas and check the carrier API error rate on the Grafana board TMS-Overview.

## Carrier API failures

If more than 20 percent of calls to a carrier API fail for 10 minutes, switch that carrier to the fallback EDI channel and open a SEV2 incident.

## Escalation

Escalate to the platform team lead after 30 minutes without mitigation.
