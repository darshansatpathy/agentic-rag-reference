---
id: eng-deployment-policy
title: "Software deployment policy"
groups: [engineering]
source: internal
---

# Software deployment policy

## Change freeze

There is a deployment freeze for tier 1 systems from 1 December to 2 January every year, the peak season for parcel volumes.

## Approvals

Production deployments need one approving code review and a green CI pipeline. Database migrations also need approval from the data platform team.

## Rollback

Every deployment must have a tested rollback plan. Rollbacks should complete within 15 minutes.
