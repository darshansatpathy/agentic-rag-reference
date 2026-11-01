---
id: it-backup-dr
title: "Backup and disaster recovery"
groups: [engineering]
source: internal
---

# Backup and disaster recovery

## Targets

Tier 1 systems (transport management and warehouse management) have a recovery point objective of 15 minutes and a recovery time objective of 4 hours. Tier 2 systems have an RPO of 24 hours and an RTO of 24 hours.

## Backups

Databases are backed up every 15 minutes to a second region in Frankfurt. Backups are immutable for 35 days.

## Testing

A full disaster recovery failover test is run twice a year, in March and September.
