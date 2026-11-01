---
id: prod-tracking-api
title: "Shipment tracking API"
groups: [all]
source: internal
---

# Shipment tracking API

## Authentication

The tracking API uses OAuth 2.0 client credentials. Access tokens are valid for one hour.

## Rate limits

Each client may make 600 requests per minute. Bursts above the limit receive HTTP 429 with a Retry-After header.

## Webhooks

Status change webhooks are retried with exponential backoff for up to 24 hours. Payloads are signed with HMAC-SHA256 in the X-Aurora-Signature header.
