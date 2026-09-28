# Northwind Cloud — API Rate Limits

## Default limits

The public REST API allows 600 requests per minute per API key on the Business plan and 3,000 requests per minute on the Enterprise plan. The Starter plan is limited to 60 requests per minute.

## Exceeding the limit

Requests above the limit receive HTTP 429 Too Many Requests with a Retry-After header indicating how many seconds to wait. Clients should retry with exponential backoff.

## Raising limits

Enterprise customers can request a higher limit by contacting their account manager. Limit increases are reviewed within 3 business days.
