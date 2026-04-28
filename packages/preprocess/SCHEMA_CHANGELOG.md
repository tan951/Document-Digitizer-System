# Schema Changelog

## preprocess.schema.json
- **1.0.0** (2026-04-28)
  - Initial schema for Module 1 output
  - Required: request_id, document_id, module, version, timestamp, status, payload
  - Payload includes width/height/rotation/is_blank
  - Optional: blank_ratio, crop_applied, deskew_angle, osd

## config.schema.json
- **1.0.0** (2026-04-28)
  - Initial config schema for Module 1
  - All pipeline stages configurable