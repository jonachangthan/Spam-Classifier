## ADDED Requirements

### Requirement: Device Simulator CLI
The system SHALL provide a device simulator CLI that can publish JSON telemetry messages over MQTT to a configurable broker.

#### Scenario: Publish a burst of temperature telemetry
- **GIVEN** a running MQTT broker
- **WHEN** the simulator is invoked with `--device-count 3 --rate 1 --duration 10s --profile temperature`
- **THEN** the simulator SHALL publish 3 messages/second for 10 seconds per device, and messages SHALL conform to the telemetry format with fields `deviceId`, `ts`, `type`, and `value`.

### Requirement: Configurable Payload Templates
The simulator SHALL accept a payload template (JSON) that defines the sensor `type` and value generation rules (e.g., fixed, random within range, or pattern).

#### Scenario: Use a humidity template
- **GIVEN** a humidity template with `value` range 30-70
- **WHEN** the simulator publishes telemetry
- **THEN** each message's `value` SHALL be a numeric value within the specified range

***

Notes:
- This spec introduces a new capability `device-simulator` and is intentionally small so it is easy to validate and implement.
