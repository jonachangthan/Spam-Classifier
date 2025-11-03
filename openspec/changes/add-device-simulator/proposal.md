## Why
Development and testing for this AIoT project require predictable device telemetry. A lightweight device simulator will enable developers and CI to generate realistic sensor data without physical hardware.

## What Changes
- Add a device simulator capability (CLI and library) that can publish JSON telemetry over MQTT.
- Provide example configurations for temperature and humidity sensors.
- Add a small integration test that runs the simulator against a local MQTT broker and validates ingestion.

**BREAKING**: None.

## Impact
- Affected specs: `device-simulator` (new capability)
- Affected code: `tools/simulator/` (new module), Docker compose for local MQTT test, CI job `ci/simulator-validate` (new)
- Data formats: telemetry JSON (see project `project.md` for canonical example)

## Migration Plan
- Add simulator scaffold and tests in a dedicated change directory.
- On approval, implement simulator and add CI step to run simulation + ingestion smoke tests.

## Open Questions
- Preferred CLI options and configuration format (YAML vs JSON)? (Assumed: JSON for simplicity)
- Do we want the simulator to run as a long-lived process or emit a finite burst for tests? (Assumed: both; default finite burst)

***

Please see `tasks.md` for the implementation checklist and `specs/device-simulator/spec.md` for the spec deltas.