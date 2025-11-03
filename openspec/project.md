# Project Context

## Purpose
This repository contains coursework and prototype code for AIoT (AI + IoT) homework (HW3). The project goal is to build a small platform that simulates IoT devices, ingests telemetry, and exercises ML/AI workflows against that telemetry for experimentation and evaluation. The repository is intended to be easy to run locally for development and to be integrated into CI for validation using OpenSpec-driven requirements.

## Tech Stack
- Node.js (LTS recommended, e.g. 18 or later)
- TypeScript for core services and tools
- npm for package management
- OpenSpec for specification-driven development and change proposals
- MQTT (e.g. Mosquitto) for device telemetry simulation
- Docker (optional) for running broker and other services locally
- Optional: Python for ML/analysis scripts (documented per script)

## Project Conventions

### Code Style
- Format: Prettier (opinionated formatting).
- Linting: ESLint with recommended TypeScript rules.
- Naming: camelCase for variables, PascalCase for types/classes, kebab-case for CLI/change ids and file/folder names under `openspec/changes`.
- Exports: prefer named exports for libraries; default exports allowed for single-file CLI entry points.

### Architecture Patterns
- Small, single-purpose capabilities (folder-per-capability inside `src/` if needed).
- Separation between simulators (device-side logic), ingestion (MQTT/HTTP bridge), and analysis (ML scripts or services).
- Keep runtime configuration in `config/` or environment variables; prefer `.env` for local development with a `.env.example` file.

### Testing Strategy
- Unit tests: Jest for TypeScript modules. Keep tests fast and isolated.
- Integration tests: run against a lightweight Docker compose that includes an MQTT broker when needed.
- Spec validation: use `openspec validate <change-id> --strict` for proposals and a CI job that runs `openspec validate --strict` before merges.

### Git Workflow
- Branching: `main` for stable course deliverable; feature branches: `feature/<short-desc>` or `change/<change-id>` for OpenSpec proposals.
- Commits: Use conventional, clear messages; prefix with change id when applicable (e.g., `add-device-simulator: add simulator CLI`).
- Pull requests: include a pointer to the OpenSpec change (if applicable) and link to `openspec/changes/<change-id>/proposal.md`.

## Domain Context
- Devices: small simulated sensors that publish telemetry messages (JSON) over MQTT.
- Telemetry: include timestamp, device id, sensor type, and measured value. Example:

```json
{
	"deviceId": "device-001",
	"ts": "2025-10-26T12:00:00Z",
	"type": "temperature",
	"value": 22.4
}
```

- Data flow: simulator -> MQTT broker -> ingestion service -> processing/analysis.

## Important Constraints
- Keep dependencies minimal for homework grading (prefer standard libraries and small dependencies).
- No external paid services required; any external APIs must be optional and documented.
- Keep runtime memory and CPU usage modest to run on a typical development laptop.

## External Dependencies
- MQTT broker (Mosquitto or equivalent) for local/dev runs. Docker compose is recommended to provision it.
- Node/npm runtime and standard build/test dev dependencies.
- Optional Python packages (documented in `requirements.txt`) for ML/analysis scripts.

## Contacts
- Primary maintainer: (fill in your name/email here)

---

Notes / Assumptions:
- I assumed this is the AIoT HW3 project (device simulators + telemetry ingestion). If the real stack differs (e.g., Python-first), tell me and I will adapt `project.md` accordingly.
