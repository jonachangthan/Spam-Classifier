## 1. Implementation
- [ ] 1.1 Scaffold `tools/simulator/` TypeScript module and CLI (node + ts-node or compiled)
- [ ] 1.2 Implement MQTT publisher with configurable message rate, device count, and payload templates
- [ ] 1.3 Add example configs: `examples/temperature.json`, `examples/humidity.json`
- [ ] 1.4 Add integration test that runs broker via Docker Compose and verifies messages arrive at ingestion
- [ ] 1.5 Document usage in README and update `project.md` if needed

## 2. Validation
- [ ] 2.1 Add spec delta under `specs/device-simulator/spec.md`
- [ ] 2.2 Run `openspec validate add-device-simulator --strict` and fix any format issues

## 3. CI
- [ ] 3.1 Add CI job to run simulator + ingestion smoke test

## 4. Cleanup
- [ ] 4.1 Move completed change to `changes/archive/YYYY-MM-DD-add-device-simulator/` after deployment
