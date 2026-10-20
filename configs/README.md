# configs/

Every experiment is defined by a YAML file in this directory. Code reads these
files; it does not hard-code model IDs, prompts, seeds, or thresholds.

| File | Contents | Added in |
|---|---|---|
| `eval_set.yaml` | Provenance and construction parameters of the frozen eval set, plus its expected SHA-256 | Phase 0 |
| `models.yaml` | Model IDs, providers, and decoding defaults | Phase 0 |
| `pricing.yaml` | Per-model token prices with a date stamp and source URL | Phase 1 |
| `prompts/` | Prompt templates | Phase 1 |
| `experiments/` | One file per experiment (E1–E6) | Phase 2 |
