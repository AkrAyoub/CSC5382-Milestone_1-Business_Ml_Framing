# Service Contract

Primary API:

- `GET /healthz`
- `GET /api/v1/runtime`
- `GET /api/v1/catalog/instances`
- `POST /api/v1/solve`
- `POST /api/v1/batch/solve`

Solve modes:

- `auto`: resolves to the deterministic baseline for production safety
- `baseline`: deterministic CBC solver only
- `llm`: baseline plus one optional LLM candidate execution
- `compare`: baseline plus one optional LLM candidate execution, intended for side-by-side evaluation

Default optional LLM candidate:

- `openai_gpt41_mini_finetuned`
- model: `ft:gpt-4.1-mini-2025-04-14:aui:uflp-symbolic-solver:DaENP2ZH`
- requires `OPENAI_API_KEY`

Input sources:

- `catalog`: solve one instance from the shared benchmark catalog by `instance_id`
- `inline`: solve raw OR-Library text supplied directly in the request

Response shape:

- top-level request metadata and status
- deterministic baseline result
- optional candidate result
- warnings indicating default fallbacks or unavailable optional runtime paths
