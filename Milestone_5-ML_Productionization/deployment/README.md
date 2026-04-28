# Deployment Notes

This milestone targets a Docker Compose deployment on a Linux VM or container host. The stack consists of:

- `llm-server`: self-hosted OpenAI-compatible vLLM runtime serving the selected base model
- `api`: FastAPI/Uvicorn production service
- `client`: Streamlit frontend client

Windows prerequisite:

- Install Docker Desktop and verify Docker Compose before running this stack.
- Setup guide: `../docs/docker_setup_windows.md`

Local stack:

```powershell
cd Milestone_5-ML_Productionization\deployment
docker compose up --build
```

Optional local stack with the self-hosted LLM server:

```powershell
cd Milestone_5-ML_Productionization\deployment
docker compose --profile selfhosted up --build
```

Recommended production deployment flow:

1. Provision a Linux VM or Docker host.
2. Install Docker Engine and Docker Compose.
3. Copy the repository to the host.
4. Export the required environment variables:
   - `SELF_HOSTED_MODEL_NAME` for the served base model, e.g. `Qwen/Qwen2.5-Coder-7B-Instruct`
   - `SELF_HOSTED_OPENAI_API_KEY` if the local runtime should enforce an API key
   - `M5_SELF_HOSTED_FINE_TUNED_MODEL_NAME` if the fine-tuned candidate slot should point at a different self-hosted model/adapters path
   - `HUGGING_FACE_HUB_TOKEN` if the vLLM container must pull a gated or private model
5. Run `docker compose --profile selfhosted up -d --build`.

The GitHub Actions workflow in `.github/workflows/milestone5-ci-cd.yml` supports this deployment pattern through an optional SSH-based deploy job when the repository secrets are configured.
