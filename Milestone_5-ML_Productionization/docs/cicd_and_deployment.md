# CI/CD and Deployment Notes

CI/CD:

- GitHub Actions workflow: [../../.github/workflows/milestone5-ci-cd.yml](../../.github/workflows/milestone5-ci-cd.yml)
- CI stages:
  - install milestone dependencies
  - run pytest
  - run API smoke test
  - build API and frontend Docker images
- CD stage:
  - optional SSH deployment to a Linux VM when deployment secrets are configured

Deployment target:

- primary target: Docker Compose on a Linux VM / DigitalOcean-style host
- local development target: Docker Desktop with Docker Compose on Windows
- stack composition:
  - `llm-server` self-hosted OpenAI-compatible vLLM runtime
  - `api` FastAPI service
  - `client` Streamlit frontend

Reason for target choice:

- matches the course deployment requirement without locking the milestone to one hosted PaaS
- keeps the service and frontend deployable together
- integrates naturally with the GitHub Actions workflow

Local Docker setup:

- Windows setup and verification steps are documented in [docker_setup_windows.md](docker_setup_windows.md).
- The standard local demo uses `docker compose up --build`.
- The optional self-hosted LLM demo uses `docker compose --profile selfhosted up --build`.
