# Docker Setup On Windows

Docker is a system requirement for the Milestone 5 containerization and deployment demo. It is not installed with `pip` and should not be added to `requirements.txt`.

## 1. Install Docker Desktop

Recommended install command:

```powershell
winget install -e --id Docker.DockerDesktop
```

If `winget` is not available, install Docker Desktop manually from:

```text
https://www.docker.com/products/docker-desktop/
```

During installation, keep the WSL 2 backend enabled.

## 2. Restart And Open Docker Desktop

After installation:

1. Restart Windows if Docker asks for it.
2. Open Docker Desktop from the Start menu.
3. Wait until Docker says the engine is running.

## 3. Verify Docker

Run:

```powershell
docker --version
docker compose version
docker run hello-world
```

Expected result:

- Docker prints a version.
- Docker Compose prints a version.
- `hello-world` downloads and runs successfully.

## 4. Run Milestone 5 With Docker

From the repository root:

```powershell
cd Milestone_5-ML_Productionization\deployment
docker compose up --build
```

Open:

```text
http://localhost:8000/docs
http://localhost:8501
```

Stop the stack:

```powershell
Ctrl+C
docker compose down
```

## 5. Optional Self-Hosted LLM Stack

The default Docker demo runs only the API and Streamlit frontend. This is enough for the productionization requirement because the deterministic baseline is the production-safe runtime.

To also start the optional self-hosted OpenAI-compatible LLM server:

```powershell
cd Milestone_5-ML_Productionization\deployment
docker compose --profile selfhosted up --build
```

This requires significantly more disk, memory, and GPU/CPU resources because it pulls and serves a model container.
