# Deploying to Azure Container Apps

`deploy.sh` builds the image in Azure Container Registry, uploads your PDFs to an Azure Files share,
and runs the app on Azure Container Apps with HTTPS, scale-to-zero and a shared access token.
**It creates paid resources. Nothing here has been run for you.**

## Before you run it

1. Install the Azure CLI and sign in: `az login` (Azure for Students gives US$100 of credit).
2. Have an Anthropic API key, or configure Azure OpenAI (below). Ollama does not run in this setup,
   so the local model and the embedding model are not available in the cloud: search uses keyword
   matching only there (the vector half of the hybrid search needs Ollama).
3. Run from the repo root: `ANTHROPIC_API_KEY=sk-ant-... ./deploy/azure/deploy.sh`

## What it costs (estimate; check the Azure pricing calculator for your region)

| Resource | Setting | Expected cost |
|---|---|---|
| Container Apps | consumption plan, 1 vCPU / 2 GiB, **min replicas 0** | Usually within the monthly free grant (180,000 vCPU-seconds, 360,000 GiB-seconds, 2 million requests) for a demo used a few hours a month |
| Container Registry | Basic | About US$5 per month |
| Storage account | Azure Files, 1 GiB share | Cents per month |
| Claude API | per token | e.g. a 21-question eval run on Claude Haiku 5.5 costs cents; Opus 5.5 costs more (see Settings → cost cap) |

Stop all charges: `az group delete -n agentic-researcher-rg --yes`

## Security

- The app has no user accounts. `AR_ACCESS_TOKEN` gates every request (open `https://<app>/?token=<token>` once;
  a secure cookie remembers it). For a team, put Container Apps built-in authentication (Microsoft Entra ID) in front instead.
- API keys are Container Apps **secrets**, not image layers or env files in the repo.
- Documents are mounted read-only from Azure Files; they are never baked into the image.

## Optional: Azure OpenAI as a provider

```bash
az cognitiveservices account create -g agentic-researcher-rg -n ar-openai -l canadaeast --kind OpenAI --sku S0
az cognitiveservices account deployment create -g agentic-researcher-rg -n ar-openai \
  --deployment-name gpt-4o-mini --model-name gpt-4o-mini --model-version 2024-07-18 --model-format OpenAI \
  --sku-name GlobalStandard --sku-capacity 10
```
Then set `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY` (as a secret) and `AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini`
on the container app. The router (`app/llm.py`) adds Azure OpenAI as the last fallback when these are set.
Model availability and prices vary by region; check before creating the deployment.
