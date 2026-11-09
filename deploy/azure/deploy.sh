#!/usr/bin/env bash
# Deploy Agentic Researcher to Azure Container Apps. Creates PAID resources: read README.md first.
# Usage: ANTHROPIC_API_KEY=sk-ant-... ./deploy/azure/deploy.sh      (run from the repo root, after `az login`)
set -euo pipefail

RG=${RG:-agentic-researcher-rg}
LOC=${LOC:-canadacentral}
ACR=${ACR:-agenticresearcher$RANDOM}
ENV_NAME=${ENV_NAME:-agentic-researcher-env}
APP=${APP:-agentic-researcher}
STORAGE=${STORAGE:-arstore$RANDOM}
TOKEN=${AR_ACCESS_TOKEN:-$(openssl rand -hex 16)}

read -rp "This creates billable Azure resources in $LOC (resource group $RG). Continue? [y/N] " ok
[[ $ok == y* ]] || exit 1

az group create -n "$RG" -l "$LOC" -o none
az acr create -g "$RG" -n "$ACR" --sku Basic --admin-enabled true -o none
az acr build -r "$ACR" -t agentic-researcher:v1 .                       # builds in Azure, no local Docker needed

# Documents live in an Azure Files share mounted at /data (never inside the image).
az storage account create -g "$RG" -n "$STORAGE" -l "$LOC" --sku Standard_LRS -o none
KEY=$(az storage account keys list -g "$RG" -n "$STORAGE" --query "[0].value" -o tsv)
az storage share-rm create -g "$RG" --storage-account "$STORAGE" -n library --quota 1 -o none
az storage directory create --account-name "$STORAGE" --account-key "$KEY" -s library -n datasheets -o none
az storage directory create --account-name "$STORAGE" --account-key "$KEY" -s library -n papers -o none
az storage file upload-batch --account-name "$STORAGE" --account-key "$KEY" -d library/datasheets -s data/datasheets -o none
az storage file upload-batch --account-name "$STORAGE" --account-key "$KEY" -d library/papers -s data/papers -o none

az containerapp env create -g "$RG" -n "$ENV_NAME" -l "$LOC" -o none
az containerapp env storage set -g "$RG" -n "$ENV_NAME" --storage-name library --azure-file-account-name "$STORAGE" \
  --azure-file-account-key "$KEY" --azure-file-share-name library --access-mode ReadOnly -o none

az containerapp create -g "$RG" -n "$APP" --environment "$ENV_NAME" \
  --image "$ACR.azurecr.io/agentic-researcher:v1" --registry-server "$ACR.azurecr.io" \
  --target-port 8770 --ingress external --min-replicas 0 --max-replicas 1 --cpu 1.0 --memory 2.0Gi \
  --secrets anthropic-key="${ANTHROPIC_API_KEY:?set ANTHROPIC_API_KEY}" access-token="$TOKEN" \
  --env-vars ANTHROPIC_API_KEY=secretref:anthropic-key AR_ACCESS_TOKEN=secretref:access-token -o none

# Mount the share at /data (volume mounts are set through YAML).
az containerapp show -g "$RG" -n "$APP" -o yaml > /tmp/ar-app.yaml
python3 - <<'PY'
import re
p = "/tmp/ar-app.yaml"; y = open(p).read()
y = y.replace("    containers:\n", "    volumes:\n    - name: library\n      storageName: library\n      storageType: AzureFile\n    containers:\n", 1)
y = re.sub(r"(\n      name: agentic-researcher\n)", r"\1      volumeMounts:\n      - volumeName: library\n        mountPath: /data\n", y, count=1)
open(p, "w").write(y)
PY
az containerapp update -g "$RG" -n "$APP" --yaml /tmp/ar-app.yaml -o none

URL=$(az containerapp show -g "$RG" -n "$APP" --query properties.configuration.ingress.fqdn -o tsv)
echo "Open: https://$URL/?token=$TOKEN"
echo "Delete everything later with: az group delete -n $RG --yes"
