#!/usr/bin/env bash
# Downloads the public documents the CI smoke eval needs (the repo never stores PDFs).
set -euo pipefail
mkdir -p data/datasheets data/papers
curl -fsSL -A "agentic-researcher-ci" -o data/datasheets/esp32_datasheet_en.pdf \
  https://www.espressif.com/sites/default/files/documentation/esp32_datasheet_en.pdf
curl -fsSL -A "agentic-researcher-ci" -o data/papers/bathroom_fall_mmwave_arxiv2603.17069.pdf \
  https://arxiv.org/pdf/2603.17069v2
ls -la data/*/
