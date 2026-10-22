# Windows equivalent of the Makefile:  .\tasks.ps1 train   (targets: see the switch below)
param([Parameter(Position = 0)][string]$Task = "help")

$ErrorActionPreference = "Stop"
$py = if (Test-Path ".venv\Scripts\python.exe") { ".venv\Scripts\python.exe" } else { "python" }

switch ($Task) {
    "install"      { uv venv --python 3.11 .venv; uv pip install --python .venv\Scripts\python.exe -r requirements-dev.txt }
    "data"         { & $py -c "from src.config import load_config, resolve; from src.data import download; c = load_config(); download(c['data']['url'], resolve(c['paths']['raw_data']))" }
    "train"        { & $py -m src.train }
    "train-fast"   { & $py -m src.train --reuse-params }
    "smoke"        { & $py -m src.train --synthetic --output-dir artifacts/smoke --no-mlflow --n-trials 3 --bootstrap 100 }
    "report"       { & $py -m src.report }
    "test"         { & $py -m pytest }
    "lint"         { & $py -m ruff check .; & $py -m ruff format --check . }
    "format"       { & $py -m ruff check --fix .; & $py -m ruff format . }
    "api"          { & $py -m uvicorn src.api.main:app --reload --port 8000 }
    "dashboard"    { & $py -m streamlit run app/dashboard.py }
    "mlflow"       { & $py -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000 }
    "docker-build" { docker build -t fraud-api . }
    "docker-run"   { docker run --rm -p 8000:8000 fraud-api }
    default        { Write-Host "Targets: install data train train-fast smoke report test lint format api dashboard mlflow docker-build docker-run" }
}
