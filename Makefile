# On Windows without make: use ./tasks.ps1 <target> (same targets).
PYTHON ?= python
IMAGE  ?= fraud-api

SPACE  ?= Zakariaebouy/fraud-decision-dashboard

.PHONY: install data train train-fast smoke report test lint format api dashboard mlflow docker-build docker-run space space-preview space-publish space-run clean

install:        ## install all dependencies (training, API, tests)
	$(PYTHON) -m pip install -r requirements-dev.txt

data:           ## download the dataset (data/raw/creditcard.csv)
	$(PYTHON) -c "from src.config import load_config, resolve; from src.data import download; c = load_config(); download(c['data']['url'], resolve(c['paths']['raw_data']))"

train:          ## full pipeline: tuning, calibration, decision, SHAP, report, MLflow
	$(PYTHON) -m src.train

train-fast:     ## same, reusing the tuned hyper-parameters (no Optuna)
	$(PYTHON) -m src.train --reuse-params

smoke:          ## end-to-end run on synthetic data (CI)
	$(PYTHON) -m src.train --synthetic --output-dir artifacts/smoke --no-mlflow --n-trials 3 --bootstrap 100

report:         ## regenerate README results + reports/fiche_resultats.md from results.json
	$(PYTHON) -m src.report

test:
	$(PYTHON) -m pytest

lint:
	ruff check . && ruff format --check .

format:
	ruff check --fix . && ruff format .

api:            ## http://localhost:8000/docs
	uvicorn src.api.main:app --reload --port 8000

dashboard:      ## http://localhost:8501
	streamlit run app/dashboard.py

mlflow:         ## http://localhost:5000
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

docker-build:
	docker build -t $(IMAGE) .

docker-run:
	docker run --rm -p 8000:8000 $(IMAGE)

space:          ## build the live demo (static Hugging Face Space) in dist/hf_space
	$(PYTHON) -m src.space build

space-preview: space  ## serve the live demo locally: http://localhost:8765
	$(PYTHON) -m http.server 8765 --bind 127.0.0.1 --directory dist/hf_space

space-publish:  ## publish the live demo (run `hf auth login` once before)
	$(PYTHON) -m src.space publish --repo-id $(SPACE)

space-run:      ## Streamlit dashboard as a Docker Space, run locally: http://localhost:8501
	$(PYTHON) -m src.space build --docker
	docker build -t fraud-dashboard dist/hf_space_docker && docker run --rm -p 8501:8501 fraud-dashboard

clean:
	rm -rf artifacts dist .pytest_cache .ruff_cache
