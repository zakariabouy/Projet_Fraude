"""Build and publish the live dashboard on Hugging Face Spaces.

    python -m src.space build                              # static Space -> dist/hf_space/
    python -m src.space publish --repo-id USER/NAME        # after `hf auth login`
    python -m src.space build --docker                     # Streamlit Space -> dist/hf_space_docker/

Static (default, free on Hugging Face): app/static/index.html + decision.js, fed with the
model's scores on the validation and test periods. The decision rules (cost model, capacity,
threshold search, top-K queue) run in the visitor's browser; tests/test_decision_js.py checks
they give exactly the Python numbers. SHAP reason codes are precomputed with the real model.

Docker (needs a paid Hugging Face plan, or run it locally with `make space-run`): the
Streamlit dashboard with the model bundle, its requirements taken from the pinned files.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CONFIG_PATH, ROOT, load_config
from src.data import RAW_FEATURES, TARGET, V_COLUMNS

DEFAULT_OUT = ROOT / "dist" / "hf_space"
DOCKER_OUT = ROOT / "dist" / "hf_space_docker"
STATIC_APP = ROOT / "app" / "static"
DASHBOARD = ROOT / "app" / "dashboard.py"
PROCESSED = ROOT / "data" / "processed"
APP_PORT = 8501
FIGURES = {
    "policies_test.png": "Policies at constant capacity, with the study's hypotheses",
    "calibration_test.png": "Calibration on the test period",
    "shap_beeswarm.png": "SHAP: which features drive the score",
    "sensitivity.png": "Sensitivity of the threshold and cost to the hypotheses",
}
# Packages of the Docker dashboard image, pinned in requirements.txt / requirements-train.txt.
SPACE_PACKAGES = {
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "joblib",
    "xgboost-cpu",
    "pyarrow",
    "streamlit",
}
# Reason codes are precomputed for every transaction that can plausibly become an alert.
REASON_TOP_PER_HOUR = 60  # the slider's maximum capacity


# ------------------------------------------------------------------------- shared


def _check_inputs() -> None:
    needed = [
        ROOT / "models" / "fraud_model.joblib",
        ROOT / "reports" / "results.json",
        PROCESSED / "predictions_test.parquet",
        PROCESSED / "predictions_valid.parquet",
    ]
    missing = [p.relative_to(ROOT).as_posix() for p in needed if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Run `make train` first, missing: {missing}")


def _repo_url() -> str | None:
    github = (load_config().get("links") or {}).get("github_repo")
    return f"https://github.com/{github}" if github else None


def _results() -> dict:
    return json.loads((ROOT / "reports" / "results.json").read_text(encoding="utf-8"))


def space_readme(results: dict, repo_url: str | None, sdk: str) -> str:
    split = results["split"]["test"]
    policies = {p["policy"]: p for p in results["policies"]}
    reduction = results["cost_reduction"]["xgb_topk_profitable_vs_xgb_raw_0.5"]
    runtime = "app_file: index.html" if sdk == "static" else f"app_port: {APP_PORT}"
    code = f"\n\nCode, method and tests: {repo_url}" if repo_url else ""
    return f"""---
title: Fraud Decision Dashboard
emoji: 🛡️
colorFrom: blue
colorTo: green
sdk: {sdk}
{runtime}
pinned: false
short_description: Choose fraud alerts by expected loss under analyst capacity
---

# Fraud detection: from score to decision

A fraud team can only investigate a few alerts per hour. This dashboard compares ways of
choosing them on a held-out period of {split["rows"]:,} real card transactions
({split["frauds"]} frauds). Ranking the alerts by expected loss under the analysts' capacity
costs {policies["xgb_topk_profitable"]["cost"]:,.0f} EUR instead of
{policies["xgb_raw_0.5"]["cost"]:,.0f} EUR with the model's default threshold
({100 * reduction["value"]:.0f}% less, 95% CI {100 * reduction["lo"]:.0f}% to
{100 * reduction["hi"]:.0f}%). Change the cost hypotheses and the capacity, then open any
alert to see its SHAP reason codes.{code}

Data: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud),
ULB Machine Learning Group and Worldline, Open Database License (ODbL 1.0). This Space
redistributes the scored validation and test periods under the same licence.
"""


# ------------------------------------------------------------------------- static


def _reason_rows(test: pd.DataFrame) -> np.ndarray:
    """Test rows that some slider setting can turn into an alert."""
    expected_loss = test["p_cal"] * test["Amount"]
    by_hour = test.assign(el=expected_loss).groupby("hour")
    top_loss = by_hour["el"].rank(ascending=False, method="first") <= REASON_TOP_PER_HOUR
    top_proba = by_hour["p_cal"].rank(ascending=False, method="first") <= REASON_TOP_PER_HOUR
    return np.flatnonzero((test["p_raw"] >= 0.5) | (test["p_cal"] >= 0.01) | top_loss | top_proba)


def _precompute_reasons(test: pd.DataFrame) -> dict:
    from src.predict import FraudScorer

    scorer = FraudScorer.from_path(ROOT / "models" / "fraud_model.joblib")
    rows = _reason_rows(test)
    codes = scorer.explain(test.iloc[rows][RAW_FEATURES])
    return {
        str(row): [
            [c["feature"], c["label"], round(c["value"], 4), round(c["contribution"], 4)]
            for c in row_codes
        ]
        for row, row_codes in zip(rows, codes, strict=True)
    }


def _write(array, path: Path, dtype: str) -> None:
    np.ascontiguousarray(array, dtype=dtype).tofile(path)


def build_static(out: Path = DEFAULT_OUT) -> Path:
    _check_inputs()
    results = _results()
    valid = pd.read_parquet(PROCESSED / "predictions_valid.parquet")
    test = pd.read_parquet(PROCESSED / "predictions_test.parquet")
    for frame in (valid, test):
        if not (np.diff(frame["hour"]) >= 0).all():
            raise ValueError("Predictions must be in chronological order")
    if not np.array_equal(test["Time"], np.round(test["Time"])):
        raise ValueError("Time is expected in whole seconds")

    if out.exists():
        shutil.rmtree(out)
    (out / "data").mkdir(parents=True)
    (out / "figures").mkdir()
    shutil.copy2(STATIC_APP / "index.html", out / "index.html")
    shutil.copy2(STATIC_APP / "decision.js", out / "decision.js")
    for name in FIGURES:
        shutil.copy2(ROOT / "reports" / "figures" / name, out / "figures" / name)

    data = out / "data"
    _write(valid["p_oof"], data / "valid_p.f64", "<f8")
    _write(valid["Amount"], data / "valid_amount.f64", "<f8")
    _write(valid[TARGET], data / "valid_y.u8", "u1")
    _write(valid["hour"], data / "valid_hour.u8", "u1")
    _write(test["p_cal"], data / "test_p_cal.f64", "<f8")
    _write(test["p_raw"], data / "test_p_raw.f64", "<f8")
    _write(test["Amount"], data / "test_amount.f64", "<f8")
    _write(test["Time"], data / "test_time.u32", "<u4")
    _write(test[TARGET], data / "test_y.u8", "u1")
    _write(test["hour"], data / "test_hour.u8", "u1")
    (data / "reasons.json").write_text(
        json.dumps(_precompute_reasons(test), separators=(",", ":")), encoding="utf-8"
    )

    summary = {
        "version": results["run"]["model_version"],
        "calibration": results["calibration"]["selected"],
        "defaults": results["decision"]["cost_params"],
        "test_fraud_rate": results["split"]["test"]["fraud_rate"],
        "score_psi": results["monitoring"]["score_psi"],
        "score_psi_status": results["monitoring"]["score_psi_status"],
        "repo_url": _repo_url(),
        "models": [
            {
                "label": m["label"],
                "pr_auc": m["pr_auc"],
                "lo": m["pr_auc_lo"],
                "hi": m["pr_auc_hi"],
                "roc_auc": m["roc_auc"],
                "brier": m["brier"],
            }
            for m in results["models"]
        ],
        "figures": [
            {"file": f"figures/{name}", "caption": caption} for name, caption in FIGURES.items()
        ],
    }
    (data / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (out / "README.md").write_text(space_readme(results, _repo_url(), "static"), encoding="utf-8")
    return out


# ------------------------------------------------------------------------- docker


def src_dependencies(entry: Path = DASHBOARD) -> list[str]:
    """Every src module `entry` imports, directly or transitively (parsed, not executed)."""
    found = {"__init__.py"}
    todo = [entry]
    while todo:
        tree = ast.parse(todo.pop().read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module == "src":
                    modules = [alias.name for alias in node.names]
                elif node.module.startswith("src."):
                    modules = [node.module.split(".")[1]]
            elif isinstance(node, ast.Import):
                modules = [a.name.split(".")[1] for a in node.names if a.name.startswith("src.")]
            for module in modules:
                path = ROOT / "src" / f"{module}.py"
                if path.exists() and path.name not in found:
                    found.add(path.name)
                    todo.append(path)
    return sorted(found)


def space_requirements() -> str:
    lines = []
    for name in ("requirements.txt", "requirements-train.txt"):
        for line in (ROOT / name).read_text(encoding="utf-8").splitlines():
            spec = line.split(";")[0].strip()
            package = re.split(r"[=<>!~ ]", spec, maxsplit=1)[0].lower()
            if package in SPACE_PACKAGES:
                lines.append(spec)
    missing = SPACE_PACKAGES - {re.split(r"=", spec)[0].lower() for spec in lines}
    if missing:
        raise ValueError(f"Not pinned in the requirements files: {sorted(missing)}")
    return "\n".join(lines) + "\n"


def dockerfile(repo_url: str | None) -> str:
    env_repo = f"ENV REPO_URL={repo_url}\n" if repo_url else ""
    return f"""FROM python:3.11-slim

# XGBoost needs the OpenMP runtime.
RUN apt-get update \\
    && apt-get install -y --no-install-recommends libgomp1 \\
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces run containers as uid 1000.
RUN useradd --create-home --uid 1000 user
USER user
ENV HOME=/home/user PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
{env_repo}WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --user -r requirements.txt

COPY --chown=user . .

EXPOSE {APP_PORT}
CMD ["python", "-m", "streamlit", "run", "app/dashboard.py", "--server.port={APP_PORT}", "--server.address=0.0.0.0"]
"""


def _slim_test_predictions(path: Path) -> pd.DataFrame:
    """Only what the dashboard reads; PCA components as float32 (XGBoost predicts in float32)."""
    frame = pd.read_parquet(path)
    frame = frame[[*RAW_FEATURES, TARGET, "hour", "p_raw", "p_cal"]].copy()
    frame[V_COLUMNS] = frame[V_COLUMNS].astype("float32")
    return frame


def build_docker(out: Path = DOCKER_OUT) -> Path:
    _check_inputs()
    if out.exists():
        shutil.rmtree(out)
    for sub in ("src", "app", "models", "reports/figures", "data/processed", ".streamlit"):
        (out / sub).mkdir(parents=True)
    for module in src_dependencies():
        shutil.copy2(ROOT / "src" / module, out / "src" / module)
    shutil.copy2(DASHBOARD, out / "app" / "dashboard.py")
    shutil.copy2(ROOT / "models" / "fraud_model.joblib", out / "models" / "fraud_model.joblib")
    shutil.copy2(ROOT / "reports" / "results.json", out / "reports" / "results.json")
    for name in FIGURES:
        shutil.copy2(ROOT / "reports" / "figures" / name, out / "reports" / "figures" / name)
    _slim_test_predictions(PROCESSED / "predictions_test.parquet").to_parquet(
        out / "data" / "processed" / "predictions_test.parquet", index=False
    )
    shutil.copy2(PROCESSED / "predictions_valid.parquet", out / "data" / "processed")
    shutil.copy2(ROOT / ".streamlit" / "config.toml", out / ".streamlit" / "config.toml")
    (out / "requirements.txt").write_text(space_requirements(), encoding="utf-8")
    (out / "Dockerfile").write_text(dockerfile(_repo_url()), encoding="utf-8")
    (out / "README.md").write_text(
        space_readme(_results(), _repo_url(), "docker"), encoding="utf-8"
    )
    return out


# ------------------------------------------------------------------------ publish


def set_space_link(repo_id: str) -> None:
    """Record the published Space in configs/config.yaml (keeps the file's comments)."""
    text = CONFIG_PATH.read_text(encoding="utf-8")
    text, n = re.subn(r"(?m)^(  hf_space:) [^#\n]*?(\s*#.*)?$", rf"\1 {repo_id}\2", text, count=1)
    if n != 1:
        raise ValueError("links.hf_space not found in configs/config.yaml")
    CONFIG_PATH.write_text(text, encoding="utf-8")


def publish(out: Path, repo_id: str, sdk: str = "static", private: bool = False) -> str:
    from huggingface_hub import HfApi

    from src import report

    api = HfApi()
    api.create_repo(repo_id, repo_type="space", space_sdk=sdk, private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(out),
        repo_id=repo_id,
        repo_type="space",
        commit_message=f"Deploy dashboard, model {_results()['run']['model_version']}",
    )
    set_space_link(repo_id)
    report.main()  # the README now shows the live-demo badge
    return f"https://huggingface.co/spaces/{repo_id}"


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("action", choices=["build", "publish"])
    parser.add_argument("--docker", action="store_true", help="Streamlit Docker Space instead")
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--repo-id", help="Hugging Face Space id, e.g. USER/fraud-decision-dashboard"
    )
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args(argv)

    if args.docker:
        out = build_docker(args.out or DOCKER_OUT)
    else:
        out = build_static(args.out or DEFAULT_OUT)
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"Space built in {out} ({size:.1f} MB)")
    if args.action == "publish":
        if not args.repo_id:
            parser.error("publish needs --repo-id USER/NAME")
        sdk = "docker" if args.docker else "static"
        print("Published:", publish(out, args.repo_id, sdk, args.private))


if __name__ == "__main__":
    main()
