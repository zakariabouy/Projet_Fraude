"""End-to-end pipeline: raw CSV -> production artefact, figures, results and report.

    python -m src.train                    # full run on the real data
    python -m src.train --n-trials 10      # quicker Optuna search
    python -m src.train --reuse-params     # skip Optuna, reuse models/best_params.json
    python -m src.train --synthetic --output-dir artifacts/smoke --no-mlflow   # CI smoke run

Steps: data -> chronological split -> baselines + tuned XGBoost -> imbalance ablation ->
calibration (cross-fitted on validation) -> cost-optimal threshold (validation) ->
one final evaluation on test -> sensitivity -> SHAP -> PSI -> artefacts, MLflow, report.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

import src
from src import plots, report
from src.calibration import compare_calibrations, fit_calibrator
from src.config import CONFIG_PATH, ROOT, load_config, resolve
from src.cost import (
    CostParams,
    capacity_is_binding,
    cost_summary,
    enforce_capacity,
    f1_optimal_threshold,
    max_alerts_per_hour,
    optimise_threshold,
    threshold_curve,
    threshold_grid,
    threshold_policy,
    topk_policy,
    transaction_costs,
)
from src.data import RAW_FEATURES, TARGET, describe_part, download, load_raw, temporal_split, xy
from src.eda import eda_summary, hourly_profile
from src.explain import global_importance, native_contributions, reason_codes, shap_explanation
from src.features import FraudFeatureEngineer, hour_index
from src.metrics import bootstrap_cost_reduction, bootstrap_pr_auc, brier, precision_at_k, roc_auc
from src.models import baseline_pipelines, xgb_classifier, xgb_pipeline
from src.monitoring import psi, psi_report, psi_status
from src.predict import make_bundle, save_bundle
from src.synthetic import make_synthetic_transactions
from src.tune import tune_xgboost

log = logging.getLogger("train")

MODEL_LABELS = {
    "dummy": "Dummy (prior)",
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "xgboost_raw": "XGBoost (uncalibrated)",
    "xgboost": "XGBoost (calibrated)",
}

POLICY_LABELS = {
    "do_nothing": "No alert at all",
    "xgb_raw_0.5": "XGBoost uncalibrated, threshold 0.5",
    "xgb_cal_0.5": "XGBoost calibrated, threshold 0.5",
    "xgb_f1": "XGBoost, F1-optimal threshold",
    "xgb_cost": "XGBoost, cost-optimal threshold t*",
    "xgb_topk": "XGBoost, top-K by expected loss",
    "xgb_topk_profitable": "XGBoost, top-K by expected loss, profitable only",
}


class Timer:
    def __init__(self):
        self.start = time.perf_counter()
        self.steps: dict[str, float] = {}
        self._last = self.start

    def lap(self, name: str) -> None:
        now = time.perf_counter()
        self.steps[name] = round(now - self._last, 1)
        log.info("%s done in %.1fs", name, now - self._last)
        self._last = now

    @property
    def total(self) -> float:
        return round(time.perf_counter() - self.start, 1)


# ------------------------------------------------------------------------ helpers


def _scored(frame: pd.DataFrame, proba: np.ndarray) -> dict:
    return {
        "y": frame[TARGET].to_numpy(),
        "p": np.asarray(proba),
        "amount": frame["Amount"].to_numpy(),
        "hours": hour_index(frame["Time"]),
    }


def imbalance_ablation(
    best_params, xgb_model, X_train, y_train, X_valid, y_valid, seed, night, boot
) -> list[dict]:
    """Same tuned XGBoost, three ways to handle the imbalance, compared on validation."""
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    unweighted = {**best_params, "scale_pos_weight": 1.0}
    variants = {
        "no_weighting": xgb_pipeline(unweighted, seed, night).fit(X_train, y_train),
        "scale_pos_weight": xgb_model,
        # SMOTE lives *inside* the training pipeline: synthetic frauds are only created
        # from the training fold, never before the split.
        "smote": ImbPipeline(
            [
                ("features", FraudFeatureEngineer(*night)),
                ("smote", SMOTE(sampling_strategy=0.1, random_state=seed)),
                ("model", xgb_classifier(unweighted, seed)),
            ]
        ).fit(X_train, y_train),
    }
    rows = []
    for name, model in variants.items():
        p = model.predict_proba(X_valid)[:, 1]
        ci = bootstrap_pr_auc(y_valid, p, boot["n_resamples"], boot["alpha"], seed)
        rows.append(
            {
                "variant": name,
                "pr_auc": ci["value"],
                "pr_auc_lo": ci["lo"],
                "pr_auc_hi": ci["hi"],
                "brier": brier(y_valid, p),
                "mean_proba": float(p.mean()),
            }
        )
    return rows


def evaluate_models(
    scores: dict, valid_scores: dict, y_valid, test: dict, params: CostParams, k_total, boot, seed
) -> list:
    rows = []
    for name, score in scores.items():
        ci = bootstrap_pr_auc(test["y"], score, boot["n_resamples"], boot["alpha"], seed)
        ci_valid = bootstrap_pr_auc(
            y_valid, valid_scores[name], boot["n_resamples"], boot["alpha"], seed
        )
        raised = score >= 0.5
        alert = enforce_capacity(raised, test["hours"], params.capacity_per_hour)
        decision = cost_summary(test["y"], alert, test["amount"], params)
        rows.append(
            {
                "model": name,
                "label": MODEL_LABELS[name],
                "pr_auc": ci["value"],
                "pr_auc_lo": ci["lo"],
                "pr_auc_hi": ci["hi"],
                "valid_pr_auc": ci_valid["value"],
                "valid_pr_auc_lo": ci_valid["lo"],
                "valid_pr_auc_hi": ci_valid["hi"],
                "roc_auc": roc_auc(test["y"], score),
                "brier": brier(test["y"], score),
                "precision_at_k": precision_at_k(test["y"], score, k_total),
                "raised_at_0.5": int(raised.sum()),
                "precision_at_0.5": decision["precision"],
                "recall_at_0.5": decision["recall"],
                "cost_at_0.5": decision["cost"],
            }
        )
    return rows


def evaluate_policies(test: dict, p_raw, params: CostParams, t_star: float, t_f1: float):
    """Every policy at the same capacity K. Returns (table, alerts by policy)."""
    y, p, amount, hours = test["y"], test["p"], test["amount"], test["hours"]
    K = params.capacity_per_hour
    raised = {
        "do_nothing": np.zeros(len(y), dtype=bool),
        "xgb_raw_0.5": p_raw >= 0.5,
        "xgb_cal_0.5": p >= 0.5,
        "xgb_f1": p >= t_f1,
        "xgb_cost": p >= t_star,
        "xgb_topk": topk_policy(p, amount, hours, K),
        "xgb_topk_profitable": topk_policy(p, amount, hours, K, params, only_profitable=True),
    }
    thresholds = {"xgb_raw_0.5": 0.5, "xgb_cal_0.5": 0.5, "xgb_f1": t_f1, "xgb_cost": t_star}
    slots = (K or 0) * len(np.unique(hours))  # investigations available over the period
    alerts, rows = {}, []
    for name, r in raised.items():
        alerts[name] = enforce_capacity(r, hours, K)
        summary = cost_summary(y, alerts[name], amount, params)
        rows.append(
            {
                "policy": name,
                "label": POLICY_LABELS[name],
                "threshold": thresholds.get(name),
                "n_raised": int(r.sum()),
                "not_investigated": int(r.sum() - alerts[name].sum()),
                "max_raised_per_hour": max_alerts_per_hour(r, hours),
                "capacity_used": alerts[name].sum() / slots if slots else None,
                **summary,
            }
        )
    return pd.DataFrame(rows), alerts


def sensitivity_analysis(sens_cfg: dict, base: CostParams, valid: dict, test: dict, grid):
    """Re-optimise t* on validation for each cost hypothesis, then score it on test."""

    def run(params: CostParams) -> dict:
        curve = threshold_curve(
            valid["y"], valid["p"], valid["amount"], valid["hours"], params, grid
        )
        t = optimise_threshold(curve)
        K = params.capacity_per_hour
        s = cost_summary(
            test["y"], threshold_policy(test["p"], t, test["hours"], K), test["amount"], params
        )
        topk = topk_policy(
            test["p"], test["amount"], test["hours"], K, params, only_profitable=True
        )
        s_topk = cost_summary(test["y"], topk, test["amount"], params)
        return {
            "t_star": t,
            "capacity_binding": capacity_is_binding(curve),
            "test_cost": s["cost"],
            "test_recall": s["recall"],
            "test_amount_recall": s["amount_recall"],
            "test_fp": s["fp"],
            "test_alerts": s["n_alerts"],
            "test_fraud_amount_lost": s["fraud_amount_lost"],
            "topk_cost": s_topk["cost"],
            "topk_amount_recall": s_topk["amount_recall"],
            "topk_fp": s_topk["fp"],
        }

    costs = pd.DataFrame(
        [
            {"c_inv": c_inv, "c_fp": c_fp, **run(replace(base, c_inv=c_inv, c_fp=c_fp))}
            for c_inv in sens_cfg["c_inv"]
            for c_fp in sens_cfg["c_fp"]
        ]
    )
    capacity = pd.DataFrame(
        [
            {"capacity": k, **run(replace(base, capacity_per_hour=k))}
            for k in sens_cfg["capacity_per_hour"]
        ]
    )
    doubled = replace(base, c_fp=2 * base.c_fp)
    fp_doubling = {
        "c_fp_base": base.c_fp,
        "c_fp_doubled": doubled.c_fp,
        "base": run(base),
        "doubled": run(doubled),
    }
    return costs, capacity, fp_doubling


def explain_cases(xgb_model, X_test, test: dict, alert, figures: Path) -> dict:
    """Waterfalls for a true positive, a false positive and a false negative of the t* policy.

    If t* raises no false positive on test, the legitimate transaction with the highest score
    is explained instead (the model's closest false alarm).
    """
    y, p, amount = test["y"].astype(bool), test["p"], test["amount"]
    picks = {}
    for case, mask, key in [
        ("true_positive", y & alert, p),
        ("false_positive", ~y & alert, p),
        ("false_negative", y & ~alert, amount),  # the most expensive missed fraud
    ]:
        if mask.any():
            candidates = np.flatnonzero(mask)
            picks[case] = int(candidates[np.argmax(key[candidates])])
    fp_title = "False positive: legitimate transaction alerted"
    if "false_positive" not in picks:
        picks["false_positive"] = int(np.flatnonzero(~y)[np.argmax(p[~y])])
        fp_title = "Most suspicious legitimate transaction (not alerted)"
    titles = {
        "true_positive": "True positive: fraud alerted",
        "false_positive": fp_title,
        "false_negative": "False negative: most expensive missed fraud",
    }
    cases = {}
    for case, i in picks.items():
        row = X_test.iloc[[i]]
        plots.plot_waterfall(
            shap_explanation(xgb_model, row)[0],
            f"{titles[case]} (p = {p[i]:.3f}, amount = {amount[i]:.2f} EUR)",
            figures / f"shap_waterfall_{case}.png",
        )
        contrib, _, X_t = native_contributions(xgb_model, row)
        cases[case] = {
            "title": titles[case],
            "alerted": bool(alert[i]),
            "row_in_test": i,
            "time_h": float(row["Time"].iloc[0] / 3600),
            "amount": float(amount[i]),
            "probability": float(p[i]),
            "reason_codes": reason_codes(contrib.iloc[0], X_t.iloc[0]),
            # What pushed the score down: the key to understanding a missed fraud.
            "legit_drivers": [
                {"feature": name, "contribution": float(value)}
                for name, value in contrib.iloc[0].sort_values().head(2).items()
            ],
        }
    return cases


def log_to_mlflow(cfg: dict, results: dict, out: Path, bundle_path: Path, figures: Path) -> str:
    import mlflow

    db = resolve(cfg["paths"]["mlflow_db"])
    artifacts = resolve(cfg["paths"]["mlflow_artifacts"])
    mlflow.set_tracking_uri(f"sqlite:///{db.as_posix()}")
    experiment = "fraud-detection"
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=artifacts.as_uri())
    mlflow.set_experiment(experiment)

    decision = results["decision"]
    policies = {row["policy"]: row for row in results["policies"]}
    with mlflow.start_run(run_name=f"xgboost-{results['run']['model_version']}") as run:
        mlflow.log_params(
            {
                "split_fractions": results["split"]["fractions_used"],
                "n_trials": results["tuning"]["n_trials"],
                "calibration": results["calibration"]["selected"],
                "t_star": round(decision["t_star"], 4),
                "t_f1": round(decision["t_f1"], 4),
                **{f"cost_{k}": v for k, v in decision["cost_params"].items()},
                **{f"xgb_{k}": v for k, v in results["tuning"]["best_params"].items()},
            }
        )
        metrics = {f"test_pr_auc_{m['model']}": m["pr_auc"] for m in results["models"]}
        metrics.update({f"test_cost_{name}": row["cost"] for name, row in policies.items()})
        metrics.update(
            {
                "test_brier_raw": results["calibration"]["test"]["brier_raw"],
                "test_brier_calibrated": results["calibration"]["test"]["brier_calibrated"],
                "cv_pr_auc_best": results["tuning"]["best_cv_pr_auc"],
                "cost_reduction_vs_raw_0.5": results["cost_reduction"]["xgb_cost_vs_xgb_raw_0.5"][
                    "value"
                ],
                "score_psi": results["monitoring"]["score_psi"],
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.log_artifacts(str(figures), "figures")
        mlflow.log_artifact(str(out / cfg["paths"]["reports_dir"] / "results.json"))
        mlflow.log_artifact(str(bundle_path), "model")
        mlflow.set_tags({"dataset": "mlg-ulb/creditcardfraud", "model": "xgboost+calibration"})
        return run.info.run_id


def _to_builtin(obj):
    if isinstance(obj, np.integer | np.bool_):
        return obj.item()
    if isinstance(obj, np.floating):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Not JSON serialisable: {type(obj)}")


# --------------------------------------------------------------------------- main


def run(cfg: dict, args: argparse.Namespace) -> dict:
    timer = Timer()
    seed = int(cfg["seed"])  # passed explicitly to every random step (split, models, SHAP sample)
    out = resolve(args.output_dir) if args.output_dir else ROOT
    models_dir = out / cfg["paths"]["models_dir"]
    reports_dir = out / cfg["paths"]["reports_dir"]
    figures = out / cfg["paths"]["figures_dir"]
    processed = out / "data" / "processed"
    for d in (models_dir, reports_dir, figures, processed):
        d.mkdir(parents=True, exist_ok=True)

    night = tuple(cfg["features"]["night_hours"])
    params = CostParams.from_config(cfg["cost"])
    boot = dict(cfg["bootstrap"])
    if args.bootstrap:
        boot["n_resamples"] = args.bootstrap
    n_trials = args.n_trials or cfg["tuning"]["n_trials"]

    # 1. Data ---------------------------------------------------------------------
    if args.synthetic:
        df = make_synthetic_transactions(seed=seed)
    else:
        raw_path = download(cfg["data"]["url"], resolve(cfg["paths"]["raw_data"]))
        df = load_raw(raw_path, cfg["data"]["expected_rows"], cfg["data"]["expected_frauds"])
    eda = eda_summary(df, night)
    plots.plot_eda(df, hourly_profile(df), figures)
    timer.lap("data + EDA")

    # 2. Chronological split -------------------------------------------------------
    split_cfg = cfg["split"]
    split = temporal_split(
        df, tuple(split_cfg["fractions"]), split_cfg["min_frauds"], split_cfg["fallback_fractions"]
    )
    X_train, y_train = xy(split.train)
    X_valid, y_valid = xy(split.valid)
    X_test, y_test = xy(split.test)
    log.info(
        "Split %s: train %d frauds, valid %d, test %d",
        split.fractions,
        y_train.sum(),
        y_valid.sum(),
        y_test.sum(),
    )
    hours_test = hour_index(split.test["Time"])
    n_hours_test = len(np.unique(hours_test))
    k_total = (params.capacity_per_hour or 0) * n_hours_test

    # 3. Baselines -----------------------------------------------------------------
    models = {
        name: pipe.fit(X_train, y_train) for name, pipe in baseline_pipelines(seed, night).items()
    }
    timer.lap("baselines")

    # 4. XGBoost: Optuna + TimeSeriesSplit on the training period ----------------
    params_path = models_dir / "best_params.json"
    if args.reuse_params and params_path.exists():
        tuning = json.loads(params_path.read_text())
        log.info("Reusing tuned parameters from %s", params_path)
    else:
        best_params, study = tune_xgboost(
            X_train,
            y_train,
            n_trials=n_trials,
            cv_splits=cfg["tuning"]["cv_splits"],
            seed=seed,
            timeout=cfg["tuning"]["timeout_s"],
            night_hours=night,
        )
        tuning = {
            "n_trials": len(study.trials),
            "n_pruned": sum(t.state.name == "PRUNED" for t in study.trials),
            "best_cv_pr_auc": study.best_value,
            "default_cv_pr_auc": study.trials[0].value,
            "best_params": best_params,
        }
        params_path.write_text(json.dumps(tuning, indent=2, default=_to_builtin))
        study.trials_dataframe().to_csv(reports_dir / "optuna_trials.csv", index=False)
    xgb_model = xgb_pipeline(tuning["best_params"], seed, night).fit(X_train, y_train)
    timer.lap("XGBoost tuning + fit")

    # 5. Imbalance handling: class weights vs SMOTE (validation) -----------------
    ablation = imbalance_ablation(
        tuning["best_params"], xgb_model, X_train, y_train, X_valid, y_valid, seed, night, boot
    )
    timer.lap("imbalance ablation")

    # 6. Calibration, cross-fitted on validation --------------------------------
    cal_cfg = cfg["calibration"]
    method, valid_probas, calib_table = compare_calibrations(
        xgb_model,
        X_valid,
        y_valid,
        cal_cfg["methods"],
        cal_cfg["cross_fit_folds"],
        cal_cfg.get("brier_tie_tolerance", 0.0),
    )
    calibrated = fit_calibrator(xgb_model, X_valid, y_valid, method)
    valid = _scored(split.valid, valid_probas[method])
    log.info("Calibration: %s selected\n%s", method, calib_table.to_string(index=False))
    timer.lap("calibration")

    # 7. Decision rule, chosen on validation only --------------------------------
    grid = threshold_grid(cfg["cost"]["threshold_grid_size"])
    curve = threshold_curve(valid["y"], valid["p"], valid["amount"], valid["hours"], params, grid)
    t_star, t_f1 = optimise_threshold(curve), f1_optimal_threshold(curve)
    curve.to_csv(reports_dir / "threshold_curve_valid.csv", index=False)
    plots.plot_cost_curve(
        curve, t_star, t_f1, params.capacity_per_hour, figures / "cost_vs_threshold.png"
    )
    log.info("t* = %.4f (cost-optimal), F1-optimal = %.4f", t_star, t_f1)
    timer.lap("threshold search")

    # 8. Final evaluation on test: used once ----------------------------------------
    p_test_raw = xgb_model.predict_proba(X_test)[:, 1]
    p_test = calibrated.predict_proba(X_test)[:, 1]
    test = _scored(split.test, p_test)
    scores = {name: m.predict_proba(X_test)[:, 1] for name, m in models.items()}
    scores["xgboost_raw"] = p_test_raw
    scores["xgboost"] = p_test
    # Validation PR-AUC is reported next to the test one: model comparisons are not
    # judged on the test set alone.
    valid_scores = {name: m.predict_proba(X_valid)[:, 1] for name, m in models.items()}
    valid_scores["xgboost_raw"] = valid_probas["raw"]
    valid_scores["xgboost"] = valid["p"]
    model_rows = evaluate_models(
        scores, valid_scores, valid["y"], test, params, k_total, boot, seed
    )
    policies, alerts = evaluate_policies(test, p_test_raw, params, t_star, t_f1)
    costs = {
        name: transaction_costs(test["y"], a, test["amount"], params) for name, a in alerts.items()
    }
    reductions = {
        f"{new}_vs_{ref}": bootstrap_cost_reduction(
            costs[ref], costs[new], boot["n_resamples"], boot["alpha"], seed
        )
        for new in ("xgb_cost", "xgb_topk", "xgb_topk_profitable")
        for ref in ("xgb_raw_0.5", "xgb_cal_0.5", "xgb_f1", "xgb_cost", "do_nothing")
        if new != ref
    }
    plots.plot_pr_curves(
        test["y"],
        {k: v for k, v in scores.items() if k not in ("dummy", "xgboost")},
        MODEL_LABELS,
        figures / "pr_curves_test.png",
    )
    plots.plot_calibration(
        test["y"],
        {
            "raw": p_test_raw,
            method: p_test,
            **{
                m: fit_calibrator(xgb_model, X_valid, y_valid, m).predict_proba(X_test)[:, 1]
                for m in cal_cfg["methods"]
                if m != method
            },
        },
        {
            "raw": "Uncalibrated (scale_pos_weight)",
            "isotonic": "Isotonic",
            "sigmoid": "Platt (sigmoid)",
        },
        figures / "calibration_test.png",
    )
    plots.plot_policies(policies[policies["policy"] != "do_nothing"], figures / "policies_test.png")
    timer.lap("test evaluation")

    # 9. Sensitivity of the decision to the cost hypotheses ---------------------
    sens_costs, sens_capacity, fp_doubling = sensitivity_analysis(
        cfg["sensitivity"], params, valid, test, grid
    )
    plots.plot_sensitivity(sens_costs, sens_capacity, figures / "sensitivity.png")
    timer.lap("sensitivity")

    # 10. SHAP ------------------------------------------------------------------------
    shap_cfg = cfg["shap"]
    sample = X_test.sample(n=min(shap_cfg["sample_size"], len(X_test)), random_state=seed)
    explanation = shap_explanation(xgb_model, sample.sort_index())
    importance = global_importance(explanation)
    plots.plot_beeswarm(explanation, figures / "shap_beeswarm.png", shap_cfg["max_display"])
    cases = explain_cases(xgb_model, X_test, test, alerts["xgb_cost"], figures)
    timer.lap("SHAP")

    # 11. Monitoring: PSI between training and test periods -----------------------
    features = xgb_model.named_steps["features"]
    psi_table = psi_report(
        features.transform(X_train), features.transform(X_test), cfg["monitoring"]["psi_bins"]
    )
    score_psi = psi(calibrated.predict_proba(X_train)[:, 1], p_test, cfg["monitoring"]["psi_bins"])
    plots.plot_psi(psi_table, figures / "psi_train_vs_test.png")
    timer.lap("monitoring")

    # 12. Artefacts ----------------------------------------------------------------------
    version = f"{src.__version__}+{datetime.now(UTC):%Y%m%dT%H%MZ}"
    test_summary = {row["policy"]: row["cost"] for row in policies.to_dict("records")}
    bundle = make_bundle(
        calibrated,
        t_star,
        params,
        method,
        version,
        metadata={
            "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "synthetic_data": bool(args.synthetic),
            "split_fractions": list(split.fractions),
            "train_period_h": [
                describe_part(split.train)["time_start_h"],
                describe_part(split.train)["time_end_h"],
            ],
            "xgb_params": tuning["best_params"],
            "t_f1": t_f1,
            "test_pr_auc": next(m["pr_auc"] for m in model_rows if m["model"] == "xgboost"),
            "test_cost_by_policy": test_summary,
            "feature_names": list(features.get_feature_names_out()),
        },
    )
    bundle_path = save_bundle(bundle, models_dir / "fraud_model.joblib")

    test_frame = split.test[RAW_FEATURES + [TARGET]].assign(
        hour=hours_test,
        p_raw=p_test_raw,
        p_cal=p_test,
        **{f"alert_{k}": v for k, v in alerts.items() if k != "do_nothing"},
    )
    test_frame.to_parquet(processed / "predictions_test.parquet", index=False)
    split.valid[["Time", "Amount", TARGET]].assign(
        hour=valid["hours"], p_oof=valid["p"]
    ).to_parquet(processed / "predictions_valid.parquet", index=False)

    psi_records = psi_table.to_dict("records")
    results = {
        "run": {
            "model_version": version,
            "synthetic_data": bool(args.synthetic),
            "seed": seed,
            "python": platform.python_version(),
            "duration_s": timer.total,
            "step_durations_s": timer.steps,
        },
        "data": eda,
        "split": {
            "fractions_requested": list(split_cfg["fractions"]),
            "fractions_used": list(split.fractions),
            "fallback_used": list(split.fractions) != list(split_cfg["fractions"]),
            "train": describe_part(split.train),
            "valid": describe_part(split.valid),
            "test": describe_part(split.test),
            "test_hours": n_hours_test,
        },
        "tuning": {**tuning, "neg_pos_ratio": float((y_train == 0).sum() / (y_train == 1).sum())},
        "imbalance_ablation": ablation,
        "calibration": {
            "selected": method,
            "cross_fit_folds": cal_cfg["cross_fit_folds"],
            "brier_tie_tolerance": cal_cfg.get("brier_tie_tolerance", 0.0),
            "valid_oof": calib_table.to_dict("records"),
            "test": {
                "brier_raw": brier(test["y"], p_test_raw),
                "brier_calibrated": brier(test["y"], p_test),
                "mean_proba_raw": float(p_test_raw.mean()),
                "mean_proba_calibrated": float(p_test.mean()),
                "fraud_rate": float(test["y"].mean()),
            },
        },
        "decision": {
            "cost_params": params.to_dict(),
            "t_star": t_star,
            "t_f1": t_f1,
            "capacity_binding": capacity_is_binding(curve),
            "unconstrained_best_threshold": float(curve.loc[curve["cost"].idxmin(), "threshold"]),
            "unconstrained_max_per_hour": int(curve.loc[curve["cost"].idxmin(), "max_per_hour"]),
            "valid_at_t_star": curve.iloc[(curve["threshold"] - t_star).abs().argmin()].to_dict(),
            "precision_at_k_total": k_total,
        },
        "models": model_rows,
        "policies": policies.to_dict("records"),
        "cost_reduction": reductions,
        "sensitivity": {
            "costs": sens_costs.to_dict("records"),
            "capacity": sens_capacity.to_dict("records"),
            "fp_cost_doubling": fp_doubling,
        },
        "shap": {
            "sample_size": len(sample),
            "importance": importance.head(15).to_dict("records"),
            "cases": cases,
        },
        "monitoring": {
            "psi": psi_records,
            "score_psi": score_psi,
            "score_psi_status": psi_status(score_psi),
        },
        "artefacts": {"bundle": bundle_path.relative_to(out).as_posix()},
    }
    results_path = reports_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2, default=_to_builtin), encoding="utf-8")
    timer.lap("artefacts")

    if not args.no_mlflow:
        try:
            run_id = log_to_mlflow(cfg, results, out, bundle_path, figures)
            log.info("MLflow run %s", run_id)
        except Exception:  # tracking must never break a training run
            log.exception("MLflow logging failed")

    report.write_reports(results, out, update_readme=not args.synthetic)
    log.info("Pipeline finished in %.1fs. Results: %s", timer.total, results_path)
    return results


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--n-trials", type=int, help="override tuning.n_trials")
    parser.add_argument("--bootstrap", type=int, help="override bootstrap.n_resamples")
    parser.add_argument("--reuse-params", action="store_true", help="skip Optuna if possible")
    parser.add_argument("--synthetic", action="store_true", help="synthetic data (smoke test)")
    parser.add_argument("--output-dir", help="write models/ and reports/ here instead of the repo")
    parser.add_argument("--no-mlflow", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
    )
    args = parse_args(argv)
    cfg = load_config(args.config)
    if args.synthetic:
        cfg["tuning"]["cv_splits"] = 3
    run(cfg, args)


if __name__ == "__main__":
    main()
