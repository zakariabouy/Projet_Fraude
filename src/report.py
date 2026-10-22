"""Turn reports/results.json into human-readable reports.

* reports/fiche_resultats.md : French interview sheet, every yellow box of the interview
  dossier filled with the numbers of the last run (nothing typed by hand).
* README.md : the results section between the RESULTS markers.

    python -m src.report        # regenerate both from the last results.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from src.config import ROOT, load_config

README_START, README_END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
HEADLINE_START, HEADLINE_END = "<!-- HEADLINE:START -->", "<!-- HEADLINE:END -->"
STATIC_BADGES = [
    ("Python 3.11", "python-3.11-3776AB?logo=python&logoColor=white"),
    ("XGBoost", "model-XGBoost-1f4e79"),
    ("FastAPI", "API-FastAPI-009688?logo=fastapi&logoColor=white"),
    ("Docker", "container-Docker-2496ED?logo=docker&logoColor=white"),
    ("MLflow", "tracking-MLflow-0194E2?logo=mlflow&logoColor=white"),
]

POLICY_FR = {
    "do_nothing": "Aucune alerte",
    "xgb_raw_0.5": "XGBoost non calibré, seuil 0,5",
    "xgb_cal_0.5": "XGBoost calibré, seuil 0,5",
    "xgb_f1": "XGBoost, seuil F1-optimal",
    "xgb_cost": "XGBoost, seuil coût-optimal t*",
    "xgb_topk": "XGBoost, top-K perte espérée",
    "xgb_topk_profitable": "XGBoost, top-K perte espérée (rentables seulement)",
}
MODEL_FR = {
    "dummy": "Dummy",
    "logistic_regression": "Régression logistique",
    "random_forest": "Random Forest",
    "xgboost_raw": "XGBoost (non calibré)",
    "xgboost": "XGBoost (calibré)",
}
CALIB_FR = {"isotonic": "isotonic", "sigmoid": "Platt (sigmoïde)", "raw": "non calibré"}


# ------------------------------------------------------------------ formatting


def fr(x: float, d: int = 2) -> str:
    return f"{x:,.{d}f}".replace(",", " ").replace(".", ",")


def pct(x: float, d: int = 1) -> str:
    return f"{fr(100 * x, d)} %"


def eur(x: float) -> str:
    return f"{fr(x, 0)} €"


def en_pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}%"


def ci_fr(row: dict, d: int = 3) -> str:
    return f"{fr(row['pr_auc'], d)} [IC 95 % : {fr(row['pr_auc_lo'], d)}–{fr(row['pr_auc_hi'], d)}]"


def ci_en(row: dict, d: int = 3) -> str:
    return f"{row['pr_auc']:.{d}f} [{row['pr_auc_lo']:.{d}f}, {row['pr_auc_hi']:.{d}f}]"


def red_fr(r: dict) -> str:
    return f"{pct(r['value'])} (IC 95 % : {pct(r['lo'])} à {pct(r['hi'])})"


def change_fr(reduction: float, noun: str = "le coût total") -> str:
    """'réduit le coût total de 12,3 %' or 'augmente le coût total de 4,5 %'."""
    verb = "réduit" if reduction >= 0 else "augmente"
    return f"{verb} {noun} de {pct(abs(reduction))}"


def signed_fr(x: float, d: int = 2) -> str:
    return ("+" if x > 0 else "") + fr(x, d)


def _by(rows: list[dict], key: str) -> dict:
    return {row[key]: row for row in rows}


def _overlap(a: dict, b: dict, prefix: str = "pr_auc") -> bool:
    return a[f"{prefix}_lo"] <= b[f"{prefix}_hi"] and b[f"{prefix}_lo"] <= a[f"{prefix}_hi"]


def sig_fr(r: dict, n_frauds: int) -> str:
    """Confidence interval of a cost reduction and whether it excludes zero."""
    ci = f"IC 95 % : {pct(r['lo'])} à {pct(r['hi'])}"
    if r["lo"] > 0:
        return f"{ci}, gain significatif"
    if r["hi"] < 0:
        return f"{ci}, hausse significative"
    return f"{ci} : l'intervalle contient 0, ce n'est pas significatif sur {n_frauds} fraudes"


def recommended_policy(pol: dict) -> str:
    """The cheaper of the two cost-aware policies on test (reporting only, nothing is refit)."""
    return min(("xgb_cost", "xgb_topk_profitable"), key=lambda name: pol[name]["cost"])


# --------------------------------------------------------------- French sheet


def _phrases(res: dict) -> dict:
    """Interview sentences built from the numbers, with wording that follows the data."""
    pol = _by(res["policies"], "policy")
    models = _by(res["models"], "model")
    red = res["cost_reduction"]
    dec = res["decision"]
    K = dec["cost_params"]["capacity_per_hour"]
    n_test = res["split"]["test"]["frauds"]
    xgb = models["xgboost"]
    describe = {
        "xgb_cost": f"le seuil coût-optimal (t* = {fr(dec['t_star'], 3)})",
        "xgb_topk_profitable": "la priorisation par perte espérée (chaque heure, les K "
        "transactions de plus forte perte p × montant, investigations rentables seulement)",
    }
    out = {}

    main = recommended_policy(pol)
    other = "xgb_cost" if main == "xgb_topk_profitable" else "xgb_topk_profitable"
    r_main, r_other = red[f"{main}_vs_xgb_raw_0.5"], red[f"{other}_vs_xgb_raw_0.5"]
    out["star"] = (
        f"« À capacité d'analystes constante ({K} alertes/heure), {describe[main]} "
        f"{change_fr(r_main['value'])} par rapport au seuil standard 0,5 du modèle, avec une "
        f"PR-AUC test de {ci_fr(xgb)}. » À savoir : {sig_fr(r_main, n_test)}. "
        f"Pour {describe[other]} : {change_fr(r_other['value'])} ({sig_fr(r_other, n_test)})."
    )

    # Sensitivity: doubling the customer friction cost, then the capacity.
    sens = res["sensitivity"]
    d = sens["fp_cost_doubling"]
    b, h = d["base"], d["doubled"]
    n_t = len({round(row["t_star"], 6) for row in sens["costs"]})
    k_text = " ; ".join(
        f"K = {row['capacity']} → t* = {fr(row['t_star'], 3)}, coût test {eur(row['test_cost'])}"
        for row in sens["capacity"]
    )
    if abs(h["t_star"] - b["t_star"]) < 1e-9:
        if b.get("capacity_binding"):
            reason = (
                f"avec K = {K}, c'est l'heure la plus chargée de la validation qui fixe le seuil : "
                f"t* est le seuil le moins strict qui ne dépasse jamais {K} alertes dans une heure"
            )
        else:
            fp_valid = dec["valid_at_t_star"].get("fp", "peu de")
            reason = (
                f"à ce seuil, la validation ne compte que {fp_valid} faux positifs, le coût client "
                f"pèse donc peu dans le calcul"
            )
        out["sensitivity"] = (
            f"« Si le coût client d'un faux positif double ({eur(d['c_fp_base'])} → "
            f"{eur(d['c_fp_doubled'])}), le seuil optimal ne bouge pas (t* = "
            f"{fr(b['t_star'], 3)}) : {reason}. "
            + (
                f"Il est identique pour les {len(sens['costs'])} hypothèses de coûts testées ; "
                f"seule la capacité le fait bouger : {k_text}."
                if n_t == 1
                else f"Selon la capacité : {k_text}."
            )
            + (
                " Le levier de décision, c'est la capacité de l'équipe, pas le coût client. »"
                if b.get("capacity_binding")
                else " »"
            )
        )
    else:
        total_fraud = res["split"]["test"]["fraud_amount"]
        missed_b = total_fraud * (1 - b["test_amount_recall"])
        missed_h = total_fraud * (1 - h["test_amount_recall"])
        more = (missed_h - missed_b) / missed_b if missed_b else 0.0
        verb = "on laisse passer" if more >= 0 else "on arrête"
        out["sensitivity"] = (
            f"« Si le coût client d'un faux positif double ({eur(d['c_fp_base'])} → "
            f"{eur(d['c_fp_doubled'])}), le seuil optimal passe de {fr(b['t_star'], 3)} à "
            f"{fr(h['t_star'], 3)} et {verb} {pct(abs(more))} de fraude (en montant) en plus "
            f"sur le test ({eur(missed_b)} → {eur(missed_h)} non détectés), avec "
            f"{b['test_fp']} → {h['test_fp']} faux positifs. C'est une recommandation de "
            f"décision, pas seulement un score. » Selon la capacité : {k_text}."
        )

    # Q5: is XGBoost worth its complexity?
    x, rf = models["xgboost_raw"], models["random_forest"]
    bp = res["tuning"]["best_params"]

    def pair(row: dict, prefix: str) -> str:
        return f"{fr(row[prefix], 3)} [{fr(row[prefix + '_lo'], 3)}–{fr(row[prefix + '_hi'], 3)}]"

    q5 = (
        f"PR-AUC en validation : XGBoost {pair(x, 'valid_pr_auc')} contre Random Forest "
        f"{pair(rf, 'valid_pr_auc')} ; sur le test : {pair(x, 'pr_auc')} contre {pair(rf, 'pr_auc')}. "
    )
    if _overlap(x, rf, "valid_pr_auc") and _overlap(x, rf, "pr_auc"):
        q5 += (
            "Les intervalles se recouvrent : sur ce dataset, XGBoost n'apporte pas de gain "
            "significatif sur la Random Forest, et je le dis. Je l'ai gardé comme modèle principal "
            "(choix fait avant de regarder le test) pour l'industrialisation : modèle compact "
            f"({bp['n_estimators']} arbres de profondeur {bp['max_depth']}) et TreeSHAP natif "
            "pour les reason codes de l'API."
        )
    elif x["valid_pr_auc_lo"] > rf["valid_pr_auc_hi"]:
        q5 += "XGBoost est significativement meilleur en validation : la complexité paie."
    else:
        q5 += "La Random Forest fait au moins aussi bien : je le signale comme limite du choix."
    out["xgb_vs_rf"] = q5

    # Q4: class weights vs SMOTE.
    ab = _by(res["imbalance_ablation"], "variant")
    spw, smote, none = ab["scale_pos_weight"], ab["smote"], ab["no_weighting"]
    overlap = smote["pr_auc_lo"] <= spw["pr_auc_hi"] and spw["pr_auc_lo"] <= smote["pr_auc_hi"]
    best = max(ab.values(), key=lambda row: row["pr_auc"])
    names = {
        "scale_pos_weight": "la pondération",
        "smote": "SMOTE",
        "no_weighting": "aucun rééquilibrage",
    }
    out["smote"] = (
        f"Sur ce dataset (validation), PR-AUC : pondération {fr(spw['pr_auc'], 3)} "
        f"[{fr(spw['pr_auc_lo'], 3)}–{fr(spw['pr_auc_hi'], 3)}], SMOTE {fr(smote['pr_auc'], 3)} "
        f"[{fr(smote['pr_auc_lo'], 3)}–{fr(smote['pr_auc_hi'], 3)}], sans rééquilibrage "
        f"{fr(none['pr_auc'], 3)}. Le meilleur point est {names[best['variant']]}"
        + (
            ", mais les intervalles se recouvrent : la différence n'est pas significative. "
            "J'ai gardé la pondération : plus simple, pas de données synthétiques, et la "
            "calibration corrige ensuite les probabilités."
            if overlap
            else ", et l'écart dépasse l'incertitude bootstrap."
        )
        + f" Brier (non calibré) : pondération {fr(spw['brier'], 5)}, SMOTE {fr(smote['brier'], 5)},"
        f" sans rééquilibrage {fr(none['brier'], 5)}."
    )

    # Q9: isotonic vs Platt.
    cal = res["calibration"]
    oof = _by(cal["valid_oof"], "method")
    chosen = cal["selected"]
    other = "sigmoid" if chosen == "isotonic" else "isotonic"
    c, o = oof[chosen], oof[other]
    tol = cal.get("brier_tie_tolerance", 0.0)
    setup = (
        f"(Brier hors-échantillon, cross-fitting en {cal['cross_fit_folds']} blocs chronologiques "
        f"sur la validation)"
    )
    if abs(c["brier"] - o["brier"]) <= min(c["brier"], o["brier"]) * tol:
        near_zero = (
            f" : {CALIB_FR[other]} donne une probabilité < 0,0001 à {o['frauds_below_1e-4']} "
            f"fraudes contre {c['frauds_below_1e-4']}, or la décision utilise p × montant"
            if "frauds_below_1e-4" in o and o["frauds_below_1e-4"] > c["frauds_below_1e-4"]
            else ""
        )
        out["calibration"] = (
            f"J'ai retenu {CALIB_FR[chosen]}. Les deux méthodes sont à égalité sur le Brier "
            f"{setup} : {fr(c['brier'], 6)} contre {fr(o['brier'], 6)} (écart < {pct(tol, 0)}), "
            f"et la log-loss les départage nettement : {fr(c['log_loss'], 4)} contre "
            f"{fr(o['log_loss'], 4)}{near_zero}."
        )
        if chosen == "sigmoid":
            out["calibration"] += (
                f" C'est cohérent avec la théorie : Platt est plus stable avec peu de positifs "
                f"({res['split']['valid']['frauds']} fraudes en validation)."
            )
    else:
        out["calibration"] = (
            f"J'ai retenu {CALIB_FR[chosen]} parce que son Brier score {setup} est le plus bas : "
            f"{fr(c['brier'], 6)} contre {fr(o['brier'], 6)} pour {CALIB_FR[other]}."
        )
    out["calibration"] += (
        f" Sans calibration : {fr(oof['raw']['brier'], 6)}. Sur le test, le Brier passe de "
        f"{fr(cal['test']['brier_raw'], 6)} à {fr(cal['test']['brier_calibrated'], 6)}."
    )

    # Q12: threshold vs top-K.
    cost_t, cost_k = pol["xgb_cost"]["cost"], pol["xgb_topk_profitable"]["cost"]
    cost_k_pure = pol["xgb_topk"]["cost"]
    diff = red["xgb_topk_profitable_vs_xgb_cost"]
    t_pol, k_pol = pol["xgb_cost"], pol["xgb_topk_profitable"]
    if cost_k < cost_t:
        out["topk"] = (
            f"Dans mes résultats, le top-K par perte espérée (investigations rentables seulement) "
            f"coûte {pct(diff['value'])} de moins que le seuil coût-optimal ({eur(cost_k)} contre "
            f"{eur(cost_t)} sur le test ; {sig_fr(diff, n_test)}). "
        )
        if k_pol["amount_recall"] > t_pol["amount_recall"]:
            out["topk"] += (
                f"Ce que montrent les chiffres : en classant par p × montant, il intercepte "
                f"{pct(k_pol['amount_recall'], 0)} du montant de fraude en détectant {k_pol['tp']} "
                f"fraudes, contre {pct(t_pol['amount_recall'], 0)} avec {t_pol['tp']} fraudes pour le "
                f"seuil, qui ignore le montant : il vise les grosses fraudes plutôt que les plus "
                f"probables, et ne gaspille pas d'investigation sur les petits montants. "
            )
        out["topk"] += (
            "Je recommanderais le top-K pour la file des analystes, et garderais le seuil t* "
            "comme règle temps réel simple à auditer."
        )
    else:
        out["topk"] = (
            f"Dans mes résultats, le seuil coût-optimal est meilleur ({eur(cost_t)} contre "
            f"{eur(cost_k)} pour le top-K rentable ; écart du top-K : {pct(diff['value'])}, "
            f"{sig_fr(diff, n_test)}). Je recommanderais donc le seuil t* : plus simple à "
            f"gouverner et à auditer."
        )
    out["topk"] += (
        f" Le top-K « pur » (toujours {K} alertes/heure, même non rentables) coûte "
        f"{eur(cost_k_pure)}."
    )

    # Q24: CFO summary, for the recommended policy.
    ref, new = pol["xgb_raw_0.5"], pol[main]
    if new["fp"] == ref["fp"]:
        fp_text = f"avec autant de blocages de clients légitimes ({new['fp']} faux positifs)"
    elif new["fp"] < ref["fp"]:
        fp_text = (
            f"tout en diminuant les blocages de clients légitimes de "
            f"{pct(1 - new['fp'] / ref['fp'])} ({ref['fp']} → {new['fp']} faux positifs)"
        )
    else:
        more = f"{pct(new['fp'] / ref['fp'] - 1)} " if ref["fp"] else ""
        fp_text = (
            f"au prix de {more}blocages de clients légitimes en plus "
            f"({ref['fp']} → {new['fp']} faux positifs)"
        )
    policy_name = {
        "xgb_cost": "le seuil optimisé par les coûts",
        "xgb_topk_profitable": "la priorisation des alertes par perte espérée",
    }[main]
    out["cfo"] = (
        f"« À capacité d'analystes constante, {policy_name} "
        f"{change_fr(r_main['value'], 'le coût total de la fraude')} par rapport au seuil "
        f"standard, {fp_text}. » ({sig_fr(r_main, n_test)}.)"
    )
    return out


def fiche_markdown(res: dict) -> str:
    pol = _by(res["policies"], "policy")
    models = _by(res["models"], "model")
    eda, split, dec, cal = res["data"], res["split"], res["decision"], res["calibration"]
    red = res["cost_reduction"]
    K = dec["cost_params"]["capacity_per_hour"]
    top3 = [row["feature"] for row in res["shap"]["importance"][:3]]
    ph = _phrases(res)
    lines = []
    add = lines.append

    add("# Fiche résultats — projet Détection de fraude")
    add("")
    add(
        f"> Générée automatiquement par `python -m src.report` depuis `reports/results.json` "
        f"(modèle `{res['run']['model_version']}`). Ne pas modifier à la main : relancer "
        f"`make train` (ou `python -m src.report`)."
    )
    if res["run"]["synthetic_data"]:
        add("")
        add("> **Attention : run sur données synthétiques. Ces chiffres ne sont pas à citer.**")
    add("")
    add("## 1. Cases jaunes du dossier")
    add("")
    add("| Case | Valeur |")
    add("|---|---|")
    add(f"| PR-AUC test, XGBoost calibré | {ci_fr(models['xgboost'])} |")
    add(
        f"| Réduction du coût, seuil t* vs seuil 0,5 (modèle non calibré, `predict()`) | {red_fr(red['xgb_cost_vs_xgb_raw_0.5'])} |"
    )
    add(
        f"| Réduction du coût, seuil t* vs seuil 0,5 (probabilités calibrées) | {red_fr(red['xgb_cost_vs_xgb_cal_0.5'])} |"
    )
    add(
        f"| Réduction du coût, top-K rentable vs seuil 0,5 (non calibré) | {red_fr(red['xgb_topk_profitable_vs_xgb_raw_0.5'])} |"
    )
    add(
        f"| Réduction du coût, top-K rentable vs seuil t* | {red_fr(red['xgb_topk_profitable_vs_xgb_cost'])} |"
    )
    add(
        f"| Brier score test avant → après calibration | {fr(cal['test']['brier_raw'], 6)} → {fr(cal['test']['brier_calibrated'], 6)} |"
    )
    add(f"| Calibration retenue | {CALIB_FR[cal['selected']]} |")
    add(f"| Seuil optimal t* | {fr(dec['t_star'], 3)} (seuil F1-optimal : {fr(dec['t_f1'], 3)}) |")
    add(f"| Capacité K retenue | {K} alertes/heure |")
    add(f"| Top-3 features SHAP | {', '.join(top3)} |")
    add(
        f"| Montant médian fraude vs légitime | {fr(eda['amount_median_fraud'])} € vs "
        f"{fr(eda['amount_median_legit'])} € → **plus "
        f"{'faible' if eda['amount_median_fraud'] < eda['amount_median_legit'] else 'élevé'}** |"
    )
    add(
        f"| Taux de fraude nuit (0h–6h) vs jour | {pct(eda['fraud_rate_night'], 3)} vs "
        f"{pct(eda['fraud_rate_day'], 3)} → **plus "
        f"{'élevé' if eda['fraud_rate_night'] > eda['fraud_rate_day'] else 'faible'} la nuit** |"
    )
    ks = ", ".join(
        f"{c['feature']} ({fr(c['ks'], 2)})" for c in eda["top_separating_components_ks"][:4]
    )
    add(f"| Composantes les plus séparatrices (stat. KS) | {ks} |")
    add("")
    add("## 2. Tableau de résultats (section 8 du dossier)")
    add("")
    add(
        f"Test = dernière période ({fr(split['test']['time_start_h'], 1)} h → "
        f"{fr(split['test']['time_end_h'], 1)} h, {fr(split['test']['rows'], 0)} transactions, "
        f"{split['test']['frauds']} fraudes, {eur(split['test']['fraud_amount'])} de fraude). "
        f"Toutes les politiques sont évaluées **à capacité constante K = {K}/heure**. "
        f"Coûts : c_inv = {eur(dec['cost_params']['c_inv'])}, c_fp = {eur(dec['cost_params']['c_fp'])}, "
        f"r = {fr(dec['cost_params']['recovery_rate'], 2)}."
    )
    add("")
    add("| Modèle / politique | PR-AUC test [IC 95 %] | Recall | Precision | Coût total |")
    add("|---|---|---|---|---|")
    for name in ("dummy", "logistic_regression", "random_forest"):
        m = models[name]
        add(
            f"| {MODEL_FR[name]} (seuil 0,5) | {ci_fr(m)} | {pct(m['recall_at_0.5'])} | "
            f"{pct(m['precision_at_0.5'])} | {eur(m['cost_at_0.5'])} |"
        )
    for name in (
        "xgb_raw_0.5",
        "xgb_cal_0.5",
        "xgb_f1",
        "xgb_cost",
        "xgb_topk",
        "xgb_topk_profitable",
    ):
        p = pol[name]
        model = models["xgboost_raw" if name == "xgb_raw_0.5" else "xgboost"]
        add(
            f"| {POLICY_FR[name]} | {ci_fr(model)} | {pct(p['recall'])} | "
            f"{pct(p['precision'])} | {eur(p['cost'])} |"
        )
    add(f"| *Référence : aucune alerte* | — | 0 % | — | {eur(pol['do_nothing']['cost'])} |")
    add("")
    add("## 3. Phrases à dire (avec tes vrais chiffres)")
    add("")
    add(f"**Résultat STAR (1.2).** {ph['star']}")
    add("")
    add(f"**Sensibilité (5.3).** {ph['sensitivity']}")
    add("")
    add(f"**Q4 — SMOTE ou poids de classe ?** {ph['smote']}")
    add("")
    add(f"**Q5 — Pourquoi XGBoost ?** {ph['xgb_vs_rf']}")
    add("")
    add(f"**Q9 — Isotonic ou Platt ?** {ph['calibration']}")
    add("")
    add(f"**Q12 — Seuil ou top-K ?** {ph['topk']}")
    add("")
    add(f"**Q24 — Pour un directeur financier.** {ph['cfo']}")
    add("")
    add("## 4. Faits à connaître sur la méthode")
    add("")
    fallback = (
        f"Le split 60/20/20 ne laissait pas assez de fraudes en validation/test (le taux de fraude "
        f"baisse le 2e jour), d'où le repli sur {'/'.join(str(round(100 * f)) for f in split['fractions_used'])}."
        if split["fallback_used"]
        else "Split 60/20/20 conservé."
    )
    add(
        f"- **Split temporel** : train {split['train']['frauds']} fraudes, validation "
        f"{split['valid']['frauds']}, test {split['test']['frauds']}. {fallback}"
    )
    tun = res["tuning"]
    add(
        f"- **Tuning** : {tun['n_trials']} essais Optuna (TPE, {tun.get('n_pruned', 0)} élagués), "
        f"PR-AUC CV temporelle {fr(tun['best_cv_pr_auc'], 3)} (paramètres par défaut : "
        f"{fr(tun['default_cv_pr_auc'], 3)}). Meilleurs paramètres : "
        + ", ".join(
            f"`{k}={v:.3g}`" if isinstance(v, float) else f"`{k}={v}`"
            for k, v in tun["best_params"].items()
        )
        + "."
    )
    if "neg_pos_ratio" in tun:
        add(
            f"- **scale_pos_weight** : Optuna a retenu {fr(tun['best_params']['scale_pos_weight'], 1)} "
            f"alors que n_neg/n_pos vaut {fr(tun['neg_pos_ratio'], 0)} sur le train (point de départ "
            f"de la recherche, PR-AUC CV {fr(tun['default_cv_pr_auc'], 3)}). Un poids extrême n'est "
            f"pas optimal en PR-AUC ; la calibration corrige de toute façon les probabilités."
        )
    if dec.get("capacity_binding"):
        add(
            f"- **Pourquoi t* = {fr(dec['t_star'], 3)}** : sans contrainte, le coût minimal en "
            f"validation serait à t = {fr(dec['unconstrained_best_threshold'], 3)}, mais avec jusqu'à "
            f"{dec['unconstrained_max_per_hour']} alertes dans une heure. t* est le seuil le moins "
            f"strict qui ne dépasse jamais K = {K} alertes dans aucune heure de validation."
        )
    used = [
        f"{POLICY_FR[name]} {pct(pol[name]['capacity_used'], 0)}"
        for name in ("xgb_cost", "xgb_topk", "xgb_topk_profitable")
        if pol[name].get("capacity_used") is not None
    ]
    if used:
        add(
            f"- **Créneaux d'investigation utilisés sur le test** ({K} × {split['test_hours']} heures) : "
            + " ; ".join(used)
            + "."
        )
    add(
        f"- **Accuracy trompeuse** : prédire « jamais fraude » donne "
        f"{pct(eda['accuracy_never_fraud'], 2)} d'accuracy."
    )
    cp = pol["xgb_cost"]
    add(
        f"- **Politique t* sur le test** : {cp['n_alerts']} alertes investiguées "
        f"({cp['tp']} fraudes, {cp['fp']} faux positifs), {pct(cp['amount_recall'])} du montant "
        f"de fraude intercepté, {fr(cp['fp_per_1000'], 2)} faux positifs pour 1 000 transactions, "
        f"{cp['not_investigated']} alertes au-delà de la capacité (max {cp['max_raised_per_hour']}/heure)."
    )
    r05 = pol["xgb_raw_0.5"]
    add(
        f"- **Seuil 0,5 non calibré** : {r05['n_raised']} alertes levées dont "
        f"{r05['not_investigated']} non traitables faute de capacité."
    )
    mon = res["monitoring"]
    drift = [p for p in mon["psi"] if p["status"] != "stable"]
    add(
        f"- **Monitoring** : PSI du score train → test = {fr(mon['score_psi'], 3)} "
        f"({mon['score_psi_status']}). Features hors zone stable : "
        + (", ".join(f"{p['feature']} ({fr(p['psi'], 2)})" for p in drift[:6]) or "aucune")
        + "."
    )
    for case, label in [
        ("true_positive", "Vrai positif (fraude alertée par t*)"),
        ("false_positive", "Faux positif"),
        ("false_negative", "Faux négatif le plus coûteux (non alerté par t*)"),
    ]:
        c = res["shap"]["cases"].get(case)
        if c:
            if case == "false_positive" and not c.get("alerted", True):
                label = (
                    "Transaction légitime la plus suspecte (t* n'a aucun faux positif sur le test)"
                )
            codes = ", ".join(
                f"{r['feature']} ({signed_fr(r['contribution'])})" for r in c["reason_codes"]
            )
            line = (
                f"- **SHAP — {label}** : montant {fr(c['amount'])} €, p = {fr(c['probability'], 3)} ; "
                f"reason codes : {codes}."
            )
            if case == "false_negative" and c.get("legit_drivers"):
                down = ", ".join(
                    f"{r['feature']} ({signed_fr(r['contribution'])})" for r in c["legit_drivers"]
                )
                line += f" Ce qui l'a fait passer pour légitime : {down}."
            add(line + f" Figure : `reports/figures/shap_waterfall_{case}.png`.")
    add("")
    add("## 5. Q23 — matière pour « ce que je ferais différemment »")
    add("")
    add(
        f"- Seulement {split['valid']['frauds']} fraudes en validation et {split['test']['frauds']} "
        f"en test : les IC sont larges, le seuil est choisi sur peu de positifs."
    )
    add(
        f"- Le taux de fraude varie dans le temps (train {pct(split['train']['fraud_rate'], 3)}, "
        f"validation {pct(split['valid']['fraud_rate'], 3)}, test {pct(split['test']['fraud_rate'], 3)}) : "
        f"recalibrer régulièrement est indispensable."
    )
    add(
        "- Les coûts sont des hypothèses : à valider avec l'équipe fraude et la finance "
        "(voir l'analyse de sensibilité)."
    )
    add(
        "- Pas d'identifiant client : pas de features comportementales (fréquence, montant habituel)."
    )
    add("")
    return "\n".join(lines)


# --------------------------------------------------------------------- README


def readme_section(res: dict) -> str:
    pol = _by(res["policies"], "policy")
    models = _by(res["models"], "model")
    red = res["cost_reduction"]
    dec, cal, split = res["decision"], res["calibration"], res["split"]
    K = dec["cost_params"]["capacity_per_hour"]
    lines = [README_START, ""]
    add = lines.append
    if res["run"]["synthetic_data"]:
        add("> Synthetic-data run: numbers below are not meaningful.")
        add("")
    add(
        f"Test period: last {en_pct(split['fractions_used'][2], 0)} of the timeline "
        f"({split['test']['rows']:,} transactions, {split['test']['frauds']} frauds, "
        f"{split['test']['fraud_amount']:,.0f} EUR of fraud). All policies at a constant capacity "
        f"of **K = {K} alerts/hour**; costs c_inv = {dec['cost_params']['c_inv']:g} EUR, "
        f"c_fp = {dec['cost_params']['c_fp']:g} EUR, recovery r = {dec['cost_params']['recovery_rate']:g}."
    )
    add("")
    add(
        "| Model / policy | PR-AUC test [95% CI] | Recall | Precision | Alerts | False positives | Total cost (EUR) |"
    )
    add("|---|---|---|---|---|---|---|")
    for name in ("dummy", "logistic_regression", "random_forest"):
        m = models[name]
        add(
            f"| {m['label']}, threshold 0.5 | {ci_en(m)} | {en_pct(m['recall_at_0.5'])} | "
            f"{en_pct(m['precision_at_0.5'])} | — | — | {m['cost_at_0.5']:,.0f} |"
        )
    best = recommended_policy(pol)
    for name in (
        "xgb_raw_0.5",
        "xgb_cal_0.5",
        "xgb_f1",
        "xgb_cost",
        "xgb_topk",
        "xgb_topk_profitable",
    ):
        p = pol[name]
        model = models["xgboost_raw" if name == "xgb_raw_0.5" else "xgboost"]
        bold = "**" if name == best else ""
        add(
            f"| {bold}{p['label']}{bold} | {ci_en(model)} | {en_pct(p['recall'])} | "
            f"{en_pct(p['precision'])} | {p['n_alerts']} | {p['fp']} | {bold}{p['cost']:,.0f}{bold} |"
        )
    add(f"| *Reference: no alert at all* | — | 0% | — | 0 | 0 | {pol['do_nothing']['cost']:,.0f} |")
    add("")

    def ci(r: dict) -> str:
        verdict = "significant" if r["lo"] > 0 or r["hi"] < 0 else "**not significant**"
        return f"95% CI [{en_pct(r['lo'])}, {en_pct(r['hi'])}], {verdict}"

    add("Key numbers (cost reductions with a paired bootstrap over test transactions):")
    add("")
    r_t, r_k = red["xgb_cost_vs_xgb_raw_0.5"], red["xgb_topk_profitable_vs_xgb_raw_0.5"]
    r_kt = red["xgb_topk_profitable_vs_xgb_cost"]
    add(
        f"- **Top-K by expected loss** (profitable investigations only) vs the default "
        f"`predict()` threshold 0.5: **{en_pct(r_k['value'])}** cost reduction ({ci(r_k)}); "
        f"vs the cost-optimal threshold: {en_pct(r_kt['value'])} ({ci(r_kt)})."
    )
    add(
        f"- **Cost-optimal threshold t\\* = {dec['t_star']:.3f}** (chosen on validation) vs 0.5: "
        f"{en_pct(r_t['value'])} ({ci(r_t)})."
    )
    t_pol, k_pol = pol["xgb_cost"], pol["xgb_topk_profitable"]
    if k_pol["amount_recall"] > t_pol["amount_recall"]:
        add(
            f"- Why top-K wins: ranking by p × amount, it stops {en_pct(k_pol['amount_recall'], 0)} "
            f"of the fraud amount with {k_pol['tp']} frauds caught, vs "
            f"{en_pct(t_pol['amount_recall'], 0)} with {t_pol['tp']} for the threshold, which "
            f"ignores the amount."
        )
    if dec.get("capacity_binding"):
        add(
            f"- With K = {K}, t\\* is set by capacity, not by costs: it is the loosest threshold that "
            f"never exceeds K alerts in any validation hour (the cheapest threshold, "
            f"{dec['unconstrained_best_threshold']:.3f}, would put "
            f"{dec['unconstrained_max_per_hour']} alerts in one hour). It is the same for all 16 "
            f"cost hypotheses of the sensitivity analysis."
            if len({round(r["t_star"], 6) for r in res["sensitivity"]["costs"]}) == 1
            else f"- With K = {K}, t\\* is constrained by the busiest validation hour."
        )
    x, rf = models["xgboost_raw"], models["random_forest"]
    add(
        f"- XGBoost vs random forest, PR-AUC validation {x['valid_pr_auc']:.3f} vs "
        f"{rf['valid_pr_auc']:.3f}, test {x['pr_auc']:.3f} vs {rf['pr_auc']:.3f}"
        + (
            ": the intervals overlap, XGBoost is not significantly better here."
            if _overlap(x, rf, "valid_pr_auc") and _overlap(x, rf, "pr_auc")
            else "."
        )
    )
    add(
        f"- Calibration: **{cal['selected']}** selected (out-of-fold Brier on validation, log-loss "
        f"tie-break). Test Brier {cal['test']['brier_raw']:.6f} → {cal['test']['brier_calibrated']:.6f}."
    )
    top3 = ", ".join(row["feature"] for row in res["shap"]["importance"][:3])
    add(f"- Top-3 SHAP features: {top3}.")
    add(
        f"- Score PSI train → test: {res['monitoring']['score_psi']:.3f} "
        f"({res['monitoring']['score_psi_status']})."
    )
    add("")
    add(
        "Full French interview sheet: [reports/fiche_resultats.md](reports/fiche_resultats.md). Raw numbers: [reports/results.json](reports/results.json)."
    )
    add("")
    add(README_END)
    return "\n".join(lines)


def readme_headline(res: dict, links: dict) -> str:
    """Top of the README: badges, the result in one sentence, the demo GIF, quick links."""
    space, repo = links.get("hf_space"), links.get("github_repo")
    test = res["split"]["test"]
    red = res["cost_reduction"]["xgb_topk_profitable_vs_xgb_raw_0.5"]
    badges = []
    if space:
        badges.append(
            f"[![Live demo](https://img.shields.io/badge/live%20demo-Hugging%20Face-FFD21E"
            f"?logo=huggingface&logoColor=black)](https://huggingface.co/spaces/{space})"
        )
    if repo:
        badges.append(
            f"[![CI](https://github.com/{repo}/actions/workflows/ci.yml/badge.svg)]"
            f"(https://github.com/{repo}/actions/workflows/ci.yml)"
        )
    badges += [f"![{alt}](https://img.shields.io/badge/{spec})" for alt, spec in STATIC_BADGES]
    try_it = (
        f"**[Try the live dashboard](https://huggingface.co/spaces/{space})**"
        if space
        else "Run the dashboard: `make dashboard`"
    )
    sig = "" if red["lo"] > 0 else ", not significant"
    lines = [
        HEADLINE_START,
        " ".join(badges),
        "",
        f"**On a held-out future period of {test['rows']:,} real card transactions, ranking the "
        f"alerts by expected loss under the analysts' capacity cuts the total fraud cost by "
        f"{100 * red['value']:.0f}% compared with the model's default threshold "
        f"(95% CI {100 * red['lo']:.0f}% to {100 * red['hi']:.0f}%{sig}).**",
        "",
        "![Dashboard: policies compared at constant capacity, then the SHAP reason codes of an "
        "alert](docs/dashboard_demo.gif)",
        "",
        f"{try_it} · [Results](#results) · [Method](#method) · [API](#api) · "
        f"[French interview sheet](reports/fiche_resultats.md)",
        HEADLINE_END,
    ]
    return "\n".join(lines)


def replace_block(readme: Path, start: str, end: str, block: str) -> None:
    text = readme.read_text(encoding="utf-8")
    pattern = re.compile(re.escape(start) + ".*?" + re.escape(end), re.DOTALL)
    if not pattern.search(text):
        raise ValueError(f"{start} ... {end} markers not found in {readme}")
    readme.write_text(pattern.sub(lambda _: block, text), encoding="utf-8")


def write_reports(results: dict, out_root: Path, update_readme: bool = True) -> None:
    cfg = load_config()
    reports_dir = out_root / cfg["paths"]["reports_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "fiche_resultats.md").write_text(fiche_markdown(results), encoding="utf-8")
    readme = out_root / "README.md"
    if update_readme and readme.exists():
        replace_block(readme, README_START, README_END, readme_section(results))
        if HEADLINE_START in readme.read_text(encoding="utf-8"):
            headline = readme_headline(results, cfg.get("links") or {})
            replace_block(readme, HEADLINE_START, HEADLINE_END, headline)


def main() -> None:
    cfg = load_config()
    path = ROOT / cfg["paths"]["reports_dir"] / "results.json"
    results = json.loads(path.read_text(encoding="utf-8"))
    write_reports(results, ROOT, update_readme=not results["run"]["synthetic_data"])
    print("Reports written: reports/fiche_resultats.md and README.md")


if __name__ == "__main__":
    main()
