"""The static dashboard recomputes the decision in JavaScript (app/static/decision.js).
These tests run it with Node and check it gives exactly the Python numbers."""

import json
import shutil
import subprocess

import numpy as np
import pytest

from src.config import ROOT
from src.cost import (
    CostParams,
    cost_summary,
    optimise_threshold,
    threshold_curve,
    threshold_grid,
    threshold_policy,
    topk_policy,
)

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="Node.js is not installed")

RUNNER = """
const D = require(process.argv[2]);
const data = JSON.parse(require("fs").readFileSync(0, "utf8"));
const v = data.valid, t = data.test;
const out = data.cases.map((c) => {
  const curve = D.thresholdCurve(v.y, v.p, v.amount, v.hour, c.params, D.thresholdGrid(c.grid));
  const tStar = D.optimiseThreshold(curve);
  const K = c.params.capacity_per_hour;
  return {
    t_star: tStar,
    curve_costs: curve.map((row) => row.cost),
    threshold: D.costSummary(t.y, D.thresholdPolicy(t.p, tStar, t.hour, K), t.amount, c.params),
    topk: D.costSummary(t.y, D.topkProfitable(t.p, t.amount, t.hour, K, c.params), t.amount, c.params),
  };
});
process.stdout.write(JSON.stringify(out));
"""

CASES = [
    (CostParams(5, 10, 0.8, 10), 299),
    (CostParams(0, 0, 1.0, 3), 299),
    (CostParams(20, 50, 0.5, 40), 999),
    (CostParams(2, 25, 0.9, None), 299),
]


def _period(rng, n=4000):
    hour = np.sort(rng.integers(0, 12, n))
    y = (rng.random(n) < 0.02).astype(int)
    p = np.clip(rng.beta(0.3, 8, n) + 0.6 * y * rng.random(n), 0, 1)
    p[rng.random(n) < 0.05] = 0.5  # ties in the score
    amount = np.round(rng.lognormal(3, 1.3, n), 2)
    amount[rng.random(n) < 0.05] = 100.0  # ties in the expected loss
    return {"y": y, "p": p, "amount": amount, "hour": hour}


@pytest.fixture(scope="module")
def js_and_python(tmp_path_factory):
    rng = np.random.default_rng(7)
    valid, test = _period(rng), _period(rng)
    payload = {
        "valid": {k: v.tolist() for k, v in valid.items()},
        "test": {k: v.tolist() for k, v in test.items()},
        "cases": [{"params": params.to_dict(), "grid": grid} for params, grid in CASES],
    }
    runner = tmp_path_factory.mktemp("js") / "runner.js"
    runner.write_text(RUNNER, encoding="utf-8")
    result = subprocess.run(
        [NODE, str(runner), str(ROOT / "app" / "static" / "decision.js")],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=True,
    )
    return valid, test, json.loads(result.stdout)


@pytest.mark.parametrize("case", range(len(CASES)))
def test_javascript_matches_python(js_and_python, case):
    valid, test, js = js_and_python
    params, grid = CASES[case]
    out = js[case]
    K = params.capacity_per_hour

    curve = threshold_curve(
        valid["y"], valid["p"], valid["amount"], valid["hour"], params, threshold_grid(grid)
    )
    t_star = optimise_threshold(curve)
    assert out["t_star"] == pytest.approx(t_star, abs=1e-12)
    np.testing.assert_allclose(out["curve_costs"], curve["cost"], rtol=1e-9)

    policies = {
        "threshold": threshold_policy(test["p"], t_star, test["hour"], K),
        "topk": topk_policy(
            test["p"], test["amount"], test["hour"], K, params, only_profitable=True
        ),
    }
    for name, alert in policies.items():
        expected = cost_summary(test["y"], alert, test["amount"], params)
        for key in ("n_alerts", "tp", "fp", "fn", "tn"):
            assert out[name][key] == expected[key], (name, key)
        assert out[name]["cost"] == pytest.approx(expected["cost"], rel=1e-9)
        assert out[name]["amount_recall"] == pytest.approx(expected["amount_recall"], rel=1e-9)
