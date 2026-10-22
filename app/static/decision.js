/*
 * The decision rules of src/cost.py, in JavaScript, for the static dashboard.
 * tests/test_decision_js.py checks that they give the same numbers as Python.
 *
 * Cost of a transaction of amount a:
 *   fraud not alerted -> a ; fraud alerted -> c_inv + (1 - r) a
 *   legit alerted -> c_inv + c_fp ; legit not alerted -> 0
 * At most K alerts are investigated per hour. Arrays are in chronological order.
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.Decision = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  // Keep the first `capacity` alerts of each hour (first come, first served).
  function firstComeFirstServed(raised, hour, capacity) {
    const kept = new Uint8Array(raised.length);
    let current = null;
    let used = 0;
    for (let i = 0; i < raised.length; i++) {
      if (hour[i] !== current) {
        current = hour[i];
        used = 0;
      }
      if (raised[i] && (capacity === null || used < capacity)) {
        kept[i] = 1;
        used++;
      }
    }
    return kept;
  }

  function maxAlertsPerHour(raised, hour) {
    let best = 0;
    let current = null;
    let count = 0;
    for (let i = 0; i < raised.length; i++) {
      if (hour[i] !== current) {
        current = hour[i];
        count = 0;
      }
      if (raised[i]) {
        count++;
        if (count > best) best = count;
      }
    }
    return best;
  }

  function costSummary(y, alert, amount, params) {
    let cost = 0, tp = 0, fp = 0, fn = 0, tn = 0, fraudAmount = 0, caught = 0;
    for (let i = 0; i < y.length; i++) {
      const a = amount[i];
      if (y[i]) {
        fraudAmount += a;
        if (alert[i]) {
          tp++;
          caught += a;
          cost += params.c_inv + (1 - params.recovery_rate) * a;
        } else {
          fn++;
          cost += a;
        }
      } else if (alert[i]) {
        fp++;
        cost += params.c_inv + params.c_fp;
      } else {
        tn++;
      }
    }
    return {
      cost,
      n_alerts: tp + fp,
      tp, fp, fn, tn,
      precision: tp + fp ? tp / (tp + fp) : 0,
      recall: tp + fn ? tp / (tp + fn) : 0,
      amount_recall: fraudAmount ? caught / fraudAmount : 0,
    };
  }

  function above(p, threshold) {
    const raised = new Uint8Array(p.length);
    for (let i = 0; i < p.length; i++) raised[i] = p[i] >= threshold ? 1 : 0;
    return raised;
  }

  function thresholdPolicy(p, threshold, hour, capacity) {
    return firstComeFirstServed(above(p, threshold), hour, capacity);
  }

  // Same values as numpy.linspace(0, 1, size + 2)[1:-1].
  function thresholdGrid(size) {
    const step = 1 / (size + 1);
    const grid = [];
    for (let i = 1; i <= size; i++) grid.push(i * step);
    return grid;
  }

  // One pass per threshold: same result as thresholdPolicy + costSummary + maxAlertsPerHour,
  // fast enough to rerun on every slider move.
  function thresholdCurve(y, p, amount, hour, params, grid) {
    const K = params.capacity_per_hour;
    const fpCost = params.c_inv + params.c_fp;
    const lostShare = 1 - params.recovery_rate;
    return grid.map((threshold) => {
      let cost = 0, alerts = 0, tp = 0, frauds = 0, maxPerHour = 0;
      let current = null, raisedInHour = 0, usedInHour = 0;
      for (let i = 0; i < p.length; i++) {
        if (hour[i] !== current) {
          current = hour[i];
          raisedInHour = 0;
          usedInHour = 0;
        }
        let alerted = false;
        if (p[i] >= threshold) {
          raisedInHour++;
          if (raisedInHour > maxPerHour) maxPerHour = raisedInHour;
          if (K === null || usedInHour < K) {
            usedInHour++;
            alerted = true;
          }
        }
        if (y[i]) {
          frauds++;
          if (alerted) {
            tp++;
            alerts++;
            cost += params.c_inv + lostShare * amount[i];
          } else {
            cost += amount[i];
          }
        } else if (alerted) {
          alerts++;
          cost += fpCost;
        }
      }
      return {
        threshold,
        cost,
        n_alerts: alerts,
        recall: frauds ? tp / frauds : 0,
        max_per_hour: maxPerHour,
        feasible: K === null || maxPerHour <= K,
      };
    });
  }

  // t* = cheapest threshold that never exceeds K alerts in an hour; middle of the plateau.
  function optimiseThreshold(curve) {
    const anyFeasible = curve.some((row) => row.feasible);
    const candidates = anyFeasible ? curve.filter((row) => row.feasible) : curve;
    const best = Math.min(...candidates.map((row) => row.cost));
    const isBest = curve.map((row) => row.cost <= best + 1e-9 && (!anyFeasible || row.feasible));
    const i = isBest.indexOf(true);
    let j = i;
    while (j + 1 < isBest.length && isBest[j + 1]) j++;
    return (curve[i].threshold + curve[j].threshold) / 2;
  }

  function expectedNetBenefit(p, a, params) {
    return p * params.recovery_rate * a - params.c_inv - (1 - p) * params.c_fp;
  }

  // Each hour, the K profitable transactions with the largest expected loss p * a.
  function topkProfitable(p, amount, hour, capacity, params) {
    const kept = new Uint8Array(p.length);
    let start = 0;
    while (start < p.length) {
      let end = start;
      while (end < p.length && hour[end] === hour[start]) end++;
      const candidates = [];
      for (let i = start; i < end; i++) {
        if (expectedNetBenefit(p[i], amount[i], params) > 0) candidates.push(i);
      }
      candidates.sort((u, v) => p[v] * amount[v] - p[u] * amount[u] || u - v);
      const take = capacity === null ? candidates.length : Math.min(capacity, candidates.length);
      for (let k = 0; k < take; k++) kept[candidates[k]] = 1;
      start = end;
    }
    return kept;
  }

  return {
    firstComeFirstServed,
    maxAlertsPerHour,
    costSummary,
    thresholdPolicy,
    thresholdGrid,
    thresholdCurve,
    optimiseThreshold,
    expectedNetBenefit,
    topkProfitable,
  };
});
