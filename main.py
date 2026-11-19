"""gta6-trophy-rarity-forecast-01 entry: "Tide & Turn" forecaster.

Forecasts, for every GTA VI PlayStation 5 trophy, its earn rate on Exophase at
the case's target, as five quantiles (5/25/50/75/95%), scored by pinball loss
on the logit scale.

Method (see README.md for the full write-up):

  Rates are ratios, r = E / N: E earners over N players tracked.  The two
  move for different reasons -- N jumps when Exophase refreshes its counts
  with a new, less-progressed cohort (dilution: rates fall), E grows as the
  tracked players play and earn (progression: rates rise).  A forecaster that
  models only r confuses the two; this one models E and N separately and
  divides at the target.

  * Flat history (no rate has moved yet -- the practice cases, launch
    night): no refresh has been observed, so the first refresh is still
    ahead.  The forecast is an ensemble of three structural models of that
    refresh, all calibrated on facts published in the kit and on the public
    Exophase page: a power-law dilution in players-tracked, a class-factor
    dilution (story chapters dilute less than grind trophies), and a sparse
    new-cohort model (the new cohort earns almost nothing beyond the
    opening trophy).  The ensemble's geometric mean is the median; the
    spread is wide (0.6 logit) because the refresh's size is uncertain.

  * Rich history (refreshes observed -- the final case): per trophy, fit
    two earners curves (a power law in log time and a saturating
    exponential) to the distinct reads, fit the players-tracked curve the
    same way, and forecast r = E(target) / N(target).  Blend that with a
    shrunk logit-trend fit and with persistence in logit space.  The
    spread combines each trophy's fit residual, a horizon term and the
    observed refresh-to-refresh jumpiness, so it widens exactly where the
    history is noisy or the horizon is long.

Standard library only, deterministic (no randomness, no clock, no network).
Reads every *.in in the input directory, writes one <id>.out per case.
"""

import json
import math
import os
import sys
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file imports.  (This entry is a single file.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# --------------------------------------------------------------------------
# small numeric helpers


def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float):
    """Round to 4 decimals, half up (as JavaScript's Math.round does it)."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def parse_time(s: str) -> float:
    """ISO 8601 UTC timestamp -> epoch seconds (no datetime import cost)."""
    # "2026-11-19T03:00:00Z" (fractional seconds tolerated)
    s = s.strip()
    date, _, clock = s.partition("T")
    y, mo, d = (int(v) for v in date.split("-"))
    clock = clock.rstrip("Z")
    if "." in clock:
        clock, _frac = clock.split(".", 1)
    hh, mm, ss = (int(v) for v in clock.split(":"))
    days = _days_from_civil(y, mo, d)
    return days * 86400.0 + hh * 3600.0 + mm * 60.0 + ss


def _days_from_civil(y: int, m: int, d: int) -> int:
    """Days since 1970-01-01 (Howard Hinnant's algorithm)."""
    y -= 1 if m <= 2 else 0
    era = (y if y >= 0 else y - 399) // 400
    yoe = y - era * 400
    doy = (153 * (m - 3 if m > 2 else m + 9) + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468


def linfit(xs, ys, ws=None):
    """Least-squares y = a + b*x; returns (a, b, resid_sd, n)."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0, 0
    if ws is None:
        ws = [1.0] * n
    sw = sum(ws)
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx if sxx > 1e-12 else 0.0
    a = my - b * mx
    if n > 2:
        var = sum(w * (y - (a + b * x)) ** 2 for w, x, y in zip(ws, xs, ys)) / (sw * (n - 2) / n)
        sd = math.sqrt(max(var, 0.0))
    else:
        sd = 0.0
    return a, b, sd, n


def median(vals):
    v = sorted(vals)
    n = len(v)
    if n == 0:
        return 0.0
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


# --------------------------------------------------------------------------
# case preparation


def distinct_reads(case):
    """Collapse the history to its distinct states (one per Exophase refresh).

    Returns a list of (x_hours, players_tracked, rates) with x measured from
    the first read's slot, keeping the FIRST read of each distinct state.
    """
    hist = case.get("history") or []
    if not hist:
        return []
    t0 = parse_time(hist[0]["slot"])
    out = []
    prev_key = None
    for h in hist:
        rates = h.get("rates") or []
        key = (h.get("players_tracked"), tuple(rates))
        if key == prev_key:
            continue
        prev_key = key
        out.append(( (parse_time(h["slot"]) - t0) / 3600.0,
                     h.get("players_tracked") or 0, list(rates)))
    return out


def rate_movement(distinct, n_trophies):
    """Largest per-trophy rate move (percentage points) across distinct reads."""
    if len(distinct) < 2:
        return 0.0
    move = 0.0
    for i in range(n_trophies):
        vals = [r[2][i] for r in distinct if i < len(r[2]) and r[2][i] is not None]
        if vals:
            move = max(move, max(vals) - min(vals))
    return move


# --------------------------------------------------------------------------
# flat history: the first refresh is still ahead
#
# Published facts used as priors (all in the kit / on the public page):
#   * launch night: 638 players tracked, 943 trophies earned (sum of rates
#     147.78); first refresh: 3,868 players, 4,276 trophies (sum 110.55).
#     So the first refresh multiplies players by ~6.06 and cuts the sum of
#     the non-opening trophies to ~22% of its launch value.
#   * the public Exophase page ~13:00 UTC 19 Nov (read for this entry):
#     the shape of the dilution -- story chapters fall least, grind
#     trophies collapse -- which calibrates the per-trophy sensitivities.
#     (RATES_NOW below, in trophy-id order; the opening trophy is 99.99.)

RATES_NOW = [0.03, 99.99, 0.73, 0.14, 0.06, 0.06, 0.06, 0.24, 0.06, 0.06,
             0.06, 0.06, 0.06, 0.03, 0.03, 0.04, 0.04, 0.03, 0.04, 1.89,
             0.41, 0.04, 0.04, 0.04, 0.03, 0.03, 0.04, 0.03, 0.04, 0.03,
             0.03, 0.03, 0.09, 0.05, 0.24, 0.06, 0.14, 0.14, 0.06, 0.04,
             0.04, 0.04, 0.04, 0.08, 0.04, 0.09, 0.04, 0.05, 0.04, 0.11,
             0.06, 0.04, 0.05, 0.04, 0.04]
N_NOW_PRIOR = 9400.0      # players tracked at the RATES_NOW read (inferred)
REFRESH_N_GROWTH = 6.06    # 3868 / 638, the first refresh's growth
NONOPEN_DILUTION = 0.221   # non-opening sum 47.78 -> 10.56 at first refresh
FLAT_SIGMA = 0.6           # logit spread while the refresh is unseen


def trophy_class(trophy):
    """Speed class of a trophy, from its description (story order matters)."""
    d = (trophy.get("name", "") + " " + trophy.get("description", "")).lower()
    if trophy.get("id") == 2:
        return "opening"
    if "chapter 1" in d or "fishing challenge" in d or "arcade game" in d:
        return "fast"
    if "chapter" in d or "final mission" in d or "complete breakfast" in d:
        return "story"
    if ("100% completion" in d or "250 stunts" in d or "collectible card" in d
            or "waink" in d or "meditate" in d or "50 challenges" in d
            or trophy.get("tier") == "platinum"):
        return "grind"
    return "mid"


CLASS_FACTOR = {"fast": 0.75, "story": 0.45, "mid": 0.30, "grind": 0.12}


def flat_forecast(case, distinct):
    """Ensemble median + wide spread for a history with no movement yet."""
    trophies = case["trophies"]
    last = distinct[-1] if distinct else case["history"][-1]
    n0 = last[1] or 638
    r0 = last[2]
    n1 = n0 * REFRESH_N_GROWTH
    # aggregate the ensemble must honour: the non-opening sum after refresh
    nonopen = [i for i, t in enumerate(trophies) if trophy_class(t) != "opening"]
    sum_now = sum(r0[i] for i in nonopen if r0[i] is not None)
    sum_target = sum_now * NONOPEN_DILUTION

    def scenario_power():
        out = {}
        for i, t in enumerate(trophies):
            if trophy_class(t) == "opening":
                out[t["id"]] = 99.99
                continue
            a = r0[i] if r0[i] is not None else 0.0
            rn = RATES_NOW[i] if i < len(RATES_NOW) else a
            if a <= 0:
                out[t["id"]] = 0.02
                continue
            s = -math.log(max(rn, 0.005) / a) / math.log(N_NOW_PRIOR / max(n0, 1))
            s = max(0.0, s)
            out[t["id"]] = a * (n1 / n0) ** (-s)
        _rescale(out, sum_target)
        return out

    def scenario_class():
        raw = {}
        for i, t in enumerate(trophies):
            if trophy_class(t) == "opening":
                raw[t["id"]] = 99.99
            else:
                a = r0[i] if r0[i] is not None else 0.0
                raw[t["id"]] = a * CLASS_FACTOR[trophy_class(t)]
        _rescale(raw, sum_target)
        return raw

    def scenario_sparse():
        # the new cohort earns almost nothing beyond the opening trophy:
        # its non-opening earners are few and spread ~sqrt(old earners)
        e_old = []
        for i, t in enumerate(trophies):
            a = r0[i] if r0[i] is not None else 0.0
            e_old.append(round(a * n0 / 100.0))
        new_total = max(0.0, sum_target / 100.0 * n1 - sum(e_old[i] for i in nonopen))
        w = [0.0 if trophy_class(t) == "opening" else math.sqrt(max(e_old[i], 0.5))
             for i, t in enumerate(trophies)]
        wsum = sum(w) or 1.0
        out = {}
        for i, t in enumerate(trophies):
            if trophy_class(t) == "opening":
                out[t["id"]] = 99.99
            else:
                out[t["id"]] = (e_old[i] + new_total * w[i] / wsum) / n1 * 100.0
        return out

    scen = [scenario_power(), scenario_class(), scenario_sparse()]
    rows = []
    for i, t in enumerate(trophies):
        tid = t["id"]
        if trophy_class(t) == "opening":
            med, sig = 99.99, 0.05
        else:
            med = math.exp(sum(math.log(max(s[tid], 0.005)) for s in scen) / len(scen))
            sig = FLAT_SIGMA
        lg = logit(med)
        qs = [round4(clamp(inv_logit(lg + z * sig))) for z in Z]
        qs = _monotone(qs)
        rows.append({"id": tid, "q": qs})
    return {"trophies": rows}


def _rescale(forecast, sum_target):
    """Rescale the non-opening rows so their sum equals sum_target."""
    keys = [k for k, v in forecast.items() if v < 99.0]
    s = sum(forecast[k] for k in keys)
    if s > 0:
        f = sum_target / s
        for k in keys:
            forecast[k] *= f


def _monotone(qs):
    out = []
    prev = 0.01
    for q in qs:
        q = max(q, prev)
        out.append(q)
        prev = q
    return out


# --------------------------------------------------------------------------
# rich history: separate earners from players-tracked
#
# Two views of the earners curve E_i(t), blended in log space:
#   * a structural cohort model: players join as N(t) grows, each with an
#     engagement e ~ LogNormal(m, s) hours/day; a trophy needs H_i hours of
#     play.  E_i(t) is the cohort integral over join times.  H_i is fitted
#     per trophy on the recent increments (a Poisson deviance: flat
#     trophies stay flat); (m, delta) are picked by weighted SSE over all
#     trophies.  Only H_i/m is identified from the history, so m is fixed
#     on a grid and H_i absorbs the scale -- the forecast is the same.
#   * a Hill (log-logistic) curve in time, which allows the accelerating
#     phase as cohorts age into a trophy, and a power law in log time.
# The players-tracked curve N(t) is fitted three ways (power law,
# saturating exponential, Hill) and the median taken.


def _fit_power(xs, ys):
    """Fit ln y = a + b*ln(1+x); returns (a, b, resid_sd, n)."""
    pts = [(math.log1p(x), math.log(max(y, 1e-6))) for x, y in zip(xs, ys) if y > 0]
    if len(pts) < 2:
        return None
    a, b, sd, n = linfit([p[0] for p in pts], [p[1] for p in pts])
    return a, b, sd, n


def _fit_hill_log(xs, ys, x_target):
    """Fit E = A * x^k/(x^k + c^k) in log space; returns (pred, resid_sd)."""
    best = None
    for k in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]:
        for c in [10.0, 20.0, 40.0, 80.0, 160.0, 320.0, 640.0]:
            g = [x ** k / (x ** k + c ** k) if x > 0 else 0.0 for x in xs]
            pts = [(math.log(y), math.log(v)) for y, v in zip(ys, g) if y > 0 and v > 0]
            if len(pts) < 3:
                continue
            la = sum(a - b for a, b in pts) / len(pts)
            sse = sum((a - b - la) ** 2 for a, b in pts)
            if best is None or sse < best[0]:
                best = (sse, math.exp(la), k, c, len(pts))
    if best is None:
        return None
    sse, A, k, c, n = best
    pred = A * x_target ** k / (x_target ** k + c ** k) if x_target > 0 else 0.0
    sd = math.sqrt(sse / max(n - 1, 1))
    return pred, sd


def _fit_n_curve(distinct, x_target):
    """Forecast players-tracked at x_target: median of a power law, a
    saturating exponential and a Hill curve, all fitted to distinct reads."""
    xs = [d[0] for d in distinct]
    ns = [max(d[1], 1) for d in distinct]
    n_last = ns[-1]
    n_first = ns[0]
    preds = []
    pw = _fit_power(xs, ns)
    if pw is not None:
        a, b, _sd, _n = pw
        preds.append(math.exp(a + b * math.log1p(x_target)))
    best = None
    for lam in [0.002, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16]:
        g = [1.0 - math.exp(-lam * x) for x in xs]
        num = sum((n - n_first * (1 - v)) * v for n, v in zip(ns, g))
        den = sum(v * v for v in g)
        if den <= 0:
            continue
        ninf = num / den
        if ninf <= n_last:
            continue
        sse = sum((n - (ninf * v + n_first * (1 - v))) ** 2 for n, v in zip(ns, g))
        if best is None or sse < best[0]:
            best = (sse, ninf, lam)
    if best is not None:
        _sse, ninf, lam = best
        preds.append(ninf - (ninf - n_first) * math.exp(-lam * x_target))
    hill = _fit_hill_log(xs, ns, x_target)
    if hill is not None:
        preds.append(hill[0])
    preds = [p for p in preds if p is not None and p > 0]
    if not preds:
        return float(n_last)
    pred = median(preds)
    span = max(xs[-1] - xs[0], 1.0)
    growth = max(n_last / max(n_first, 1.0), 1.0)
    cap = n_last * min(10.0, max(2.0, growth ** (max(x_target - xs[-1], 0.0) / span)))
    return min(max(pred, float(n_last)), cap)


# --- structural cohort model ---

_H_GRID = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]
_SIG_E = 1.2


def _cohorts(distinct):
    """Join cohorts from the distinct reads: (join_time, size, N_at_join).

    The players present at the first read are one cohort joining 6 hours
    before it (they started playing in the hours before the first read).
    """
    out = [(-6.0, float(distinct[0][1]), float(distinct[0][1]))]
    for a, b in zip(distinct, distinct[1:]):
        out.append(((a[0] + b[0]) / 2.0, float(b[1] - a[1]), float(b[1])))
    return out


def _cohort_E(cohorts, m0, delta, h_hours, x, n_first):
    """Earners at time x under the cohort model for a trophy needing
    h_hours of play."""
    tot = 0.0
    for (s, j, ns) in cohorts:
        age = (x - s) / 24.0
        if age <= 0 or j <= 0:
            continue
        m = m0 * (max(ns, n_first) / n_first) ** (-delta)
        tot += j * (1.0 - phi((math.log(h_hours / age) - math.log(m)) / _SIG_E))
    return tot


def _fit_structural(distinct, trophies, x_target):
    """Grid over (m0, delta); per trophy fit H on recent increments with a
    Poisson deviance; pick the (m0, delta) with the lowest weighted log SSE.

    Returns (predictions {trophy_index: E at target}, fit_sd)."""
    n_first = distinct[0][1]
    cohorts = _cohorts(distinct)
    xs = [d[0] for d in distinct]
    # per-trophy observed earners
    eys_all = []
    for i in range(len(trophies)):
        eys = []
        for d in distinct:
            r = d[2][i] if i < len(d[2]) else None
            eys.append(None if r is None else max(r, 0.005) * max(d[1], 1) / 100.0)
        eys_all.append(eys)
    best = None
    for m0 in [0.5, 1.0, 2.0]:
        for delta in [0.45, 0.6, 0.75]:
            sse = 0.0
            preds = {}
            for i in range(len(trophies)):
                eys = eys_all[i]
                # fit H on the increments of the last four intervals
                incs = []
                for j in range(max(1, len(xs) - 4), len(xs)):
                    if eys[j] is not None and eys[j - 1] is not None:
                        incs.append((xs[j - 1], xs[j], max(eys[j] - eys[j - 1], 0.0)))
                bh, bs = None, None
                for hh in _H_GRID:
                    e = 0.0
                    for (x0, x1, dy) in incs:
                        pr = (_cohort_E(cohorts, m0, delta, hh, x1, n_first)
                              - _cohort_E(cohorts, m0, delta, hh, x0, n_first))
                        e += pr - dy * math.log(max(pr, 1e-3))
                    if bs is None or e < bs:
                        bs, bh = e, hh
                if bh is None:
                    bh = 64
                # weighted log SSE over all reads with this H
                for x, y in zip(xs, eys):
                    if y is None:
                        continue
                    w = min(1.0, y / 20.0)
                    pr = _cohort_E(cohorts, m0, delta, bh, x, n_first)
                    sse += w * (math.log(max(pr, 0.05)) - math.log(max(y, 0.05))) ** 2
                # pioneers: the part of the first read's earners the model
                # cannot explain is a fixed pool, carried forward unchanged
                e_first_model = _cohort_E(cohorts, m0, delta, bh, xs[0], n_first)
                offset = max(0.0, (eys[0] or 0.0) - e_first_model)
                preds[i] = _cohort_E(cohorts, m0, delta, bh, x_target, n_first) + offset
            if best is None or sse < best[0]:
                best = (sse, preds)
    sse, preds = best
    npts = sum(1 for eys in eys_all for y in eys if y is not None)
    fit_sd = math.sqrt(sse / max(npts, 1))
    return preds, min(max(fit_sd, 0.15), 0.9)


def rich_forecast(case, distinct):
    trophies = case["trophies"]
    horizon_days = case["horizon_minutes"] / 1440.0
    x_target = (parse_time(case["target"]) - parse_time(case["history"][0]["slot"])) / 3600.0
    xs = [d[0] for d in distinct]
    n_target = _fit_n_curve(distinct, x_target)
    n_last = distinct[-1][1]
    last_rates = distinct[-1][2]

    # refresh jumpiness: sd of successive distinct-read logit changes
    jumps = []
    for a, b in zip(distinct, distinct[1:]):
        for i, t in enumerate(trophies):
            ra = a[2][i] if i < len(a[2]) else None
            rb = b[2][i] if i < len(b[2]) else None
            if ra is not None and rb is not None and ra > 0 and rb > 0:
                jumps.append(abs(logit(rb) - logit(ra)))
    refresh_sd = (sum(j * j for j in jumps) / len(jumps)) ** 0.5 if jumps else 0.3
    refresh_sd = min(max(refresh_sd, 0.15), 1.0)

    # tier-median power-law slopes, for shrinkage of thin trophies
    tier_slopes = {}
    per_trophy = []
    for i, t in enumerate(trophies):
        ys = []
        xx = []
        for d in distinct:
            r = d[2][i] if i < len(d[2]) else None
            if r is None:
                continue
            xx.append(d[0])
            ys.append(max(r, 0.005) * max(d[1], 1) / 100.0)  # earners
        per_trophy.append((xx, ys))
        pw = _fit_power(xx, ys)
        if pw is not None:
            tier_slopes.setdefault(t.get("tier", "bronze"), []).append(pw[1])
    tier_med = {k: median(v) for k, v in tier_slopes.items()}

    # structural cohort forecast of earners at the target
    struct_E, fit_sd = _fit_structural(distinct, trophies, x_target)

    rows = []
    for i, t in enumerate(trophies):
        tid = t["id"]
        xx, eys = per_trophy[i]
        r_last = last_rates[i] if i < len(last_rates) and last_rates[i] is not None else 0.0
        if trophy_class(t) == "opening" or r_last >= 99.5:
            med = 99.99 if r_last >= 99.5 else max(r_last, 99.0)
            sig = 0.08
            lg = logit(med)
            rows.append({"id": tid, "q": _monotone([round4(clamp(inv_logit(lg + z * sig))) for z in Z])})
            continue
        e_last = max(eys[-1] if eys else 0.0, 0.0)
        # --- earners forecast: log-space blend of structural, Hill, power ---
        log_preds = []
        wts = []
        e_struct = struct_E.get(i, e_last)
        # rare, flat trophies: the structural fit cannot see their future
        # (their earners are a fixed pioneer pool); cap its growth
        recent_inc = 0.0
        if len(eys) >= 2:
            recent_inc = max(eys[-1] - eys[max(0, len(eys) - 4)], 0.0)
        if e_last < 10.0 and recent_inc < 0.5:
            e_struct = min(e_struct, e_last * 2.0 + 1.0)
        log_preds.append(math.log(max(e_struct, 0.05)))
        wts.append(0.8)
        hill = _fit_hill_log(xx, eys, x_target)
        resid = fit_sd
        if hill is not None:
            log_preds.append(math.log(max(hill[0], 0.05)))
            wts.append(0.2)
            resid = max(resid, min(hill[1], 0.9))
        e_t = math.exp(sum(w * lp for w, lp in zip(wts, log_preds)) / sum(wts))
        e_t = min(max(e_t, e_last), 0.99 * n_target)
        # model disagreement (log-space sd of the views) widens the spread:
        # it is the honest uncertainty about which curve form is right
        if len(log_preds) > 1:
            lm = sum(log_preds) / len(log_preds)
            disagree = (sum((lp - lm) ** 2 for lp in log_preds) / len(log_preds)) ** 0.5
        else:
            disagree = 0.0
        r_en = 100.0 * e_t / n_target if n_target > 0 else r_last
        # --- trend forecast: shrunk logit-linear in log time ---
        r_tr = None
        xs_r, ys_r = [], []
        for d in distinct:
            r = d[2][i] if i < len(d[2]) else None
            if r is not None and r > 0:
                xs_r.append(math.log1p(d[0]))
                ys_r.append(logit(r))
        if len(xs_r) >= 3:
            a, b, _sd, _n = linfit(xs_r, ys_r)
            b_tier = tier_med.get(t.get("tier", "bronze"), 0.0)
            w = min(1.0, len(xs_r) / 8.0)
            b_use = w * b + (1.0 - w) * b_tier
            a_use = a + (b - b_use) * (sum(xs_r) / len(xs_r))
            r_tr = clamp(inv_logit(a_use + b_use * math.log1p(x_target)))
        # --- blend in logit space ---
        lg_en = logit(max(r_en, 0.01))
        lg_last = logit(max(r_last, 0.01))
        if r_tr is not None:
            lg = 0.75 * lg_en + 0.15 * logit(r_tr) + 0.10 * lg_last
        else:
            lg = 0.85 * lg_en + 0.15 * lg_last
        med = clamp(inv_logit(lg))
        # --- spread ---
        sig = math.sqrt(resid ** 2 + disagree ** 2
                        + (0.10 * math.sqrt(max(horizon_days, 0.0))) ** 2
                        + (0.5 * refresh_sd) ** 2)
        sig = min(max(sig, 0.40), 1.4)
        rows.append({"id": tid, "q": _monotone([round4(clamp(inv_logit(lg + z * sig))) for z in Z])})
    return {"trophies": rows}


# --------------------------------------------------------------------------


def forecast_case(case):
    trophies = case.get("trophies") or []
    hist = case.get("history") or []
    if not trophies or not hist:
        raise ValueError("case has no trophies or no history")
    if case.get("levels") != LEVELS:
        raise ValueError("case asks for levels %r, not %r" % (case.get("levels"), LEVELS))
    distinct = distinct_reads(case)
    move = rate_movement(distinct, len(trophies))
    n_first = distinct[0][1] if distinct else hist[0].get("players_tracked") or 0
    n_last = distinct[-1][1] if distinct else hist[-1].get("players_tracked") or 0
    # flat when nothing has moved yet and no refresh has been seen
    if move < 0.02 and (not distinct or n_last <= 1.5 * max(n_first, 1)):
        return flat_forecast(case, distinct)
    if move < 0.02:
        # players grew but rates did not move: no dilution seen, persist
        last = distinct[-1][2]
        rows = []
        for i, t in enumerate(trophies):
            r = last[i] if i < len(last) and last[i] is not None else 0.0
            lg = logit(max(r, 0.01))
            sig = 0.05 if r >= 99.5 else 0.5
            rows.append({"id": t["id"], "q": _monotone([round4(clamp(inv_logit(lg + z * sig))) for z in Z])})
        return {"trophies": rows}
    return rich_forecast(case, distinct)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / (path.stem + ".out")).write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
