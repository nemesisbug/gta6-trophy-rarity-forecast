"""Exact scorer (per RULES.txt) + baselines + synthetic practice answers."""
import json, math

KIT = "/home/hatch/workspace/makefun-gta6-trophy-rarity-forecast-01-kit"
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

def clamp(x): return min(99.99, max(0.01, x))
def logit(p): return math.log(clamp(p) / (100.0 - clamp(p)))
def inv_logit(l): return 100.0 / (1.0 + math.exp(-l))

def pinball_loss(forecast, actual):
    """forecast: {id: [q05..q95]}; actual: {id: rate}. Mean over all items."""
    tot, n = 0.0, 0
    for tid, y in actual.items():
        ly = logit(y)
        for tau, q in zip(LEVELS, forecast[tid]):
            d = ly - logit(q)
            tot += 2.0 * max(tau * d, (tau - 1.0) * d)
            n += 1
    return tot / n

def typical_miss(forecast, actual):
    return sum(abs(y - forecast[i][2]) for i, y in actual.items()) / len(actual)

def coverage90(forecast, actual):
    return sum(1 for i, y in actual.items() if forecast[i][0] <= y <= forecast[i][4]) / len(actual)

def persistence_forecast(case):
    last = case["history"][-1]["rates"]
    d = case["horizon_minutes"] / 1440.0
    spread = 0.5 * math.sqrt(d)
    out = {}
    for i, t in enumerate(case["trophies"]):
        r = last[i]
        lg = logit(r if r is not None else 0.0)
        out[t["id"]] = [round(inv_logit(lg + z * spread), 4) for z in Z]
    return out

def trend_forecast(case):
    """House curve baseline approx: logit-linear in ln(1+hours since 19 Nov 00:00Z),
    last 24h of reads, least squares; spread = random-walk + residual."""
    import datetime
    t0 = datetime.datetime(2026, 11, 19, tzinfo=datetime.timezone.utc)
    def hours(at):
        dt = datetime.datetime.fromisoformat(at.replace("Z", "+00:00"))
        return (dt - t0).total_seconds() / 3600.0
    hist = case["history"]
    # last 24h of reads
    t_end = hours(hist[-1]["at"])
    pts = [(hours(h["at"]), h["rates"]) for h in hist if hours(h["at"]) >= t_end - 24]
    d = case["horizon_minutes"] / 1440.0
    rw = 0.5 * math.sqrt(d)
    out = {}
    h_t = hours(case["target"])
    x_t = math.log1p(max(h_t, 0.0))
    for i, t in enumerate(case["trophies"]):
        xs, ys = [], []
        for hh, rates in pts:
            r = rates[i]
            if r is None: continue
            xs.append(math.log1p(max(hh, 0.0))); ys.append(logit(r))
        if len(xs) < 3:
            f = persistence_forecast(case)[t["id"]]
            out[t["id"]] = f; continue
        mx = sum(xs)/len(xs); my = sum(ys)/len(ys)
        sxx = sum((x-mx)**2 for x in xs); sxy = sum((x-mx)*(y-my) for x,y in zip(xs,ys))
        b = sxy/sxx if sxx > 0 else 0.0
        a = my - b*mx
        resid = [y-(a+b*x) for x,y in zip(xs,ys)]
        sig = math.sqrt(sum(e*e for e in resid)/max(len(resid)-2,1))
        spread = math.sqrt(rw*rw + sig*sig)
        lg = a + b*x_t
        out[t["id"]] = [round(clamp(inv_logit(lg + z*spread)),4) for z in Z]
    return out

# ---- synthetic practice answers ----
def build_answers():
    case = json.load(open(f"{KIT}/input/p00.in"))
    r0 = case["history"][-1]["rates"]
    r_now = [0.03,99.99,0.73,0.14,0.06,0.06,0.06,0.24,0.06,0.06,0.06,0.06,0.06,0.03,0.03,0.04,0.04,0.03,0.04,1.89,0.41,0.04,0.04,0.04,0.03,0.03,0.04,0.03,0.04,0.03,0.03,0.03,0.09,0.05,0.24,0.06,0.14,0.14,0.06,0.04,0.04,0.04,0.04,0.08,0.04,0.09,0.04,0.05,0.04,0.11,0.06,0.04,0.05,0.04,0.04]
    ids = [t["id"] for t in case["trophies"]]
    # SYN-A: power decay in N, N_now=9400 (sum-calibrated to 110.55)
    def syn_a(N_now=9400):
        out = {}
        for i, a in enumerate(r0):
            if a >= 99.99: out[ids[i]] = 99.99; continue
            s = max(0.0, -math.log(max(r_now[i],0.005)/a)/math.log(N_now/638.0))
            out[ids[i]] = a * (3868.0/638.0)**(-s)
        # rescale non-100 to hit sum 110.55 exactly
        ssum = sum(v for k,v in out.items() if v < 99)
        target = 110.55 - 99.99
        f = target/ssum
        return {k: (v if v>=99 else v*f) for k,v in out.items()}
    # SYN-B: class-factor dilution, rescaled to sum 110.55
    def syn_b():
        def cls(t):
            d = (t["name"]+" "+t["description"]).lower()
            if t["id"] == 2: return 1.0
            if "chapter 1" in d or "fishing challenge" in d or "arcade game" in d: return 0.75
            if "chapter" in d or "final mission" in d or "complete breakfast" in d: return 0.45
            if "100% completion" in d or "platinum" in d.lower() or "250 stunts" in d or "collectible card" in d: return 0.10
            if "complete 50 challenges" in d or "waink" in d or "meditate" in d: return 0.15
            return 0.30
        raw = {t["id"]: r0[i]*cls(t) for i,t in enumerate(case["trophies"])}
        ssum = sum(v for k,v in raw.items() if k != 2)
        f = (110.55-99.99)/ssum
        return {k: (99.99 if k==2 else v*f) for k,v in raw.items()}
    # SYN-C: new cohort earns ~103 non-fishing trophies, distributed ~proportional to sqrt(E_old)
    def syn_c():
        import math as m
        E_old = [round(r*638/100) for r in r0]
        w = [m.sqrt(max(e,0.5)) if i!=1 else 0 for i,e in enumerate(E_old)]
        tot = sum(w)
        new_e = [103.0*x/tot for x in w]
        out = {}
        for i,t in enumerate(case["trophies"]):
            if i==1: out[t["id"]] = 99.99; continue
            out[t["id"]] = (E_old[i]+new_e[i])/3868.0*100
        return out
    return {"A_power": syn_a(), "B_class": syn_b(), "C_sparse": syn_c()}

if __name__ == "__main__":
    ans = build_answers()
    for name, a in ans.items():
        print(name, "sum", round(sum(a.values()),2), "ch1 %.3f ch2 %.3f ch3 %.3f plat %.3f fish-ch %.3f arcade %.3f" % (a[3],a[4],a[5],a[1],a[20],a[21]))
    for cid in ["p00","p01"]:
        case = json.load(open(f"{KIT}/input/{cid}.in"))
        pf = persistence_forecast(case); tf = trend_forecast(case)
        print(f"--- {cid} baselines ---")
        for name, a in ans.items():
            print(f"  {name}: persistence loss {pinball_loss(pf,a):.4f} (miss {typical_miss(pf,a):.2f} cov {coverage90(pf,a):.2f}) | trend loss {pinball_loss(tf,a):.4f}")
