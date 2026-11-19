"""Cohort simulator v3: pioneers + regulars.

- t = hours since first unlock 2026-11-18T11:00Z.
- Pioneers: fixed pool P0 (early-access players), play rate r_p ~ LogNormal(ln 10, 0.4)
  h/day, headstart 48h (they have played since before unlock).
- Regulars: join per N(t) anchors; engagement e ~ LogNormal(ln E_med(s), sig_e),
  E_med(s) = E0 * (N(s)/638)^(-delta).
- Trophy i needs H_i play-hours. Earn when cumulative play > H_i.
- Refreshes every U(3,8)h from t=17.5.
"""
import json, math, random

KIT = "/home/hatch/workspace/makefun-gta6-trophy-rarity-forecast-01-kit"
BASE = json.load(open(f"{KIT}/input/p00.in"))
TROPHIES = BASE["trophies"]

def phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

def median_hours(t):
    d = (t["name"] + " " + t["description"]).lower()
    if t["id"] == 2: return 0.0
    if "chapter 1" in d: return 8.0
    if "chapter 2" in d: return 16.0
    if "chapter 3" in d: return 26.0
    if "chapter 4" in d: return 36.0
    if "final mission" in d and "for " not in d: return 48.0
    if "fishing challenge" in d: return 6.0
    if "arcade game" in d: return 8.0
    if "complete breakfast" in d: return 20.0
    if "100% completion" in d or t["tier"] == "platinum": return 260.0
    if "250 stunts" in d: return 220.0
    if "collectible card" in d: return 180.0
    if "waink" in d or "meditate" in d or "50 challenges" in d: return 150.0
    if "final mission for" in d: return 60.0
    if "strangers" in d: return 90.0
    if "discover 100 vehicles" in d: return 70.0
    if "earn $250,000" in d or "purchase all garage" in d or "spend $100,000" in d: return 110.0
    if "explore all of leonida" in d: return 55.0
    return 45.0

H = [median_hours(t) for t in TROPHIES]

class Sim:
    def __init__(self, seed=0, E0=1.0, delta=0.5, sig_e=1.3, N_cp=90000, N_tgt=300000, P0=12, N26=9200.0, rp_med=14.0, rp_sig=0.35, headstart=72.0, lag=4.0):
        self.rnd = random.Random(seed)
        self.E0, self.delta, self.sig_e, self.P0 = E0, delta, sig_e, P0
        self.rp_med, self.rp_sig, self.headstart, self.lag = rp_med, rp_sig, headstart, lag
        self.anchors = [(12.8, 638.0), (17.5, 3868.0), (26.0, N26),
                        (72.5, float(N_cp)), (181.0, float(N_tgt))]
        self.refreshes = [17.5]
        while self.refreshes[-1] < 185:
            self.refreshes.append(self.refreshes[-1] + self.rnd.uniform(3.0, 8.0))
        self.HRS = 190
        self.Nh = [self.N(t) for t in range(self.HRS)]
        self.joins = [max(0.0, self.Nh[h+1]-self.Nh[h]) if h+1 < self.HRS else 0.0 for h in range(self.HRS)]
        # pioneer play rates (deterministic quantiles, no MC noise)
        self.p_rates = [math.exp(random.Random(1000+i).gauss(math.log(rp_med), rp_sig)) for i in range(P0)]

    def N(self, t):
        a = self.anchors
        if t <= 0: return 0.0
        if t <= a[0][0]:
            return a[0][1] * (t / a[0][0]) ** 2
        for (t1, n1), (t2, n2) in zip(a, a[1:]):
            if t <= t2:
                f = (math.log(t) - math.log(t1)) / (math.log(t2) - math.log(t1))
                return math.exp(math.log(n1) + f * (math.log(n2) - math.log(n1)))
        return a[-1][1]

    def Emed(self, s):
        n_eff = max(self.N(s), 638.0)
        return min(3.0 * self.E0, self.E0 * (n_eff / 638.0) ** (-self.delta))

    def rates_at(self, t):
        # 0.5h cohort grid; pioneers separate; cache by t
        if not hasattr(self, "_cache"):
            self._cache = {}
        key = round(t, 2)
        if key in self._cache:
            return self._cache[key]
        E = [0.0] * len(TROPHIES)
        for i in range(len(TROPHIES)):
            c = 0
            for r in self.p_rates:
                if r * (t + self.headstart) / 24.0 > H[i]:
                    c += 1
            E[i] = float(c)
        # cohort integral on 0.5h grid; joins per half hour from N()
        step = 0.5
        nsteps = int(t / step)
        prev = 0.0
        for k in range(nsteps + 1):
            t1 = k * step
            t2 = min((k + 1) * step, t)
            if t2 <= t1:
                break
            j = (self.N(t2) - self.N(t1))
            if j <= 0:
                prev = self.N(t2)
                continue
            age_d = (t - (t1 + t2) / 2.0 + self.lag) / 24.0
            if age_d <= 0:
                prev = self.N(t2)
                continue
            em = self.Emed((t1 + t2) / 2.0)
            for i in range(len(TROPHIES)):
                if H[i] <= 0:
                    E[i] += j * 0.9999
                    continue
                need = H[i] / age_d
                p = 1.0 - phi((math.log(need) - math.log(em)) / self.sig_e)
                E[i] += j * p
            prev = self.N(t2)
        n = self.N(t)
        out = (n, [100.0 * e / n if n > 0 else 0.0 for e in E])
        self._cache[key] = out
        return out

    def observed(self, t):
        r = max([x for x in self.refreshes if x <= t], default=12.8)
        return self.rates_at(r)

def make_case(sim, checkpoint_h, target_h):
    import datetime
    t0 = datetime.datetime(2026, 11, 18, 11, 0, tzinfo=datetime.timezone.utc)
    def iso(h):
        return (t0 + datetime.timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M:%SZ")
    hist = []
    t = 12.8
    while t <= checkpoint_h:
        n, rates = sim.observed(t)
        hist.append({"slot": iso(t), "at": iso(t), "players_tracked": int(round(n)),
                     "trophies_earned": int(round(sum(r*n/100 for r in rates))),
                     "rates": [round(r, 2) for r in rates]})
        t += (0.5 if t < 24 else 1/6)
    ans = None
    for r in sim.refreshes:
        if target_h <= r < target_h + 1.0:
            _, rates = sim.observed(r); ans = [round(x, 2) for x in rates]; break
    if ans is None:
        _, rates = sim.observed(target_h + 0.99); ans = [round(x, 2) for x in rates]
    case = {"id": "sim", "at": iso(checkpoint_h), "horizon_minutes": int((target_h-checkpoint_h)*60),
            "target": iso(target_h), "levels": [0.05,0.25,0.5,0.75,0.95],
            "game": BASE["game"], "trophies": TROPHIES, "history": hist, "notes": "simulated"}
    return case, {t["id"]: ans[i] for i, t in enumerate(TROPHIES)}

if __name__ == "__main__":
    for (E0, delta, sig_e) in [(1.0, 0.5, 1.3), (1.5, 0.5, 1.2), (0.8, 0.6, 1.4), (1.2, 0.4, 1.1)]:
        sim = Sim(seed=1, E0=E0, delta=delta, sig_e=sig_e)
        print(f"E0={E0} delta={delta} sig={sig_e}")
        for t, label in [(12.8, "launch"), (17.5, "04:30Z"), (26.0, "13:00Z"), (72.5, "deadline"), (181.0, "target")]:
            n, r = sim.rates_at(t)
            print(f"  {label} t={t}: N={n:.0f} sum={sum(r):.1f} ch1={r[2]:.2f} ch2={r[3]:.2f} ch3={r[4]:.2f} final={r[6]:.2f} plat={r[0]:.2f} fish-ch={r[19]:.2f}")
        print("  anchors: launch sum 147.78 ch1 1.25 ch2 1.10 plat 0.63 | 04:30 sum 110.55 | 13:00 sum 106.12 ch1 0.73 ch2 0.14 plat 0.03")
