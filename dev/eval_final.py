import json, math, sys, importlib.util
sys.path.insert(0, '.')
from simulator import Sim, make_case
from evalharness import pinball_loss, typical_miss, coverage90, persistence_forecast, trend_forecast

spec = importlib.util.spec_from_file_location("mainmod", "../main.py")
mainmod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mainmod)

def run_case(case, ans):
    fc = {r["id"]: r["q"] for r in mainmod.forecast_case(case)["trophies"]}
    pf = persistence_forecast(case)
    tf = trend_forecast(case)
    return (pinball_loss(fc, ans), typical_miss(fc, ans), coverage90(fc, ans),
            pinball_loss(pf, ans), pinball_loss(tf, ans))

# family of simulated finals: checkpoint 72.5h (deadline) -> target 181h
configs = []
for E0 in [0.8, 1.5, 2.5]:
    for N_cp, mult in [(60000, 2.0), (90000, 3.3), (150000, 2.0)]:
        configs.append(dict(E0=E0, N_cp=N_cp, N_tgt=int(N_cp*mult)))
tot = {"mine": [], "pers": [], "trend": []}
for cfg in configs:
    for seed in [1, 2]:
        sim = Sim(seed=seed, delta=0.5, sig_e=1.2, **cfg)
        case, ans = make_case(sim, 72.5, 181.0)
        m, miss, cov, p, t = run_case(case, ans)
        tot["mine"].append(m); tot["pers"].append(p); tot["trend"].append(t)
        print(f"E0={cfg['E0']} Ncp={cfg['N_cp']} Ntgt={cfg['N_tgt']} seed={seed}: "
              f"mine {m:.4f} (miss {miss:.2f} cov {cov:.2f}) pers {p:.4f} trend {t:.4f} "
              f"ch1 last {case['history'][-1]['rates'][2]:.2f}->ans {ans[3]:.2f}")
print()
for k, v in tot.items():
    print(f"{k}: mean {sum(v)/len(v):.4f} worst {max(v):.4f} n={len(v)}")
