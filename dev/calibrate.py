from simulator import *
import itertools, math
# anchors: (t, sum, {id: rate})
ANCH = [
 (12.8, 147.78, {1:0.63, 3:1.25, 4:1.10, 20:2.04, 2:100.0}),
 (17.5, 110.55, {2:100.0}),
 (26.0, 106.12, {1:0.03, 3:0.73, 4:0.14, 20:1.89, 2:99.99}),
]
def err(params):
    sim = Sim(seed=1, **params)
    tot = 0.0
    for t, ssum, rates in ANCH:
        n, r = sim.rates_at(t)
        tot += ((sum(r)-ssum)/ssum)**2
        for tid, v in rates.items():
            tot += ((r[tid-1]-v)/max(v,0.05))**2 * 0.5
    return tot
best = []
for E0 in [0.8, 1.2, 1.8]:
    for delta in [0.4, 0.55]:
        for sig_e in [1.1, 1.3, 1.5]:
            for rp_med in [12.0, 16.0]:
                for hs in [48.0, 72.0]:
                    p = dict(E0=E0, delta=delta, sig_e=sig_e, rp_med=rp_med, headstart=hs)
                    e = err(p)
                    best.append((e, p))
best.sort(key=lambda x: x[0])
for e, p in best[:6]:
    print(round(e,3), p)
# show best in detail
sim = Sim(seed=1, **best[0][1])
for t, label in [(12.8,"launch"),(17.5,"04:30"),(26.0,"13:00"),(72.5,"deadline"),(181.0,"target")]:
    n, r = sim.rates_at(t)
    print(f"{label}: N={n:.0f} sum={sum(r):.1f} ch1={r[2]:.2f} ch2={r[3]:.2f} ch3={r[4]:.2f} final={r[6]:.2f} plat={r[0]:.2f} fish-ch={r[19]:.2f} id2={r[1]:.2f}")
