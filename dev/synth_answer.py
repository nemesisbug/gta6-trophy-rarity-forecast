"""Build plausible practice answers (04:30Z / 05:00Z reads) from public facts.

Facts used (all published in the kit / live public page):
- Launch-night rates r0 (638 players, 943 trophies earned, sum 147.78).
- 04:30Z refresh: 3868 players, 4276 trophies earned -> sum of rates 110.55.
- Live Exophase page ~13:00Z 19 Nov (browser fetch): per-trophy rates r_now.
Model: rate decays as a power of players-tracked: r(N) = r0 * (N/638)^(-s_i),
with per-trophy sensitivity s_i from the live page given an assumed N_now.
N_now is pinned by requiring sum of predicted 04:30 rates == 110.55.
"""
import json, math

KIT = "/home/hatch/workspace/makefun-gta6-trophy-rarity-forecast-01-kit"
case = json.load(open(f"{KIT}/input/p00.in"))
r0 = case["history"][-1]["rates"]
names = [t["name"] for t in case["trophies"]]

# Live page rates at ~13:00Z 19 Nov, in trophy-id order (from browser fetch).
r_now = [0.03,99.99,0.73,0.14,0.06,0.06,0.06,0.24,0.06,0.06,0.06,0.06,0.06,0.03,0.03,0.04,0.04,0.03,0.04,1.89,0.41,0.04,0.04,0.04,0.03,0.03,0.04,0.03,0.04,0.03,0.03,0.03,0.09,0.05,0.24,0.06,0.14,0.14,0.06,0.04,0.04,0.04,0.04,0.08,0.04,0.09,0.04,0.05,0.04,0.11,0.06,0.04,0.05,0.04,0.04]

def predict_0430(N_now):
    # s_i from live page; trophy 2 (100% -> 99.99) handled separately (stays ~100)
    out = []
    for a, b in zip(r0, r_now):
        if a >= 99.99:
            out.append(99.99); continue
        s = -math.log(max(b, 0.005) / a) / math.log(N_now / 638.0)
        s = max(0.0, s)
        out.append(a * (3868.0 / 638.0) ** (-s))
    return out

# find N_now that makes sum == 110.55
for N in [8000, 9000, 10000, 12000, 15000, 20000, 30000, 50000, 80000, 120000, 200000]:
    p = predict_0430(N)
    print(N, round(sum(p), 2))
