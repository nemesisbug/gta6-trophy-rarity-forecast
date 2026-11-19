import json, math
from evalharness import *

ans = build_answers()
case = json.load(open(f"{KIT}/input/p00.in"))
names = ["A_power","B_class","C_sparse"]

def fc_from_median_sigma(median, sigma):
    out = {}
    for tid, m in median.items():
        lg = logit(m)
        out[tid] = [round(clamp(inv_logit(lg + z*sigma)),4) for z in Z]
    return out

# candidate medians
med_geo = {}
for tid in ans["A_power"]:
    med_geo[tid] = math.exp(sum(math.log(ans[s][tid]) for s in names)/3)
med_a = ans["A_power"]; med_b = ans["B_class"]; med_c = ans["C_sparse"]
med_prop = {}  # proportional: non-100 trophies x0.221
r0 = case["history"][-1]["rates"]
for i,t in enumerate(case["trophies"]):
    med_prop[t["id"]] = 99.99 if t["id"]==2 else r0[i]*0.221

for mname, med in [("geomean",med_geo),("A",med_a),("B",med_b),("C",med_c),("prop0.221",med_prop)]:
    for sigma in [0.5,0.7,0.9,1.1,1.3,1.5,1.8]:
        fc = fc_from_median_sigma(med, sigma)
        losses = [pinball_loss(fc, ans[s]) for s in names]
        covs = [coverage90(fc, ans[s]) for s in names]
        print(f"{mname:9s} sig={sigma:.1f} mean={sum(losses)/3:.4f} worst={max(losses):.4f} " +
              " ".join(f"{s}:{l:.3f}" for s,l in zip(names,losses)) +
              f" cov={sum(covs)/3:.2f}")
    print()
