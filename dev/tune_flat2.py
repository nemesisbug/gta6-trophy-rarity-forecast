import json, math, random
from evalharness import *

ans = build_answers()
names = ["A_power","B_class","C_sparse"]
med_geo = {tid: math.exp(sum(math.log(ans[s][tid]) for s in names)/3) for tid in ans["A_power"]}

# cross-scenario log-sd per trophy
sd_cross = {}
for tid in med_geo:
    logs = [math.log(ans[s][tid]) for s in names]
    m = sum(logs)/3
    sd_cross[tid] = math.sqrt(sum((x-m)**2 for x in logs)/3)
print("cross-scenario log-sd: mean %.2f median %.2f max %.2f" % (
    sum(sd_cross.values())/55, sorted(sd_cross.values())[27], max(sd_cross.values())))

random.seed(7)
def mc_eval(median_fn, sigma_fn, s_true, n=400):
    """truth ~ median * exp(N(0,s_true)) per trophy; expected loss + coverage."""
    tot, cov, miss = 0.0, 0.0, 0.0
    for _ in range(n):
        truth = {tid: clamp(med_geo[tid]*math.exp(random.gauss(0, s_true))) for tid in med_geo}
        truth[2] = 99.99
        fc = {}
        for tid in med_geo:
            m = median_fn(tid); sg = sigma_fn(tid)
            lg = logit(m)
            fc[tid] = [round(clamp(inv_logit(lg + z*sg)),4) for z in Z]
        tot += pinball_loss(fc, truth); cov += coverage90(fc, truth); miss += typical_miss(fc, truth)
    return tot/n, cov/n, miss/n

for s_true in [0.0, 0.3, 0.6, 0.9]:
    print(f"--- truth noise s_true={s_true} ---")
    for sig in [0.4, 0.5, 0.6, 0.7, 0.9, 1.1]:
        l, c, m = mc_eval(lambda tid: med_geo[tid], lambda tid, s=sig: s, s_true, n=200)
        print(f"  const sigma={sig:.1f}: loss {l:.4f} cov {c:.2f} miss {m:.2f}")
    # adaptive: sigma_i = sqrt(floor^2 + (k*sd_cross_i)^2)
    for floor, k in [(0.4, 1.0), (0.5, 1.0), (0.4, 1.5), (0.6, 1.0)]:
        l, c, m = mc_eval(lambda tid: med_geo[tid], lambda tid, f=floor, kk=k: math.sqrt(f*f + (kk*sd_cross[tid])**2), s_true, n=200)
        print(f"  adapt floor={floor} k={k}: loss {l:.4f} cov {c:.2f} miss {m:.2f}")
