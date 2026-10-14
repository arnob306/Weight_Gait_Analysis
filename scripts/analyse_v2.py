import json, collections, math, sys

rows = [json.loads(l) for l in open(sys.argv[1])]
REPS = ["raw_video", "skeleton", "overlay", "montage"]
PROMPTS = ["Z_zero_shot", "R_rubric"]


def binom_p(k, n, p=1 / 3):
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))


def mcnemar(a, b):
    n01 = sum(1 for x, y in zip(a, b) if x and not y)
    n10 = sum(1 for x, y in zip(a, b) if y and not x)
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    k = min(n01, n10)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(0, k + 1)) / 2**n)
    return n01, n10, p


by = collections.defaultdict(dict)
for r in rows:
    by[(r["prompt"], r["rep"])][r["instance"]] = r

print("== class accuracy and predicted-class spread ==")
for p in PROMPTS:
    for rep in REPS:
        rs = list(by[(p, rep)].values())
        ok = sum(r["pred"] == r["true_class"] for r in rs)
        spread = collections.Counter(r["pred"] for r in rs)
        per = {c: sum(r["pred"] == c for r in rs if r["true_class"] == c) for c in (0, 1, 2)}
        print(f"{p:12s} {rep:10s} {ok:2d}/30 p={binom_p(ok,30):.3f} pred0/1/2={spread[0]}/{spread[1]}/{spread[2]}  correct per true class 0/1/2={per[0]}/{per[1]}/{per[2]}")

print("\n== blind ==")
for r in rows:
    if r["rep"] == "blind":
        print(r["prompt"], "pred", r["pred"], "conf", r["confidence"], "legs", r["left_pred"], r["right_pred"])

print("\n== Z vs R paired (same clips) ==")
for rep in REPS:
    ids = sorted(by[("R_rubric", rep)])
    z = [by[("Z_zero_shot", rep)][i]["pred"] == by[("Z_zero_shot", rep)][i]["true_class"] for i in ids]
    rr = [by[("R_rubric", rep)][i]["pred"] == by[("R_rubric", rep)][i]["true_class"] for i in ids]
    same = sum(by[("Z_zero_shot", rep)][i]["pred"] == by[("R_rubric", rep)][i]["pred"] for i in ids)
    n01, n10, p = mcnemar(z, rr)
    print(f"{rep:10s} Z-only-right={n01} R-only-right={n10} McNemar p={p:.3f}; identical class answer on {same}/30 clips")

print("\n== leg-level (left impaired in classes 1,2; right impaired in class 2) ==")
for p in PROMPTS:
    for rep in REPS:
        rs = list(by[(p, rep)].values())
        ok_l = [r for r in rs if r["left_pred"] in (True, False)]
        ok_r = [r for r in rs if r["right_pred"] in (True, False)]
        l_hit = sum(r["left_pred"] == (r["true_class"] >= 1) for r in ok_l)
        r_hit = sum(r["right_pred"] == (r["true_class"] == 2) for r in ok_r)
        c1 = [r for r in rs if r["true_class"] == 1]
        flag = collections.Counter(
            ("L" if r["left_pred"] is True else "") + ("R" if r["right_pred"] is True else "") or "none" for r in c1
        )
        print(f"{p:12s} {rep:10s} left {l_hit}/{len(ok_l)} right {r_hit}/{len(ok_r)} | class-1 clips flagged: {dict(flag)}")

print("\n== events ==")
for p in PROMPTS:
    for rep in REPS:
        rs = list(by[(p, rep)].values())
        fz_t = [r for r in rs if r["freeze_true"]]
        ob_t = [r for r in rs if r["obstacle_true"]]
        fz_hit = sum(r["freeze_pred"] is True for r in fz_t)
        ob_hit = sum(r["obstacle_pred"] is True for r in ob_t)
        fz_fp = sum(r["freeze_pred"] is True for r in rs if not r["freeze_true"])
        ob_fp = sum(r["obstacle_pred"] is True for r in rs if not r["obstacle_true"])
        print(f"{p:12s} {rep:10s} freeze found {fz_hit}/{len(fz_t)} false alarms {fz_fp}/{30-len(fz_t)} | obstacle found {ob_hit}/{len(ob_t)} false alarms {ob_fp}/{30-len(ob_t)}")

print("\n== consistency, confidence, 'unclear' use, invalid values ==")
for p in PROMPTS:
    for rep in REPS:
        rs = list(by[(p, rep)].values())
        cons = collections.Counter(r["consistent"] for r in rs)
        conf = collections.Counter(r["confidence"] for r in rs)
        unc = inv = 0
        n_fields = 0
        for r in rs:
            pa = r["parsed"]
            for leg in ("left_leg", "right_leg"):
                for f in ("step_length", "foot_lift", "knee_bend_in_swing"):
                    n_fields += 1
                    unc += pa[leg].get(f) == "unclear"
            inv += len(pa["_checks"]["invalid_values"]) > 0
        print(f"{p:12s} {rep:10s} consistent {dict(cons)} confidence {dict(conf)} 'unclear' {unc}/{n_fields} fields; clips with invalid values {inv}")

print("\n== leg fields (all clips) ==")
for p in PROMPTS:
    for rep in REPS:
        rs = list(by[(p, rep)].values())
        c = collections.Counter()
        for r in rs:
            for leg in ("left_leg", "right_leg"):
                for f in ("step_length", "foot_lift", "knee_bend_in_swing"):
                    c[r["parsed"][leg].get(f)] += 1
        print(f"{p:12s} {rep:10s} {dict(c)}")

print("\n== class-1 clips predicted 1: which leg flagged vs walking direction ==")
tab = collections.Counter()
for (p, rep), d in by.items():
    if rep == "blind":
        continue
    for r in d.values():
        if r["true_class"] == 1 and r["pred"] == 1:
            side = "L" if r["left_pred"] is True and r["right_pred"] is not True else ("R" if r["right_pred"] is True and r["left_pred"] is not True else "other")
            tab[(r["direction"], side)] += 1
print(dict(tab))
print("(truth is always the LEFT leg; near leg = direction side)")
print("images per clip:", min(r["n_images"] for r in rows), "to", max(r["n_images"] for r in rows))
