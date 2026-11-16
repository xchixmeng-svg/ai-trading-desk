import json, numpy as np, itertools, datetime as dt
DIS = json.load(open('feat_2024-01_2026-08.json')); HOL = json.load(open('feat_2023-01_2023-12.json'))
names = sorted(DIS[0]['f'])
def mat(rows, fam):
    rs = [r for r in rows if r['fam'] == fam]
    F = np.array([[r['f'][k] for k in names] for r in rs], bool); R = np.array([r['R'] for r in rs])
    Y = np.array([dt.datetime.fromtimestamp(r['t'] / 1000, dt.UTC).year for r in rs]); return F, R, Y
def scan(F, R, Y, years):
    base = {y: R[Y == y].mean() for y in years}; ok = []
    combos = [(i,) for i in range(len(names))] + list(itertools.combinations(range(len(names)), 2)) + list(itertools.combinations(range(len(names)), 3))
    ym = {y: Y == y for y in years}
    for cb in combos:
        m = F[:, cb[0]].copy()
        for j in cb[1:]: m &= F[:, j]
        good = True; st = []
        for y in years:
            v = R[m & ym[y]]
            if len(v) < 20: good = False; break
            mu = v.mean(); vs = np.sort(v)[:-3].mean()
            if mu < base[y] + .05 or mu < .10 or vs <= 0: good = False; break
            st.append((len(v), mu, vs))
        if good: ok.append((cb, st))
    return ok, base
for fam in ('突破', '飆幣做空'):
    F, R, Y = mat(DIS, fam); Fh, Rh, Yh = mat(HOL, fam)
    ok, base = scan(F, R, Y, (2024, 2025, 2026))
    rng = np.random.default_rng(0); chance = []
    for _ in range(10):
        Rs = R.copy()
        for y in (2024, 2025, 2026): idx = np.flatnonzero(Y == y); Rs[idx] = rng.permutation(Rs[idx])
        chance.append(len(scan(F, Rs, Y, (2024, 2025, 2026))[0]))
    print(f"== {fam}: 基準 " + ' '.join(f"{y} {base[y]:+.2f}R" for y in base) + f" | 2023 基準 {Rh.mean():+.2f}R ({len(Rh)}筆)")
    print(f"   通過 2024–26 的組合: {len(ok)} 個；打亂結果後純運氣平均 {np.mean(chance):.0f} 個（最多 {max(chance)}）")
    res = []
    for cb, st in ok:
        m = Fh[:, cb[0]].copy()
        for j in cb[1:]: m &= Fh[:, j]
        v = Rh[m]
        res.append((cb, st, len(v), v.mean() if len(v) else np.nan, np.sort(v)[:-3].mean() if len(v) > 5 else np.nan))
    hold_ok = [x for x in res if x[2] >= 20 and x[3] >= Rh.mean() + .05 and x[3] >= .10]
    print(f"   其中 2023 也成立（≥20筆、≥+0.10R、比基準好 0.05R）: {len(hold_ok)} 個 / {len(res)}")
    res.sort(key=lambda x: -min(s[1] for s in x[1]))
    for cb, st, n, mu, vs in res[:15]:
        print(f"   {' + '.join(names[i] for i in cb):60s} " + ' '.join(f"{a}筆{b:+.2f}" for a, b, c in st) + f" | 2023 {n}筆 {mu:+.2f}R 去3 {vs:+.2f}")
    json.dump([[[names[i] for i in cb], st, n, mu] for cb, st, n, mu, vs in res], open(f'search_{fam}.json', 'w'), default=float)
