"""`asof` δ 축의 β 스윕 — **음수 격자 포함**, 배포 상태 위에서.

왜 다시 재는가
    §14-6 이 `gap_p_reverse` 를 "최적 β = +0.000, 6/6, 모든 β 가 손해" 로 닫았는데,
    팀원이 **β = −0.017** 로 열어 **LB +8.68** 을 냈다. 같은 절에 답이 적혀 있었다 —
    `gap_p_success ↔ gap_p_reverse` 상관이 **−0.7430** 이다. 이미 배포된 축과
    반대로 움직이면 최적 β 는 **음수**여야 자연스럽다.

    같은 종결 결함이 우리 큐 #5 에도 있다:
    "`gap_middle`·`gap_strike` 종결 — 전이 0.380 / **미측정**".
    `gap_strike` 는 아예 안 재봤다.

설계 (§14-6 의 구조를 그대로, 격자만 고침)
    · 배포 상태 위에서 잰다 — 이미 있는 축과의 중복을 통제한다
    · 격자에 **음수를 포함**한다
    · **계측기 검증**: 배포된 축 하나를 빼고 그 β 를 되찾는지 본다
    · **위약**: 투수를 이력 십분위 안에서 섞는다
    · 두 fold (2023 · 2024)

⚠️ 베이스 모델이 전 시즌을 학습해 잔차가 in-sample 이다. §14-6 도 같은 조건이었고
   계측기 검증이 배포 β 를 복원했으므로 상대 비교는 유효하다. 절대값은 믿지 않는다.
"""
import argparse
import copy
import os
import sys

import joblib
import numpy as np
import pandas as pd


def bss(p, y):
    r = y.mean()
    return 1e5 * (1 - ((p - y) ** 2).mean() / (r * (1 - r)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip-dir", required=True, help="풀어 둔 제출 zip")
    ap.add_argument("--r-col", required=True, help="예: asof_pitcher_middle_rate")
    ap.add_argument("--name", required=True)
    ap.add_argument("--work", required=True,
                    help="asofdelta.py 가 있는 디렉터리. **템플릿 복사는 금지** — "
                         "k_off 가 r_col 마다 달라서 축이 통째로 틀린다 (§22-4)")
    ap.add_argument("--betas", default="-0.05,-0.03,-0.02,-0.017,-0.01,-0.005,0,0.005,0.01,0.02,0.03,0.05")
    ap.add_argument("--drop", default=None, help="계측기 검증: 이 축을 빼고 β 를 되찾아 본다")
    a = ap.parse_args()

    sys.path.insert(0, a.zip_dir)
    from features import build_features, logit_adjust  # noqa
    from predict import Predictor

    bundle = joblib.load(os.path.join(a.zip_dir, "model", "artifacts.joblib"))
    art0 = bundle["artifacts"]
    sys.path.insert(0, a.work)
    import asofdelta as AD
    S = float(art0["asof"][0].get("strength", 35.0))
    AD.ASOF_SPEC[a.name] = dict(id="pitcher_id", n_col="asof_pitcher_n", r_col=a.r_col)
    df0 = pd.read_csv("/Users/hyejin/Aimers_9th/data/train.csv", encoding="utf-8-sig")

    # G0 재현 관문 — 기존 축을 다시 만들어 대조한다 (§22-4 의 k_off 사고 대책)
    have = [(x["name"], x["beta"]) for x in art0["asof"] if x["name"] in AD.ASOF_SPEC]
    redo = {x["name"]: x for x in AD.fit_asof_axes(df0, have, strength=S, ct=art0.get("contam"))}
    for nm, _ in have:
        d0 = next(x for x in art0["asof"] if x["name"] == nm)
        if not (np.allclose(redo[nm]["k_off"], d0["k_off"]) and abs(redo[nm]["mu"] - d0["mu"]) < 1e-8):
            raise SystemExit(f"[중단] {nm} 재현 실패 — 축 제작 경로가 다르다")
    print(f"G0 재현 관문 통과 ({len(have)}축)")

    new = AD.fit_asof_axes(df0, [(a.name, 1.0)], strength=S, ct=art0.get("contam"))[0]
    new["beta"] = 0.0
    print(f"새 축 {a.name}  r_col={a.r_col}  k_off 합 {new['k_off'].sum():,.1f}")
    print(f"배포 축: {[(x['name'], x['beta']) for x in art0['asof']]}\n")

    df = pd.read_csv("/Users/hyejin/Aimers_9th/data/train.csv", encoding="utf-8-sig")
    P = Predictor(model_dir=os.path.join(a.zip_dir, "model"))

    betas = [float(x) for x in a.betas.split(",")]
    out = {}
    for s in (2023, 2024):
        d = df[df.season == s].reset_index(drop=True)
        y = d["control_success"].to_numpy(float)
        src = d.drop(columns=["control_success"])

        art = copy.deepcopy(art0)
        if a.drop:
            art["asof"] = [x for x in art["asof"] if x["name"] != a.drop]
        base_axes = copy.deepcopy(art["asof"])

        # 기준: 새 축 없이
        P.art = art
        p0 = P.predict(src)
        b0 = bss(p0, y)
        z0 = np.log(p0 / (1 - p0))

        # 새 축의 δ 를 한 번만 계산 (β=1 로 두고 뽑는다)
        probe = copy.deepcopy(art)
        one = copy.deepcopy(new); one["beta"] = 1.0
        probe["asof"] = base_axes + [one]
        P.art = probe
        zc, hits = logit_adjust(src, probe)
        z_base, _ = logit_adjust(src, art)
        dv = zc - z_base                      # 새 축의 δ (β=1 기준)
        P.art = art0

        row = {}
        for b in betas:
            p = 1 / (1 + np.exp(-(z0 + b * dv)))
            row[b] = bss(p, y) - b0
        out[s] = row
        print(f"{s}: 기준 BSS {b0:.2f} · 조회 적중 {hits.get(a.name, float('nan')):.3f}")

    print(f"\n{'β':>8} {'Δ2023':>10} {'Δ2024':>10}   판정")
    best = None
    for b in betas:
        d23, d24 = out[2023][b], out[2024][b]
        ok = (d23 > 0 and d24 > 0)
        tag = "★ 양 fold 양수" if ok else ""
        if ok and (best is None or min(d23, d24) > best[1]):
            best = (b, min(d23, d24))
        print(f"{b:>8.4f} {d23:>+10.3f} {d24:>+10.3f}   {tag}")
    print(f"\n최적(양 fold 양수 중 최소값 최대) : {best}" if best
          else "\n양 fold 모두 양수인 β 가 없다")


if __name__ == "__main__":
    main()
