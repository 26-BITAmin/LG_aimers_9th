"""★ 마지막 고리 — 시즌 진행분이 **배포 모델의 잔차**에 남아 있는가.

여기까지의 상태
    ✅ §7.54 H1  "선수마다 다르다"가 참이다. 반쪽 상관 +0.677, 신뢰도 0.807
                 (안정 4분위 χ²/df 0.95 = 진짜로 안 바뀜 · 들썩 4분위 직구 13.5%p)
    ✅ 배포 형태  `asofdelta`가 2025 행에서 그 투수의 **올해 믹스·올해 성공률**을
                 직접 준다. Trackman도, 형질 가정도 필요 없다
    ❓ 남은 것    그것이 **제구로 이어지는가**

⚠️ 지금까지 두 번 쟀는데 둘 다 **조절변수** 틀이었다
        §7.52 P3  믹스 변동 ↔ |Δ성공률|/SE      +0.035
        §7.53 M4  믹스 변동 ↔ 전이오차 z²        +0.033
    둘 다 "이력이 덜 넘어가나"를 물었다. **주효과** 틀은 안 쟀다 —
    브레이킹볼을 13%p 더 던지게 된 투수는 실패 유형 구성이 달라지므로
    성공률 **자체**가 달라야 한다. 모델은 지금 커리어 믹스만 보고 있고,
    그건 '들썩' 4분위에게 낡은 값이다.

방법
    배포 상태(블렌드 + δ 2축)의 잔차 r = y − p 에 각 피처를 회귀한다.
    학습이 없다. t가 0이면 모델이 이미 안다.

⚠️ t가 커도 **채택 근거가 아니다** — §7.9가 홀드아웃 +13.3에서 LB −13.6을 두 번
   맞았다. t는 "모델이 모른다"만 말한다. 부호가 양 fold 일치해야 다음 단계로 간다.

대조군
    `shuf` = 검증 시즌 안에서 pitcher_id를 무작위 재배정해 오프셋을 조회한다.
    컬럼 수·주변분포·결측 패턴이 같고 **투수↔피처 연결만 끊긴다** (§7.51의 그 대조군).

실행:  .venv/Scripts/python.exe -u src/mixresid.py 2>&1 | tee experiments/mixresid.log
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np
import pandas as pd

import asofdelta
import ctxfit
import foldcache
from config import (ASOF_AXES, ASOF_STRENGTH, CTX_AXES, CTX_K, MLP_ALPHA, SEED,
                    TARGET_COL, TRAIN_CSV)
from ctxsplit import resid_t
from data import load_csv
from features import logit_adjust
from validation import bss, shift_to_prior

FOLDS = [2023, 2024]
MIX = ["asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate",
       "asof_pitcher_offspeed_rate"]


def chi2_rows(c_szn, c_pre):
    """행 단위 2×3 동질성 χ²/df — '이 투수가 올해 얼마나 바꿨나'를 노이즈 정규화해서.

    시즌 진행분 vs **그 이전까지**(커리어 − 시즌분)를 비교한다. 두 표본이
    겹치지 않아야 χ²의 귀무분포가 성립한다.
    """
    n1, n2 = c_szn.sum(1), c_pre.sum(1)
    tot = n1 + n2
    ok = (n1 > 0) & (n2 > 0)
    p = np.divide(c_szn + c_pre, np.maximum(tot, 1)[:, None])
    out = np.zeros(len(c_szn))
    for c, n in ((c_szn, n1), (c_pre, n2)):
        e = n[:, None] * p
        out += np.where(e > 0, (c - e) ** 2 / np.maximum(e, 1e-9), 0.0).sum(1)
    return np.where(ok, out / 2.0, 0.0)


# 아직 잔차를 물어본 적 없는 축들 (§7.57). `DELTA_SPEC`에 다 들어 있는데
# §7.54는 success · middle · 믹스 3개만 봤다.
#   reverse  실패 3유형 중 "포수 요구와 반대 방향". §7.49의 오염 보정이 건드린 축이다
#   ball/strike  판정 결과 쪽. success와 다른 것을 잰다
#   타자 쪽  §7.34 G가 batter_id 축 천장을 −25.7로 쟀지만, 그건 **레벨**이었다.
#            시즌 진행분은 레벨이 아니라 **변화**다 — 같은 관문이 아니다
GAP_SPEC = [
    ("gap_reverse", "asof_pitcher_reverse_rate", "asof_pitcher_n"),
    ("gap_ball", "asof_pitcher_ball_rate", "asof_pitcher_n"),
    ("gap_strike", "asof_pitcher_strike_rate", "asof_pitcher_n"),
    ("gap_bsuccess", "asof_batter_success_rate", "asof_batter_n"),
    ("gap_bmiddle", "asof_batter_middle_rate", "asof_batter_n"),
]


def build(d, art, league, priors=None):
    """검증 시즌 행에 대해 시즌 진행분 피처를 만든다. **행 단위 조회뿐.**

    `priors`: {rate 컬럼: 리그 평균}. cold-start(커리어 결측) 행의 대체값으로만 쓴다.
    없으면 0.2로 둔다 — 그 행들은 어차피 소수이고, 없어도 축의 존폐 판정은 바뀌지 않는다.
    """
    priors = priors or {}
    st = asofdelta.season_to_date(d, art)
    out = {}

    n_p = st["asof_pitcher_n"]
    n_m = st["asof_pitcher_pitchmix_n"]
    out["n_szn_pitch"] = n_p
    out["n_szn_mix"] = n_m

    # ── 성공률: 올해 − 커리어 ─────────────────────────────────────
    r_car = d["asof_pitcher_success_rate"].to_numpy(dtype=np.float64)
    r_szn = asofdelta.shrink(st["asof_pitcher_success_rate"], n_p, r_car, league)
    out["r_szn_success"] = r_szn
    out["gap_success"] = r_szn - np.where(np.isfinite(r_car), r_car, league)

    r_mid = d["asof_pitcher_middle_rate"].to_numpy(dtype=np.float64)
    out["gap_middle"] = (asofdelta.shrink(st["asof_pitcher_middle_rate"], n_p, r_mid, 0.2)
                         - np.where(np.isfinite(r_mid), r_mid, 0.2))

    # ── 미측정 축들 (§7.57) ───────────────────────────────────────
    for name, rate, ncol in GAP_SPEC:
        car = d[rate].to_numpy(dtype=np.float64)
        pr = float(priors.get(rate, 0.2))
        base = np.where(np.isfinite(car), car, pr)
        out[name] = asofdelta.shrink(st[rate], st[ncol], car, pr) - base
        out["n_szn_" + name.split("_")[1]] = st[ncol]

    # ── 믹스: 올해 − 커리어 (사용자 관찰의 직접 표현) ──────────────
    n_car = d["asof_pitcher_pitchmix_n"].to_numpy(dtype=np.float64)
    c_szn = np.stack([st[c] for c in MIX], axis=1)
    c_car = np.stack([np.nan_to_num(d[c].to_numpy(dtype=np.float64)) * n_car
                      for c in MIX], axis=1)
    p_szn = c_szn / np.maximum(c_szn.sum(1, keepdims=True), 1e-9)
    p_car = c_car / np.maximum(c_car.sum(1, keepdims=True), 1e-9)
    dif = p_szn - p_car
    for i, nm in enumerate(("ff", "br", "os")):
        out[f"gap_{nm}"] = np.where(n_m > 0, dif[:, i], 0.0)
    out["mix_move_l1"] = np.where(n_m > 0, np.abs(dif).sum(1), 0.0)
    out["mix_chi2df"] = np.log1p(chi2_rows(c_szn, np.maximum(c_car - c_szn, 0.0)))
    return out


def main():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    cached = foldcache.load(FOLDS, verbose=False)
    rng = np.random.default_rng(SEED)

    print(f"기준선 = 블렌드(α={MLP_ALPHA}) + δ{CTX_AXES}  ← 배포 상태\n")
    keep = {}
    for fold in FOLDS:
        tr, va = s < fold, s == fold
        d = df[va].copy()
        yv = cached[fold][2]
        z = (1 - MLP_ALPHA) * cached[fold][0] + MLP_ALPHA * cached[fold][1]
        ctx = ctxfit.fit_ctx(df[tr], TARGET_COL, CTX_AXES, CTX_K)
        adj, _ = logit_adjust(d, {"ctx": ctx})
        p0 = shift_to_prior(1.0 / (1.0 + np.exp(-(z + adj))), yv.mean())
        r = yv - p0

        art = asofdelta.fit_offsets(df[tr])
        league = float(df.loc[tr, TARGET_COL].mean())
        # cold-start 대체값은 **학습 시즌에서만** 뽑는다
        priors = {r: float(np.nanmean(df.loc[tr, r].to_numpy(dtype=np.float64)))
                  for _, r, _ in GAP_SPEC}
        feat = build(d, art, league, priors)

        # 대조군 — 시즌 안에서 ID를 섞어 같은 조회를 돌린다.
        # ⚠️ **타자 축도 재므로 batter_id도 같이 섞어야 한다.** 투수만 섞으면
        #    타자 컬럼의 대조군이 진짜와 똑같아져서 게이트가 무력화된다.
        dsh = d.copy()
        dsh["pitcher_id"] = rng.permutation(d["pitcher_id"].to_numpy())
        dsh["batter_id"] = rng.permutation(d["batter_id"].to_numpy())
        fsh = build(dsh, art, league, priors)

        print("=" * 78)
        print(f"fold {fold}   기준 BSS {bss(yv, p0):.1f}   행 {len(d):,}"
              f"   n_szn 중앙 {np.median(feat['n_szn_pitch']):.0f}")
        print("=" * 78)
        print(f"  {'피처':<18}{'잔차 t':>10}{'대조군 t':>11}{'차':>9}")
        for k in feat:
            _, t = resid_t(r, feat[k])
            _, tc = resid_t(r, fsh[k])
            keep.setdefault(k, []).append((t, tc))
            mark = "  ✓" if abs(t) > 2 and abs(t) > 2 * abs(tc) else ""
            print(f"  {k:<18}{t:>10.2f}{tc:>11.2f}{t - tc:>9.2f}{mark}")

    print("\n" + "=" * 78)
    print("판정 — 부호 일치 + 대조군 초과")
    print("=" * 78)
    print(f"  {'피처':<18}{'t(2023)':>10}{'t(2024)':>10}{'부호일치':>10}{'대조군초과':>12}")
    for k, v in keep.items():
        (t1, c1), (t2, c2) = v
        same = (t1 > 0) == (t2 > 0)
        over = abs(t1) > 2 * abs(c1) and abs(t2) > 2 * abs(c2)
        print(f"  {k:<18}{t1:>10.2f}{t2:>10.2f}{'✓' if same else '✗':>10}"
              f"{'✓' if over else '✗':>12}")
    print("\n  둘 다 ✓ 이고 |t| 가 큰 것만 다음 단계(β 스윕 / 컬럼 추가)로 간다.")
    print("  ✗ 가 하나라도 있으면 §7.51 믹스(부호 반전)와 같은 자리다 → 닫는다.")

    incremental(df, s, cached, [k for k, v in keep.items()
                                if (v[0][0] > 0) == (v[1][0] > 0)
                                and all(abs(t) > 2 * abs(c) for t, c in v)])


def incremental(df, s, cached, cands):
    """★ **배포된 `gap_success`를 얹은 뒤** 남는 잔차에 다시 묻는다.

    왜 필요한가
        `success = 1 − (몰림 + 반대 + …)` 이므로 `gap_reverse`는 `gap_success`와
        구조적으로 겹친다. 위 1단계 기준선에는 배포된 asof 항이 **빠져 있어서**
        같은 신호를 두 번 세고 있을 수 있다.

        §7.37이 count4를 "배포 상태(블렌드 + 1.5·δ_hand) 위의 **증분**으로 쟀다"고
        한 것과 같은 이유다. 축을 하나씩 쌓을 때는 항상 증분으로 재야 한다.
    """
    if not cands:
        return
    print("\n" + "=" * 78)
    print(f"★ 증분 — 배포 상태(+ asof {[n for n, _ in ASOF_AXES]}) 뒤 남는 잔차")
    print("=" * 78)
    rng = np.random.default_rng(SEED)
    out = {}
    for fold in FOLDS:
        tr, va = s < fold, s == fold
        d = df[va].copy()
        yv = cached[fold][2]
        z = (1 - MLP_ALPHA) * cached[fold][0] + MLP_ALPHA * cached[fold][1]
        ctx = ctxfit.fit_ctx(df[tr], TARGET_COL, CTX_AXES, CTX_K)
        a = asofdelta.fit_asof_axes(df[tr], ASOF_AXES, ASOF_STRENGTH)
        adj, _ = logit_adjust(d, {"ctx": ctx, "asof": a})       # ← 배포 상태 전부
        p0 = shift_to_prior(1.0 / (1.0 + np.exp(-(z + adj))), yv.mean())
        r = yv - p0

        art = asofdelta.fit_offsets(df[tr])
        league = float(df.loc[tr, TARGET_COL].mean())
        priors = {c: float(np.nanmean(df.loc[tr, c].to_numpy(dtype=np.float64)))
                  for _, c, _ in GAP_SPEC}
        feat = build(d, art, league, priors)
        dsh = d.copy()
        dsh["pitcher_id"] = rng.permutation(d["pitcher_id"].to_numpy())
        dsh["batter_id"] = rng.permutation(d["batter_id"].to_numpy())
        fsh = build(dsh, art, league, priors)

        print(f"\nfold {fold}  기준 BSS {bss(yv, p0):.1f}  (배포 전부 포함)")
        print(f"  {'피처':<18}{'증분 t':>10}{'대조군 t':>11}{'1단계 t':>10}")
        for c in cands:
            _, t = resid_t(r, feat[c])
            _, tc = resid_t(r, fsh[c])
            out.setdefault(c, []).append(t)
            print(f"  {c:<18}{t:>10.2f}{tc:>11.2f}")

    print("\n  읽는 법: 1단계 t가 컸는데 증분 t가 0으로 죽으면 **`gap_success`와 같은 신호**다.")
    print("           증분에서도 살아남아야 새 축이다.")


if __name__ == "__main__":
    main()
