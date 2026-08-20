"""세부 구종 예측 축 — §7.58의 출력을 3그룹 → 7구종으로 (§7.76 후보)

사용자 제안
    투수 ID 대신 **투수 스탯 시그니처**(as-of 3그룹 믹스율)와 상황만으로 세부
    구종(슬라이더/커브/스플리터…)을 예측하는 전역 함수를 trackman에서 학습해
    main에 꽂는다. §7.58과 같은 운반체(전역 함수, 행마다 입력 갱신, ID 매핑
    불필요)에 출력 입도만 올린 것. "솔직히 안될 것 같긴 한데" — 동의하고,
    측정으로 닫는 것이 목적이다.

무엇이 §7.58과 다른가 — 세부 입도의 고유분
    3그룹 축(pt_*, r3_*)은 §7.58~59에서 전부 닫혔다. 이번 후보의 고유 주장은
    **그룹 내부 조성**(예: 같은 breaking 40%라도 슬라이더형 vs 커브형)이다.
    따라서 대조가 둘 필요하다:
      · implied 분포 = P₃(그룹) × P_league(세부|그룹)  ← 세부 모델이 이걸 못
        이기면 원천에서부터 고유분이 없다 (관문 0)
      · main 잔차에서 세부 피처의 t가 **자기 그룹 축(§7.58의 죽은 축)의 t**를
        넘어야 한다 (죽은 축 = 영가설, 메모리 "use-a-known-dead-axis-as-the-null")

라벨 정규화
    tagged_pitch_type을 소문자화 → {fastball, slider, curveball, changeup,
    splitter, sinker, cutter} 7종 (전체의 99.3%). four-seam(42)·sweeper·
    knuckleball·other·undefined는 제외.

사전 예측 (실행 전에 박아둔다)
    P0  원천 fine 증류량 (implied − full) > +0.01 nat      예측: **그렇다**
        (그룹내 조성이 믹스율·상황과 무관하지는 않을 것)
    P1  main 잔차에서 세부 고유축 생존(부호 일치 + 셔플 2배 + 그룹 대조 초과) = 0개
        예측: **그렇다**
    P2  pt3_* 축들은 §7.58 프로파일 재현 (죽은 축 영가설 정상 동작)   예측: **그렇다**

실행:  .venv/Scripts/python.exe -u src/pitchfine.py 2>&1 | tee experiments/pitchfine.log
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import lightgbm as lgb
import numpy as np
import pandas as pd

import asofdelta
import ctxfit
import foldcache
from config import (ASOF_AXES, ASOF_STRENGTH, CTX_AXES, CTX_K, DATA_DIR,
                    MLP_ALPHA, SEED, TARGET_COL, TRAIN_CSV)
from ctxsplit import resid_t
from data import load_csv
from features import logit_adjust
from pitchtype import (GROUPS, PT_PARAMS, PT_ROUNDS, entropy, main_matrix,
                       mlogloss, tm_matrix)
from validation import bss, shift_to_prior

TM_CSV = DATA_DIR / "trackman_history.csv"
FOLDS = [2023, 2024]
FINE = ["fastball", "slider", "curveball", "changeup", "splitter", "sinker",
        "cutter"]
# 세부 → 3그룹 (다수결. cutter는 fastball 22k vs breaking 15k → fastball)
FINE_GROUP = {"fastball": "fastball", "sinker": "fastball", "cutter": "fastball",
              "slider": "breaking", "curveball": "breaking",
              "changeup": "offspeed", "splitter": "offspeed"}
# main 잔차 보고에서 각 세부 피처의 「죽은 그룹 축」 대조 상대
PAIR = {"ptf_ent": "pt3_ent", "ptf_condent": "pt3_ent", "ptf_top1": "pt3_top1",
        "pt_slider": "pt3_br", "pt_curveball": "pt3_br",
        "pt_sinker": "pt3_ff", "pt_cutter": "pt3_ff", "pt_splitter": "pt3_os"}


def load_tm():
    tm = pd.read_csv(TM_CSV, usecols=[
        "season", "game_date", "trackman_game_id", "pitch_no", "inning",
        "balls_before", "strikes_before", "outs_before",
        "pitcher_trackman_id", "pitcher_hand", "batter_hand",
        "pitch_type_group", "tagged_pitch_type"])
    tm = tm.sort_values(["game_date", "trackman_game_id", "pitch_no"],
                        kind="stable").reset_index(drop=True)
    # as-of 3그룹 커리어 믹스 (§7.58과 동일: 현재 투구 제외 누적)
    oh = np.stack([(tm["pitch_type_group"] == g).to_numpy(np.float64)
                   for g in GROUPS], axis=1)
    cum = (pd.DataFrame(oh, columns=GROUPS)
           .groupby(tm["pitcher_trackman_id"].to_numpy(), sort=False)
           .cumsum().to_numpy()) - oh
    n3 = cum.sum(1)
    tm["n3"] = n3
    for i, g in enumerate(GROUPS):
        tm["r3_" + g] = np.divide(cum[:, i], np.maximum(n3, 1.0))
    tm["y"] = -1
    for i, g in enumerate(GROUPS):
        tm.loc[tm["pitch_type_group"] == g, "y"] = i
    fine = tm["tagged_pitch_type"].astype(str).str.strip().str.lower()
    tm["yf"] = -1
    for i, f in enumerate(FINE):
        tm.loc[fine == f, "yf"] = i
    return tm


def main():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    cached = foldcache.load(FOLDS, verbose=False)
    rng = np.random.default_rng(SEED)

    tm = load_tm()
    ok = (tm["y"].to_numpy() >= 0) & (tm["yf"].to_numpy() >= 0)
    print(f"trackman {len(tm):,}행  3그룹∩7구종 {ok.sum():,}행 ({ok.mean():.3f})")
    print("  세부 구성 " + " ".join(
        f"{f} {float((tm['yf'] == i).mean()):.3f}" for i, f in enumerate(FINE)))

    keep = {}
    for fold in FOLDS:
        fit = ok & (tm["season"].to_numpy() < fold)
        hld = ok & (tm["season"].to_numpy() == fold)
        y3f = tm.loc[fit, "y"].to_numpy()
        yff = tm.loc[fit, "yf"].to_numpy()
        league3 = np.array([float((y3f == i).mean()) for i in range(3)])
        xf = tm_matrix(tm[fit], league3)

        m3 = lgb.train(dict(PT_PARAMS, num_class=3),
                       lgb.Dataset(xf, label=y3f), PT_ROUNDS)
        m7 = lgb.train(dict(PT_PARAMS, num_class=len(FINE)),
                       lgb.Dataset(xf, label=yff), PT_ROUNDS)

        # 리그 그룹내 조성 M[g, fine] — implied 분포용
        M = np.zeros((3, len(FINE)))
        for gi in range(3):
            m_ = yff[y3f == gi]
            for fi in range(len(FINE)):
                M[gi, fi] = float((m_ == fi).mean())

        # ── 관문 0: 원천 holdout — 세부 모델이 implied를 이기는가 ──────
        yfh = tm.loc[hld, "yf"].to_numpy()
        xh = tm_matrix(tm[hld], league3)
        p3h = m3.predict(xh)
        ll = dict(
            const=mlogloss(yfh, np.tile(
                np.array([float((yff == i).mean()) for i in range(len(FINE))]),
                (len(yfh), 1))),
            implied=mlogloss(yfh, p3h @ M),
            full=mlogloss(yfh, m7.predict(xh)),
        )
        print("=" * 82)
        print(f"fold {fold}  세부 구종 예측기 (fit {fit.sum():,} → holdout {hld.sum():,})")
        print(f"  7-class logloss  상수 {ll['const']:.4f}  "
              f"implied(3그룹×리그조성) {ll['implied']:.4f}  full {ll['full']:.4f}")
        print(f"  ★세부 고유 증류량 (implied − full) = {ll['implied'] - ll['full']:+.4f} nat")

        # ── 배포 전체 잔차 (§7.58 ③과 동일) ─────────────────────────────
        tr, va = s < fold, s == fold
        d = df[va].copy()
        yv = cached[fold][2]
        z = (1 - MLP_ALPHA) * cached[fold][0] + MLP_ALPHA * cached[fold][1]
        ctx = ctxfit.fit_ctx(df[tr], TARGET_COL, CTX_AXES, CTX_K)
        a = asofdelta.fit_asof_axes(df[tr], ASOF_AXES, ASOF_STRENGTH)
        adj, _ = logit_adjust(d, {"ctx": ctx, "asof": a})
        p0 = shift_to_prior(1.0 / (1.0 + np.exp(-(z + adj))), yv.mean())
        r = yv - p0

        # ── main 행 예측 → 피처 (+ 죽은 그룹 축 대조) ───────────────────
        xm, r3m, tot = main_matrix(d, league3)
        pt3 = m3.predict(xm)
        ptf = m7.predict(xm)

        def pack(p3, pf):
            f = {"pt3_ff": p3[:, 0], "pt3_br": p3[:, 1], "pt3_os": p3[:, 2],
                 "pt3_ent": entropy(p3), "pt3_top1": p3.max(1),
                 "ptf_ent": entropy(pf), "ptf_top1": pf.max(1)}
            f["ptf_condent"] = f["ptf_ent"] - f["pt3_ent"]
            for i, nm in enumerate(FINE):
                if nm != "fastball" and nm != "changeup":
                    f["pt_" + nm] = pf[:, i]
            return f

        feat = pack(pt3, ptf)
        # 셔플 대조군 — 믹스 입력 4열만 행 셔플 (§7.58과 동일한 정의)
        perm = rng.permutation(len(xm))
        xs = xm.copy()
        xs[:, 6:10] = xm[perm][:, 6:10]
        fsh = pack(m3.predict(xs), m7.predict(xs))

        print(f"\nfold {fold}  기준 BSS {bss(yv, p0):.1f}  행 {len(d):,}")
        print(f"  {'피처':<14}{'잔차 t':>9}{'셔플 t':>9}{'그룹대조 t':>11}")
        for k in feat:
            _, t = resid_t(r, feat[k])
            _, tc = resid_t(r, fsh[k])
            tp = ""
            if k in PAIR:
                _, tg = resid_t(r, feat[PAIR[k]])
                tp = f"{tg:>11.2f}"
                mark = ("  ✓" if abs(t) > 2 and abs(t) > 2 * abs(tc)
                        and abs(t) > abs(tg) else "")
            else:
                tp, mark = f"{'—':>11}", ""
            keep.setdefault(k, []).append(t)
            print(f"  {k:<14}{t:>9.2f}{tc:>9.2f}{tp}{mark}")

    print("\n" + "=" * 82)
    print("판정 — 세부 고유축: 부호 일치 + 셔플 2배 + 자기 그룹 죽은 축 초과")
    print("=" * 82)
    for k, v in keep.items():
        tag = "세부" if k.startswith(("ptf", "pt_")) else "그룹(영가설)"
        same = "✓" if (v[0] > 0) == (v[1] > 0) else "✗"
        print(f"  {k:<14} t = {v[0]:>7.2f} / {v[1]:>7.2f}   부호 {same}   [{tag}]")


if __name__ == "__main__":
    main()
