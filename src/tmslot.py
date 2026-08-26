"""E2 — Trackman 을 s29 배포 층 잔차에 다시 묻는다 (§7.116 사전 등록).

§7.39와 무엇이 다른가
    §7.39의 tmresid는 **블렌드 + δ 2축** 시절의 측정이다. 그 뒤 gap 축 2개(§7.55·60),
    init_score(§7.111), 소프트 라벨(§7.70)이 들어와 스택이 다르다. 그리고 §7.39의
    t는 전부 **OLS**였다 — 투수 상수 회귀자의 OLS t는 잔차의 투수 내 상관 때문에
    √(행/투수)≈24배까지 부풀 수 있다 (§7.105). 여기서는 **투수 CR0**를 나란히 찍는다.

세 부분 (전부 학습 0회 — 캐시된 z 위 벡터 연산)
    A. §7.9 컬럼 10개의 잔차 t (OLS + CR0) — §7.39 반전 패턴이 현 스택에서 유지되나.
    B. 슬롯(형질 mean) × same_hand(행별 상태) 상호작용 — "투수 레벨"이 아닌 운반체의
       첫 시험. 대조군 3개 동시: 슬롯 단독(레벨 — 이게 같이 뜨면 재포장), same_hand
       단독(기존 축), 죽은축 널(relpoint_disp × sh — §7.58 죽은축 영가설 기법).
    C. β 스윕 — tm_relh_std (§7.39 표에서 부호 일관인데 유일하게 미스윕된 컬럼.
       방향 사전 지정: **음**) + tm_ext_std (죽은축 널, §7.39 방향 **양**, 고원 2배로
       사망했던 프로파일) + B의 상호작용 (방향 미지 → 대칭 그리드).

판정 (사전 등록 §7.116)
    B 통과선: 양 fold CR0 |t|>2 & 부호 일치 & 상호작용 |t| > 슬롯 단독 |t| & 널과 구별.
    C 채택 논의는 고원 ≥8배 + 크기비 ≤3 + mixdiag 전부 통과할 때만.
    사전 예측: A는 mean류 CR0 |t|<2, B는 죽는다, C는 고원 ≤2배.

모델: s29 배포 층 (pclip.py 와 같은 z — LGBM init_score + MLP α + δ축 4개, clean 행).
실행:  .venv/Scripts/python.exe -u src/tmslot.py 2>&1 | tee experiments/tmslot.log
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

import foldcache as _fc

KEY = _fc._key()

import jointbeta as J                                              # noqa: E402
from config import (ASOF_AXES, CTX_AXES, MLP_ALPHA, ROOT,          # noqa: E402
                    TARGET_COL, TRAIN_CSV)
from ctxsplit import resid_t                                       # noqa: E402
from data import load_csv                                          # noqa: E402
from validation import bss, shift_to_prior                         # noqa: E402

FOLDS = [2023, 2024]
CACHE = ROOT / "experiments" / "cache"
TM_CSV = ROOT / "model_s28" / "trackman_pitcher.csv"

KEY_COLS = ["tm_relh_std", "tm_rels_std", "tm_relpoint_disp",
            "tmfb_relh_std", "tmfb_rels_std", "tmfb_relpoint_disp",
            "tm_ext_std", "tm_relh_mean", "tm_rels_mean", "tm_ext_mean"]
BETAS = [0.005, 0.01, 0.02, 0.03, 0.04, 0.06, 0.10, 0.15, 0.25]


def resid_t_cr0(r, x, cl, n_id):
    """투수 클러스터 강건 t (CR0, §7.105). 기울기는 OLS와 같고 t만 바뀐다."""
    xc = x - x.mean()
    den = float((xc ** 2).sum())
    if den <= 0:
        return 0.0, 0.0
    sl = float((xc * r).sum() / den)
    e = r - r.mean() - sl * xc
    g = np.bincount(cl, weights=xc * e, minlength=n_id)
    v = float((g ** 2).sum()) / den ** 2
    return sl, (sl / np.sqrt(v) if v > 0 else 0.0)


def zscore_trainside(tm, col, fold, idx_vals):
    """열을 학습 시즌 통계로 표준화해 idx(행별 값)로 돌려준다. 결측 → 0(=평균)."""
    trv = tm[tm.season <= fold][col].to_numpy(dtype=np.float64)
    mu, sd = np.nanmean(trv), np.nanstd(trv)
    v = (idx_vals - mu) / (sd if sd > 0 else 1.0)
    return np.where(np.isnan(v), 0.0, v)


def main():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    gt = df["game_type"].to_numpy()
    y = df[TARGET_COL].to_numpy(dtype=np.float64)
    codes, uniq = pd.factorize(df["pitcher_id"].to_numpy())
    n_id = len(uniq)
    mclean = ~((gt == "F") & (s < 2023))

    tm = pd.read_csv(TM_CSV)
    # 파생 슬롯 축 — mean에서만 만든다 (산포 아님)
    tm["tm_abs_rels"] = tm.tm_rels_mean.abs()
    tm["tm_angle"] = np.degrees(np.arctan2(tm.tm_abs_rels, tm.tm_relh_mean))
    print(f"train {df.shape}   trackman 테이블 {tm.shape}   지문 {KEY}   모델 = s29\n")

    state = {}
    for fold in FOLDS:
        Xax, _, _, _ = J.axis_matrix(df, s, fold)
        dep = np.array([b for _, b in CTX_AXES] + [b for _, b in ASOF_AXES])
        vi = np.where(s == fold)[0]
        with np.load(CACHE / f"fold{fold}_{KEY}.npz") as d:
            z_mlp, yc = d["z_mlp"], d["y"]
        assert np.allclose(yc, y[vi]), "캐시 y 불일치 — 지문이 틀렸다"
        z_lgb = np.load(CACHE / f"initdep_fold{fold}_ctr0.npz")["z"]
        z = (1 - MLP_ALPHA) * z_lgb + MLP_ALPHA * z_mlp + Xax @ dep

        mm = mclean[vi]
        yv = y[vi][mm]
        z0 = z[mm]
        p0 = shift_to_prior(1 / (1 + np.exp(-z0)), yv.mean())
        base = bss(yv, p0)
        r = yv - p0

        d = df.iloc[vi[mm]]
        pid = d["pitcher_id"].to_numpy(dtype=np.int64)
        cl = codes[vi][mm]
        sub = tm[tm.season == fold].set_index("pitcher_id")
        hit = np.isin(pid, sub.index.to_numpy())
        state[fold] = dict(yv=yv, z0=z0, p0=p0, base=base, r=r, d=d,
                           pid=pid, cl=cl, sub=sub)
        print(f"fold {fold}   기준 BSS {base:.1f}   n={len(yv):,}   "
              f"Trackman 조회 적중 {hit.mean():.1%}")

    # ── A. §7.9 컬럼 10개 — 현 스택 잔차 t ─────────────────────────────
    print(f"\n{'='*96}\n█ A — §7.9 컬럼 10개의 잔차 t (현 스택. §7.39 값은 옛 스택·OLS)\n{'='*96}")
    print(f"  {'컬럼':<22}" + "".join(f"{f'f{f} OLS':>11}{f'f{f} CR0':>10}" for f in FOLDS)
          + f"{'§7.39 (OLS)':>16}")
    OLD = {"tm_relh_std": "−3.01/−2.86", "tm_rels_std": "+3.25/−2.34",
           "tm_relpoint_disp": "+1.50/−2.96", "tmfb_relh_std": "—",
           "tmfb_rels_std": "+3.13/−2.29", "tmfb_relpoint_disp": "+2.33/−2.94",
           "tm_ext_std": "+6.63/+3.96", "tm_relh_mean": "—", "tm_rels_mean": "—",
           "tm_ext_mean": "—"}
    for c in KEY_COLS:
        row = f"  {c:<22}"
        for f in FOLDS:
            st = state[f]
            v = st["sub"][c].reindex(st["pid"]).to_numpy(dtype=np.float64)
            v = np.where(np.isnan(v), np.nanmean(v), v)
            _, t_ols = resid_t(st["r"], v)
            _, t_cr0 = resid_t_cr0(st["r"], v, st["cl"], n_id)
            row += f"{t_ols:>+11.2f}{t_cr0:>+10.2f}"
        print(row + f"{OLD.get(c, '—'):>16}")

    # ── B. 슬롯 × same_hand 상호작용 ───────────────────────────────────
    print(f"\n{'='*96}\n█ B — 슬롯(형질) × same_hand(행별) 상호작용 + 대조군 3개\n{'='*96}")
    print("  통과선: 양 fold CR0 |t|>2 & 부호 일치 & 상호작용 |t| > 슬롯 단독 |t| & 널과 구별")
    SLOTS = ["tm_relh_mean", "tm_abs_rels", "tm_angle"]
    for f in FOLDS:
        st = state[f]
        d = st["d"]
        sh = (d["pitcher_hand"].to_numpy() == d["batter_hand"].to_numpy()).astype(np.float64)
        shc = sh - sh.mean()
        print(f"\n  ── fold {f} (same_hand 비율 {sh.mean():.3f}) " + "─" * 55)
        print(f"  {'변수':<28}{'OLS t':>9}{'CR0 t':>9}")
        _, t = resid_t(st["r"], shc)
        _, tc = resid_t_cr0(st["r"], shc, st["cl"], n_id)
        print(f"  {'same_hand 단독 (기존 축)':<28}{t:>+9.2f}{tc:>+9.2f}")
        for slot in SLOTS:
            raw = st["sub"][slot].reindex(st["pid"]).to_numpy(dtype=np.float64)
            vz = zscore_trainside(tm, slot, f, raw)
            for name, v in ((f"{slot} 단독(레벨)", vz),
                            (f"{slot} × sh", vz * shc)):
                _, t = resid_t(st["r"], v)
                _, tc = resid_t_cr0(st["r"], v, st["cl"], n_id)
                print(f"  {name:<28}{t:>+9.2f}{tc:>+9.2f}")
        raw = st["sub"]["tm_relpoint_disp"].reindex(st["pid"]).to_numpy(dtype=np.float64)
        vz = zscore_trainside(tm, "tm_relpoint_disp", f, raw)
        _, t = resid_t(st["r"], vz * shc)
        _, tc = resid_t_cr0(st["r"], vz * shc, st["cl"], n_id)
        print(f"  {'널: relpoint_disp × sh':<28}{t:>+9.2f}{tc:>+9.2f}   ← 죽은축 영가설")

    # ── C. β 스윕 — 사후 로짓 가산 (학습 없음) ─────────────────────────
    print(f"\n{'='*96}\n█ C — β 스윕 (tm_relh_std 사전 방향 음 / tm_ext_std 죽은축 널, 방향 양)\n{'='*96}")
    CANDS = [("−tm_relh_std", "tm_relh_std", -1.0),
             ("+tm_ext_std (널)", "tm_ext_std", +1.0)]
    print("  " + " ".join(f"{b:>7}" for b in BETAS))
    gains = {}
    for label, col, sign in CANDS:
        for f in FOLDS:
            st = state[f]
            raw = st["sub"][col].reindex(st["pid"]).to_numpy(dtype=np.float64)
            v = sign * zscore_trainside(tm, col, f, raw)
            row = []
            for b in BETAS:
                p = shift_to_prior(1 / (1 + np.exp(-(st["z0"] + b * v))), st["yv"].mean())
                row.append(bss(st["yv"], p) - st["base"])
            gains[(label, f)] = row
            print(f"  {label:<20} f{f}  " + " ".join(f"{g:>+7.2f}" for g in row))
    print(f"\n  고원(양 fold 양수인 연속 β 구간의 max/min — 채택 논의 기준 ≥8배):")
    for label, _, _ in CANDS:
        ok = [b for b, g23, g24 in zip(BETAS, gains[(label, 2023)], gains[(label, 2024)])
              if g23 > 0 and g24 > 0]
        if ok:
            print(f"    {label:<20} [{min(ok)}, {max(ok)}]  폭 {max(ok)/min(ok):.1f}배")
        else:
            print(f"    {label:<20} 양 fold 양수 β 없음")

    # 상호작용은 방향 미지 → 대칭 그리드 (참고용 — B를 통과했을 때만 의미)
    print(f"\n  상호작용 (대칭 그리드, 참고용):")
    SGRID = [-0.10, -0.06, -0.03, -0.015, 0.015, 0.03, 0.06, 0.10]
    print("  " + " ".join(f"{b:>7}" for b in SGRID))
    for slot in SLOTS:
        for f in FOLDS:
            st = state[f]
            d = st["d"]
            sh = (d["pitcher_hand"].to_numpy() == d["batter_hand"].to_numpy()).astype(np.float64)
            shc = sh - sh.mean()
            raw = st["sub"][slot].reindex(st["pid"]).to_numpy(dtype=np.float64)
            v = zscore_trainside(tm, slot, f, raw) * shc
            row = []
            for b in SGRID:
                p = shift_to_prior(1 / (1 + np.exp(-(st["z0"] + b * v))), st["yv"].mean())
                row.append(bss(st["yv"], p) - st["base"])
            print(f"  {slot + '×sh':<20} f{f}  " + " ".join(f"{g:>+7.2f}" for g in row))

    print("\n" + "=" * 96)
    print("읽는 법: 어떤 결과든 채택 근거가 아니다 — LB가 이 가계에 −13.6을 두 번 답했다.")
    print("통과해도 mixdiag(조건 ④) + LB 실측(§0-7)이 남는다. 사전 예측은 §7.116에 있다.")


if __name__ == "__main__":
    main()
