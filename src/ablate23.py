"""드리프트 거리 × 블렌드 가중 α* — "매끄러운 함수는 드리프트에 강한가"

동기
    `SUBMISSIONS.md`가 사전 등록한 가설이다. LB는 매끄러운 성분의 비중을 올릴 때마다
    보상했다 (MLP 도입 +29.5, α 0.30→0.40 +6.7). 홀드아웃은 두 번 다 과소평가했다.
    가설: **트리는 학습 분포에 맞춘 임계값으로 공간을 쪼개고 잎마다 고정값을 박는다.
    분포가 밀린 곳에서 그 잎 값은 낡은 지역 평균이 된다. 매끄러운 함수는 더 완만하게
    무너진다.** 참이면 α*는 드리프트 거리를 따라 올라가야 한다.

    ⚠️ 검증 대상은 **α 값이 아니라 위 가설**이다. LB로 α를 맞추는 것은 역추론이다
    (`SUBMISSIONS.md` 규칙 2). 여기서 나오는 것은 train 안에서만 잰 숫자다.

왜 새 설계인가
    `SUBMISSIONS.md`가 제안한 원안은 "fold를 2021·2022로 늘려 α* 추세를 본다"였다.
    그런데 같은 문단이 교란도 적어뒀다 — 학습 시즌이 2개뿐이면 MLP가 약해져 α*가
    아래로 끌린다. **사전 등록된 실험이 설계상 가설에 불리하게 편향돼 있다.**

    여기서는 **학습기를 고정하고 검증 연도만 밀어낸다.**

        W2021 (학습 ≤2021) → 검증 2022(h=1) 2023(h=2) 2024(h=3)
        W2022 (학습 ≤2022) → 검증 2023(h=1) 2024(h=2)
        W2023 (학습 ≤2023) → 검증 2024(h=1)          ← foldcache 재사용

    같은 z_lgb·z_mlp가 h=1,2,3을 전부 예측하므로 **모델 실력이 상수**다. α*가 h를
    따라 움직이면 그건 순수하게 드리프트 효과다. 덤으로 hl=0.5라 유효 표본이
    창 길이와 무관하게 ~1.33시즌으로 고정돼 원안의 교란이 대부분 사라진다.

    W2022가 이 실험의 핵심 행이다. h=1(2023)·h=2(2024) 둘 다 2022→2023 단절을
    넘으므로 **단절을 고정한 채 h만 비교**할 수 있다. W2021은 h=1만 단절 이전이라
    h와 단절이 섞인다 — 그래서 보조 증거로만 읽는다.

사전 예측 (실행 전에 박아둔다. `SUBMISSIONS.md`의 0승 4패 기록을 이어간다)
    P1  W2022: α*(2024,h=2) − α*(2023,h=1) ≥ +0.05      내 예측: **아니오** (50/50에 가깝지만
                                                        ablate17에서 0.30~0.50이 평평했다)
    P2  W2021: α*가 2022→2023→2024로 단조 증가            내 예측: **아니오** (노이즈가 단조성을 깬다)
    P3  h=1 고정, 달력 연도별 α* 추세 (2022→2023→2024)     내 예측: **평평, ~0.4 근처**

    즉 나는 이 실험이 **가설을 확증하지 못할 것**으로 본다. 그래도 돌리는 이유:
    음성이면 "α는 0.40에서 멈춘다"가 CV 근거로 확정되고, 마지막 열린 축이 닫힌다.
    그것도 결과다.

    ⚠️ **α* 하나로 판정하지 않는다.** BSS(α) 곡선이 정점 근처에서 평평해
    (`alpha_se.py`: 0.30↔0.40 쌍체 SE = 1.16 BSS) 격자 argmax는 노이즈에 흔들린다.
    그래서 행마다 **부트스트랩 신뢰구간**을 같이 낸다. CI가 겹치면 추세가 아니다.

실행:  .venv/Scripts/python.exe -u src/ablate23.py 2>&1 | tee experiments/ablate_run23.log
"""
import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import lightgbm as lgb
import numpy as np

import mlp
from config import (DROP_FEATURES, ENSEMBLE_VARIANTS, FINAL_LGB_PARAMS, MLP_ALPHA,
                    MLP_SEEDS, MLP_VARIANTS, N_ROUNDS, RECENCY_HALF_LIFE,
                    SMOOTH_K_BATTER, SMOOTH_K_PITCHER, TARGET_COL, TRAIN_CSV,
                    USE_PLAYER_IDS)
from data import load_csv, recency_weight
from features import build_features, fit_artifacts
from foldcache import CACHE, LGB_SEEDS, _key
from mlp_infer import encode_apply
from validation import shift_to_prior

# 다른 프로젝트 잡 3개가 16코어 중 ~10을 쓰고 있다. 4로 묶어 나눠 쓴다.
THREADS = 4

# 학습 창 → 검증 시즌들. 검증은 항상 창보다 미래다.
WINDOWS = {2021: [2022, 2023, 2024], 2022: [2023, 2024]}
CACHED = {2023: 2024}            # 학습창 → foldcache의 fold 이름 (재사용)

# 0.80에서 끊었더니 W2021→2022의 α* CI가 격자 끝에 닿았다. 1.00까지 연다
# (덤으로 α=1.0 = MLP 단독이라 `paired_gap`이 쓸 행이 생긴다).
ALPHAS = np.round(np.arange(0.0, 1.001, 0.05), 2)
N_BOOT = 400
BOOT_CHUNK = 50


# ── 학습 ────────────────────────────────────────────────────────────

def featurize(d, art):
    x = build_features(d, art)
    if USE_PLAYER_IDS:
        for c in ("pitcher_id", "batter_id"):
            x[c] = d[c].to_numpy(dtype=np.float32)
    return x.drop(columns=[c for c in DROP_FEATURES if c in x.columns])


def ckpt(w, kind, tag):
    return CACHE / f"h_w{w}_{_key()}_{kind}_{tag}.npz"


def member_list(kind):
    if kind == "lgb":
        return [f"{vn}_{s}" for vn in ENSEMBLE_VARIANTS for s in LGB_SEEDS]
    return [f"{vn}_{s}" for vn, (_, ns) in MLP_VARIANTS.items() for s in MLP_SEEDS[:ns]]


def try_load(wmax, valids):
    """멤버 체크포인트가 전부 있으면 z만 모아 돌려준다 (피처·인코딩 전부 생략).

    분석 코드를 고칠 때마다 20분씩 다시 학습할 이유가 없다. 여기가 있어야
    "분석만 다시" 가 몇 초로 끝난다.
    """
    todo = [(k, t) for k in ("lgb", "mlp") for t in member_list(k)]
    if not all(ckpt(wmax, k, t).exists() for k, t in todo):
        return None
    acc = {k: {v: np.zeros(0) for v in valids} for k in ("lgb", "mlp")}
    for k, t in todo:
        with np.load(ckpt(wmax, k, t)) as z:
            for v in valids:
                a = acc[k][v]
                acc[k][v] = z[str(v)] if a.size == 0 else a + z[str(v)]
    for k in acc:
        n = len(member_list(k))
        for v in valids:
            acc[k][v] /= n
    return acc


def run_window(df, wmax, valids, t0):
    """학습 ≤wmax 로 한 번 학습하고 valids 전부를 예측한다.

    체크포인트는 **멤버 단위**다. foldcache가 fold 단위로 만들었다가 MLP 학습 도중
    세션이 내려가면서 LGBM 349초어치를 날린 적이 있다 (MODEL.md §7.19).
    """
    season = df.season.to_numpy()
    yv = {v: df[season == v][TARGET_COL].to_numpy(dtype=np.float64) for v in valids}
    mtr = season <= wmax
    ptr = float(np.average(df[mtr][TARGET_COL].to_numpy(dtype=np.float64),
                           weights=recency_weight(season[mtr], RECENCY_HALF_LIFE,
                                                  ref_season=wmax)))

    pre = try_load(wmax, valids)
    if pre is not None:
        print(f"\n=== 학습창 W{wmax}  체크포인트 재사용 (학습 생략) ===")
        return {v: (pre["lgb"][v], pre["mlp"][v], yv[v]) for v in valids}, ptr

    tr = df[season <= wmax]
    w = recency_weight(tr.season.to_numpy(), RECENCY_HALF_LIFE, ref_season=wmax)
    art = fit_artifacts(tr, weights=w, k_pitcher=SMOOTH_K_PITCHER,
                        k_batter=SMOOTH_K_BATTER)
    Xtr = featurize(tr, art)
    ytr = tr[TARGET_COL].to_numpy(dtype=np.float64)
    print(f"\n=== 학습창 W{wmax}  n={len(Xtr):,}  피처 {Xtr.shape[1]}개"
          f"  유효표본 {w.sum():,.0f} ===")

    va = {}
    for v in valids:
        d = df[season == v]
        va[v] = [featurize(d, art), d[TARGET_COL].to_numpy(dtype=np.float64)]
    del tr
    gc.collect()

    # LightGBM — 배포와 같은 4변형, fold 실험 관례대로 3시드
    z_lgb = {v: np.zeros(len(y)) for v, (_, y) in va.items()}
    n = 0
    for vname, over in ENSEMBLE_VARIANTS.items():
        for s in LGB_SEEDS:
            p = ckpt(wmax, "lgb", f"{vname}_{s}")
            if p.exists():
                with np.load(p) as z:
                    for v in va:
                        z_lgb[v] += z[str(v)]
            else:
                par = dict(FINAL_LGB_PARAMS, **over, seed=s, bagging_seed=s,
                           feature_fraction_seed=s, data_random_seed=s,
                           num_threads=THREADS)
                bst = lgb.train(par, lgb.Dataset(Xtr, ytr, weight=w),
                                num_boost_round=N_ROUNDS)
                cur = {str(v): bst.predict(X, raw_score=True) for v, (X, _) in va.items()}
                np.savez_compressed(p, **cur)
                for v in va:
                    z_lgb[v] += cur[str(v)]
                del bst, cur
            n += 1
            print(f"  lgb {vname}_{s}  ({n}/{len(ENSEMBLE_VARIANTS)*len(LGB_SEEDS)})"
                  f"  ({time.time()-t0:.0f}s)")
    for v in va:
        z_lgb[v] /= n

    # MLP — 인코더는 학습셋으로 한 번 fit, 검증셋들에 apply
    enc = mlp.fit_encoder(Xtr, art)
    A, Ctr = encode_apply(Xtr, enc)
    del Xtr
    gc.collect()
    enc_va = {}
    for v in list(va):
        enc_va[v] = encode_apply(va[v][0], enc)
        va[v][0] = None              # LGBM은 끝났다. 원본 프레임은 놓아준다
    gc.collect()

    # train_mlp은 에폭마다 valid 로짓을 계산한다. 진짜 예측은 infer로 따로 하므로
    # 여기엔 더미 조각만 넘겨 에폭당 25만행 추론 8회를 아낀다.
    Bd, Cd = A[:512], Ctr[:512]
    z_mlp = {v: np.zeros(len(y)) for v, (_, y) in va.items()}
    m_n = 0
    for vname, (cfg, nseed) in MLP_VARIANTS.items():
        for s in MLP_SEEDS[:nseed]:
            p = ckpt(wmax, "mlp", f"{vname}_{s}")
            if p.exists():
                with np.load(p) as z:
                    for v in va:
                        z_mlp[v] += z[str(v)]
            else:
                m, _ = mlp.train_mlp(A, Ctr, ytr, w, Bd, Cd, enc["sizes"],
                                     seed=s, verbose=False, **cfg)
                cur = {str(v): mlp.infer(m, B, Cv) for v, (B, Cv) in enc_va.items()}
                np.savez_compressed(p, **cur)
                for v in va:
                    z_mlp[v] += cur[str(v)]
                del m, cur
            m_n += 1
            print(f"  mlp {vname}_{s}  ({m_n}/{sum(k for _, k in MLP_VARIANTS.values())})"
                  f"  ({time.time()-t0:.0f}s)")
    for v in va:
        z_mlp[v] /= m_n

    out = {v: (z_lgb[v], z_mlp[v], va[v][1]) for v in va}
    del A, Ctr, enc_va, va
    gc.collect()
    return out, ptr


# ── 분석 ────────────────────────────────────────────────────────────

def alpha_curve(z_lgb, z_mlp, y):
    """α 격자마다 BSS(oracle)과 행별 제곱오차를 낸다.

    oracle shift와 BS_ref는 **전체 표본 값으로 고정**한다. 부트스트랩마다 다시 풀면
    α 비교에 표본 평균의 요동이 섞인다. 여기서 묻는 것은 쌍체 비교(어느 α가 나은가)라
    두 성분을 고정하는 쪽이 맞다.
    """
    ref = y.mean() * (1 - y.mean())
    err = np.empty((len(ALPHAS), len(y)), dtype=np.float32)
    for i, a in enumerate(ALPHAS):
        z = (1 - a) * z_lgb + a * z_mlp
        p = shift_to_prior(1 / (1 + np.exp(-z)), y.mean())
        err[i] = (p - y) ** 2
    return 100000.0 * (1 - err.mean(1, dtype=np.float64) / ref), err, ref


def refine(bss):
    """격자 argmax 주변 3점에 포물선을 맞춰 α*를 격자보다 촘촘히 읽는다."""
    j = int(np.argmax(bss))
    if j in (0, len(bss) - 1):
        return float(ALPHAS[j])
    y0, y1, y2 = bss[j - 1], bss[j], bss[j + 1]
    den = y0 - 2 * y1 + y2
    if den >= 0:                       # 볼록이면 정점이 아니다
        return float(ALPHAS[j])
    step = float(ALPHAS[1] - ALPHAS[0])
    return float(ALPHAS[j] + 0.5 * step * (y0 - y2) / den)


def boot_alpha(err, ref, seed=42):
    """α*의 부트스트랩 분포. 포아송 가중으로 행을 재표집한다.

    err는 (α, n) 행렬이므로 재표집이 그냥 가중 평균이다 — 예측을 다시 만들 필요가 없다.

    ⚠️ **기준 α의 행을 빼고 곱한다.** 정점 근처 α들의 BSS 차이는 0.2~0.3 (평균 제곱오차로
    5e-6 규모)인데 오차값 자체는 0.25 규모다. 원본을 그대로 float32로 25만항 누산하면
    반올림 오차가 그 차이와 같은 자릿수에 온다. 중심화하면 argmin은 그대로면서
    (열마다 상수를 더하는 것뿐) 유효숫자가 전부 차이 쪽에 실린다.
    """
    rng = np.random.default_rng(seed)
    n = err.shape[1]
    base = err[int(np.argmin(err.mean(1, dtype=np.float64)))]
    E = (err - base).astype(np.float32)
    stars = []
    for _ in range(0, N_BOOT, BOOT_CHUNK):
        W = rng.poisson(1.0, size=(n, BOOT_CHUNK)).astype(np.float32)
        m = E @ W                                 # (α, chunk) — tot로 나눌 필요 없다
        stars.append(ALPHAS[np.argmin(m, axis=0)])
        del W, m
    return np.concatenate(stars)


def paired_boot(errs, seed=1234):
    """같은 검증셋을 예측한 여러 창의 α*를 **같은 재표집**으로 동시에 낸다.

    검증 2024는 W2021(h=3)·W2022(h=2)·W2023(h=1)이 전부 예측한다 — 행도 라벨도 같다.
    독스트링의 W2022 행 비교(2023 vs 2024)는 검증셋이 달라 표본 변동이 그대로 섞이지만,
    여기서는 그 성분이 **차분에서 소거**된다. 주변 CI(±0.10)로는 못 보는 차이가 보인다.
    """
    rng = np.random.default_rng(seed)
    n = errs[0].shape[1]
    E = [(e - e[int(np.argmin(e.mean(1, dtype=np.float64)))]).astype(np.float32)
         for e in errs]
    stars = [[] for _ in errs]
    for _ in range(0, N_BOOT, BOOT_CHUNK):
        W = rng.poisson(1.0, size=(n, BOOT_CHUNK)).astype(np.float32)
        for i, e in enumerate(E):
            stars[i].append(ALPHAS[np.argmin(e @ W, axis=0)])
        del W
    return np.array([np.concatenate(s) for s in stars])          # (창, N_BOOT)


def paired_gap(errs, refs, seed=99):
    """창마다 (BSS_lgb − BSS_mlp)를 **같은 재표집**으로 낸다.

    α*는 정점이 평평해(0.30~0.50이 1.1점 안) argmax가 흔들린다. 이건 가설의 **기전을
    직접** 재는 통계다 — 모델이 낡을수록 트리가 MLP보다 빨리 무너지는가.
    부호가 h를 따라 뒤집히면 α*가 왜 움직이는지까지 설명된다.
    """
    rng = np.random.default_rng(seed)
    n = errs[0].shape[1]
    out = [[] for _ in errs]
    gaps = [(e[0] - e[-1]).astype(np.float32) for e in errs]     # 행별 lgb오차 − mlp오차
    for _ in range(0, N_BOOT, BOOT_CHUNK):
        W = rng.poisson(1.0, size=(n, BOOT_CHUNK)).astype(np.float32)
        tot = W.sum(0)
        for i, (g, ref) in enumerate(zip(gaps, refs)):
            out[i].append(-100000.0 * (g @ W) / tot / ref)
        del W
    return np.array([np.concatenate(o) for o in out])


def main():
    t0 = time.time()
    CACHE.mkdir(parents=True, exist_ok=True)
    df = load_csv(TRAIN_CSV)

    rows = []
    for wmax, valids in WINDOWS.items():
        res, ptr = run_window(df, wmax, valids, t0)
        for v in valids:
            rows.append((wmax, v, ptr) + res[v])
    del df
    gc.collect()

    # W2023 → 2024 는 foldcache가 이미 갖고 있다 (같은 지문, 같은 구성)
    for wmax, fold in CACHED.items():
        import foldcache
        d = foldcache.load([fold], verbose=True)
        z_lgb, z_mlp, y, _ = d[fold]
        rows.append((wmax, fold, float("nan"), z_lgb, z_mlp, y))

    print("\n" + "=" * 104)
    print(f"{'창':>6} {'검증':>6} {'h':>3} {'n':>9} {'ȳ':>7} {'단절':>5} "
          f"{'lgb':>8} {'mlp':>8} {'corr':>6} {'α=0':>8} {'α=.40':>8} "
          f"{'α*':>6} {'α* 95%CI':>14}")
    print("=" * 104)

    table = []
    for wmax, v, ptr, z_lgb, z_mlp, y in rows:
        bss, err, ref = alpha_curve(z_lgb, z_mlp, y)
        star, boot = refine(bss), boot_alpha(err, ref)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        i40 = int(np.where(ALPHAS == 0.40)[0][0])
        brk = "O" if (wmax <= 2022 and v >= 2023) else "."     # 2022→2023 단절을 넘는가
        print(f"{'W'+str(wmax):>6} {v:>6} {v-wmax:>3} {len(y):>9,} {y.mean():>7.4f} "
              f"{brk:>5} {mlp.score(z_lgb, y):>8.1f} {mlp.score(z_mlp, y):>8.1f} "
              f"{np.corrcoef(z_lgb, z_mlp)[0,1]:>6.3f} {bss[0]:>8.1f} {bss[i40]:>8.1f} "
              f"{star:>6.3f} {'['+f'{lo:.2f}, {hi:.2f}'+']':>14}")
        table.append(dict(w=wmax, v=v, h=v - wmax, star=star, lo=lo, hi=hi,
                          bss=bss, boot=boot, err=err, ref=ref, ptr=ptr,
                          ybar=float(y.mean())))

    print("\nBSS(oracle) vs α — 전체 격자")
    print(f"{'창→검증':>12} " + " ".join(f"{a:>7.2f}" for a in ALPHAS))
    for r in table:
        print(f"{'W'+str(r['w'])+'→'+str(r['v']):>12} "
              + " ".join(f"{b:>7.1f}" for b in r["bss"]))

    # ── 쌍체 비교 (같은 검증셋) ─────────────────────────────────────
    print("\n" + "=" * 104)
    print("같은 검증셋 위 쌍체 비교 — 행·라벨이 동일하므로 표본 변동이 차분에서 소거된다")
    print("가설이 참이면 Δα*(h 큰 쪽 − 작은 쪽) > 0")
    print("=" * 104)
    for v in sorted({r["v"] for r in table}):
        grp = sorted([r for r in table if r["v"] == v], key=lambda r: -r["h"])
        if len(grp) < 2:
            continue
        st = paired_boot([r["err"] for r in grp])
        gp = paired_gap([r["err"] for r in grp], [r["ref"] for r in grp])
        print(f"\n  검증 {v} (n={len(grp[0]['err'][0]):,})")
        print("    " + "   ".join(f"W{r['w']} h={r['h']}: α*={r['star']:.3f}"
                                  f" lgb−mlp={g.mean():+7.1f}"
                                  for r, g in zip(grp, gp)))
        for i in range(len(grp) - 1):
            for j in range(i + 1, len(grp)):
                d, dg = st[i] - st[j], gp[i] - gp[j]
                lo, hi = np.percentile(d, [2.5, 97.5])
                glo, ghi = np.percentile(dg, [2.5, 97.5])
                print(f"    h={grp[i]['h']} − h={grp[j]['h']}:"
                      f"  Δα* {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] P(>0)={(d > 0).mean():.2f}"
                      f"   |  Δ(lgb−mlp) {dg.mean():+7.1f} [{glo:+.1f}, {ghi:+.1f}]"
                      f" P(<0)={(dg < 0).mean():.2f}")

    # ── 사전 예측 채점 ──────────────────────────────────────────────
    def get(w, v):
        return next((r for r in table if r["w"] == w and r["v"] == v), None)

    print("\n" + "=" * 104)
    print("사전 예측 채점 (독스트링에 실행 전 박아둔 것)")
    print("=" * 104)

    a, b = get(2022, 2023), get(2022, 2024)
    if a and b:
        d = b["star"] - a["star"]
        ov = not (b["lo"] > a["hi"] or a["lo"] > b["hi"])
        print(f"  P1  W2022 단절 고정, h=1→2:  α* {a['star']:.3f} → {b['star']:.3f}"
              f"  (Δ {d:+.3f})   CI 겹침 {'O' if ov else 'X'}")
        print(f"      가설(Δ≥+0.05) {'성립' if d >= 0.05 else '불성립'}"
              f"   내 예측 '아니오' {'적중' if d < 0.05 else '빗나감'}")

    w21 = [get(2021, v) for v in (2022, 2023, 2024)]
    if all(w21):
        st = [r["star"] for r in w21]
        mono = st[0] <= st[1] <= st[2]
        print(f"  P2  W2021 h=1→2→3:  α* {st[0]:.3f} → {st[1]:.3f} → {st[2]:.3f}"
              f"   단조증가 {'O' if mono else 'X'}"
              f"   내 예측 '아니오' {'적중' if not mono else '빗나감'}")

    h1 = [r for r in table if r["h"] == 1]
    if len(h1) >= 2:
        h1.sort(key=lambda r: r["v"])
        s = " → ".join(f"{r['v']}:{r['star']:.3f}" for r in h1)
        rng_ = max(r["star"] for r in h1) - min(r["star"] for r in h1)
        print(f"  P3  h=1 달력 추세:  {s}   폭 {rng_:.3f}"
              f"   내 예측 '평평(~0.4)' {'적중' if rng_ <= 0.10 else '빗나감'}")

    print(f"\n  배포값 α={MLP_ALPHA}.  위 세 줄이 전부 음성이면 α는 여기서 멈춘다 —"
          f" 그것도 CV 근거다.")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
