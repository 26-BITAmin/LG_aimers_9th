"""§7.219 — `fspec` 슬롯의 학습기 교체: LightGBM → `HistGradientBoosting`.

base: **우리 fold 하네스** (`foldcache`) — 챔피언 층이 아니다. CLAUDE.md §7 3종 세트:
    ① 베이스 명시    이 줄.
    ② 층 격차 감수   우리 층은 `FSPEC_ALPHA=0.458` 로 F 전문가를 섞지만, 챔피언
                     `hj05_r047f330.zip` 은 그 경로가 **꺼져 있고**(`fspec_alpha=0.0`,
                     `fspec_members=[]`) 대신 라우팅 층에서 `user_fspec_{42,7,2024}.txt`
                     를 **F행 가중 0.330** 으로 섞는다 (`predict.py:180`).
                     같은 물건이지만 가중이 0.458 vs 0.330 이고, 앞단 스택도 다르다
                     (우리 = LGBM12+MLP8, 챔피언 = 혜진+Compact 라우팅).
                     → 그래서 **배포 가중 a=0.330 을 1차 판정점으로 고정**하고,
                       하네스가 고른 정직한 a* 는 참고로만 같이 찍는다.
    ③ 재베이스 계획  `user_fspec_*.txt` 3개를 HGB 산출물로 교체 + `predict.py` 의
                     로더/호출 2곳 수정 (`lgb.Booster(model_file=)` → `joblib.load`,
                     `.predict(raw_score=True)` → `.decision_function()`).
                     `tools_hj/rebase_routing.py` 가 이미 이 층을 다루므로 그 옆에
                     `rebase_fspec_learner.py` 를 붙이는 형태. 관문은 `verify_zip.py`.
                     ⚠️ sklearn 아티팩트가 zip 에 새로 들어가지만 `scikit-learn==1.6.1`
                        은 이미 챔피언 `requirements.txt` 에 있다 — 새 의존성 0.

fold 2023 이 없는 이유 (§7 "fold 2023·2024 필수" 에 대한 명시적 예외)
    표준 fold 2023 은 train ≤2022 인데 `FSPEC_MIN_SEASON=2023` 이라 깨끗한 F 학습행이
    **0개**다. 이 슬롯에서 fold 2023 은 정의되지 않는다. 이 저장소가 이미 쓰는 대체
    구간 `2023H2` (≤2023.06 학습 / 2023.07~09 검증) 를 쓴다 — `fspec_verify.py` ·
    `fspec_a.py` 가 정한 전례와 같다. 두 구간의 **부호 일치**를 채택 조건으로 본다.

팔 (전부 같은 X·같은 학습행·같은 라운드 선택 규약)
    L   LGBM 전문가 x 3시드 (42, 7, 2024)          <- 현행 = 배포 `user_fspec`
    H   HGB  전문가 x 3시드 (42, 7, 2024)          <- 학습기 교체
    LH  0.5*L + 0.5*H   (6멤버)                    <- 앙상블 편입
    LL  0.5*L + 0.5*L'  (6멤버, L' = LGBM 시드 1234/31337/99)

    ⭐ `LL` 이 `LH` 의 **영가설**이다. `LH > L` 만 보면 그건 멤버가 3->6 으로 는 효과와
       구분이 안 된다. 판정해야 할 양은 **`LH - LL`** 이다.
       [[use-a-known-dead-axis-as-the-null]] · [[variance-reduction-overlaps-in-ensembles]]

    관문 0 은 corr(z_L, z_H) 다. HGB 는 LightGBM 의 재구현(히스토그램 비닝 · leaf-wise ·
    native NaN)이라 상관이 매우 높을 것으로 **예상**한다. 0.99 대면 a 를 어떻게 줘도
    이득이 노이즈 안이다. [[overlap-is-about-shared-rows-not-branch-count]]

HGB 쪽에서 조심한 것
    ⚠️ `early_stopping="auto"` 는 `n_samples > 10000` 이면 **자동으로 켜지고**
       (`gradient_boosting.py:601`) `validation_fraction=0.1` 의 **랜덤** 분할을 쓴다.
       학습행이 14.5k / 25.7k 라 둘 다 켜진다. 시간 기반 검증이 기본인 이 문제에서
       랜덤 홀드아웃은 규약 위반이다 -> `early_stopping=False` 로 끄고 라운드 선택을
       **LGBM 과 같은 절차**(학습셋 내부 시간분할 + patience 100)로 직접 한다.
    ⚠️ `random_state` 기본값이 None 이다. 시드 고정이 이 저장소의 전제라 명시한다.
    ⚠️ HGB 에는 `bagging_fraction` 이 **없다**. 행 배깅 없이 열 서브샘플링
       (`max_features`) 만 흉내낼 수 있다 -> `H` 3시드의 내부 다양성이 `L` 보다 작고,
       그건 `LH` 의 이득을 **깎는 방향**이다. 미리 장부에 적는다.
       [[averaging-bias-diverse-members-isnt-variance-reduction]]

실행:
    .venv/Scripts/python.exe -u src/fspec_hgb.py 2>&1 | tee experiments/fspec_hgb.log
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

# HGB 는 OpenMP 로 돈다. 동시 세션이 있을 수 있으므로 LGBM(num_threads=6)과 맞춘다.
os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "6")

import lightgbm as lgb
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

import foldcache
from config import MLP_ALPHA, TARGET_COL, TRAIN_CSV
from data import load_csv
from validation import bss, shift_to_prior

FOLDS = ["2023H2", 2024]
SEEDS = (42, 7, 2024)               # 배포 `user_fspec` 과 같은 시드
SEEDS2 = (1234, 31337, 99)          # 영가설 팔 L' 전용
FSPEC_MIN_SEASON = 2023
A_DEPLOY = 0.330                    # 챔피언 predict.py:180 의 F행 가중
GRID = np.round(np.arange(0.0, 0.801, 0.05), 3)
N_BOOT = 2000                       # [[bootstrap-B-is-a-hyperparameter]] — 400 은 부족했다
MAX_ROUNDS = 3000
PATIENCE = 100

# 현행 F 전문가 (fspec_verify.py 와 **동일**하게 유지한다 — 여기서 바꾸면 비교가 깨진다)
SMALL_LGB = dict(objective="binary", learning_rate=0.03, num_leaves=15,
                 min_data_in_leaf=100, feature_fraction=0.7, bagging_fraction=0.7,
                 bagging_freq=1, lambda_l2=10.0, max_bin=127, num_threads=6,
                 verbosity=-1)

# HGB 대응. API 가 허용하는 한 노브를 1:1 로 맞춘다. 맞출 수 없는 것은 위 주석 참조.
#   learning_rate  0.03  <- learning_rate 0.03
#   num_leaves     15    <- max_leaf_nodes 15
#   min_data_in_leaf 100 <- min_samples_leaf 100
#   lambda_l2      10.0  <- l2_regularization 10.0
#   max_bin        127   <- max_bins 127
#   feature_fraction 0.7 <- max_features 0.7
#   bagging_fraction 0.7 <- (없음)
HGB_PARAMS = dict(loss="log_loss", learning_rate=0.03, max_leaf_nodes=15,
                  min_samples_leaf=100, l2_regularization=10.0, max_bins=127,
                  max_features=0.7, early_stopping=False)


def _inner_split(month_f):
    """학습셋 **내부** 시간분할 (뒤 25%). 검증 fold 는 보지 않는다."""
    cut = np.quantile(month_f, 0.75)
    itr, iva = month_f <= cut, month_f > cut
    return itr, iva


def _usable(ytr_f, iva):
    return iva.sum() >= 300 and len(np.unique(ytr_f[iva])) >= 2


def spec_lgb(Xtr_f, ytr_f, month_f, Xva, seeds):
    """현행 LGBM 전문가. `fspec_verify.specialist()` 와 같은 절차."""
    itr, iva = _inner_split(month_f)
    if not _usable(ytr_f, iva):
        itr, iva = np.ones(len(ytr_f), bool), np.zeros(len(ytr_f), bool)
    zs, rounds = [], []
    for s in seeds:
        prm = dict(SMALL_LGB, seed=s, bagging_seed=s, feature_fraction_seed=s,
                   data_random_seed=s)
        if iva.any():
            b0 = lgb.train(prm, lgb.Dataset(Xtr_f[itr], ytr_f[itr]),
                           num_boost_round=MAX_ROUNDS,
                           valid_sets=[lgb.Dataset(Xtr_f[iva], ytr_f[iva])],
                           callbacks=[lgb.early_stopping(PATIENCE, verbose=False)])
            nr = max(50, b0.best_iteration)
        else:
            nr = 200
        b = lgb.train(prm, lgb.Dataset(Xtr_f, ytr_f), num_boost_round=nr)
        zs.append(b.predict(Xva, raw_score=True))
        rounds.append(nr)
    return np.mean(zs, axis=0), rounds


def spec_hgb(Xtr_f, ytr_f, month_f, Xva, seeds):
    """HGB 전문가. 라운드 선택을 LGBM 과 **같은 절차**로 직접 구현한다.

    sklearn 은 커스텀 검증셋을 못 받는다 (`validation_fraction` 랜덤 분할뿐).
    그래서 `staged_decision_function` 으로 내부 시간분할 위의 손실 곡선을 직접 훑고,
    patience 100 으로 멈춘 뒤 그 라운드 수로 전체 학습셋에 재적합한다.
    이렇게 해야 두 팔의 차이가 **학습기 하나**로 좁혀진다.
    """
    itr, iva = _inner_split(month_f)
    if not _usable(ytr_f, iva):
        itr, iva = np.ones(len(ytr_f), bool), np.zeros(len(ytr_f), bool)
    yi = ytr_f.astype(np.int8)
    zs, rounds = [], []
    for s in seeds:
        if iva.any():
            m0 = HistGradientBoostingClassifier(**HGB_PARAMS, max_iter=MAX_ROUNDS,
                                                random_state=s)
            m0.fit(Xtr_f[itr], yi[itr])
            yv = ytr_f[iva]
            best_i, best_ll = 0, np.inf
            for i, d in enumerate(m0.staged_decision_function(Xtr_f[iva])):
                d = np.asarray(d).ravel()
                ll = float(np.mean(np.logaddexp(0.0, d) - yv * d))   # 로지스틱 손실
                if ll < best_ll - 1e-9:
                    best_ll, best_i = ll, i + 1
                elif (i + 1) - best_i >= PATIENCE:
                    break
            nr = max(50, best_i)
        else:
            nr = 200
        m = HistGradientBoostingClassifier(**HGB_PARAMS, max_iter=nr, random_state=s)
        m.fit(Xtr_f, yi)
        zs.append(m.decision_function(Xva).ravel())
        rounds.append(nr)
    return np.mean(zs, axis=0), rounds


def main():
    df = load_csv(TRAIN_CSV)
    print("=" * 78)
    print("  §7.219  fspec 슬롯 학습기 교체 — LightGBM vs HistGradientBoosting")
    print("  base: foldcache (우리 fold 하네스). 챔피언 층 아님 — 헤더 ②③ 참조.")
    print("=" * 78)
    cache = foldcache.load(FOLDS)
    t0 = time.time()

    P = {}
    for f in FOLDS:
        z_lgb, z_mlp, y, _ = cache[f]
        z_dep = (1 - MLP_ALPHA) * z_lgb + MLP_ALPHA * z_mlp     # 배포 스택 로짓

        mtr, mva = foldcache.SPLITS[f](df.season.to_numpy(), df.game_month.to_numpy())
        tr, va = df[mtr], df[mva]
        assert len(va) == len(y) and np.allclose(va[TARGET_COL].to_numpy(), y), \
            "행 정렬 불일치"

        Xtr, ytr, _, Xva, yva, _, _ = foldcache.fold_data(df, f)

        gtr = tr["game_type"].to_numpy().astype(str)
        gva = va["game_type"].to_numpy().astype(str)
        mF = (gtr == "F") & (tr["season"].to_numpy() >= FSPEC_MIN_SEASON)
        vF = gva == "F"

        Xf = Xtr[mF].to_numpy(dtype=np.float32)
        yf = ytr[mF]
        mo = tr["game_month"].to_numpy(dtype=np.float64)[mF]
        Xv = Xva.to_numpy(dtype=np.float32)

        # HGB 는 inf 를 거부한다 (LGBM 은 통과시킨다). 두 팔이 **같은 X** 를 보게
        # 하려면 여기서 한 번에 처리해야 한다 — 한쪽만 씻으면 그게 곧 교란이다.
        n_inf = int(np.isinf(Xf).sum() + np.isinf(Xv).sum())
        if n_inf:
            Xf[np.isinf(Xf)] = np.nan
            Xv[np.isinf(Xv)] = np.nan
        assert set(np.unique(yf)) <= {0.0, 1.0}, "전문가 학습 라벨이 0/1 이 아니다"

        zL, rL = spec_lgb(Xf, yf, mo, Xv, SEEDS)
        zH, rH = spec_hgb(Xf, yf, mo, Xv, SEEDS)
        zL2, rL2 = spec_lgb(Xf, yf, mo, Xv, SEEDS2)

        spec = {"L": zL, "H": zH,
                "LH": 0.5 * zL + 0.5 * zH,
                "LL": 0.5 * zL + 0.5 * zL2}
        P[f] = dict(z_dep=z_dep, spec=spec, y=yva, vF=vF)

        print(f"\n  fold {f}: 검증 {len(yva):,}행 (F {int(vF.sum()):,}) · "
              f"전문가 학습 {int(mF.sum()):,}행 · inf→nan {n_inf}개  "
              f"({time.time()-t0:.0f}s)")
        print(f"    라운드  L {rL}   H {rH}   L' {rL2}")
        # 관문 0 — F 검증행 위의 전문가 로짓 상관
        a, b, c = zL[vF], zH[vF], zL2[vF]
        print(f"    corr(F행)  L~H {np.corrcoef(a, b)[0,1]:.4f}   "
              f"L~L' {np.corrcoef(a, c)[0,1]:.4f}   "
              f"(L~L' 이 같은 계열의 상한 = 기준선)")
        print(f"    sd(z)      L {a.std():.4f}   H {b.std():.4f}   L' {c.std():.4f}")

    ARMS = ["L", "H", "LH", "LL"]

    def score(f, arm, a):
        """F행만 전문가와 섞은 뒤 **전체** fold BSS (오라클 시프트)."""
        d = P[f]
        z = d["z_dep"].copy()
        vf = d["vF"]
        z[vf] = (1 - a) * z[vf] + a * d["spec"][arm][vf]
        return bss(d["y"], shift_to_prior(1 / (1 + np.exp(-z)), d["y"].mean()))

    def score_F(f, arm, a):
        """F행만 떼어 그 자신의 prior 로 시프트 — 서브그룹 진단용."""
        d = P[f]
        vf = d["vF"]
        zf = (1 - a) * d["z_dep"][vf] + a * d["spec"][arm][vf]
        yf = d["y"][vf]
        return bss(yf, shift_to_prior(1 / (1 + np.exp(-zf)), yf.mean()))

    # ── a 스윕 ────────────────────────────────────────────────────────────
    for f in FOLDS:
        print(f"\n  a 스윕 — fold {f} 전체 BSS  (a=0 이 배포 스택 그대로)")
        print(f"  {'a':>6}" + "".join(f"{arm:>10}" for arm in ARMS)
              + "   |" + "".join(f"{'F '+arm:>10}" for arm in ARMS))
        for a in GRID:
            print(f"  {a:>6.2f}" + "".join(f"{score(f, arm, a):>10.1f}" for arm in ARMS)
                  + "   |" + "".join(f"{score_F(f, arm, a):>10.1f}" for arm in ARMS))

    # ── 1차 판정: 배포 가중 a=0.330 고정 ─────────────────────────────────
    print("\n" + "=" * 78)
    print(f"  1차 판정 — 배포 가중 a={A_DEPLOY} 고정 (챔피언 predict.py:180)")
    print("=" * 78)
    print(f"  {'팔':<6}" + "".join(f"{'fold '+str(f):>16}" for f in FOLDS)
          + "".join(f"{'F행 '+str(f):>16}" for f in FOLDS))
    for arm in ARMS:
        row = "".join(f"{score(f, arm, A_DEPLOY) - score(f, 'L', A_DEPLOY):>+16.2f}"
                      for f in FOLDS)
        rowF = "".join(f"{score_F(f, arm, A_DEPLOY) - score_F(f, 'L', A_DEPLOY):>+16.2f}"
                       for f in FOLDS)
        print(f"  {arm:<6}" + row + rowF + ("   <- 기준" if arm == "L" else ""))
    print("  (전부 L 대비 Δ. a=0 대비 L 자신의 값은 아래 절대치 표에서.)")
    print(f"\n  {'':<6}" + "".join(f"{'fold '+str(f):>16}" for f in FOLDS))
    for arm in ["a=0(전문가없음)"] + ARMS:
        vals = [score(f, "L", 0.0) if arm.startswith("a=0")
                else score(f, arm, A_DEPLOY) for f in FOLDS]
        print(f"  {arm:<6}" + "".join(f"{v:>16.1f}" for v in vals))

    # ── 정직한 a* (2023H2 에서만 고른다) ─────────────────────────────────
    inner = "2023H2"
    print("\n" + "=" * 78)
    print(f"  참고 — 팔마다 {inner} 에서 고른 정직한 a* 를 2024 에 적용")
    print("=" * 78)
    for arm in ARMS:
        a_star = float(GRID[int(np.argmax([score(inner, arm, a) for a in GRID]))])
        d0, d1 = score(2024, arm, 0.0), score(2024, arm, a_star)
        i0, i1 = score(inner, arm, 0.0), score(inner, arm, a_star)
        a_or = float(GRID[int(np.argmax([score(2024, arm, a) for a in GRID]))])
        bias = score(2024, arm, a_or) - d1
        print(f"  {arm:<4} a*={a_star:.2f}  {inner} {i1-i0:>+8.2f}   "
              f"2024 {d1-d0:>+8.2f}   부호일치 {'O' if (i1-i0>0)==(d1-d0>0) else 'X'}"
              f"   | 오라클 a={a_or:.2f} 편향 {bias:>+7.2f}")

    # ── 오차막대 — 쌍체 부트스트랩 ───────────────────────────────────────
    print("\n" + "=" * 78)
    print(f"  쌍체 부트스트랩 — 행 재표집 {N_BOOT}회, a={A_DEPLOY} 고정")
    print("  같은 재표집 인덱스를 모든 팔에 쓴다 (쌍체). 시프트는 표집 안에서 다시 푼다.")
    print("=" * 78)
    CONTRASTS = [("H  - L ", "H", "L"),
                 ("LH - L ", "LH", "L"),
                 ("LL - L ", "LL", "L"),
                 ("LH - LL", "LH", "LL")]      # <- 진짜 질문
    for f in FOLDS:
        d = P[f]
        y, vf = d["y"], d["vF"]
        zarm = {}
        for arm in ARMS:
            z = d["z_dep"].copy()
            z[vf] = (1 - A_DEPLOY) * z[vf] + A_DEPLOY * d["spec"][arm][vf]
            zarm[arm] = z
        n = len(y)
        rb = np.random.default_rng(7)
        boots = {k: np.empty(N_BOOT) for k, _, _ in CONTRASTS}
        for t in range(N_BOOT):
            j = rb.integers(0, n, n)
            yj = y[j]
            m = yj.mean()
            sc = {arm: bss(yj, shift_to_prior(1 / (1 + np.exp(-zarm[arm][j])), m))
                  for arm in ARMS}
            for k, x, b in CONTRASTS:
                boots[k][t] = sc[x] - sc[b]
        print(f"\n  fold {f}")
        for k, _, _ in CONTRASTS:
            v = boots[k]
            lo, hi = np.percentile(v, [2.5, 97.5])
            width = hi - lo
            # [[bootstrap-B-can-flip-a-borderline-gate]] — 구간 끝이 0 에서 폭의 10%
            # 안이면 B 를 바꾸면 뒤집힐 수 있는 자리다. 그때는 '경계' 라고 부른다.
            if lo > 0 or hi < 0:
                edge = min(abs(lo), abs(hi))
                verdict = "0 제외" if edge > 0.10 * width else "0 제외(경계 — B 의존)"
            else:
                verdict = "0 포함 — 판별 불가"
            print(f"    {k}  Δ {v.mean():>+8.2f}   95% [{lo:>+8.2f}, {hi:>+8.2f}]"
                  f"   폭 {width:>7.2f}   {verdict}")

    print(f"\n  총 {time.time()-t0:.0f}s")
    print("\n  판정 규약 (사전 등록):")
    print("    관문 0  corr(L,H) 가 corr(L,L') 보다 **낮아야** 한다.")
    print("            같거나 높으면 HGB 는 시드보다 다양성이 없다 → 여기서 종료.")
    print("    관문 1  `LH - LL` 이 두 구간에서 **부호 일치** + 둘 다 0 제외.")
    print("    관문 2  효과 크기가 시드 노이즈(4~9 BSS)와 비교해 어디 있는지 명시.")
    print("    ⚠️ 이 층의 이득이 챔피언 층(a=0.330, 앞단 스택 다름)에서 그대로")
    print("       재현된다는 보장은 없다. 판정자는 LB 슬롯이다 (CLAUDE.md §7.207).")


if __name__ == "__main__":
    main()
