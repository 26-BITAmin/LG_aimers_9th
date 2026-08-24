"""FT-Transformer를 **배포 등가 층**에서 두 슬롯에 넣어본다 (사용자 질문)

배경 — 이 축은 한 번도 측정된 적이 없다
    §7.32 교육자료 전수 대조 표에서 `FT-Transformer / SAINT / NPT`는 "용량 법칙 +
    추론 예산"을 사유로 **논증으로만** 닫혔다. 코드도 로그도 없다.
    용량 법칙은 §7.25·§7.26 — 입력차원 73→223→381→823에서 fold2023/2024 모두
    완벽히 단조 하락. 기전은 §7.13의 "신호 총량 ≈ 0.004 nat".
    FT-Transformer는 피처 79개를 각각 d차원 토큰으로 올리므로 그 법칙의 나쁜 쪽
    끝에 해당한다. 논거는 강하지만 **법칙은 MLP 계열에서 측정됐고**, 어텐션은
    파라미터 수와 유효 자유도의 관계가 다르다. 그래서 잰다.

⚠️ 챔피언이 바뀌었다 — 층을 맞춰야 한다
    §7.103 시절 하네스는 `z = (1-MLP_ALPHA)z_lgb + MLP_ALPHA z_mlp` 2중 블렌드였고
    fold2024 기준이 838.3이었다. 현재 배포는 다음 순서다 (predict.py):
        z  = z_lgb(12)
        z  = (1-MLP_ALPHA)z + MLP_ALPHA z_mlp(8)          MLP_ALPHA = 0.75
        z[F] = (1-FSPEC_ALPHA)z[F] + FSPEC_ALPHA z_fspec   FSPEC_ALPHA = 0.458
        z  = (1-CAT_ALPHA)z + CAT_ALPHA z_cat(12)          CAT_ALPHA = 0.10
    s35layer.py 실측 앵커: s29 층 924.5 → s35 등가 층 930.0 → CatBoost +2.47.
    **낡은 층에서 재면 F 전문가 기여가 1/(1-0.10) = 1.11배 부풀려진다.**

설계 — 선택 fold를 안 쓴다
    s35 층은 **fold2024에서만 정의된다**(fspec 학습셋이 `F & season>=2023`이라
    fold2023에는 깨끗한 F가 없다 — s35layer.py 주석). 그래서 "2023H2에서 고르고
    2024에서 확인"이 불가능하다.

    대신 **같은 슬롯 · 같은 α · 모델만 교체**하는 쌍체 비교를 한다. α를 새로 고르지
    않으므로 커닝할 자유도가 0이고, 대조군이 현행 그 자체다. 이게 선택 fold보다
    강한 설계다 — §7.103에서 선택 fold의 모양을 두 번이나 잘못 읽었다.

      A) fspec 슬롯:  z_fspec(LightGBM 3시드)  ↔  z_ft(F행 학습)      α=0.458 고정
      B) MLP 슬롯:    z_mlp(8멤버)             ↔  z_ft(전체 학습)     α=0.75 고정

    그리고 §7.103의 마지막 교훈대로 **draw 분포**를 낸다. 단일 실현으로 판정하지
    않는다 (TabPFN이 +4.56 → draw 4개 평균 +1.28±1.23이었다).

실행:
    .venv-gpu/Scripts/python.exe -u src/ftx.py            # A만
    FTX_PART_B=1 .venv-gpu/Scripts/python.exe -u src/ftx.py   # A + B (B는 120만행 학습)
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "6")
sys.path.insert(0, str(Path(__file__).parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import warnings

warnings.filterwarnings("ignore")

import numpy as np

CACHE = Path(__file__).parent.parent / "experiments" / "cache"
EXP = Path(__file__).parent.parent / "experiments"
TMP = Path("C:/Users/강원우/AppData/Local/Temp")
WORK = TMP / "ftx_work"
LOG = EXP / "ftx.log"

FOLD = 2024
# draw 분포용. 파트 B 는 시드당 ~70분이라 환경변수로 좁힐 수 있게 둔다.
FT_SEEDS = [int(x) for x in os.environ.get("FT_SEEDS", "42,7,2024").split(",")]
FIT_TIME_LIMIT = int(os.environ.get("FIT_TIME_LIMIT", 2700))
DO_B = os.environ.get("FTX_PART_B", "0") == "1"


def emit(msg=""):
    print(msg)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(msg + "\n")


# ─────────────────────────────────────────────────────────────
# 워커 — fit 하나가 프로세스 하나. §7.103에서 AutoGluon+torch의
# 메모리 누수가 큰 학습을 체계적으로 불리하게 만든 편향을 겪었다.
# ─────────────────────────────────────────────────────────────
def worker(npz_path, seed, out_path):
    """FT-Transformer 를 `pytabkit.FTT_D_Classifier` 로 직접 적합한다.

    AutoGluon 의 `FT_TRANSFORMER` 키는 `autogluon.tabular.models.automm.*` 경로라
    `autogluon.multimodal` 을 요구하는데 설치돼 있지 않다(수 GB · transformers/timm
    을 건드린다). pytabkit 은 §7.33 의 RealMLP 를 잴 때 이미 깔린 패키지고
    Apache-2.0 이며 같은 벤치마크 계열의 FT-Transformer 기본 튜닝판을 제공한다.

    ⚠️ 비대칭 하나를 의도적으로 남긴다 — pytabkit 은 `val_fraction=0.2` 무작위
       내부 분할로 조기종료하고, 대조군인 LightGBM F 전문가는 **시간 기반**
       (뒤 25%) 분할을 쓴다. 즉 도전자가 더 후한 대접을 받는다. 이 방향이면
       **지는 결과가 결정적**이므로 그대로 두고, 이기면 그때 맞춰서 재검증한다.
    """
    import torch
    from pytabkit import FTT_D_Classifier

    import pandas as pd

    d = np.load(npz_path, allow_pickle=True)
    num_cols = [str(c) for c in d["num_cols"]]
    emb_cols = [str(c) for c in d["emb_cols"]]
    ytr = d["ytr"].astype(int)

    def frame(A, C):
        f = pd.DataFrame(A, columns=num_cols)
        for j, c in enumerate(emb_cols):
            # 정수 인덱스를 category dtype 으로 준다 — 그래야 pytabkit 이
            # 연속이 아니라 범주로 분류하고 임베딩을 만든다(MLP 와 같은 취급).
            f[c] = pd.Categorical(C[:, j])
        return f

    dtr = frame(d["A_tr"], d["C_tr"])
    dva = frame(d["A_va"], d["C_va"])

    # ⚠️ FTT_D 기본은 max_epochs=300 · batch_size=256 이다. 120만 행에서는
    #    166초/에폭 → **시드당 13.8시간**이라 그대로는 못 돌린다.
    #
    #    배치를 4096 으로 올려봤다가 되돌렸다 — 더 느렸다. 커널 런치 오버헤드
    #    병목에서 벗어나는 대신 어텐션 비용(배치 x 토큰^2)에 묶여 VRAM 7.9GB 를
    #    다 쓰며 대역폭에 걸린다. 78토큰 x 192차원이면 그렇다.
    #
    #    그래서 **에폭 상한**으로 간다. 근거는 이 모델 자신의 검증곡선이다 —
    #    batch=256 실행에서 valid_loss 가 epoch 6 에 이미 평평했다
    #    (0.6838 / 0.6834(ep9 최저) / 0.6838). 상수 모델이 0.6926 이니 개선폭
    #    0.009 로, §7.13 의 "신호 총량 0.004 nat" 과 정합이다. 300 에폭은 그 뒤
    #    290 에폭을 더 도는 것이다. 25 에폭 = 약 69분/시드.
    big = len(dtr) > 200_000
    kw = dict(max_epochs=25) if big else {}
    clf = FTT_D_Classifier(device="cuda" if torch.cuda.is_available() else "cpu",
                           random_state=int(seed), n_cv=1, verbosity=1, **kw)
    clf.fit(dtr, ytr)

    CH = 50000
    parts = [clf.predict_proba(dva.iloc[i:i + CH])[:, 1]
             for i in range(0, len(dva), CH)]
    p = np.clip(np.concatenate(parts), 1e-6, 1 - 1e-6)
    np.save(out_path, np.log(p / (1 - p)))


def run_fit(npz, seed, tag):
    out = WORK / f"z_{tag}_{seed}.npy"
    if out.exists():
        return np.load(out)
    cmd = [sys.executable, "-u", str(Path(__file__).resolve()),
           "--fit", str(npz), str(seed), str(out)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace",
                           timeout=FIT_TIME_LIMIT + 900)
    except subprocess.TimeoutExpired:
        return None
    if r.returncode != 0 or not out.exists():
        # 마지막 줄을 그냥 집으면 안 된다. 라이브러리가 예외 **뒤에도** 정보성
        # 출력을 찍어서 오늘 세 번 엉뚱한 줄을 원인으로 읽었다
        # ("Columns classified as categorical: []" 등). 예외 줄을 찾아 쓴다.
        lines = [l.rstrip() for l in
                 (r.stderr or "").splitlines() + (r.stdout or "").splitlines()
                 if l.strip()]
        keys = ("Error:", "Exception:", "error:")
        exc = [l for l in lines if any(k in l for k in keys)
               and not l.lstrip().startswith(("File ", "~", "^"))]
        why = exc[-1].strip() if exc else (lines[-1] if lines else "(출력 없음)")
        emit(f"      FAIL {tag} seed={seed}: {why[:180]}")
        return None
    return np.load(out)



# ─────────────────────────────────────────────────────────────
# s35layer.py 에서 그대로 옮겨온 두 함수.
# import 하지 않는 이유: s35layer 는 `from catens import ...` 로 catboost 를
# 끌어오는데 .venv-gpu 에는 catboost 가 없다(제출용 .venv 에만). 우리에게 필요한
# 것은 CatBoost **캐시 예측**뿐이라 모듈은 필요 없다.
# ─────────────────────────────────────────────────────────────
def prep_full(df, fold):
    from config import (CONTAM_AXES, DROP_FEATURES, RECENCY_HALF_LIFE,
                        SMOOTH_K_BATTER, SMOOTH_K_PITCHER, TARGET_COL,
                        USE_PLAYER_IDS)
    from data import recency_weight
    from features import build_features, fit_artifacts

    s = df["season"].to_numpy()
    tr, va = df[s < fold], df[s == fold]
    w = recency_weight(tr.season.to_numpy(), RECENCY_HALF_LIFE,
                       ref_season=int(tr.season.max()))
    art = fit_artifacts(tr, weights=w, k_pitcher=SMOOTH_K_PITCHER,
                        k_batter=SMOOTH_K_BATTER, contam_axes=CONTAM_AXES,
                        center_asof=False)

    def X(d):
        x = build_features(d, art)
        if USE_PLAYER_IDS:
            for c in ("pitcher_id", "batter_id"):
                x[c] = d[c].to_numpy(dtype=np.float32)
        return x.drop(columns=[c for c in DROP_FEATURES if c in x.columns])

    return X, tr, va, X(va), art


def train_fspec(X, tr, X_ref, t0):
    import lightgbm as lgb
    from config import (FSPEC_MAX_ROUNDS, FSPEC_MIN_SEASON, FSPEC_PARAMS,
                        FSPEC_SEEDS, TARGET_COL)

    pth = CACHE / f"s35_fspec_fold{FOLD}.npz"
    if pth.exists():
        d = np.load(pth)
        emit(f"    fspec 캐시 적중 (학습 {int(d['n']):,}행)")
        return d["z"], int(d["n"])
    m = ((tr["game_type"].to_numpy().astype(str) == "F")
         & (tr["season"].to_numpy() >= FSPEC_MIN_SEASON))
    sub = tr[m]
    Xf = X(sub)
    yf = sub[TARGET_COL].to_numpy(dtype=np.float64)
    tt = (sub["season"].to_numpy(dtype=np.int64) * 12
          + sub["game_month"].to_numpy(dtype=np.int64))
    cut = np.quantile(tt, 0.75)
    itr, iva = tt <= cut, tt > cut
    z = np.zeros(len(X_ref))
    for seed in FSPEC_SEEDS:
        prm = dict(FSPEC_PARAMS, seed=seed, bagging_seed=seed,
                   feature_fraction_seed=seed, data_random_seed=seed)
        b0 = lgb.train(prm, lgb.Dataset(Xf[itr], yf[itr]),
                       num_boost_round=FSPEC_MAX_ROUNDS,
                       valid_sets=[lgb.Dataset(Xf[iva], yf[iva])],
                       callbacks=[lgb.early_stopping(100, verbose=False)])
        nr = max(50, b0.best_iteration)
        bst = lgb.train(prm, lgb.Dataset(Xf, yf), num_boost_round=nr)
        z += bst.predict(X_ref, raw_score=True)
        emit(f"    F전문가 seed={seed} rounds={nr} ({time.time()-t0:.0f}s)")
        del bst
    z /= len(FSPEC_SEEDS)
    np.savez_compressed(pth, z=z, n=int(m.sum()))
    return z, int(m.sum())



def encode_pair(Xtr, Xva, art):
    """배포 MLP와 **완전히 같은** 인코딩을 적용한다.

    pytabkit 의 FT-Transformer 는 연속 컬럼의 NaN 을 못 받는다
    (`NaN values in continuous columns are currently not allowed!`).
    우리 피처는 `asof_*` 가 시즌 초·신인에서 대량 결측이라(CLAUDE.md §3.1)
    반드시 채워야 하는데, **임의로 채우면 아키텍처가 아니라 전처리를 재게 된다.**

    배포 MLP 는 이미 이 문제를 도메인 방식으로 풀고 있다 — 원본 rate 의 결측을
    같은 통계의 `_sm`(스무딩) 컬럼으로, 그것도 없으면 학습 prior 로 채운 뒤
    학습 mu/sd 로 표준화한다. `EMB_COLS` 7개는 임베딩으로 따로 간다.
    그 경로(`mlp.fit_encoder` + `mlp_infer.encode_apply`)를 그대로 부른다.
    """
    import mlp
    from mlp_infer import EMB_COLS, encode_apply

    enc = mlp.fit_encoder(Xtr, art)
    A_tr, C_tr = encode_apply(Xtr, enc)
    A_va, C_va = encode_apply(Xva, enc)
    assert np.isfinite(A_tr).all() and np.isfinite(A_va).all(), "NaN이 남았다"
    return (np.asarray(A_tr, np.float32), np.asarray(C_tr, np.int64),
            np.asarray(A_va, np.float32), np.asarray(C_va, np.int64),
            list(enc["num_cols"]), list(EMB_COLS))


# ─────────────────────────────────────────────────────────────
def main():
    import deploy29 as D
    from config import CAT_ALPHA, FSPEC_ALPHA, MLP_ALPHA
    from validation import bss, shift_to_prior

    t0 = time.time()
    WORK.mkdir(parents=True, exist_ok=True)
    emit("=" * 78)
    emit(f"FT-Transformer — 배포 등가 층 (fold {FOLD})")
    emit(f"  MLP_ALPHA={MLP_ALPHA}  FSPEC_ALPHA={FSPEC_ALPHA}  CAT_ALPHA={CAT_ALPHA}")
    emit(f"  파트 B(MLP 슬롯) 실행: {DO_B}")

    df, s, y = D.load_df()
    st = D.state(df, s, y, FOLD)
    with np.load(CACHE / f"fold{FOLD}_{D.KEY}.npz") as d0:
        z_mlp = d0["z_mlp"]
    z_lgb = np.load(CACHE / f"initdep_fold{FOLD}_ctr0.npz")["z"]
    mm, yv, dep = st["mm"], st["yv"], st["dep"]
    Xa = np.zeros((len(z_lgb), st["Xax"].shape[1]))
    Xa[mm] = st["Xax"]
    ax = Xa[mm] @ dep
    ybar = yv.mean()

    def sc(zb):
        return bss(yv, shift_to_prior(1 / (1 + np.exp(-(zb[mm] + ax))), ybar))

    # ── 앵커 1: s29 층 ────────────────────────────────────────
    z_s29 = (1 - MLP_ALPHA) * z_lgb + MLP_ALPHA * z_mlp
    base29 = sc(z_s29)
    assert abs(base29 - D.EXPECT_BASE[FOLD]) < 0.3, f"s29 앵커 실패 {base29:.1f}"
    emit(f"\n  앵커① s29 층 {base29:.1f}  (기대 {D.EXPECT_BASE[FOLD]:.1f}) ✅")

    X, tr, va, X_ref, art = prep_full(df, FOLD)
    z_fs, n_fs = train_fspec(X, tr, X_ref, t0)
    is_f = va["game_type"].to_numpy().astype(str) == "F"

    def add_fspec(zb, zf, a=FSPEC_ALPHA):
        z = zb.copy()
        z[is_f] = (1 - a) * zb[is_f] + a * zf[is_f]
        return z

    base35 = sc(add_fspec(z_s29, z_fs))
    emit(f"  앵커② s35 등가 층 {base35:.1f}  ({base35-base29:+.2f}; "
         f"s35layer.py 실측 930.0) {'✅' if abs(base35-930.0) < 1.0 else '⚠️'}")

    # CatBoost 12멤버 (§7.141 캐시) — s37 층
    cat_files = sorted(CACHE.glob(f"catcpu_fold{FOLD}_*.npz"))
    assert len(cat_files) == 12, f"CatBoost 캐시 {len(cat_files)}개 (12 기대)"
    z_cat = np.mean([np.load(f)["z"] for f in cat_files], axis=0)

    def add_cat(zb, a=CAT_ALPHA):
        return (1 - a) * zb + a * z_cat

    base37 = sc(add_cat(add_fspec(z_s29, z_fs)))
    emit(f"  앵커③ s37 층 {base37:.1f}  ({base37-base35:+.2f}; "
         f"s35layer.py 실측 +2.47) {'✅' if abs((base37-base35)-2.47) < 0.6 else '⚠️'}")
    emit(f"  검증 {int(mm.sum()):,}행 · F {int(is_f.sum()):,} ({is_f.mean():.1%})"
         f" · fspec 학습 {n_fs:,}행   ({time.time()-t0:.0f}s)")

    # ── 파트 A — fspec 슬롯 교체 ──────────────────────────────
    emit("")
    emit("=== A) fspec 슬롯: LightGBM 3시드  ↔  FT-Transformer   (α=%.3f 고정) ==="
         % FSPEC_ALPHA)
    from config import FSPEC_MIN_SEASON, TARGET_COL
    mF = ((tr["game_type"].to_numpy().astype(str) == "F")
          & (tr["season"].to_numpy() >= FSPEC_MIN_SEASON))
    sub = tr[mF]
    Xf = X(sub)
    npzA = WORK / "slotA.npz"
    if not npzA.exists():
        A_tr, C_tr, A_va, C_va, ncol, ecol = encode_pair(Xf, X_ref, art)
        np.savez(npzA, A_tr=A_tr, C_tr=C_tr, A_va=A_va, C_va=C_va,
                 ytr=sub[TARGET_COL].to_numpy(),
                 num_cols=np.array(ncol, dtype=object),
                 emb_cols=np.array(ecol, dtype=object))
    emit(f"  학습 {Xf.shape} · 추론 {X_ref.shape}   ({time.time()-t0:.0f}s)")

    dA = []
    for seed in FT_SEEDS:
        zft = run_fit(npzA, seed, "A")
        if zft is None:
            continue
        s37_ft = sc(add_cat(add_fspec(z_s29, zft)))
        dA.append(s37_ft - base29)
        emit(f"    seed {seed:<5} s37층 {s37_ft:.1f}   Δ(vs s29) {s37_ft-base29:+.2f}"
             f"    현행 {base37-base29:+.2f}   차 {s37_ft-base37:+.2f}"
             f"   ({time.time()-t0:.0f}s)")
    if dA:
        a = np.array(dA)
        se = a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else float("nan")
        emit(f"    draw {len(a)}개  평균 {a.mean():+.2f}  SE {se:.2f}"
             f"   현행({base37-base29:+.2f}) 대비 {a.mean()-(base37-base29):+.2f}")

    # ── 파트 B — MLP 슬롯 교체 ────────────────────────────────
    if DO_B:
        emit("")
        emit("=== B) MLP 슬롯: MLP 8멤버  ↔  FT-Transformer   (α=%.2f 고정) ==="
             % MLP_ALPHA)
        npzB = WORK / "slotB.npz"
        if not npzB.exists():
            Xt = X(tr)
            A_tr, C_tr, A_va, C_va, ncol, ecol = encode_pair(Xt, X_ref, art)
            np.savez(npzB, A_tr=A_tr, C_tr=C_tr, A_va=A_va, C_va=C_va,
                     ytr=tr[TARGET_COL].to_numpy(),
                     num_cols=np.array(ncol, dtype=object),
                     emb_cols=np.array(ecol, dtype=object))
            del Xt, A_tr, C_tr, A_va, C_va
        emit(f"  학습 {len(tr):,}행 · 추론 {len(X_ref):,}행   ({time.time()-t0:.0f}s)")

        dB = []
        for seed in FT_SEEDS:
            zft = run_fit(npzB, seed, "B")
            if zft is None:
                continue
            z_b = (1 - MLP_ALPHA) * z_lgb + MLP_ALPHA * zft
            s37_b = sc(add_cat(add_fspec(z_b, z_fs)))
            dB.append(s37_b - base37)
            emit(f"    seed {seed:<5} s37층 {s37_b:.1f}   현행 {base37:.1f}"
                 f"   차 {s37_b-base37:+.2f}   ({time.time()-t0:.0f}s)")
        if dB:
            b = np.array(dB)
            se = b.std(ddof=1) / np.sqrt(len(b)) if len(b) > 1 else float("nan")
            emit(f"    draw {len(b)}개  평균 {b.mean():+.2f}  SE {se:.2f}")

    emit("")
    emit(f"완료 ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--fit":
        worker(sys.argv[2], int(sys.argv[3]), sys.argv[4])
    else:
        main()
