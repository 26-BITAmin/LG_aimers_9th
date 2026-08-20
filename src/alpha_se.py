"""쌍체 표준오차 — **임의의 두 예측 쌍**에 대해 잰다.

원래는 α 비교 전용이었다(§7.17). 일반화한 계기는 §7.56이다 —
α 쌍에서 잰 `±1.16`을 s15/s16(**트리 재학습이 낀 쌍**)의 유의성 판정에
그대로 인용했다. 그 차용은 검증된 적이 없다.

    d_i = (p_B,i − y_i)² − (p_A,i − y_i)² = (p_B,i − p_A,i)(p_B,i + p_A,i − 2 y_i)
                                            └── 이 인수가 쌍마다 다르다 ──┘
    SE(mean d) = std(d_i) / √n          BSS 환산: SE / BS_ref × 1e5

**쌍체 SE는 대회의 상수가 아니라 쌍의 성질이다.** 두 예측이 붙어 있으면 작고
떨어져 있으면 크다. 그래서 이 스크립트는 어떤 모드로 돌리든 `std(logit 차)`를
항상 같이 찍고, 기준 쌍(§7.17) 대비 몇 배인지로 **빌려쓰기 가부**를 판정한다.

⚠️ `deploy` 소스는 2019~2024 학습 모델을 2024에 돌리므로 **in-sample**이다.
   BSS 수준값은 낙관적이라 의미가 없다. 필요한 건 수준이 아니라 (p_A − p_B)의
   스케일이고 그건 in-sample에서도 대표성이 있다 — 유의성 판정에만 쓴다.
   out-of-sample이 필요하면 `--source foldcache`를 쓴다.

⚠️ 배포는 설정마다 `logit_delta`를 다시 풀어 예측 평균을 prior에 **핀으로 고정**한다.
   기본 경로는 §7.17 재현을 위해 단일 delta를 쓰므로 **배포가 제거할 레벨 차이**가
   SE에 섞인다. `--repin`을 주면 각 팔의 평균을 공통 prior에 맞춘 뒤 다시 잰다.

사용법
    # ① α 프리셋 (§7.17 재현. 인자 없이 돌리면 이것)
    .venv/Scripts/python.exe -u src/alpha_se.py
    .venv/Scripts/python.exe -u src/alpha_se.py alpha --repin

    # ② 홀드아웃 fold에서 (out-of-sample. 캐시 적중이면 즉시)
    .venv/Scripts/python.exe -u src/alpha_se.py alpha --source foldcache --fold 2024

    # ③ 임의의 두 예측 벡터 — 확률 .npy / .csv(control_success 컬럼) 아무거나
    .venv/Scripts/python.exe -u src/alpha_se.py pair A.npy B.npy --y 2024
    .venv/Scripts/python.exe -u src/alpha_se.py pair a.csv b.csv --y y.npy

    # ④ 여러 후보를 한 기준에 대고 줄세우기 (첫 인자가 기준)
    .venv/Scripts/python.exe -u src/alpha_se.py pair base.npy c1.npy c2.npy --y 2024
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "submit"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    # 이 스크립트는 **거절 메시지가 산출물**이다 (로짓 .npy 가드 등). stderr가
    # CP949면 그 메시지만 깨져서 읽히므로 같이 잡는다.
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np

from config import TARGET_COL, TARGET_PRIOR_2025, TRAIN_CSV
from validation import shift_to_prior

N_ROWS = 250_000          # 평가 테스트셋(245,789)과 비슷하게 맞춘다
ALPHAS = (0.30, 0.40, 0.50, 0.60)
EPS = 1e-12

# §7.17에서 실측한 기준 쌍. 빌려쓰기 판정의 대조군이다.
#   "이 SE를 다른 쌍에 써도 되나"는 곧 "그 쌍의 std(logit 차)가 여기 가까운가"다.
REF_PAIR = dict(name="α .30↔.40 (§7.17)", logit_sd=0.0059, se_bss=1.16)


# ── 판정 기준 ────────────────────────────────────────────────────────────
# ⚠️ **여기가 판단 지점이다.** 아래 1.5는 임시값이고, 이 저장소가 조건 ①②③④를
#    정한 것과 같은 성격의 선택이다 — 근거를 붙여 확정해야 한다.
#      · 느슨하면(예: 3.0) §7.56 같은 차용이 다시 통과한다
#      · 빡빡하면(예: 1.2) 사실상 매번 재측정이라 게이트가 무의미해진다
#    SE는 std(logit 차)에 **대략** 비례하지만 정확히 비례하지는 않는다
#    (두 번째 인수 (p_A + p_B − 2y)도 쌍마다 다르다) — 그래서 비율이 1에 가까울
#    때만 안전하다.
BORROW_MAX_RATIO = 1.5


def borrow_verdict(logit_sd, ref=REF_PAIR, max_ratio=BORROW_MAX_RATIO):
    """이 쌍의 SE를 기준 쌍의 SE로 **대신할 수 있는가**.

    반환: (가부: bool, 비율: float, 사람이 읽을 한 줄: str)
    """
    ratio = logit_sd / ref["logit_sd"] if ref["logit_sd"] else float("inf")
    ok = (1.0 / max_ratio) <= ratio <= max_ratio
    if ok:
        msg = (f"기준 쌍과 같은 대역 (×{ratio:.2f}) — {ref['name']}의 "
               f"SE {ref['se_bss']:.2f}를 대용해도 된다")
    else:
        side = "넓다" if ratio > 1 else "좁다"
        msg = (f"기준 쌍보다 {ratio:.2f}배 {side} — {ref['name']}의 SE를 "
               f"빌려 쓰면 안 된다. 이 표의 SE를 쓸 것")
    return ok, ratio, msg


# ── 핵심 프리미티브 ──────────────────────────────────────────────────────
def _logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def paired_se(p_a, p_b, y, bs_ref=None):
    """기준 A → 후보 B의 쌍체 Brier 차이와 그 표준오차. 반환 단위는 BSS.

    부호 규약은 원본과 같다 — **ΔBSS 양수 = B가 낫다.**

    쌍체인 것이 핵심이다. 같은 행에서 두 예측을 비교하므로 "이 투구가 어려운가"가
    빠져나가고 (p_B − p_A)만 남는다. 두 제출이 상관 0.99+로 붙어 있으면 SE가
    100배 가까이 작아진다 — 그래서 소수점 이하 차이를 읽을 수 있다.
    """
    p_a = np.asarray(p_a, dtype=np.float64)
    p_b = np.asarray(p_b, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if not (len(p_a) == len(p_b) == len(y)):
        raise ValueError(f"길이 불일치: A {len(p_a):,} / B {len(p_b):,} / y {len(y):,}")

    if bs_ref is None:
        bs_ref = y.mean() * (1.0 - y.mean())

    d = (p_b - y) ** 2 - (p_a - y) ** 2
    se = d.std(ddof=1) / np.sqrt(len(d))
    dz = _logit(p_b) - _logit(p_a)
    return dict(
        n=len(d),
        dbrier=float(d.mean()),
        dbss=float(-d.mean() / bs_ref * 1e5),
        se_bss=float(se / bs_ref * 1e5),
        t=float(abs(d.mean()) / se) if se > 0 else float("inf"),
        logit_sd=float(dz.std()),
        prob_sd=float((p_b - p_a).std()),
        corr=float(np.corrcoef(p_a, p_b)[0, 1]),
    )


HEAD = (f"{'비교':<22s} {'ΔBSS':>9s} {'SE(BSS)':>9s} {'t':>6s} "
        f"{'std(Δlogit)':>12s} {'corr':>7s}")


def report(rows, note=""):
    """rows: [(이름, paired_se 결과), ...]"""
    print(f"\n{HEAD}")
    print("-" * 72)
    for name, r in rows:
        print(f"{name:<22s} {r['dbss']:>+9.2f} {r['se_bss']:>9.2f} {r['t']:>6.1f} "
              f"{r['logit_sd']:>12.4f} {r['corr']:>7.4f}")
    if note:
        print(f"\n  ※ {note}")

    print("\n  빌려쓰기 판정 (기준 = §7.17 α .30↔.40, std(Δlogit) 0.0059 / SE 1.16)")
    for name, r in rows:
        ok, _, msg = borrow_verdict(r["logit_sd"])
        print(f"    {'○' if ok else '×'} {name:<20s} {msg}")


# ── 예측 벡터 조달 ───────────────────────────────────────────────────────
def load_probs(path):
    """확률 벡터를 읽는다. `.npy` 또는 제출 형식 `.csv`.

    ⚠️ 로짓을 저장해둔 `.npy`를 그대로 넣는 사고를 막는다 — [0,1] 밖이면 거절한다.
       조용히 시그모이드를 씌우면 "무슨 단위였나"가 기록에서 사라진다.
    """
    p = Path(path)
    if p.suffix == ".npy":
        v = np.load(p).astype(np.float64).ravel()
    elif p.suffix == ".csv":
        import pandas as pd
        d = pd.read_csv(p)
        col = TARGET_COL if TARGET_COL in d.columns else d.columns[-1]
        v = d[col].to_numpy(dtype=np.float64)
    else:
        raise ValueError(f"모르는 형식: {p.suffix} (.npy 또는 .csv)")
    if v.min() < 0.0 or v.max() > 1.0:
        raise ValueError(
            f"{p.name}: 값이 [0,1] 밖이다 (min {v.min():.4g} max {v.max():.4g}). "
            f"로짓이면 시그모이드를 씌워 저장할 것 — 여기서 추측하지 않는다.")
    return v


def load_y(spec, n=None):
    """`--y`: `.npy` 경로이거나 시즌 정수(그 시즌 train 행의 라벨)."""
    s = str(spec)
    if s.isdigit():
        from data import load_csv
        df = load_csv(TRAIN_CSV)
        y = df.loc[df.season == int(s), TARGET_COL].to_numpy(dtype=np.float64)
        if n is not None and len(y) > n:
            y = y[:n]              # 예측 쪽도 앞에서 잘랐다고 가정 (deploy 소스 규약)
        return y
    return np.load(s).astype(np.float64).ravel()


def source_deploy(season=2024, n_rows=N_ROWS):
    """배포 모델을 그 시즌에 돌려 z_lgb / z_mlp / y / delta 를 만든다.

    §7.17이 쓴 경로 그대로다. **in-sample**임에 주의 (docstring 상단 경고).
    """
    from data import load_csv
    from features import build_features
    from predict import Predictor

    t0 = time.time()
    df = load_csv(TRAIN_CSV)
    df = df[df.season == season]
    if len(df) > n_rows:
        df = df.iloc[:n_rows]
    y = df[TARGET_COL].to_numpy(dtype=np.float64)
    print(f"표본 {len(df):,}행 ({season}, 배포모델 in-sample)  ({time.time()-t0:.0f}s)")

    P = Predictor(model_dir=str(Path(__file__).parent.parent / "submit" / "model"))

    # predict()는 블렌드된 확률만 돌려주므로 두 성분을 여기서 따로 만든다.
    # 배포 파일(predict.py)은 검증을 통과한 상태라 건드리지 않는다.
    X = build_features(df, P.art)
    if P.art.get("use_player_ids"):
        for c in ("pitcher_id", "batter_id"):
            X[c] = df[c].to_numpy(dtype=np.float32)
    X = X[P.cols]

    z_lgb = np.zeros(len(X))
    for bst in P.boosters:
        z_lgb += bst.predict(X, raw_score=True)
    z_lgb /= len(P.boosters)

    z_mlp = np.zeros(len(X))
    for m in P.mlps:
        z_mlp += m.logit(X)
    z_mlp /= len(P.mlps)
    print(f"예측 완료  lgb {len(P.boosters)}개 / mlp {len(P.mlps)}개"
          f"  ({time.time()-t0:.0f}s)")
    return z_lgb, z_mlp, y, P.delta


def source_foldcache(fold=2024):
    """홀드아웃 fold 예측 (**out-of-sample**).

    ⚠️ 캐시 미스면 여기서 fold 학습이 통째로 돈다 (20분+). 지문이 config에 걸려
       있으므로 설정을 바꿨다면 미스가 정상이다.
    """
    import foldcache
    z_lgb, z_mlp, y, _ = foldcache.load([fold])[fold]
    print(f"표본 {len(y):,}행 (fold {fold}, 홀드아웃)  지문 {foldcache._key()}")
    return z_lgb, z_mlp, y, 0.0


# ── 모드 ─────────────────────────────────────────────────────────────────
def _blend(z_lgb, z_mlp, a, delta, repin, target):
    z = (1.0 - a) * z_lgb + a * z_mlp + delta
    p = 1.0 / (1.0 + np.exp(-z))
    # 배포는 설정마다 logit_delta를 다시 풀어 평균을 prior에 고정한다.
    # 그걸 흉내내지 않으면 배포가 제거할 레벨 차이가 SE에 섞인다.
    return shift_to_prior(p, target) if repin else p


def run_alpha(z_lgb, z_mlp, y, delta, alphas, repin, target):
    print(f"\nstd(z_lgb)={z_lgb.std():.4f}  std(z_mlp)={z_mlp.std():.4f}"
          f"  corr={np.corrcoef(z_lgb, z_mlp)[0,1]:.4f}")
    print(f"std(z_mlp − z_lgb) = {(z_mlp - z_lgb).std():.4f}")

    base = _blend(z_lgb, z_mlp, alphas[0], delta, repin, target)
    rows = [(f"α {alphas[0]:.2f} → {a:.2f}",
             paired_se(base, _blend(z_lgb, z_mlp, a, delta, repin, target), y))
            for a in alphas[1:]]
    report(rows, note=("각 팔의 평균을 prior에 재고정했다 (--repin) — 배포 형태"
                       if repin else
                       "단일 delta 경로(§7.17 재현). 배포는 팔마다 delta를 다시 "
                       "풀므로 여기 SE에는 레벨 차이가 섞여 있다 → --repin 비교"))


def run_pair(paths, y, repin, target):
    vecs = [(Path(p).stem, load_probs(p)) for p in paths]
    base_name, base = vecs[0]
    if repin:
        vecs = [(n, shift_to_prior(v, target)) for n, v in vecs]
        base = vecs[0][1]
    print(f"기준 = {base_name}   n={len(base):,}")
    rows = [(f"{base_name} → {n}", paired_se(base, v, y)) for n, v in vecs[1:]]
    report(rows, note=("각 팔의 평균을 prior에 재고정했다 (--repin)" if repin else
                       "레벨 차이를 포함한 원단위 비교. 배포 형태를 보려면 --repin"))


def main():
    ap = argparse.ArgumentParser(description="쌍체 표준오차 — 임의의 두 예측 쌍")
    ap.add_argument("mode", nargs="?", default="alpha", choices=("alpha", "pair"))
    ap.add_argument("paths", nargs="*", help="pair 모드: 기준 먼저, 그다음 후보들")
    ap.add_argument("--y", help="pair 모드: 라벨 .npy 경로 또는 시즌(예: 2024)")
    ap.add_argument("--source", default="deploy", choices=("deploy", "foldcache"),
                    help="alpha 모드의 z 출처 (기본 deploy = in-sample)")
    ap.add_argument("--fold", default=2024, help="foldcache 소스의 fold")
    ap.add_argument("--season", type=int, default=2024, help="deploy 소스의 시즌")
    ap.add_argument("--alphas", type=float, nargs="+", default=list(ALPHAS))
    ap.add_argument("--repin", action="store_true",
                    help="각 팔의 예측 평균을 prior에 재고정한다 (배포 형태)")
    # ⚠️ 기본값은 **배포 형태**(2025 목표 prior)다. 홀드아웃 fold에 그대로 쓰면
    #    그 fold의 실제 평균(2024는 .486)과 어긋나 ΔBSS 수준값이 눌린다. 팔끼리는
    #    같은 목표로 고정되므로 비교 자체는 성립하고 SE도 거의 안 변하지만,
    #    fold의 수준값을 읽고 싶으면 `--prior <그 fold의 y 평균>`을 준다.
    ap.add_argument("--prior", type=float, default=TARGET_PRIOR_2025)
    a = ap.parse_args()

    t0 = time.time()
    if a.mode == "pair":
        if len(a.paths) < 2 or not a.y:
            ap.error("pair 모드는 예측 2개 이상과 --y 가 필요하다")
        vec0 = load_probs(a.paths[0])
        run_pair(a.paths, load_y(a.y, n=len(vec0)), a.repin, a.prior)
    else:
        fold = int(a.fold) if str(a.fold).isdigit() else a.fold
        src = (source_foldcache(fold) if a.source == "foldcache"
               else source_deploy(a.season))
        run_alpha(*src, tuple(a.alphas), a.repin, a.prior)

    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
