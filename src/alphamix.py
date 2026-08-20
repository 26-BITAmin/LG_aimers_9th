"""블렌드 가중 α를 **행에 따라** 바꾼다 — mixture-of-experts 형태의 국소 모델 선택.

가설 (도메인 + 문헌)
--------------------
배포는 `z = (1−α)·z_lgb + α·z_mlp`, α=0.40 **전역 상수**다. 두 학습기의
상대적 강점이 행마다 다르다면 상수는 최적이 아니다.

  트리는 학습 분포를 축으로 쪼개고 잎마다 상수를 박는다 → **데이터가 두꺼운 곳**에서 강하다.
  매끄러운 함수는 외삽/보간이 완만하다 → **데이터가 얇은 곳**에서 덜 무너진다.
  (Grinsztajn et al. NeurIPS 2022가 정리한 트리 vs MLP의 귀납 편향 차이.
   이 저장소의 §7.21 가설 "매끄러운 함수는 드리프트에 강하다"와 같은 축이다.)

  → **α는 `asof_pitcher_n`이 작을수록 커야 한다.**

그리고 MLP 풀 안에서:
  `emb4`는 투수/타자 ID 임베딩을 쓴다. **train에 없던 투수**는 그 임베딩이
  학습된 적이 없다 → `noid`가 그 구간에서 상대적으로 나아야 한다.
  → **noid 비중도 미지 투수 구간에서 커야 한다.**

왜 이건 §7.21·§7.22가 닫은 질문이 아닌가
----------------------------------------
그쪽은 **α의 수준**을 물었고 CV의 분해능(±1)이 실제 효과(0.1당 6.7)보다 나빠서
식별이 안 됐다. 여기서 묻는 것은 **버킷 간 차이** α*(얇음) − α*(두꺼움)이다.
수준 성분이 상쇄되므로 같은 데이터에서 검정력이 훨씬 높다.
[[report-effect-size-with-paired-se]] 와 같은 논리다.

s10과도 다르다. s10은 noid 비중의 **전역** 이동이었고 −4.97을 맞았다.
여기서는 전역 비중을 건드리지 않고 **구간별로만** 다르게 준다 (가중 평균은 유지).

채택 기준 — 자기 fold에서 α*를 고르고 자기 fold에서 재면 당연히 이긴다.
**한 fold에서 고른 α 프로파일을 다른 fold에 적용**해서 이겨야 한다.
(§7.22 · `eraclust.py`가 era 축을 죽인 바로 그 검정)

대조군
------
  `game_dayofweek` 버킷 — 어느 학습기가 나은지가 요일에 따라 달라질 이유가 없다.
  여기서도 α*가 흔들리면 그건 신호가 아니라 버킷별 노이즈다. **필수 대조군.**

실행:  .venv/Scripts/python.exe -u src/alphamix.py 2>&1 | tee experiments/alphamix.log
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np

import foldcache
from config import (CTX_AXES, CTX_K, MLP_ALPHA, MLP_VARIANTS, TARGET_COL,
                    TRAIN_CSV)
from ctxsplit import build_delta, lookup
from data import load_csv
from validation import bss, shift_to_prior

FOLDS = [2023, 2024]
K = CTX_K
ALPHAS = np.round(np.arange(0.10, 0.86, 0.05), 2)
N_BOOT = 300

# asof_pitcher_n 버킷. 경계는 §7.34 F의 `asof_n_bucket`과 맞춘다 (새 노브를 안 만든다)
N_EDGES = [0, 200, 1000, 3000, 8000, 20000, 10 ** 9]


def bucketize(v, edges):
    return np.clip(np.searchsorted(edges, v, side="right") - 1, 0, len(edges) - 2)


def deployed_z0(fold, df, s, y, pid, bh, c4, cached):
    """배포 상태의 로짓 — 단, 블렌드 **직전**의 두 성분을 따로 돌려준다."""
    tr, va = s < fold, s == fold
    z_lgb, z_mlp, yv, mem = cached[fold]
    adj = np.zeros_like(z_lgb)
    for name, beta in CTX_AXES:
        code, m = (bh, 8) if name == "hand" else (c4, 4)
        u, d, _ = build_delta(pid[tr], code[tr], y[tr], m, K)
        drow, _ = lookup(u, d, pid[va], code[va], m)
        adj = adj + beta * drow
    return z_lgb, z_mlp, adj, yv, mem


def score_alpha(z_lgb, z_mlp, adj, yv, a):
    z = (1 - a) * z_lgb + a * z_mlp + adj
    return bss(yv, shift_to_prior(1.0 / (1.0 + np.exp(-z)), yv.mean()))


def main():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    y = df[TARGET_COL].to_numpy(dtype=np.float64)
    pid = df["pitcher_id"].to_numpy(dtype=np.int64)
    an = df["asof_pitcher_n"].to_numpy(dtype=np.float64)
    dow = df["game_dayofweek"].to_numpy(dtype=np.int64)
    bh = df["batter_hand"].to_numpy(dtype=np.int64)
    b = df["balls_before"].to_numpy(dtype=np.int64)
    st = df["strikes_before"].to_numpy(dtype=np.int64)
    c4 = (b >= 3).astype(np.int64) + 2 * (st >= 2).astype(np.int64)

    cached = foldcache.load(FOLDS, verbose=False)
    nv_emb = MLP_VARIANTS["emb4"][1]
    print(f"배포 α={MLP_ALPHA}   MLP 멤버 emb4 {nv_emb} + noid "
          f"{MLP_VARIANTS['noid'][1]}\n")

    prof = {}          # (버킷기준, fold) → α* 배열
    state = {}

    for fold in FOLDS:
        tr, va = s < fold, s == fold
        z_lgb, z_mlp, adj, yv, mem = deployed_z0(fold, df, s, y, pid, bh, c4, cached)
        base = score_alpha(z_lgb, z_mlp, adj, yv, MLP_ALPHA)

        seen = np.isin(pid[va], np.unique(pid[tr]))
        nb = bucketize(an[va], N_EDGES)
        state[fold] = dict(z_lgb=z_lgb, z_mlp=z_mlp, adj=adj, yv=yv, mem=mem,
                           nb=nb, seen=seen, base=base, pid=pid[va], dow=dow[va])

        print("=" * 100)
        print(f"fold {fold}   valid {va.sum():,}   기준 BSS(배포 α) {base:.1f}"
              f"   미지 투수 행 {(~seen).mean():.1%}")
        print("=" * 100)

        for label, code, names in (
            ("asof_n", nb, [f"{lo:,}~" for lo in N_EDGES[:-1]]),
            ("미지투수", (~seen).astype(int), ["기지", "미지"]),
            ("요일(대조)", dow[va], [f"dow{i}" for i in range(7)]),
        ):
            print(f"\n  [{label}] 버킷별 최적 α  (전역 α*도 같이)")
            print(f"      {'버킷':<12}{'행수':>10}{'비중':>7}{'α*':>7}"
                  f"{'α*에서 이득':>12}{'α=0.40 대비 곡선 폭':>20}")
            stars = []
            for cval in range(int(code.max()) + 1):
                m = code == cval
                if m.sum() < 2000:
                    stars.append(np.nan)
                    continue
                zl, zm, ad, yy = z_lgb[m], z_mlp[m], adj[m], yv[m]
                # 버킷 안에서만 재려면 그 버킷의 평균으로 shift 해야 공정하다
                cur = np.array([bss(yy, shift_to_prior(
                    1.0 / (1.0 + np.exp(-((1 - a) * zl + a * zm + ad))), yy.mean()))
                    for a in ALPHAS])
                i = int(np.argmax(cur))
                b0 = float(cur[np.argmin(np.abs(ALPHAS - MLP_ALPHA))])
                stars.append(float(ALPHAS[i]))
                nm = names[cval] if cval < len(names) else str(cval)
                print(f"      {nm:<12}{m.sum():>10,}{m.mean():>7.1%}"
                      f"{ALPHAS[i]:>7.2f}{cur[i] - b0:>12.2f}"
                      f"{cur.max() - cur.min():>20.1f}")
            prof[(label, fold)] = np.array(stars)

    # ── 버킷 프로파일이 fold 간 일치하는가 ────────────────────────
    print("\n" + "=" * 100)
    print("α* 프로파일의 fold 간 일치 — 신호라면 같은 방향으로 기울어야 한다")
    print("=" * 100)
    for label in ("asof_n", "미지투수", "요일(대조)"):
        a, bb = prof[(label, FOLDS[0])], prof[(label, FOLDS[1])]
        ok = ~(np.isnan(a) | np.isnan(bb))
        c = float(np.corrcoef(a[ok], bb[ok])[0, 1]) if ok.sum() > 2 else float("nan")
        print(f"  {label:<12} f2023 " + " ".join(f"{v:.2f}" if v == v else "  - " for v in a))
        print(f"  {'':<12} f2024 " + " ".join(f"{v:.2f}" if v == v else "  - " for v in bb)
              + f"    corr {c:+.3f}   산포 {np.nanstd(a):.3f}/{np.nanstd(bb):.3f}")

    # ── 교차 fold 적용 — 이것만이 채택 근거다 ─────────────────────
    print("\n" + "=" * 100)
    print("교차 fold 적용 — 한 fold에서 고른 α 프로파일을 **다른 fold**에 적용")
    print("(자기 fold에서 고르고 자기 fold에서 재는 것은 근거가 아니다)")
    print("=" * 100)
    rng = np.random.default_rng(42)
    for label, key in (("asof_n", "nb"), ("미지투수", "seen"), ("요일(대조)", "dow")):
        for src, dst in ((FOLDS[0], FOLDS[1]), (FOLDS[1], FOLDS[0])):
            stars = prof[(label, src)]
            d = state[dst]
            code = (d["nb"] if key == "nb" else
                    (~d["seen"]).astype(int) if key == "seen" else d["dow"])
            a_row = np.full(len(code), MLP_ALPHA)
            for cval, av in enumerate(stars):
                if av == av:
                    a_row[code == cval] = av
            z = (1 - a_row) * d["z_lgb"] + a_row * d["z_mlp"] + d["adj"]
            g = bss(d["yv"], shift_to_prior(1.0 / (1.0 + np.exp(-z)), d["yv"].mean())) \
                - d["base"]

            # 투수 블록 부트스트랩
            ug, inv = np.unique(d["pid"], return_inverse=True)
            order = np.argsort(inv, kind="stable")
            starts = np.searchsorted(inv[order], np.arange(len(ug)))
            idx_by_g = [order[a_:b_] for a_, b_ in
                        zip(starts, np.append(starts[1:], len(order)))]
            pnew = 1.0 / (1.0 + np.exp(-z))
            pold = 1.0 / (1.0 + np.exp(-((1 - MLP_ALPHA) * d["z_lgb"]
                                         + MLP_ALPHA * d["z_mlp"] + d["adj"])))
            boot = np.empty(N_BOOT)
            for t_ in range(N_BOOT):
                pick = rng.integers(0, len(ug), len(ug))
                ii = np.concatenate([idx_by_g[q] for q in pick])
                yb = d["yv"][ii]
                boot[t_] = (bss(yb, shift_to_prior(pnew[ii], yb.mean()))
                            - bss(yb, shift_to_prior(pold[ii], yb.mean())))
            lo, hi = np.percentile(boot, [2.5, 97.5])
            print(f"  {label:<12} {src} → {dst}   이득 {g:+7.2f}"
                  f"   95%CI [{lo:+6.2f}, {hi:+6.2f}]  {'0 배제' if lo > 0 else '0 포함'}")


if __name__ == "__main__":
    main()
