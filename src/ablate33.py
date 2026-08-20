"""RealMLP 스케일링 층 — 교육자료 6종에서 **유일하게 남은 미측정 후보**

교육자료 대조 결과
    강의자료(01~06 + 실습 노트북)에 나온 방법을 이 저장소 이력과 대조했다.
    거의 전부가 이미 측정·기각됐다:

        LR/kNN/DT/RF          — 자명하거나 §7.25(rf 모드 −83,000)
        XGBoost / CatBoost    — §5 · §7.14 기각
        PLE / 주기 임베딩      — §7.26 기각 (입력차원 단조 법칙)
        ResNet-MLP (깊이·잔차) — §7.25 기각
        TabPFN / TabICL       — §7.13 기각 (≤10k 레짐 · 추론 예산 · 라이선스)
        LLM 직렬화             — 대회 규정상 원격 API 금지
        SSL(VIME/SCARF/SubTab)— 쓸 수 있는 무라벨 데이터가 없다. test 행 사용은 실격
        어텐션 / 검색(TabR)    — 용량 법칙 + 10분 추론 예산

    남은 하나가 **RealMLP**(Holzmüller et al., NeurIPS 2024, 자료 03 p13-15)다.

왜 이건 다른가 — 네 가지 사망 원인(§7.27) 어디에도 안 걸린다
    스케일링 층은 입력에 학습되는 대각 스케일 `s`를 곱하는 것뿐이다.
    **표현력이 0만큼 는다** — Linear(diag(s)·x)는 그냥 다른 행렬이고, 함수족이
    똑같다. §7.26이 세운 법칙("입력이든 은닉이든 차원을 늘리면 진다")의 적용
    대상이 아니다. 정보를 더하지도, 학습 분포를 바꾸지도, 새 함수족을 들이지도
    않는다.

    바뀌는 것은 **wd의 의미**다. ‖W‖² + ‖s‖² 를 W·diag(s) = M 제약에서 최소화하면

        min ‖W‖² + ‖s‖²  s.t. W·diag(s) = M   =   2·Σ_i ‖M_:,i‖

    즉 ridge가 **입력 피처 단위의 group lasso**로 바뀐다. 배포 MLP의 wd는 1e-2로
    세다 — 전환이 실제로 작동할 조건이다.

우리 실측과 맞물리는 지점
    자료 03 p3이 인용한 Grinsztajn et al.(NeurIPS 2022)의 세 발견 중 두 번째:
    **"무정보 피처는 MLP 계열에 더 크게 해롭다."** 이건 우리가 이미 측정한 사실의
    기전이다 — §7.27에서 `noid`(투수/타자 임베딩을 **뺀** 변형)가 양 fold에서
    `emb4`보다 **셌다**(+69.0 / +5.3). 입력을 빼는 게 이득인 모델이다.
    스케일링 층은 그 조작의 연속판이고, 어느 입력을 줄일지 손실이 고른다.

    그리고 MLP 단독은 fold2024에서 653인데 LGBM은 752.5다. §7.24가 정한 후보
    조건("단독 점수가 오르는가")을 겨냥할 자리가 100점 남아 있다.

배포 비용 0
    diag(s)는 첫 Linear에 접힌다 (`mlp.export_weights`). 임베딩 스칼라는 임베딩
    테이블에 접힌다. `mlp_infer.py`는 한 줄도 안 바뀌고 npz 규약도 그대로다.
    → 채택되면 추론 코드 변경 없이 가중치만 바뀐다.

후보
    scale     수치 입력에 대각 스케일 + 같은 wd    ← RealMLP 그대로
    scalenw   같은데 스케일에만 wd 없음             ← 동역학만 남긴다
    scalewd10 스케일에만 wd 10배                    ← group lasso 축을 실제로 흔든다
    scaleall  scale + 임베딩 블록마다 스칼라 하나   ← soft `noid`

    ⚠️ wd 축을 배수로 연 이유. AdamW의 decoupled decay는 스텝당 (1 − lr·wd)라
    9에폭(약 1,320스텝)에서 배수 1.0이면 s가 **1.6%밖에 안 줄어든다.** 반면 Adam은
    경사를 RMS로 정규화하므로 s는 스텝당 ~lr 만큼 자유롭게 움직인다 (스모크 테스트에서
    2스텝에 std 0.0016 = 2×lr 확인). 즉 `scale` vs `scalenw`의 차이는 기전을 가르기엔
    너무 작다 — 거기서 null이 나와도 "group lasso가 아니다"의 근거가 못 된다.
    그 판정은 `scalewd10`이 한다.

통제
    · 대조군 `emb4ctl`은 **ablate26이 학습한 캐시를 그대로 재사용**한다.
      그래서 `THREADS=4`를 지켜야 한다 (스레드 수가 BLAS 리덕션 순서를 바꿔
      결과를 1e-6에서 흔든다 — ablate26 실측).
    · 셋 다 `emb4` cfg를 물려받고 스케일 축 하나만 바꾼다. 에폭 9 고정,
      마지막 에폭 값을 읽는다 (§7.13 규약: valid argmax 금지).
    · **비트 동일성 검사를 먼저 돌린다.** `mlp.py`를 건드렸으므로 scale=None
      경로가 예전과 같은지 fold2023 seed42로 확인하고, 다르면 즉시 중단한다.
      (다르면 대조군 캐시가 무효라 이 실험 전체가 무의미해진다.)
    · β 스윕에 **대조군도 같은 방식으로 넣는다.** 새 멤버를 풀에 섞으면 품질과
      무관하게 분산 축소 보너스가 붙는다. Δscale − Δctl 이라야 구조 효과다.
      (ablate26은 이 대조가 없었다.)
    · Δ에는 쌍체 SE를 붙인다 (§7.28 규약).

사전 예측 (실행 전에 박아둔다)
    P1  `scale` 단독이 양 fold에서 대조군 이상 (Δ ≥ 0)
        내 예측: **그렇다**  ← 긍정 예측이다. §7.26 이후 내 긍정 예측은 2/2로
                              틀렸으므로 이 항목은 근거로 쓰지 않고 채점만 한다.
    P2  wd 축이 단조다: `scalenw` ≤ `scale` ≤ `scalewd10` (fold2024 단독 기준)
        내 예측: **아니다**. group lasso가 작동해도 최적 강도가 있을 테니 정점이
        가운데일 것이고, `scale`↔`scalenw` 차이는 위 이유로 노이즈에 묻힐 것이다.
        단조가 나오면 wd 전환이 진짜 기전이라는 강한 증거다.
    P3  `scaleall` ≥ `scale`
        내 예측: **아니다**. `noid`는 하드 제거가 이득이었고, wd 하의 소프트
        게이트는 0까지 안 가서 중간값에 머문다.
    P4  조건 ①②③ 통과 후보 수
        내 예측: **0개**. 이 세션의 벽은 아이디어가 아니라 fold2024의 판별력이다
        (§7.27: 남은 노브가 전부 ±3.3 노이즈 대역 안).

    P1이 맞고 P4가 0이면 그것도 결과다 — "단독은 올랐는데 시스템이 안 움직인다"는
    §7.24 기준 자체의 한계를 처음으로 보여주는 사례가 된다.

실행:  .venv/Scripts/python.exe -u src/ablate33.py 2>&1 | tee experiments/ablate_run33.log
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

import numpy as np

import ablate23 as A23
import foldcache
import mlp
from config import (MLP_ALPHA, MLP_SEEDS, MLP_VARIANTS, RECENCY_HALF_LIFE,
                    SMOOTH_K_BATTER, SMOOTH_K_PITCHER, TARGET_COL, TRAIN_CSV)
from data import load_csv, recency_weight
from features import fit_artifacts
from foldcache import CACHE, _key
from mlp_infer import encode_apply
from validation import shift_to_prior

THREADS = 4                # ⚠️ ablate26 캐시를 재사용한다. 바꾸면 대조군이 무효다
mlp.THREADS = THREADS

FOLDS = [2023, 2024]
SEEDS = MLP_SEEDS[:3]
EMB4 = MLP_VARIANTS["emb4"][0]

ARCH = {
    "emb4ctl":  dict(EMB4),                                        # 대조군 (캐시 재사용)
    "scale":     dict(EMB4, scale={"emb": False, "wd": True}),
    "scalenw":   dict(EMB4, scale={"emb": False, "wd": False}),
    "scalewd10": dict(EMB4, scale={"emb": False, "wd": 10.0}),
    "scaleall":  dict(EMB4, scale={"emb": True, "wd": True}),
}

# 0.625 = emb4 몫 전체를 갈아끼우는 지점 (배포 풀은 emb4 .625 / noid .375)
BETAS = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.625, 0.80, 1.0]


def ck(fold, name, seed):
    """ablate26과 **같은 이름 규약** — 그래야 emb4ctl 캐시가 그대로 잡힌다."""
    return CACHE / f"arch_mlp_{name}_f{fold}_{_key()}_{seed}.npz"


def prob(z, y):
    return shift_to_prior(1.0 / (1.0 + np.exp(-z)), y.mean())


def paired(zA, zB, y):
    """BSS(A) − BSS(B)와 쌍체 SE. 같은 행이라 표본 변동이 차분에서 소거된다."""
    pA, pB = prob(zA, y), prob(zB, y)
    c = 100000.0 / (y.mean() * (1 - y.mean()))
    d = (pB - y) ** 2 - (pA - y) ** 2
    return c * d.mean(), c * d.std(ddof=1) / np.sqrt(len(d))


def fold_inputs(df, fold):
    season = df.season.to_numpy()
    tr, va = df[season < fold], df[season == fold]
    w = recency_weight(tr.season.to_numpy(), RECENCY_HALF_LIFE, ref_season=fold - 1)
    art = fit_artifacts(tr, weights=w, k_pitcher=SMOOTH_K_PITCHER,
                        k_batter=SMOOTH_K_BATTER)
    Xtr, Xva = A23.featurize(tr, art), A23.featurize(va, art)
    ytr = tr[TARGET_COL].to_numpy(dtype=np.float64)
    enc = mlp.fit_encoder(Xtr, art)
    An, Ctr = encode_apply(Xtr, enc)
    B, Cva = encode_apply(Xva, enc)
    del tr, va, Xtr, Xva
    gc.collect()
    return dict(ytr=ytr, w=w, An=An, Ctr=Ctr, B=B, Cva=Cva, enc=enc)


def train(D, name, seed):
    """z와 함께 **학습된 스케일 벡터**를 돌려준다 — 기전이 실제로 발동했는지
    상관계수로 추측하지 않고 직접 본다."""
    m, _ = mlp.train_mlp(D["An"], D["Ctr"], D["ytr"], D["w"],
                         D["An"][:512], D["Ctr"][:512], D["enc"]["sizes"],
                         seed=seed, verbose=False, **ARCH[name])
    z = mlp.infer(m, D["B"], D["Cva"])
    sn = None if m.snum is None else m.snum.detach().numpy()
    se = None if m.semb is None else m.semb.detach().numpy()
    del m
    return z, sn, se


def main():
    t0 = time.time()
    CACHE.mkdir(parents=True, exist_ok=True)
    cached = foldcache.load(FOLDS)
    Z = {f: (cached[f][0], cached[f][1], cached[f][2]) for f in FOLDS}

    todo = [(f, n, s) for f in FOLDS for n in ARCH for s in SEEDS]
    # 비트 동일성 검사도 fold2023을 필요로 하므로 캐시가 다 있어도 데이터는 읽는다
    df = load_csv(TRAIN_CSV)

    got, scl, checked = {}, {}, False
    for fold in FOLDS:
        D = None
        for name, s in [(n, s) for (f, n, s) in todo if f == fold]:
            p = ck(fold, name, s)
            if p.exists():
                with np.load(p) as d:
                    got[(fold, name, s)] = d["z"]
                    if "snum" in d.files:
                        scl[(fold, name, s)] = (d["snum"],
                                                d["semb"] if "semb" in d.files else None)
                continue
            if D is None:
                D = fold_inputs(df, fold)
                print(f"\n=== fold {fold} 준비 완료 ({time.time()-t0:.0f}s) ===")
            z, sn, se = train(D, name, s)
            extra = {}
            if sn is not None:
                extra["snum"] = sn
                scl[(fold, name, s)] = (sn, se)
            if se is not None:
                extra["semb"] = se
            np.savez_compressed(p, z=z, **extra)
            got[(fold, name, s)] = z
            print(f"  {name}_{s}  단독 {mlp.score(z, Z[fold][2]):>8.1f}"
                  + ("" if sn is None else
                     f"   s: std {sn.std():.3f} min {sn.min():+.3f} max {sn.max():+.3f}")
                  + f"  ({time.time()-t0:.0f}s)")

        # ── 0. 비트 동일성 — mlp.py를 건드렸다. scale=None 경로가 그대로인가 ──
        if not checked and (fold, "emb4ctl", SEEDS[0]) in got:
            if D is None:
                D = fold_inputs(df, fold)
            zr, _, _ = train(D, "emb4ctl", SEEDS[0])
            d = np.abs(zr - got[(fold, "emb4ctl", SEEDS[0])]).max()
            print(f"\n[비트검사] fold{fold} emb4ctl_{SEEDS[0]} 재학습 vs ablate26 캐시"
                  f"   max|diff| = {d:.3e}   {'일치' if d == 0 else '⚠️ 불일치'}")
            if d != 0:
                raise SystemExit(
                    "scale=None 경로가 달라졌다. 대조군 캐시가 무효이므로 중단한다.\n"
                    "mlp.py의 Net.__init__ 생성 순서(스케일 파라미터는 맨 마지막)를 확인할 것.")
            checked = True

        del D
        gc.collect()
    del df
    gc.collect()

    def avg(fold, name):
        return np.mean([got[(fold, name, s)] for s in SEEDS], axis=0)

    def bl(zl, zm, y, a=MLP_ALPHA):
        return (1 - a) * zl + a * zm

    # ── 1. 단독 (§7.24 기준) ────────────────────────────────────────
    print("\n" + "=" * 100)
    print("1. 단독 성능 — §7.24 기준은 '단독 점수가 오르는가'다 (3시드 평균, Δ는 대조군 대비)")
    print("=" * 100)
    print(f"{'변형':>10} " + " ".join(f"{'fold'+str(f):>9}" for f in FOLDS)
          + " " + " ".join(f"{'Δ'+str(f)+'±SE':>18}" for f in FOLDS)
          + f" {'corr(z_lgb)':>12}")
    for name in ARCH:
        row, dl = [], []
        for f in FOLDS:
            y = Z[f][2]
            row.append(mlp.score(avg(f, name), y))
            if name == "emb4ctl":
                dl.append("        —         ")
            else:
                m_, se = paired(avg(f, name), avg(f, "emb4ctl"), y)
                dl.append(f"{m_:>+9.2f} ± {se:<5.2f} (t{m_/se:>+5.1f})")
        print(f"{name:>10} " + " ".join(f"{v:>9.1f}" for v in row)
              + " " + " ".join(f"{v:>18}" for v in dl)
              + f" {np.corrcoef(avg(2024, name), Z[2024][0])[0,1]:>12.3f}")

    # ── 2. 스케일이 실제로 무엇을 했나 ──────────────────────────────
    print("\n" + "=" * 100)
    print("2. 학습된 스케일 — 기전이 발동했는가 (추측하지 않고 s를 직접 본다)")
    print("=" * 100)
    print("   group lasso가 작동했다면 s가 1에서 흩어지고 일부가 0 근처로 눌린다.")
    print("   s가 전부 1.0 근처면 층이 켜져 있을 뿐 아무 일도 안 한 것이다.")
    print(f"\n{'변형':>10} {'fold':>6} {'std(s)':>8} {'min':>8} {'max':>8}"
          f" {'|s|<0.1':>8} {'|s|<0.3':>8} {'corr(대조군)':>13}")
    for name in list(ARCH)[1:]:
        for f in FOLDS:
            ss = [scl[(f, name, s)][0] for s in SEEDS if (f, name, s) in scl]
            c = np.corrcoef(avg(f, name), avg(f, "emb4ctl"))[0, 1]
            if not ss:
                print(f"{name:>10} {f:>6}  (스케일 미저장 — 캐시가 옛 형식이다)"
                      f"{c:>40.4f}")
                continue
            v = np.concatenate([np.abs(x) for x in ss])
            print(f"{name:>10} {f:>6} {np.mean([x.std() for x in ss]):>8.3f}"
                  f" {min(x.min() for x in ss):>8.3f} {max(x.max() for x in ss):>8.3f}"
                  f" {(v < 0.1).mean():>8.1%} {(v < 0.3).mean():>8.1%} {c:>13.4f}")
    for name in list(ARCH)[1:]:
        e = [scl[(f, name, s)][1] for f in FOLDS for s in SEEDS
             if (f, name, s) in scl and scl[(f, name, s)][1] is not None]
        if e:
            print(f"\n{name:>10}  임베딩 스칼라 (fold2023 seed{SEEDS[0]}, ecols 순): "
                  f"{np.round(e[0], 3)}")

    # ── 3. 시스템 판정 — 풀에 β로 섞는다 ────────────────────────────
    print("\n" + "=" * 100)
    print(f"3. 채택 판정 — MLP 풀에 β로 섞고 트리와 α={MLP_ALPHA}로 블렌드 (기준 = 현행 8멤버 풀)")
    print("=" * 100)
    print("   ⚠️ 대조군 행이 '새 멤버를 넣기만 해도 붙는 분산 축소 보너스'다.")
    print("      구조 효과는 (후보 − 대조군)이지 후보 행 자체가 아니다.")
    print(f"\n{'후보':>10} {'fold':>6} " + " ".join(f"{b:>8.3f}" for b in BETAS))
    dl_tab = {}
    for name in ARCH:
        for f in FOLDS:
            zl, zm, y = Z[f]
            zn, out = avg(f, name), []
            for b in BETAS:
                m_, _ = paired(bl(zl, (1 - b) * zm + b * zn, y), bl(zl, zm, y), y)
                out.append(m_)
            out[0] = 0.0
            dl_tab[(name, f)] = out
            print(f"{name:>10} {f:>6} " + " ".join(f"{v:>+8.2f}" for v in out))

    print(f"\n{'구조 효과 (후보 − 대조군)':>26}")
    print(f"{'후보':>10} {'fold':>6} " + " ".join(f"{b:>8.3f}" for b in BETAS))
    for name in list(ARCH)[1:]:
        for f in FOLDS:
            d = [a - b for a, b in zip(dl_tab[(name, f)], dl_tab[("emb4ctl", f)])]
            print(f"{name:>10} {f:>6} " + " ".join(f"{v:>+8.2f}" for v in d))

    # β=0.625(emb4 몫 전체 교체)에서 쌍체 SE를 붙여 한 번 더 본다
    print(f"\n  β=0.625 (emb4 몫 전체 교체) — 대조군 대비, 쌍체 SE")
    for name in list(ARCH)[1:]:
        cells = []
        for f in FOLDS:
            zl, zm, y = Z[f]
            b = 0.625
            zc = bl(zl, (1 - b) * zm + b * avg(f, name), y)
            zk = bl(zl, (1 - b) * zm + b * avg(f, "emb4ctl"), y)
            m_, se = paired(zc, zk, y)
            cells.append(f"fold{f} {m_:>+7.2f} ± {se:<5.2f} (t{m_/se:>+5.1f})")
        print(f"{name:>10}  " + "   ".join(cells))

    print("\n  채택 조건: ① |Δ| > 2·SE  ② 양 fold 부호 일치  ③ fold간 크기비 ≤ 3배")
    print("  조건 ②는 5전 5중, 조건 ③은 4전 4중이다. 넘기지 않는다.")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
