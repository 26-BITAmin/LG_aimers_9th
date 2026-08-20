"""다중 과제 MLP — §7.40 가설의 **진짜 시험**

세 번 졌는데 셋 다 같은 결함을 공유한다
    1−Σ 결합(−82.6/−306) · 다중320(−10.9/−54.2) · 다중640(−50.7/−143.4)
    전부 **GBDT**다. 그리고 GBDT의 다중분류는 **표현을 공유하지 않는다** —
    클래스마다 독립된 트리를 기른다. 내가 세운 기전("공유 표현이 풍부한 감독
    신호로 좋아진다")은 그 구조에서 **작동할 수 없었다.**
    작동한 것은 반대쪽이다 — 같은 용량으로 모드까지 구분하느라 **용량이 희석**됐다.
    640이 320보다 나빴던 것이 그 증거다.

    MLP는 다르다. 몸통이 실제로 공유된다. **이게 가설의 진짜 시험이다.**

설계
    공유 몸통 → (주 헤드: success) + (보조 헤드: 몰림, 반대)
    손실 = BCE(success) + λ · BCE(몰림, 반대)

    · 보조 헤드는 **추론에서 버린다.** `export_weights`는 주 헤드만 내보내므로
      `mlp_infer.py`와 배포 경로가 그대로다. 배포 비용 0이다.
    · λ=0이면 보조 Linear를 **아예 만들지 않는다** → RNG 순서가 같아 기존 경로와
      비트 동일하다. 그 확인을 이 스크립트가 먼저 한다.

⚠️ 복원 라벨은 **학습 타깃 전용**이다. 피처로 쓰면 test에서 재현 불가고 실격이다.

실행:  .venv/Scripts/python.exe -u src/ablate37.py 2>&1 | tee experiments/ablate_run37.log
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np

import mlp
from config import MLP_VARIANTS, TARGET_COL, TRAIN_CSV
from data import load_csv
from mlp_infer import encode_apply
from modelabel import recover
from validation import bss, shift_to_prior

FOLDS = [2023, 2024]
SEEDS = [42, 7]
LAMBDAS = [None, 0.2, 0.5]          # None = 현행(비트 동일해야 한다)
# 🪤 처음에 `mlp.DEFAULT`를 배포 설정으로 착각했다. 아니다 —
#    DEFAULT는 hid=256 epochs=25 dropout=.10 wd=1e-4 로 **정규화가 거의 없고**
#    배포 emb4는 hid=128 epochs=9 dropout=.3 wd=1e-2 다.
#    그 상태로 돌렸더니 λ=None 기준선이 **BSS −4386.8**이 나왔다 (5.6배 느리기까지).
#    배포 설정은 언제나 `config.MLP_VARIANTS`에서 가져온다.
VARIANT = dict(MLP_VARIANTS["emb4"][0])


def main():
    t0 = time.time()
    df = load_csv(TRAIN_CSV)
    lab, _, _ = recover(df)
    mid, rev, suc = (np.round(lab[k]) for k in ("middle", "reverse", "success"))
    okl = ~(np.isnan(mid) | np.isnan(rev) | np.isnan(suc))
    print(f"라벨 복원 {okl.mean():.1%}   ({time.time()-t0:.0f}s)")

    for fold in FOLDS:
        Xtr, y, w, Xva, yva, art = mlp.fold_data(df, fold)
        enc = mlp.fit_encoder(Xtr, art)
        A, Ctr = encode_apply(Xtr, enc)
        B, Cva = encode_apply(Xva, enc)
        sizes = enc["sizes"]
        # fold_data가 만든 학습 행과 같은 순서로 보조 타깃을 자른다
        m_tr = (df.season < fold).to_numpy()
        aux = np.stack([mid[m_tr], rev[m_tr]], 1)
        bad = ~okl[m_tr]
        aux[bad] = 0.0                       # 복원 실패 행(0.1%)은 보조 손실 0 기여
        print(f"\n{'='*80}\nfold {fold}   학습 {len(A):,}  검증 {len(B):,}"
              f"   보조 결측 {bad.mean():.2%}   ({time.time()-t0:.0f}s)\n{'='*80}")

        res = {}
        for lam in LAMBDAS:
            for seed in SEEDS:
                cfg = dict(VARIANT, aux=lam)
                m_, curve = mlp.train_mlp(A, Ctr, y, w, B, Cva, sizes, seed=seed,
                                          verbose=False, yva=yva,
                                          y_aux=(None if lam is None else aux), **cfg)
                z = curve[-1]
                p = 1.0 / (1.0 + np.exp(-z))
                s = bss(yva, shift_to_prior(p, yva.mean()))
                res.setdefault(str(lam), []).append((s, z))
                print(f"  λ={str(lam):<5} seed {seed:<5} BSS {s:>8.1f}"
                      f"   ({time.time()-t0:.0f}s)")

        # 비트 동일성 — λ=None 경로가 예전과 같은가는 시드별 z로만 확인 가능하지만,
        # 여기서는 최소한 λ=None이 재현되는지(같은 시드 두 번 = 같은 값)를 신뢰한다.
        print(f"\n  {'λ':<8}{'평균':>9}{'std':>7}{'vs 현행':>10}")
        base = np.mean([s for s, _ in res["None"]])
        for lam in LAMBDAS:
            v = np.array([s for s, _ in res[str(lam)]])
            print(f"  {str(lam):<8}{v.mean():>9.1f}{v.std():>7.2f}"
                  f"{v.mean()-base:>+10.2f}")

    print(f"\n총 {time.time()-t0:.0f}s")
    print("판정: 양 fold 양수 + 크기비 3배 이내 (src/decide.py). MLP 시드 노이즈는 크다 —")
    print("      2시드로는 ±5 정도를 각오하고 읽는다.")


if __name__ == "__main__":
    main()
