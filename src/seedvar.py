"""(b) 분산 장부 — 시드 노이즈가 지금 BSS 를 몇 점 갉아먹는가 (§7.161). fit 0회.

멤버 예측을 `p_{f,i}(r) = p̄_f(r) + η_i(r)`, 시드 노이즈 η 가 멤버 간 독립·평균 0,
행별 분산 `s_f²(r)` 로 놓으면 `N_f` 개 평균의 Brier 기여는

    E[BS] = BS(p̄_∞) + Σ_f w_f² · E_r[s_f²] / N_f

이고 확장 이득이 닫힌형이다. **자유 외삽이 아니라 이론이 함수형을 고정하는 경우**라
§7.158 의 학습곡선과 성격이 다르다.

### ⚠️ 정정 두 개 — 그대로 쓰면 이득이 과대평가된다

**① 변형 간 분산은 축소되지 않는다.** LGBM 20 = 4변형×5시드, CatBoost 12 = 4변형×3시드,
MLP 8 = 2변형(emb4 5 · noid 3). **전 멤버에 걸쳐 `s_f²` 를 재면** 변형 간 체계적 차이가
섞이는데, 그것은 시드를 늘려도 안 줄어든다 (고정된 V점 평균이지 iid 노이즈가 아니다).
→ **변형 내부·시드 간** 분산만 축소 가능하다:

    Var(z̄_f) = Σ_v W_v² · s_v² / n_v        (W_v = 변형 v 의 family 내 가중, Σ W_v = 1)

저장소 교훈: "편향이 다른 것들의 평균은 분산 축소가 아니다 — 하이퍼파라미터 평균은
그 축 위의 보간일 뿐."

**② 이 파이프라인은 로짓 공간에서 평균한다.** 멤버 평균이 `z` 에서 일어나므로
확률공간 Brier 로 옮기려면 국소 미분을 물려야 한다:

    Var(p) ≈ σ'(r)² · Var(z̄),   σ' = p(1−p)

### 블렌드 가중 (로짓 공간, R 행)

    w_lgb = (1−α_mlp)(1−α_cat) = 0.25 × 0.90 = 0.225
    w_mlp = α_mlp(1−α_cat)     = 0.75 × 0.90 = 0.675
    w_cat = α_cat              = 0.10

### 출력

family × 변형별로 **현재 손실**(= N→∞ 이득의 상한)과 **한 시드 추가의 한계 이득**
`W_v² s_v² / n_v(n_v+1)` 을 낸다. 컴퓨트를 어디에 쓸지가 표에서 바로 나온다.

⚠️ 멤버별 캐시가 있는 것만 잰다 — CatBoost(양 fold) · MLP(fold2023).
**LGBM 은 평균만 캐시돼 있어 12 fit 이 필요하다.**

실행:  .venv/Scripts/python.exe -u src/seedvar.py 2>&1 | tee experiments/seedvar.log
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

import numpy as np

import deploy29 as D
from calslope import build_layers
from catens import SEEDS, VARIANTS
from config import CAT_ALPHA, MLP_ALPHA
from predict import PROB_CENTER, PROB_SCALE
from validation import shift_to_prior

CACHE = Path(__file__).parent.parent / "experiments" / "cache"
W_LGB = (1 - MLP_ALPHA) * (1 - CAT_ALPHA)
W_MLP = MLP_ALPHA * (1 - CAT_ALPHA)
W_CAT = CAT_ALPHA


def load_family(fold):
    """(family, 변형, family내 가중 W_v, [멤버 z 배열]) 목록. 캐시에 있는 것만."""
    out = []
    for v in VARIANTS:                                   # CatBoost 4변형 × 3시드
        zs = []
        for sd in SEEDS:
            f = CACHE / f"catcpu_fold{fold}_{v}_{sd}.npz"
            if f.exists():
                zs.append(np.load(f)["z"])
        if len(zs) >= 2:
            out.append(("CatBoost", v, W_CAT, 1.0 / len(VARIANTS), zs))
    # MLP — mem_{variant}_f{fold}_{key}_{seed}.npz
    for v, Wv in (("emb4", 0.875), ("noid", 0.125)):     # §7.120 `_vw` noid 몫 0.125
        zs = []
        for f in sorted(CACHE.glob(f"mem_{v}_f{fold}_*.npz")):
            zs.append(np.load(f)["z"] if "z" in np.load(f) else None)
        zs = [z for z in zs if z is not None]
        if len(zs) >= 2:
            out.append(("MLP", v, W_MLP, Wv, zs))
    return out


def main():
    df, s, y = D.load_df()
    print(f"블렌드 로짓 가중  LGBM {W_LGB:.3f} · MLP {W_MLP:.3f} · CatBoost {W_CAT:.3f}\n")
    for fold in D.FOLDS:
        st = D.state(df, s, y, fold)
        yv, mm = st["yv"], st["mm"]
        ax = st["Xax"] @ st["dep"]
        _, zb = build_layers(df, s, fold, mm)[-1]
        pb = shift_to_prior(1 / (1 + np.exp(-(zb + ax))), yv.mean())
        p = shift_to_prior(PROB_CENTER + PROB_SCALE * (pb - PROB_CENTER), yv.mean())
        dpdz2 = (p * (1 - p)) ** 2                       # σ'(r)²
        UNC = yv.mean() * (1 - yv.mean())
        K = 1e5 / UNC

        fam = load_family(fold)
        if not fam:
            print(f"fold{fold}: 멤버 캐시 없음 — 건너뜀\n")
            continue
        print(f"{'='*98}\nfold{fold}   {len(yv):,}행   UNC {UNC:.5f}\n{'='*98}")
        print(f"  {'family':<10s}{'변형':<12s}{'n_v':>5s}{'W_f':>7s}{'W_v':>7s}"
              f"{'sd(z) 시드간':>13s}{'현재 손실':>11s}{'+1시드 이득':>12s}")
        tot = {}
        for f_, v, Wf, Wv, zs in fam:
            Z = np.stack([z[mm] for z in zs])            # (n_v, 행)
            n_v = Z.shape[0]
            s2 = Z.var(axis=0, ddof=1)                   # 행별 시드 간 분산
            base = Wf ** 2 * Wv ** 2 * float(np.mean(dpdz2 * s2))
            cur = base / n_v * K                          # 지금 이 변형이 내는 손실
            marg = base / (n_v * (n_v + 1)) * K           # 시드 1개 추가 시 이득
            tot[f_] = tot.get(f_, 0.0) + cur
            print(f"  {f_:<10s}{v:<12s}{n_v:>5d}{Wf:>7.3f}{Wv:>7.3f}"
                  f"{float(np.sqrt(s2.mean())):>13.4f}{cur:>11.3f}{marg:>12.3f}")
        print(f"\n  family 합계 (= N→∞ 이득의 상한):")
        for f_, v_ in tot.items():
            print(f"    {f_:<10s} {v_:+8.3f} BSS")
        print(f"    {'측정된 합':<10s} {sum(tot.values()):+8.3f} BSS")
        print(f"    ⚠️ LGBM(w={W_LGB:.3f}, 20멤버)은 멤버별 캐시가 없어 미측정\n")


if __name__ == "__main__":
    main()
