"""보조 헤드 확장 — 볼/스트라이크 축을 더하면 이득이 커지는가

배경
    s13이 채택한 보조 헤드는 (몰림, 반대) 2개다 (LB 987.41, +2.66).
    `modelabel`은 같은 증분 방식으로 (볼, 스트라이크)도 99.93% 복원한다.

    `src/auxinfo.py`가 사전 검사를 통과시켰다 — 기존 축 (성공, 몰림, 반대)가
    주어졌을 때 ball은 88.5%, strike는 95.1%의 엔트로피가 **남는다**.
    (대조: 이미 채택된 mid는 success 조건에서 70.2%, rev는 61.3%밖에 안 남았다.
     새 축이 기존 축보다 더 독립적이다.)
    카운트 12셀 프로파일 상관도 갈렸다 — mid–rev는 +0.990으로 사실상 한 축인데
    ball–strike는 −0.322다. 지금 2헤드는 독립 축을 하나쯤 주고 있었던 셈이다.

무엇을 가려야 하는가 — 헤드를 늘리면 **정규화도 같이 는다**
    ablate39b: 라벨을 행 단위로 섞은 `rand`도 fold2023에서 +32.9를 냈다.
    보조 헤드 이득의 상당 부분은 모드 지식이 아니라 정규화다. 헤드 2→4는
    정규화를 확실히 늘리므로, 대조군 없이 이득을 보면 **반드시 속는다.**

    그래서 다섯 구성을 **같은 실행 안에서** 돌린다:
        None    λ=0                          — 바닥
        mr      [몰림, 반대]                  — 배포 중 (s13). **판정의 기준선**
        bs      [볼, 스트라이크]              — 헤드 수가 같다. 축만 갈아끼운 대조
        mrbs    [몰림, 반대, 볼, 스트라이크]   — 후보
        rand4   mrbs를 행 단위로 섞음          — 4헤드의 정규화 성분만

    `bs`가 핵심이다. `mr`와 헤드 수가 같으므로 정규화가 맞춰져 있고,
    차이는 **오직 어느 축을 아는가**다. 여기서 갈리면 정보 논증이 확인된다.

판정
    채택 여부는 `None` 대비가 아니라 **`mr` 대비**로 정한다. 지금 배포된 것이
    `mr`이므로, 갈아탈 이유는 "mrbs − mr"이 조건 ①②③을 넘는 것뿐이다.

    ablate40은 "블렌드는 풀 평균 하나뿐이라 쌍체 SE가 없다"고 적었는데, 그건
    **시드 노이즈** 이야기였다. 같은 행에 대한 두 예측의 쌍체 SE는 언제나 잴 수
    있다 (decide.paired_se). 두 축은 다른 것이고 둘 다 넘어야 한다 — 여기서는
    행 쌍체 SE를 직접 재고, 시드 축은 ablate38의 측정(12.9~14.0)으로 읽는다.

⚠️ GPU 전용. 절대 점수를 CPU 기록과 대조하지 말 것 — 같은 실행 안의 A/B만 읽는다.

실행:  MLP_DEVICE=cuda .venv-gpu/Scripts/python.exe -u src/ablate41.py
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

import ctxfit
import foldcache
import mlp
from config import (CTX_AXES, CTX_K, MLP_ALPHA, MLP_AUX_LAMBDA, MLP_SEEDS,
                    MLP_VARIANT_WEIGHT, MLP_VARIANTS, TARGET_COL, TRAIN_CSV)
from data import load_csv
from decide import decide, paired_se
from features import logit_adjust
from mlp_infer import encode_apply
from modelabel import recover
from validation import bss, shift_to_prior

FOLDS = [2023, 2024]
LAM = float(sys.argv[1]) if len(sys.argv) > 1 else MLP_AUX_LAMBDA

# (이름, 축 목록 또는 None, 행을 섞는가)
CONFIGS = [
    ("None",  None,                                        False),
    ("mr",    ("middle", "reverse"),                       False),
    ("bs",    ("ball", "strike"),                          False),
    ("mrbs",  ("middle", "reverse", "ball", "strike"),      False),
    ("rand4", ("middle", "reverse", "ball", "strike"),      True),
]
BASE = "mr"          # 판정의 기준선 = 지금 배포된 구성


def pool_z(A, Ctr, y, w, B, Cva, sizes, aux):
    """배포와 같은 8멤버 풀(emb4 5시드 + noid 3시드)의 가중 평균 로짓.

    보조 헤드는 **두 변형 모두**에 단다 (MODEL.md §7.41 — `둘 다`가 `emb4만`보다
    양 fold에서 낫다). aux=None이면 λ도 None이라 헤드 자체가 안 만들어진다.
    """
    tot, acc = 0.0, np.zeros(len(B))
    for vname, (vcfg, nseed) in MLP_VARIANTS.items():
        wt = MLP_VARIANT_WEIGHT[vname] / nseed        # 멤버당 가중
        cfg = dict(vcfg, aux=(None if aux is None else LAM))
        for seed in MLP_SEEDS[:nseed]:
            _, curve = mlp.train_mlp(A, Ctr, y, w, B, Cva, sizes, seed=seed,
                                     verbose=False, yva=None, y_aux=aux, **cfg)
            acc += wt * curve[-1]
            tot += wt
    return acc / tot


def main():
    t0 = time.time()
    df = load_csv(TRAIN_CSV)
    season = df["season"].to_numpy()
    lab, _, _ = recover(df)
    rounded = {k: np.round(v) for k, v in lab.items()}
    # 유효 행: 요청될 수 있는 모든 축 + 검증용 success 가 전부 복원된 행
    need = ["success", "middle", "reverse", "ball", "strike"]
    okl = ~np.any([np.isnan(rounded[k]) for k in need], axis=0)

    cached = foldcache.load(FOLDS, verbose=False)
    print(f"라벨 복원 {okl.mean():.1%}   device={mlp._device()}   λ={LAM}"
          f"   풀={MLP_VARIANT_WEIGHT}   ({time.time()-t0:.0f}s)")
    print(f"판정 기준선 = '{BASE}' (지금 배포 중인 구성)")

    store = {}
    for fold in FOLDS:
        Xtr, y, w, Xva, yva, art = mlp.fold_data(df, fold)
        enc = mlp.fit_encoder(Xtr, art)
        A, Ctr = encode_apply(Xtr, enc)
        B, Cva = encode_apply(Xva, enc)
        m_tr = season < fold

        z_lgb = cached[fold][0]
        ctx = ctxfit.fit_ctx(df[m_tr], TARGET_COL, CTX_AXES, CTX_K)
        d_adj, _ = logit_adjust(df[season == fold], {"ctx": ctx})

        # 섞기는 **학습셋 안에서** 한다 — 행 정렬만 깨고 주변 분포는 보존한다.
        # 시드는 ablate39b와 같은 12345 (대조군끼리 비교 가능하게).
        rng = np.random.default_rng(12345)
        perm = rng.permutation(int(m_tr.sum()))

        def make_aux(modes, shuffle):
            if modes is None:
                return None
            a = np.stack([rounded[k][m_tr] for k in modes], 1)
            a[~okl[m_tr]] = 0.0
            a = a.astype(np.float32)
            return a[perm] if shuffle else a

        print(f"\n{'='*84}\nfold {fold}   학습 {len(A):,}   검증 {len(B):,}"
              f"   ({time.time()-t0:.0f}s)\n{'='*84}")
        print(f"  {'설정':<8}{'헤드':>5}{'MLP풀':>9}{'블렌드':>9}"
              f"{'vs None':>10}{f'vs {BASE}':>10}")
        res = {}
        for name, modes, shuffle in CONFIGS:
            aux = make_aux(modes, shuffle)
            zm = pool_z(A, Ctr, y, w, B, Cva, enc["sizes"], aux)
            p_solo = shift_to_prior(1 / (1 + np.exp(-zm)), yva.mean())
            zb = (1 - MLP_ALPHA) * z_lgb + MLP_ALPHA * zm + d_adj
            p_blend = shift_to_prior(1 / (1 + np.exp(-zb)), yva.mean())
            res[name] = {"solo": bss(yva, p_solo), "blend": bss(yva, p_blend),
                         "p": p_blend}
            dn = res[name]["blend"] - res["None"]["blend"] if "None" in res else 0.0
            db = (res[name]["blend"] - res[BASE]["blend"]) if BASE in res else float("nan")
            nh = 0 if modes is None else len(modes)
            print(f"  {name:<8}{nh:>5}{res[name]['solo']:>9.1f}"
                  f"{res[name]['blend']:>9.1f}{dn:>+10.2f}{db:>+10.2f}"
                  f"   ({time.time()-t0:.0f}s)")
        res["_y"] = yva
        store[fold] = res

    # ── 판정 ────────────────────────────────────────────────────────────
    print("\n" + "=" * 84)
    print(f"판정 — 블렌드에서 '{BASE}' 대비 이득 (배포 구성: 8멤버 + LGBM + δ)")
    print("=" * 84)
    print(f"  {'설정':<8}{'f2023':>9}{'±SE':>7}{'f2024':>9}{'±SE':>7}"
          f"{'크기비':>8}   판정")
    for name, modes, _ in CONFIGS:
        if name == BASE:
            continue
        ds, ses = [], []
        for fold in FOLDS:
            r = store[fold]
            d, se = paired_se(r["_y"], r[BASE]["p"], r[name]["p"])
            ds.append(float(d))
            ses.append(float(se))
        a, b = ds
        v, why, _ = decide(name, a, b)
        ratio = max(abs(a), abs(b)) / max(min(abs(a), abs(b)), 1e-9)
        print(f"  {name:<8}{a:>+9.2f}{ses[0]:>7.2f}{b:>+9.2f}{ses[1]:>7.2f}"
              f"{ratio:>8.2f}   {v}  {why.splitlines()[0]}")

    print("\n" + "-" * 84)
    print("정보 대 정규화 — 헤드 수를 맞춘 대조")
    print("-" * 84)
    for fold in FOLDS:
        r = store[fold]
        print(f"  fold{fold}   mr {r['mr']['blend']:>8.1f}   "
              f"bs {r['bs']['blend']:>8.1f}   (bs−mr {r['bs']['blend']-r['mr']['blend']:+.2f})"
              f"      mrbs {r['mrbs']['blend']:>8.1f}   "
              f"rand4 {r['rand4']['blend']:>8.1f}   "
              f"(mrbs−rand4 {r['mrbs']['blend']-r['rand4']['blend']:+.2f})")
    print("\n  · bs−mr : 헤드 수가 같다 → 정규화가 맞춰져 있고 차이는 **축의 정보**뿐이다.")
    print("  · mrbs−rand4 : 4헤드끼리 → 4헤드 정규화를 뺀 **순수 모드 지식**.")
    print("    rand4가 mrbs를 따라오면 늘린 것은 정보가 아니라 잡음 주입이다.")
    print("\n  ⚠️ 위 ±SE는 **행 쌍체**다 (같은 행의 두 예측). 시드 축은 별개이고,")
    print("     ablate38에서 잰 MLP 단독 쌍체 t가 12.9~14.0이었다.")
    print(f"\n총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
