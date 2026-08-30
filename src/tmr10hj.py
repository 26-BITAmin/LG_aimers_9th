"""Trackman 재구축 10단계 — **hj03 파이프라인 그대로** 적용해서 다시 잰다

사용자 지시 (2회)
    *"hj03 거기에 있는 학습이 다 완료된 거 기준으로 하라는 것이 아니라,
      그 파이프라인을 그냥 그대로 적용하라는 말이다"*

지금까지 세 층을 헷갈렸다 — 셋 다 hj03 이 아니다
    ① 내 3~6단계 `foldcache`      LGBM(init_score 없음) + MLP dim4 + CatBoost **없음**
    ② `tmarm`/`tmeval` B1 하네스   LGBM(init_score) + MLP **dim4·콜드패치 없음** + CatBoost 12
                                   ★ 앵커 807.68/925.75 는 `bcoldfold.py` 가 명시하듯
                                     **dim 4 앵커**다 — 챔피언이 아니다
    ③ **이 파일** = hj03 파이프라인 LGBM(init_score) + MLP **D8C(dim 8 + 콜드패치)** + CatBoost 12
                                   B1 변형가중(.875/.125) · W_CAT .10 · 배포 δ 4축

    §7.197 이 확정한 것: `sy02_c4`(LB +7.44) 의 유일한 실질 변경이 **emb4 의
    `pitcher_id` 임베딩 4→8** 이었다. §7.206: hj03 = 그 위에 콜드스타트 패치.
    그러므로 dim 4 로 잰 모든 trackman 숫자는 **챔피언 파이프라인의 답이 아니다.**

부품 (전부 이미 캐시돼 있다 — 재학습은 trackman 팔의 LGBM·CatBoost 뿐)
    LGBM 12    base `initdepm_fold*`          / focus `tmarm_focus_lgb_f*`
    CatBoost12 base `catcpu_fold*`            / focus `tmarm_focus_cat_f*`
    MLP 8      **`champ_D8C_f*_emb4_*` 5개 + `champ_noid_f*` 3개**  ← 양 팔 공통
               (MLP 는 재학습하지 않는다: `tmeval` 격자가 손해 출처를 MLP 로 특정했고
                — lgb만 tm +1.48/−0.08 vs mlp만 tm −5.63/−10.29 —
                trackman 을 MLP 에 주지 않는 것이 이 팔의 설계다)

보고
    · D4 앵커(807.68/925.75) 와 **D8C 앵커**를 나란히 → 층이 실제로 갈렸는지 확인
    · 학습기 배정 격자 (lgb × cat) 각 칸의 f2023 · f2024 BSS 와 Δ
    · 레벨 oracle / frozen 둘 다 (oracle 만 보면 레벨 손해가 지워진다)
    · 투수 클러스터 부트스트랩 95% CI · P(Δ>0)

실행:  .venv/Scripts/python.exe -u src/tmr10hj.py 2>&1 | tee experiments/tmr10hj.log
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

import deploy29 as D
import tmeval
from config import MLP_SEEDS

FOLDS = [2023, 2024]
CACHE = D.CACHE
FEATSET = "focus"
D4_ANCHOR = {2023: 807.68, 2024: 925.75}


def mlp_members(fold, arm="D8C"):
    """hj03 의 MLP 층 — emb4 5멤버(dim 8 + 콜드패치) + noid 3멤버."""
    emb = [np.load(CACHE / f"champ_{arm}_f{fold}_emb4_{s}.npz")["z"] for s in MLP_SEEDS[:5]]
    nd = np.load(CACHE / f"champ_noid_f{fold}.npz")["z"]
    noid = [nd[i] for i in range(nd.shape[0])] if nd.ndim == 2 else [nd]
    assert len(noid) == 3, f"noid 멤버 {len(noid)}개 (기대 3)"
    return np.stack(emb + noid, axis=1).astype(np.float64)


def evaluate(st, M, mode="oracle"):
    """`tmeval.score` / `tmeval.probs` 를 그대로 쓴다 (배포 δ 4축 · clean 마스크)."""
    z = tmeval.blend_z(M)
    b, _pm = tmeval.score(st, z, mode)
    return tmeval.probs(st, z, mode=mode), b


def main():
    t0 = time.time()
    print("Trackman × **hj03 파이프라인** (LGBM+init_score · MLP D8C dim8+cold · CatBoost12)")
    print(f"지문 {D.KEY}   B1 변형가중 .875/.125 · W_CAT {tmeval.W_CAT} · 배포 δ 4축\n")
    df, s, y = D.load_df()

    ST, FAM = {}, {}
    for f in FOLDS:
        ST[f] = D.state(df, s, y, f)
        ST[f]["ybar"] = float(ST[f]["yv"].mean())   # deploy29.state 에 없는 키
        FAM[f] = {}
        for fs in ("base", "tm", FEATSET):
            FAM[f][fs] = {fam: tmeval.load_family(fs, fam, f) for fam in ("lgb", "cat")}
        FAM[f]["mlpD4"] = tmeval.load_family("base", "mlp", f)
        FAM[f]["mlpD8C"] = mlp_members(f, "D8C")

    # ── 앵커 — 층이 실제로 갈렸는가 ────────────────────────────
    print("=" * 92)
    print("앵커 — dim4 B1 vs **D8C(hj03)**. 두 값이 같으면 층이 안 갈린 것이다")
    print("=" * 92)
    base_p, anchors = {}, {}
    for f in FOLDS:
        M4 = dict(lgb=FAM[f]["base"]["lgb"], mlp=FAM[f]["mlpD4"], cat=FAM[f]["base"]["cat"])
        M8 = dict(lgb=FAM[f]["base"]["lgb"], mlp=FAM[f]["mlpD8C"], cat=FAM[f]["base"]["cat"])
        _p4, b4 = evaluate(ST[f], M4)
        p8, b8 = evaluate(ST[f], M8)
        ok = "✅" if abs(b4 - D4_ANCHOR[f]) < 0.05 else "❌"
        print(f"  fold{f}   D4 {b4:>8.2f} (기대 {D4_ANCHOR[f]}) {ok}"
              f"   **D8C {b8:>8.2f}**   차 {b8-b4:+.2f}")
        base_p[f], anchors[f] = p8, b8
        _pf, bf = evaluate(ST[f], M8, "frozen")
        anchors[(f, "frozen")] = bf
        anchors[(f, "ybar_raw")] = float(_pf.mean())

    print("\n" + "=" * 92)
    print("학습기 배정 격자 — hj03 층. s7.9 = §7.9 원래 10컬럼 / focus = 재구축 집중 3컬럼")
    print("=" * 92)
    print(f"  {'lgb':>6}{'cat':>6}{'f2023':>9}{'Δ':>8}{'f2024':>9}{'Δ':>8}"
          f"{'f2024 95% CI':>21}{'P>0':>7}{'frozenΔ24':>11}")
    rng = np.random.default_rng(42)
    for lg in ("base", "tm", FEATSET):
        for ct in ("base", "tm", FEATSET):
            cells, ci, pgt, fz = [], "", "", ""
            for f in FOLDS:
                M = dict(lgb=FAM[f][lg]["lgb"], mlp=FAM[f]["mlpD8C"],
                         cat=FAM[f][ct]["cat"])
                p, b = evaluate(ST[f], M)
                cells.append((b, b - anchors[f]))
                if f == 2024 and not (lg == "base" and ct == "base"):
                    lo, hi, pr = tmeval.boot(ST[f], p, base_p[f], rng)
                    ci, pgt = f"[{lo:+7.2f},{hi:+7.2f}]", f"{pr:.3f}"
                    _pf, bf = evaluate(ST[f], M, "frozen")
                    fz = f"{bf - anchors[(f,'frozen')]:+.2f}"
            tag = lambda x: {"base":"base","tm":"s7.9","focus":"focus"}[x]
            print(f"  {tag(lg):>6}{tag(ct):>6}{cells[0][0]:>9.2f}{cells[0][1]:>+8.2f}"
                  f"{cells[1][0]:>9.2f}{cells[1][1]:>+8.2f}{ci:>21}{pgt:>7}{fz:>11}")

    print("\n" + "=" * 92)
    print("읽는 법")
    print("=" * 92)
    print("  · MLP 를 안 건드리므로 이 격자의 잡음 바닥은 §7.199 의 lgb 0.22/0.40 ·")
    print("    cat 0.23/0.22 다 (MLP 4.8/7.1 이 아니다). Δ ±1 도 의미가 있다.")
    print("  · frozenΔ 가 oracleΔ 와 크게 다르면 trackman 이 **레벨을 통해** 움직인 것 —")
    print("    그 몫은 배포에서 PROB_SHIFT 재보정으로 사라지므로 refinement 로 세면 안 된다.")
    print("  · 사전 예측(5·6단계)은 **음수**였다: hb_std 4시즌 BSS +4.3/+4.6/+10.8/−4.8,")
    print("    2024 기울기가 앞 세 시즌의 15%(= 이 저장소가 3회 기록한 전이 감쇠 c≈0.16).")
    print("  ⚠️ 라우팅(hj03 은 t=−0.25)과 PROB_SHIFT 는 이 하네스에 없다. 홀드아웃 층의 한계다.")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
