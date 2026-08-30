"""Trackman 재구축 9단계 — **재구축 컬럼을 챔피언 층에 올린다** (남은 마지막 칸)

왜 이 칸이 남았나
    `tmarm`/`tmeval` (다른 세션, 오늘 13:26) 이 챔피언 등가 32멤버 층에서
    trackman 을 이미 쟀다. 앵커 807.68/925.75 재현, 학습기 배정 격자, δ·γ 재적합,
    레벨 3종, 상황 게이트 — 전부. 결과는 전 학습기 tm 이 −6.35 / −12.85.

    ★ 단 그 실험이 쓴 것은 **§7.9 가 배포했던 원래 10컬럼**(`load_table("release")`)이다.
      이 세션이 원본에서 다시 만든 변형도, 5·6단계가 유일하게 살려낸 컬럼도 안 들어갔다.

    ★★ 그리고 그 격자가 방향을 특정했다:
          lgb 만 tm   f2023 **+1.48** / f2024 **−0.08**   (잡음 0.3)  ← 중립, 무해
          mlp 만 tm   f2023 −5.63 / f2024 −10.29          (잡음 5~7)  ← 손해는 전부 여기
      [[augmentation-targets-learners-not-just-rows]]

    → 남은 질문 하나: **재구축 집중 컬럼을 MLP 빼고 LGBM+CatBoost 에만 주면?**

컬럼 (사전 등록 — 5·6단계 결과로 정했고 3단계 BSS 를 보고 고르지 않았다)
    tm_hb_std     4시즌 기울기 t 2.41 · Q 3.22(과분산 아님) · 부호 ++++
    tm_ext_drift  t 2.00 · Q 2.56 · ++++   (시즌간 표류 — 기존 작업이 준 적 없는 신호)
    tm_n          표본 크기
    ⚠️ 둘 다 4시즌 BSS 는 전부 양수가 아니었다 (hb_std +4.3/+4.6/+10.8/**−4.8**).
       그 감쇠(2024 β 가 앞 세 시즌의 15%)가 이 실험의 사전 예측이다 — **음수 예상**.
       그래도 돌리는 이유: 사용자 지시가 "기각 함부로 하지 말 것" 이고,
       사후 β 선형 가산과 트리 재학습은 다른 질문이다.

남의 파일은 안 건드린다
    `tmarm.tm_table` 만 몽키패치한다. `tmarm.py` 원본은 그대로 둔다 (병렬 세션 공존).
    MLP 는 학습하지 않는다 — 격자가 손해 출처를 MLP 로 특정했고 비용이 절반이다.
    MLP 칸은 base 를 그대로 쓴다 (= 배포 MLP).

실행:  .venv/Scripts/python.exe -u src/tmr9focus.py 2>&1 | tee experiments/tmr9focus.log
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
import pandas as pd

from config import ROOT

TMR_DIR = ROOT / "experiments" / "tmr"
FEATSET = "focus"

import tmarm                                                    # noqa: E402
import tmeval                                                   # noqa: E402
import deploy29 as D                                            # noqa: E402

FOCUS_COLS = ["tm_hb_std", "tm_n", "tm_ext_drift"]
_orig_tm_table = tmarm.tm_table
_orig_assert = tmarm._assert_split


def _patched(featset):
    if featset == FEATSET:
        t = pd.read_csv(TMR_DIR / "tm_focus.csv")
        print(f"  [focus] {t.shape[0]:,}행 × "
              f"{len([c for c in t.columns if c not in ('pitcher_id','season')])}컬럼")
        return t
    return _orig_tm_table(featset)


def _patched_assert(featset, Xva, fold):
    """`tmarm._assert_split` 과 **같은 의도**. 컬럼 수 기대치만 집중 세트에 맞춘다.
    [[assert-the-cache-key-actually-split]] — 끄지 않는다, 눈금만 바꾼다."""
    if featset != FEATSET:
        return _orig_assert(featset, Xva, fold)
    tmc = [c for c in Xva.columns if c.startswith(("tm_", "tmfb_"))]
    assert "tm_matched" in Xva.columns, "tm_matched 없음 — art['tm'] 미주입"
    assert len(tmc) == len(FOCUS_COLS) + 1, f"트랙맨 컬럼 {len(tmc)}개 (기대 4)"
    hit = float(Xva["tm_matched"].mean())
    fp = {c: float(np.nansum(Xva[c].to_numpy(dtype=np.float64))) for c in FOCUS_COLS}
    print(f"  fold{fold}/{featset}  X {Xva.shape}  tm 컬럼 {len(tmc)}  적중 {hit:.1%}")
    assert hit > 0.4, f"적중률이 너무 낮다 {hit:.1%}"
    return dict(hit=hit, fp=fp)


tmarm.tm_table = _patched
tmarm._assert_split = _patched_assert


def main():
    t0 = time.time()
    print("Trackman 재구축 9단계 — 재구축 집중 컬럼 × 챔피언 등가 32멤버 층")
    print(f"지문 {D.KEY}   featset={FEATSET}   MLP 는 base 고정 (재학습 안 함)\n")
    df, s, y = D.load_df()

    for fold in tmarm.FOLDS:
        print(f"\n{'='*84}\n■ fold {fold}   ({time.time()-t0:.0f}s)\n{'='*84}")
        tmarm.train_lgb(df, fold, FEATSET, t0)
        tmarm.train_cat(df, fold, FEATSET, t0)

    # ── 평가 — tmeval 기계 그대로 ─────────────────────────────
    print(f"\n{'='*84}\n앵커 확인 — base 조합이 §7.194 B1 을 재현하는가 (807.68 / 925.75)\n{'='*84}")
    ST, MEM = {}, {}
    for fold in tmarm.FOLDS:
        ST[fold] = D.state(df, s, y, fold)
        MEM[fold] = {}
        for fs in ("base", FEATSET):
            MEM[fold][fs] = {}
            for fam in ("lgb", "cat", "mlp"):
                use = "base" if fam == "mlp" else fs
                MEM[fold][fs][fam] = tmeval.load_family(use, fam, fold)

    def score(fold, pick):
        st = ST[fold]
        M = {fam: MEM[fold][pick[fam]][fam] for fam in ("lgb", "mlp", "cat")}
        z = tmeval.blend_z(M)
        from validation import bss, shift_to_prior
        p = shift_to_prior(1 / (1 + np.exp(-(z[st["mask"]] + st["ax"]))), st["ybar"])
        return p, bss(st["yv"], p)

    base_pick = dict(lgb="base", mlp="base", cat="base")
    anchors = {}
    for fold in tmarm.FOLDS:
        _p, b = score(fold, base_pick)
        anchors[fold] = b
        ref = 807.68 if fold == 2023 else 925.75
        ok = "✅" if abs(b - ref) < 0.05 else "❌"
        print(f"  fold{fold}  B1(base) = {b:.2f}   기대 {ref}   {ok}")

    print(f"\n{'='*84}")
    print("학습기 배정 격자 — 재구축 집중 컬럼 (MLP 는 항상 base)")
    print(f"{'='*84}")
    print(f"  {'lgb':>6}{'cat':>6}{'f2023':>10}{'Δ':>9}{'f2024':>10}{'Δ':>9}"
          f"{'f2024 95% CI':>22}{'P(Δ>0)':>9}")
    rng = np.random.default_rng(42)
    base_p = {f: score(f, base_pick)[0] for f in tmarm.FOLDS}
    for lg in ("base", FEATSET):
        for ct in ("base", FEATSET):
            pick = dict(lgb=lg, mlp="base", cat=ct)
            row, ci, pgt = [], "", ""
            for fold in tmarm.FOLDS:
                p, b = score(fold, pick)
                row.append((b, b - anchors[fold]))
                if fold == 2024 and not (lg == "base" and ct == "base"):
                    lo, hi, pr = tmeval.boot_delta(ST[fold], p, base_p[fold], rng)
                    ci, pgt = f"[{lo:+7.2f},{hi:+7.2f}]", f"{pr:.3f}"
            print(f"  {lg[:4]:>6}{ct[:4]:>6}{row[0][0]:>10.2f}{row[0][1]:>+9.2f}"
                  f"{row[1][0]:>10.2f}{row[1][1]:>+9.2f}{ci:>22}{pgt:>9}")

    print(f"\n  ⚠️ 사전 예측은 **음수**였다 (5단계: hb_std 4시즌 BSS +4.3/+4.6/+10.8/−4.8,")
    print( "     2024 기울기가 앞 세 시즌의 15%). 결과가 그와 다르면 그것이 정보다.")
    print( "  ⚠️ Δ 는 채택 근거가 아니다 — 다음 관문은 LB 다. §7.9 홀드아웃 +13.3 → LB −13.6.")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
