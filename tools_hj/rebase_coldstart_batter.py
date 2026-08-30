"""`batter_id` 콜드스타트 — `emb1[0]` 을 학습된 임베딩의 평균으로. 재학습 없음.

혜진 §18-7 이 **미측정**으로 남긴 채널이다. 기전은 `pitcher_id`(라운드 18, LB +3.32)와
같고 규모만 작다.

    실측 (챔피언 `hj03_dim8_cold.zip` 의 emb4 5멤버)
        학습된 타자 830명의 중심  |mean(emb1[1:])| = 0.043 ~ 0.107   <- 사실상 원점
        미지 슬롯                 |emb1[0]|        = 0.63 / 0.97 / 1.55 / 1.97 / 2.25
                                  학습 노름 분포의 1.2 ~ 74.1 백분위를 시드마다 배회

    미지 행 비율 (`data/train.csv` 직전 전체 이력 대비 신규 타자)
        2020 11.6% · 2021 8.8% · 2022 11.0% · 2023 9.4% · **2024 9.3%**
        -> 2025 의 구조적 유사물은 2024. UNK = 0.093 채택.
        (pitcher 는 19.9% 였다 — 질량비 0.47)

⚠️ `emb1` 은 dim **4** 다 (`emb0` 은 8). 정체를 더 거칠게 표현하므로 미지 슬롯이
   틀렸을 때의 손해도, 평균으로 회수되는 몫도 pitcher 와 다를 수 있다.
   혜진 §18-2 가 dim8 vs dim12 에서 관측한 것이 정확히 그 축이다 — **차원이 높을수록
   구멍이 커지고 메우기 어려워진다.** dim4 면 반대로 구멍이 작고 회수가 쉬울 수 있다.
   → 방향은 기대하되 **크기는 pitcher 에서 비례 외삽하지 않는다.**

절차는 혜진 `build_coldstart_zip.py` 와 동일하다. 마스킹 프록시가 필요한 이유도 같다 —
2024 프록시에는 미지 타자가 0명이라(학습에 2024 가 들어갔다) 그대로 재면 Δmean = 0 이다.
그래서 타자 일부의 `batter_id` 를 LUT 밖 값(−1)으로 바꿔 실제 test 의 미지 비율을 흉내낸다.

사용
    python tools_hj/rebase_coldstart_batter.py --src hj03_dim8_cold.zip --out hj07_bcold.zip
"""
import argparse
import glob
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from rebase_routing import SHIFT_RE, TRAIN_CSV, predict_in_subprocess, sha

SENTINEL = -1
UNK = 0.093          # 2024 실측 신규 타자 행 비율


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import pandas as pd
    os.makedirs("D:/t1eval", exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="bcold_", dir="D:/t1eval")
    work, orig = os.path.join(tmp, "work"), os.path.join(tmp, "orig")
    with zipfile.ZipFile(a.src) as z:
        names = z.namelist()
        z.extractall(work)
        z.extractall(orig)

    pp = os.path.join(work, "predict.py")
    src0 = open(pp, encoding="utf-8").read()
    m = SHIFT_RE.findall(src0)
    assert len(m) == 1
    shift0 = float(m[0][1])
    print(f"소스 {os.path.basename(a.src)}   PROB_SHIFT {shift0:.9f}")

    # ── 마스킹 프록시: 타자 단위로 골라 목표 행 비율을 맞춘다 ──────────
    df = pd.read_csv(TRAIN_CSV, encoding="utf-8-sig")
    df = df[df["season"] == 2024].drop(columns=["control_success"]).reset_index(drop=True)
    rng = np.random.RandomState(a.seed)
    bids = rng.permutation(np.sort(df["batter_id"].unique()))
    cnt = df["batter_id"].value_counts()
    take, tot, goal = [], 0, UNK * len(df)
    for b in bids:
        if tot >= goal:
            break
        take.append(b)
        tot += int(cnt[b])
    mask = df["batter_id"].isin(take).to_numpy()
    df.loc[mask, "batter_id"] = SENTINEL
    pq = os.path.join(tmp, "proxy.pkl")
    df.to_pickle(pq)
    print(f"프록시 2024 {len(df):,}행 · 미지 처리 {mask.sum():,} "
          f"({mask.mean():.2%}, 타자 {len(take)}명)")

    print("추론 ① 패치 전 …")
    before = predict_in_subprocess(work, pq, os.path.join(tmp, "b.npy"))

    # ── 패치: emb1[0] ← 학습된 임베딩의 평균 ─────────────────────
    files = sorted(glob.glob(os.path.join(work, "model", "mlp_emb4_*.npz")))
    assert files, "mlp_emb4_*.npz 가 없다"
    for f in files:
        d = dict(np.load(f))
        e = d["emb1"]
        assert [str(c) for c in d["ecols"]][1] == "batter_id", f"{f}: emb1 이 batter_id 가 아니다"
        assert not np.allclose(e[0], e[1:].mean(0)), f"{f}: 이미 패치돼 있다"
        d["emb1"] = np.vstack([e[1:].mean(0, keepdims=True), e[1:]]).astype(e.dtype)
        np.savez(f, **d)
    print(f"패치 {len(files)}개 · emb1[0] ← mean(emb1[1:])")

    print("추론 ② 패치 후 (레벨 미보정) …")
    mid = predict_in_subprocess(work, pq, os.path.join(tmp, "m.npy"))
    d_ = mid - before
    dm = float(d_.mean())
    print(f"  미지 행 Δp 평균 {float(d_[mask].mean()):+.6g} · 평균|Δp| {np.abs(d_[mask]).mean():.6g}")
    print(f"  알려진 행 max|Δp| {float(np.abs(d_[~mask]).max()):.3g}   <- 0 이어야 한다")
    assert np.abs(d_[~mask]).max() == 0.0, "패치가 알려진 타자 행까지 건드렸다"
    assert np.abs(d_[mask]).max() > 0.0, "패치가 예측에 반영되지 않았다"
    shift = shift0 - dm
    print(f"  전체 Δmean {dm:+.6g}   PROB_SHIFT {shift0:.9f} -> {shift:.9f}")
    open(pp, "w", encoding="utf-8").write(
        SHIFT_RE.sub(lambda mm: f"{mm.group(1)}{shift:.9f}", src0))

    print("추론 ③ 재보정 검증 …")
    after = predict_in_subprocess(work, pq, os.path.join(tmp, "a.npy"))
    r = float((after - before).mean())
    print(f"  패치 후 Δmean {r:+.3g}  {'✅' if abs(r) < 1e-9 else '⚠️'}")
    assert abs(r) < 1e-7

    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            fp = os.path.join(work, n)
            if os.path.isfile(fp):
                z.write(fp, n)
    diff = sorted(n for n in names if os.path.isfile(os.path.join(work, n))
                  and open(os.path.join(work, n), "rb").read()
                  != open(os.path.join(orig, n), "rb").read())
    exp = sorted(["predict.py"] + [f"model/{os.path.basename(f)}" for f in files])
    print(f"\n변경 격리  {len(diff)}/{len(names)}: {diff}")
    assert diff == exp, f"기대 {exp}"
    assert len(os.path.basename(a.out)) <= 30
    print(f"출력   {a.out}\n  SHA-256 {sha(a.out)}")
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
