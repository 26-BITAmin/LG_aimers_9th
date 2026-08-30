"""CatBoost α 의 R행 한정 적용(§7.192 `s48_catf0`)을 임의의 제출 zip 위에 재적용한다.

혜진 `tools/rebase_routing.py` 와 **같은 골격**이다 — 프록시·`PROB_SHIFT` 재보정·
변경 격리 검사를 그대로 import 해서 쓴다. 바꾸는 것은 패치 대상 한 줄뿐이다.

    before   z = (1.0 - α) * z + α * zc                 (전 행)
    after    z[is_r] = (1.0 - α) * z[is_r] + α * zc[is_r]   (R 행만, α_F = 0)

근거 사슬 (§7.192)
    §7.145④ 구 스택 fold-정직    cat 기여 F 초과분 −0.826 [−1.647, −0.029]
    catrf45  s45 등가층           F 초과분 −3.844 [−5.947, −1.922], 사전 게이트 3/3
    ⚠️ 단 s45 층은 F 기저의 in-sample F전문가(37.5%)로 α_F=0 쪽에 편향돼 있다.
       **크기는 못 믿고 방향만 읽는다.** 정직한 앵커는 §7.145④ -> 기대 LB +0 ~ +0.5.

검증
    · 예측이 달라진 행 비율 == F 행 비율 (변경이 R 로 새지 않았는지)
    · 재패킹 후 `predict.py` 하나만 바이트 상이

사용
    python tools_hj/rebase_catf0.py --src hj03_dim8_cold.zip --out hj06_catf0.zip
"""
import argparse
import os
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from rebase_routing import (SHIFT_RE, load_proxy, predict_in_subprocess, sha)

OLD = "            z = (1.0 - self.cat_alpha) * z + self.cat_alpha * zc"
NEW = ("            # α_F = 0 — cat 은 R 행에만 섞는다 (§7.192 · 앵커 §7.145④)\n"
       "            z[is_r] = ((1.0 - self.cat_alpha) * z[is_r]\n"
       "                       + self.cat_alpha * zc[is_r])")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import shutil
    import tempfile
    os.makedirs("D:/t1eval", exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="catf0_", dir="D:/t1eval")
    work, orig = os.path.join(tmp, "work"), os.path.join(tmp, "orig")
    with zipfile.ZipFile(a.src) as z:
        names = z.namelist()
        z.extractall(work)
        z.extractall(orig)

    pp = os.path.join(work, "predict.py")
    src0 = open(pp, encoding="utf-8").read()
    assert src0.count(OLD) == 1, f"패치 대상 {src0.count(OLD)}회 (1회여야 한다)"
    assert src0.index("is_r = ~is_f") < src0.index(OLD), "is_r 이 스코프에 없다"
    m = SHIFT_RE.findall(src0)
    assert len(m) == 1
    shift0 = float(m[0][1])
    print(f"소스 {os.path.basename(a.src)}   PROB_SHIFT {shift0:.9f}")

    proxy = load_proxy()
    fr = float((proxy["game_type"].astype(str) == "F").mean())
    pq = os.path.join(tmp, "proxy.pkl")
    proxy.to_pickle(pq)
    print(f"프록시 2024 {len(proxy):,}행 · F 비율 {fr:.4f}")

    print("추론 ① 패치 전 …")
    before = predict_in_subprocess(work, pq, os.path.join(tmp, "b.npy"))

    open(pp, "w", encoding="utf-8").write(src0.replace(OLD, NEW))
    print("추론 ② 패치 후 (레벨 미보정) …")
    mid = predict_in_subprocess(work, pq, os.path.join(tmp, "m.npy"))

    d = mid - before
    nz = float((np.abs(d) > 0).mean())
    dmean = float(d.mean())
    print(f"\n  max|Δp|  {np.abs(d).max():.6g} · 평균|Δp| {np.abs(d).mean():.6g}")
    print(f"  예측이 달라진 행 비율 {nz:.4f}   (F 비율 {fr:.4f} 와 일치해야 한다)")
    assert abs(nz - fr) < 0.005, "변경이 F 행 밖으로 샜다"
    shift = shift0 - dmean
    print(f"  Δmean {dmean:+.6g}   PROB_SHIFT {shift0:.9f} -> {shift:.9f}")

    src2 = SHIFT_RE.sub(lambda mm: f"{mm.group(1)}{shift:.9f}",
                        src0.replace(OLD, NEW))
    open(pp, "w", encoding="utf-8").write(src2)
    print("추론 ③ 재보정 검증 …")
    after = predict_in_subprocess(work, pq, os.path.join(tmp, "a.npy"))
    r = float((after - before).mean())
    print(f"  패치 후 Δmean {r:+.3g}  {'✅' if abs(r) < 1e-8 else '⚠️'}")
    assert abs(r) < 1e-7

    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            fp = os.path.join(work, n)
            if os.path.isfile(fp):
                z.write(fp, n)
    diff = [n for n in names if os.path.isfile(os.path.join(work, n))
            and open(os.path.join(work, n), "rb").read()
            != open(os.path.join(orig, n), "rb").read()]
    print(f"\n변경 격리  {len(diff)}/{len(names)}: {diff}")
    assert diff == ["predict.py"], "predict.py 외의 파일이 바뀌었다"
    assert len(os.path.basename(a.out)) <= 30
    print(f"출력   {a.out}\n  SHA-256 {sha(a.out)}")
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
