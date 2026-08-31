"""C1 계층 EB 잔차 보정을 임의의 제출 zip 위에 이식한다 (재학습 없음).

출처
    팀원 `sy12_c1.zip` 의 `model/c1_hierarchical_eb.joblib` + `predict.py` 블록.
    그 zip 은 `hj09_grbc` 베이스라 현 챔피언보다 두 층 뒤처져 있다.

★ 원본의 버그를 고친다 — **조회 키 타입 불일치**
    저장된 키는 정수 `(1, 1)` · `(21372, 1)` 인데 `predict.py` 는 `str(ph)` 로
    조회한다. 전 행 미적중이라 **보정이 정확히 0** 이었다 (실측: sy12_c1 과
    "라우팅만 바꾼 우리 빌드" 의 차이가 PROB_SHIFT 상수 하나, std 5.2e-17).
    -> 키를 `int` 로 맞춘다. 미적중 0.0% 로 회복된다.

    ⚠️ 그래서 이 축은 **한 번도 실제로 시험된 적이 없다.** 팀원이 홀드아웃에서
       이득을 봤다면 그 하네스는 배포 `predict.py` 와 다른 코드다.

적용 위치는 원본 그대로 둔다 — 확률공간 아핀 **이후**의 `p = p + alpha*corr`.
위치를 옮기면 다른 축이 된다.

행 독립성
    행 자신의 (투수, 타자, 양쪽 손) 값만 보는 조회다. 표는 학습 시점 동결
    (`source_folds` = 2022·2023·2024 forward OOF). `verify_zip` L2 가 재확인한다.

PROB_SHIFT 재보정
    보정도 PROB_SHIFT 도 확률공간 덧셈 상수라, 프록시 Δmean 만큼 shift 를 빼면
    레벨이 정확히 복원된다. `rebase_routing.py` 와 같은 절차·같은 프록시.

사용
    python3 tools/rebase_c1.py --src outputs/hj16_psucc.zip \
        --c1 outputs/sy12_c1.zip --alpha 0.5 --out outputs/hj17_c1a050.zip
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import joblib
import numpy as np

from rebase_routing import SHIFT_RE, TRAIN_CSV, load_proxy, predict_in_subprocess, sha

C1_FILE = "model/c1_hierarchical_eb.joblib"

INIT_OLD = '        self.delta = float(bundle["logit_delta"])\n'
INIT_NEW = '''        self.delta = float(bundle["logit_delta"])
        # C1 통계는 제출 전 forward OOF 잔차에서 동결된다 (§C1).
        c1_path = os.path.join(model_dir, "c1_hierarchical_eb.joblib")
        self.c1 = joblib.load(c1_path) if os.path.exists(c1_path) else None
'''

# ★ 원본 대비 고친 곳: 조회 키를 str -> int. 나머지 산술은 한 글자도 안 바꿨다.
BLOCK = '''
        # 계층 EB (투수 × 상대 타자손 / 타자 × 상대 투수손) 잔차 보정.
        # 순수 행 단위 조회다 — 평가 행으로 mean/count/groupby/fit 을 하지 않는다.
        # ★ 조회 키는 int 다. 원본은 str(...) 이라 전 행 미적중이었다.
        if self.c1 is not None:
            hand_table = self.c1["hand"]
            pitcher_table = self.c1["pitcher"]
            batter_table = self.c1["batter"]
            group_k = float(self.c1["group_k"])
            player_k = float(self.c1["player_k"])
            pitcher_weight = float(self.c1["pitcher_weight"])
            alpha = float(self.c1["alpha"])
            pitcher_ids = pd.to_numeric(test_df["pitcher_id"], errors="coerce").to_numpy()
            batter_ids = pd.to_numeric(test_df["batter_id"], errors="coerce").to_numpy()
            pitcher_hands = pd.to_numeric(test_df["pitcher_hand"], errors="coerce").to_numpy()
            batter_hands = pd.to_numeric(test_df["batter_hand"], errors="coerce").to_numpy()
            correction = np.zeros(len(test_df), dtype=np.float64)
            for i, (pid, bid, ph, bh) in enumerate(
                    zip(pitcher_ids, batter_ids, pitcher_hands, batter_hands)):
                phk = None if not np.isfinite(ph) else int(ph)
                bhk = None if not np.isfinite(bh) else int(bh)
                hs, hn = hand_table.get((phk, bhk), (0.0, 0))
                hand_effect = float(hs) / (float(hn) + group_k)
                ps, pn = ((0.0, 0) if not np.isfinite(pid) else
                          pitcher_table.get((int(pid), bhk), (0.0, 0)))
                bs, bn = ((0.0, 0) if not np.isfinite(bid) else
                          batter_table.get((int(bid), phk), (0.0, 0)))
                pitcher_effect = (float(ps) + player_k * hand_effect) / (float(pn) + player_k)
                batter_effect = (float(bs) + player_k * hand_effect) / (float(bn) + player_k)
                correction[i] = (pitcher_weight * pitcher_effect
                                 + (1.0 - pitcher_weight) * batter_effect)
            self.c1_rms = float(np.sqrt(((alpha * correction) ** 2).mean()))
            p = p + alpha * correction
'''
RET_OLD = "        return np.clip(p, CLIP_EPS, 1 - CLIP_EPS)"


def patch_predict(src: str) -> str:
    if "c1_hierarchical_eb" in src:
        raise SystemExit("[중단] 이 zip 에는 C1 이 이미 있다")
    if src.count(INIT_OLD) != 1:
        raise SystemExit("[중단] __init__ 앵커가 1개가 아니다")
    if src.count(RET_OLD) != 1:
        raise SystemExit("[중단] return 앵커가 1개가 아니다")
    src = src.replace(INIT_OLD, INIT_NEW, 1)
    src = src.replace(RET_OLD, BLOCK + RET_OLD, 1)
    for need in ("import joblib", "import os"):
        if need not in src:
            raise SystemExit(f"[중단] predict.py 에 `{need}` 가 없다")
    # 챔피언 predict.py 는 pandas 를 안 쓴다. 같은 파일 안이라 변경 격리에 영향 없다.
    if "import pandas as pd" not in src:
        src = src.replace("import numpy as np", "import numpy as np\nimport pandas as pd", 1)
    return src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="이식 대상 (현 챔피언)")
    ap.add_argument("--c1", required=True, help="C1 joblib 을 가진 zip (팀원 sy12_c1)")
    ap.add_argument("--alpha", type=float, help="생략하면 joblib 의 값을 그대로 쓴다")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix="c1_")
    work, orig = os.path.join(tmp, "work"), os.path.join(tmp, "orig")
    with zipfile.ZipFile(a.src) as z:
        z.extractall(work); z.extractall(orig)
    with zipfile.ZipFile(a.c1) as z:
        if C1_FILE not in z.namelist():
            raise SystemExit(f"[중단] {a.c1} 에 {C1_FILE} 이 없다")
        z.extract(C1_FILE, tmp)

    c1 = joblib.load(os.path.join(tmp, C1_FILE))
    print(f"소스   {os.path.basename(a.src)}")
    print(f"C1     version={c1['version']} · source_folds={c1['source_folds']} "
          f"· centered_by_year={c1['centered_by_year']}")
    print(f"       셀 hand {len(c1['hand'])} · pitcher {len(c1['pitcher'])} · batter {len(c1['batter'])}")
    print(f"       group_k {c1['group_k']} · player_k {c1['player_k']} "
          f"· pitcher_weight {c1['pitcher_weight']} · alpha {c1['alpha']}")
    if a.alpha is not None and a.alpha != c1["alpha"]:
        print(f"       alpha {c1['alpha']} -> {a.alpha}")
        c1["alpha"] = float(a.alpha)
    os.makedirs(os.path.join(work, "model"), exist_ok=True)
    joblib.dump(c1, os.path.join(work, C1_FILE))

    pp = os.path.join(work, "predict.py")
    src0 = open(pp, encoding="utf-8").read()
    shift0 = float(SHIFT_RE.findall(src0)[0][1])
    open(pp, "w", encoding="utf-8").write(patch_predict(src0))
    print(f"  현재 PROB_SHIFT {shift0:.9f}")

    print("\n프록시 2024행 적재 …")
    proxy = os.path.join(tmp, "proxy.pkl")
    load_proxy().to_pickle(proxy)
    print("추론 ① 이식 전 …")
    p0 = predict_in_subprocess(orig, proxy, os.path.join(tmp, "p0.npy"))
    print("추론 ② 이식 후 (레벨 미보정) …")
    p1 = predict_in_subprocess(work, proxy, os.path.join(tmp, "p1.npy"))
    d = p1 - p0
    print(f"  Δp 평균 {d.mean():+.9g} · 평균|Δp| {np.abs(d).mean():.7g} · max|Δp| {np.abs(d).max():.7g}")
    if np.abs(d).max() < 1e-12:
        raise SystemExit("[중단] 보정이 0 이다 — 조회 키가 여전히 안 맞는다")
    shift1 = shift0 - float(d.mean())
    print(f"  PROB_SHIFT {shift0:.9f} -> {shift1:.9f}")
    s = open(pp, encoding="utf-8").read()
    open(pp, "w", encoding="utf-8").write(
        SHIFT_RE.sub(lambda m: f"{m.group(1)}{shift1:.9f}", s))
    print("추론 ③ 재보정 검증 …")
    p2 = predict_in_subprocess(work, proxy, os.path.join(tmp, "p2.npy"))
    dm = float((p2 - p0).mean())
    print(f"  재보정 후 Δmean {dm:+.2e}  {'OK' if abs(dm) < 1e-8 else '❌ 미보정'}")
    print(f"  최종 max|Δp| {np.abs(p2 - p0).max():.7g}")

    names = sorted(set(zipfile.ZipFile(a.src).namelist()) | {C1_FILE})
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            fp = os.path.join(work, n)
            if os.path.isfile(fp):
                z.write(fp, n)
    ch = [n for n in names
          if not os.path.isfile(os.path.join(orig, n))
          or open(os.path.join(work, n), "rb").read() != open(os.path.join(orig, n), "rb").read()]
    print(f"\n변경 격리  {len(ch)}/{len(names)}: {ch}")
    print(f"출력   {a.out}\n  SHA-256 {sha(a.out)}")
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
