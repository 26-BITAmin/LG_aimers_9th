"""사용자 제안 — `runner_on_2b/3b` 를 빼도 되는가 (base_state 가 이미 포함).

전제 확인: base_state 8종(`___` `1__` `_2_` `__3` `12_` `1_3` `_23` `123`)은
세 플래그와 **정확한 전단사**다. 정보량으로는 완전 중복.
그러나 트리에서 정보 동일 != 표현 동일:
    base_state  범주형 → 8레벨을 임의 두 집합으로 1회 분할 (2^7=127 가지)
    플래그      이진   → 축 정렬 연언만
범주형 쪽이 표현력이 크다. 플래그가 장식인지 **세어서** 확인한다.

§7.11 그룹 LOO(5컬럼 통째 −2.8)는 이 질문에 답하지 않는다 — 그건 정보 제거였고
이건 정보 보존 표현 변경이다.

측정 (배포 모델 텍스트만 읽는다. 학습·데이터 0):
  P1  5컬럼의 gain·split수 + 23개 모델 간 표준편차 → 1b>2b>3b 순서가 안정적인가
  P2  base_state 범주형 분할이 실제로 쓴 부분집합 → 2루/3루 판정을 재구성하는가
  P3  플래그 분할의 트리 내 깊이 분포 → 얕으면 구조, 깊으면 노이즈 적합

실행:  .venv/Scripts/python.exe -u src/basedup.py 2>&1 | tee experiments/basedup.log
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "submit" / "model"
COLS = ["base_state", "num_runners_on", "runner_on_1b", "runner_on_2b", "runner_on_3b"]
# data_description 순서. LightGBM 범주 인덱스는 features.py 의 category 코드다.
STATES = ["___", "1__", "12_", "123", "1_3", "_23", "_2_", "__3"]


def walk(node, depth, out):
    if "split_index" not in node:
        return
    out.append((node["split_feature"], depth, node["decision_type"],
                node.get("threshold"), node.get("split_gain", 0.0)))
    walk(node["left_child"], depth + 1, out)
    walk(node["right_child"], depth + 1, out)


def main():
    files = sorted(MODEL_DIR.glob("lgbm_*.txt"))
    gains, splits, names = [], [], None
    cat_subsets = Counter()
    flag_depth = {c: [] for c in COLS}
    cat_order = None

    for f in files:
        bst = lgb.Booster(model_str=f.read_text(encoding="utf-8"))
        if names is None:
            names = bst.feature_name()
        g = np.asarray(bst.feature_importance("gain"), np.float64)
        sp = np.asarray(bst.feature_importance("split"), np.float64)
        gains.append(g / g.sum())
        splits.append(sp / sp.sum())

        dm = bst.dump_model()
        if cat_order is None:
            cat_order = dm.get("pandas_categorical")
        nodes = []
        for t in dm["tree_info"]:
            walk(t["tree_structure"], 0, nodes)
        for fi, dep, dt, thr, _ in nodes:
            nm = names[fi]
            if nm in flag_depth:
                flag_depth[nm].append(dep)
            if nm == "base_state" and dt == "==":
                ks = tuple(sorted(int(x) for x in str(thr).split("||")))
                cat_subsets[ks] += 1

    G = np.array(gains) * 100
    S = np.array(splits) * 100
    print(f"모델 {len(files)}개 · 피처 {len(names)}개")
    if cat_order:
        print(f"base_state 범주 코드 순서: {cat_order[0] if cat_order else '?'}")
    print("\n" + "=" * 92)
    print("P1. gain / split 점유율 — 23개 모델 평균 ± 표준편차")
    print("=" * 92)
    print(f"  {'컬럼':<18s}{'gain%':>9s}{'±sd':>8s}{'split%':>9s}{'±sd':>8s}"
          f"{'분할/모델':>11s}{'gain 순위 범위':>16s}")
    for c in COLS:
        i = names.index(c)
        rk = [int(np.sum(row > row[i])) + 1 for row in G]
        nsp = [int(round(row[i] / 100 * tot)) for row, tot in
               zip(S, [np.asarray(lgb.Booster(model_str=f.read_text(encoding='utf-8'))
                                  .feature_importance("split")).sum() for f in files[:1]] * len(S))]
        print(f"  {c:<18s}{G[:,i].mean():>9.3f}{G[:,i].std():>8.3f}"
              f"{S[:,i].mean():>9.3f}{S[:,i].std():>8.3f}"
              f"{np.mean(nsp):>11.0f}{f'{min(rk)}–{max(rk)}':>16s}")

    print("\n" + "=" * 92)
    print("P2. base_state 범주형 분할이 실제로 쓴 부분집합 (상위 12개)")
    print("=" * 92)
    cats = cat_order[0] if cat_order else STATES
    tot = sum(cat_subsets.values())
    print(f"  총 범주형 분할 {tot:,}회 · 서로 다른 부분집합 {len(cat_subsets)}종\n")
    print(f"  {'빈도':>8s}{'비율':>8s}   부분집합 (왼쪽으로 가는 상태)          재구성")
    want = {"2루있음": {"_2_", "12_", "_23", "123"},
            "3루있음": {"__3", "1_3", "_23", "123"},
            "1루있음": {"1__", "12_", "1_3", "123"},
            "주자없음": {"___"}, "주자있음": {"1__", "_2_", "__3", "12_", "1_3", "_23", "123"}}
    for ks, n in cat_subsets.most_common(12):
        try:
            lab = {cats[k] for k in ks}
        except Exception:
            lab = set(map(str, ks))
        rec = next((w for w, sset in want.items() if sset == lab), "")
        rec = rec or next((f"NOT {w}" for w, sset in want.items()
                           if sset == set(cats) - lab), "")
        print(f"  {n:>8,d}{100*n/tot:>7.1f}%   {'|'.join(sorted(lab)):<38s}{rec}")

    print("\n  ── 관심 분할이 실제로 나오는가 ──")
    inv = {}
    for ks, n in cat_subsets.items():
        try:
            inv[frozenset(cats[k] for k in ks)] = inv.get(frozenset(cats[k] for k in ks), 0) + n
        except Exception:
            pass
    for w, sset in want.items():
        a = inv.get(frozenset(sset), 0)
        b = inv.get(frozenset(set(cats) - sset), 0)
        print(f"    {w:<10s} {a + b:>6,d}회  ({100*(a+b)/tot:>5.2f}% of 범주형 분할)")

    print("\n" + "=" * 92)
    print("P3. 분할 깊이 — 얕으면 구조, 깊으면 노이즈 적합 (트리 최대깊이 무제한, leaves=31)")
    print("=" * 92)
    print(f"  {'컬럼':<18s}{'분할수':>10s}{'평균깊이':>10s}{'중앙':>7s}{'깊이<=2 비율':>14s}")
    for c in COLS:
        d = np.array(flag_depth[c])
        if len(d) == 0:
            print(f"  {c:<18s}{0:>10d}   (분할 없음)")
            continue
        print(f"  {c:<18s}{len(d):>10,d}{d.mean():>10.2f}{np.median(d):>7.0f}"
              f"{100*(d <= 2).mean():>13.1f}%")


if __name__ == "__main__":
    main()
