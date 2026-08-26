"""§7.164 — **GRANDE** (Marton et al., ICLR 2024) 1단계 측정.

평가 코드는 **`node.main()` 을 그대로 부른다.** `node.ARMS` 와 `node.BASE` 만
갈아끼운다 — 평가 코드가 두 벌이면 조용히 어긋난다 (MODEL.md §5 전례:
train.py 와 verify_pipeline.py 에 학습 루프가 중복돼 검증이 배포를 테스트하지
않았다). 캐시 파일명 규약도 그대로라 CTRL·T5·N128e3 이 재학습 없이 읽힌다.

### 판정이 세 갈래로 쪼개진다 (`grandenet` 상단 참조)

    G32e3 vs N128e3   GRANDE 의 ①비-oblivious + ②하드 STE 가 값을 내나
                      (게이트 파라미터화·임계값 초기화·잎값 분포를 맞춰 놨으므로
                       남는 차이가 이 둘뿐이다)
    G32e3 vs GU32e3   ③instance-wise weighting 의 값     ← §7.45 가 죽은 축
    G32e3 vs G64e3    용량-반응 (§7.151 — 단일 지점이면 "미판정" 으로 끝난다)
    G32e9             에폭 예산 대조 + 에폭 곡선 (§7.163 에서 배운 것)

### 사전 등록 판정 (결과 보기 **전에** 적는다)

  1차 관문 — 단독 점수가 **양 fold 에서 T5 를** 이길 것. §7.24 조건.
  2차 관문 — 배포 층 양 fold Δ 와 클러스터 부트스트랩 하한.
             ⚠️ 구간이 0 을 포함하면 부호만으로 자동 기각 금지 (§7.152).
  3차 관문 — draw 분포. 5 draw sd 를 Δ 와 맞댄다 (§7.161).
  4차 관문 — 에폭 곡선 (G32e9 팔).

  ✋ ③ 이 죽는 조건 — G32e3 ≈ GU32e3 이면 instance-wise weighting 이 한 일이
     없다. §7.45 와 같은 결론이 되고, 그때는 **위약(무작위 가중)까지 갈 필요도
     없다** — 절제가 이미 0 이면 위약은 그 아래를 잴 뿐이다.

  ✋ ①② 가 죽는 조건 — G32e3 ≈ N128e3 이면 "하드 분할 + 노드별 피처" 가
     oblivious 소프트 트리 대비 값이 없다는 뜻이다. 그러면 미분가능 트리 계열
     전체가 이 문제에서 닫힌다 (NODE·GRANDE 가 그 계열의 양 극단이다).

  ⚠️ 멤버 수 핸디캡은 §7.163 과 같다 — G* 유효 5 vs T5 유효 15,
     §7.161(b) 장부로 b(1/5 − 1/15) = +4.46 이 T5 쪽 공짜 이득. 더해 주지 않는다.

실행 (GPU):
    MLP_DEVICE=cuda .venv-gpu/Scripts/python.exe -u src/grande.py 2>&1 | tee experiments/grande.log
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import node as H                                                    # noqa: E402
from config import MLP_SEEDS                                        # noqa: E402

S = H.NODE_SEEDS                    # CTRL 과 같은 5시드

H.ARMS = [
    # ── 눈금 · 대조군 (전부 캐시) ─────────────────────────────────────
    ("CTRL  emb4 독립 5멤버", {}, tuple(MLP_SEEDS[:5]), "ctrl", None),
    ("T5    TabM-mini k=5", {"tabm": dict(k=5)}, H.TABM_DRAWS, "k5", None),
    ("N128e3 NODE m128 D6 ep3",
     {"node": dict(m=128, depth=6, gate="entmax")}, S, "n128e3", 3),
    # ── GRANDE ───────────────────────────────────────────────────────
    ("G32e3  GRANDE E32 D4 w",
     {"grande": dict(E=32, depth=4, weighted=True)}, S, "g32e3", 3),
    ("GU32e3 GRANDE E32 D4 균등",
     {"grande": dict(E=32, depth=4, weighted=False)}, S, "gu32e3", 3),
    ("G64e3  GRANDE E64 D4 w",
     {"grande": dict(E=64, depth=4, weighted=True)}, S, "g64e3", 3),
    ("G32e9  GRANDE E32 D4 w ep9",
     {"grande": dict(E=32, depth=4, weighted=True)}, S, "g32e9", None),
]
H.BASE = "T5    TabM-mini k=5"

if __name__ == "__main__":
    H.main()
