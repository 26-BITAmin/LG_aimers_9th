"""§7.163b · §7.164b — **aux 2×2.** §7.162 가 규칙으로 승격시킨 것을 지킨다.

> 새 정규화 후보는 주효과를 재기 전에 기존 정규화(aux · soft label · init_score ·
> dropout · wd) 와의 **2×2 를 먼저 설계한다.** 주효과 단독은 채택 근거가 될 수 없다.

§7.163(NODE)·§7.164(GRANDE) 1단계는 `tabm.py` 1단계 하네스를 물려받아 **전 팔
aux=None** 이었다. 그런데 배포는 `MLP_AUX_LAMBDA = 1.0` (aux ON) 이고, §7.162 3단계가
바로 그 자리에서 TabM 을 죽였다 (Δ +21.5/+19.8 → **+9.7 / −1.7**). NODE·GRANDE 도
정규화 장치이므로 같은 대체 관계가 있을 수 있다 → 여기서 잰다.

### 왜 aux OFF 만으로는 기각은 되고 채택은 안 되나

aux 를 빼면 그 몫이 비어 있어 **새 정규화 후보에게 유리한** 레짐이다. 거기서 지면
그 패배는 견고하다(유리한 자리에서도 졌다). 반대로 거기서 이겨도 aux ON 에서
남는지는 알 수 없다 — TabM 이 정확히 그렇게 사라졌다.

### 팔 — 2×2×2 (aux × 에폭예산 × 몸통)

    CTRL     독립 emb4 5멤버              ← 배포되는 몸통. **각 aux 층의 기준선**
    N128e3   NODE m128 D6 entmax ep3      §7.163 최선
    G32e3    GRANDE E32 D4 가중 ep3
    G32e9    GRANDE E32 D4 가중 ep9       §7.164 최선 (물려받은 기본 예산)
    GU32e9   GRANDE E32 D4 균등 ep9       ③ 절제 — **최선 예산에서** 잰다

### ⚠️ 왜 에폭 예산도 2×2 에 넣나 — 그것도 정규화 노브다

§7.164 1단계 실측: GRANDE 는 ep9 이 ep3 보다 배포층 +29.0 / +26.4 로 낫다.
**NODE 와 정반대다** (NODE 는 ep9 에서 과적합해 ep3 을 원했다). 하드 STE 는 경사가
sigmoid′ 로 눌려 더 천천히 학습하기 때문으로 보인다.

그런데 에폭도 정규화다 → **aux 와 상호작용할 수 있다.** aux 를 켜면 정규화가 늘어
최적 에폭이 내려갈 수 있고, 그러면 "ep9 이 낫다" 는 aux OFF 레짐에만 유효한 결론이다.
§7.162 가 가르친 것이 정확히 이것이므로 (하네스가 뺀 대체재가 최적점을 옮긴다)
두 예산을 두 aux 층에서 전부 잰다.

⚠️ 내가 1단계에서 ep3 을 GRANDE 의 주력으로 사전 등록한 것은 **§7.163 의 설정을
   근거 없이 물려받은 것**이었다. 물려받은 기본값(ep9)이 오히려 최선이었다.

`PAIR` 로 **aux 층을 맞춰** 부트스트랩한다 (ON 팔은 CTRL_ON 에, OFF 팔은 CTRL_OFF 에).
섞으면 aux 효과와 몸통 효과가 한 숫자에 뭉친다.

### ⚠️ noid 몫도 aux 로 갈린다

슬롯 = 0.875·emb4 + 0.125·noid 인데 noid 멤버도 보조 헤드를 달고 학습한다
(`noid.npz` vs `noidaux.npz`, corr 0.985). `node.noid_z` 가 팔의 aux 설정을 보고
고르도록 고쳤다 — 안 그러면 슬롯의 12.5% 가 다른 레짐 값이 된다.

### 판정 (결과 보기 전에 적는다)

  상호작용 = Δ(aux OFF) − Δ(aux ON),  Δ 는 같은 aux 층의 CTRL 대비.
  ✋ **상호작용이 주효과의 50% 를 넘으면 대체재로 보고 기각한다** — §7.162 가
     쓴 문턱을 그대로 쓴다 (거기서 55% / 109% 였다).
  ✋ aux ON 에서 양 fold Δ 가 양수이고 부트스트랩 하한이 양수여야 채택 논의를 연다.
     ⚠️ 구간이 0 을 포함하면 부호만으로 자동 기각하지 않는다 (§7.152).
  ★ GU 열은 덤으로 **③×aux** 를 준다 — instance-wise weighting 도 정규화라
    aux 와 겹칠 수 있다. G 와 GU 의 상호작용이 같으면 겹치는 것은 ③ 이 아니다.

실행:
    MLP_DEVICE=cuda .venv-gpu/Scripts/python.exe -u src/auxgrid.py 2>&1 | tee experiments/auxgrid.log
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import node as H                                                    # noqa: E402
from config import MLP_AUX_LAMBDA, MLP_SEEDS                        # noqa: E402

S = H.NODE_SEEDS
AUX = {"aux": MLP_AUX_LAMBDA}

NODE_CFG = {"node": dict(m=128, depth=6, gate="entmax")}
G_CFG = {"grande": dict(E=32, depth=4, weighted=True)}
GU_CFG = {"grande": dict(E=32, depth=4, weighted=False)}

H.ARMS = [
    # ── aux OFF (전부 캐시. gu32e9 만 새로) ──────────────────────────
    ("CTRL   emb4 5멤버   auxOFF", {}, tuple(MLP_SEEDS[:5]), "ctrl", None),
    ("N128e3 NODE   ep3   auxOFF", NODE_CFG, S, "n128e3", 3),
    ("G32e3  GRANDE ep3   auxOFF", G_CFG, S, "g32e3", 3),
    ("G32e9  GRANDE ep9   auxOFF", G_CFG, S, "g32e9", None),
    ("GU32e9 GRANDE 균등  auxOFF", GU_CFG, S, "gu32e9", None),
    # ── aux ON (= 배포 조건). CTRL 은 §7.162-3 캐시 ──────────────────
    ("CTRL   emb4 5멤버   auxON", AUX, tuple(MLP_SEEDS[:5]), "ctrlaux", None),
    ("N128e3 NODE   ep3   auxON", {**NODE_CFG, **AUX}, S, "n128e3a", 3),
    ("G32e3  GRANDE ep3   auxON", {**G_CFG, **AUX}, S, "g32e3a", 3),
    ("G32e9  GRANDE ep9   auxON", {**G_CFG, **AUX}, S, "g32e9a", None),
    ("GU32e9 GRANDE 균등  auxON", {**GU_CFG, **AUX}, S, "gu32e9a", None),
]

OFF = "CTRL   emb4 5멤버   auxOFF"
ON = "CTRL   emb4 5멤버   auxON"
H.BASE = ON
H.PAIR = {a[0]: (ON if a[0].endswith("auxON") else OFF) for a in H.ARMS}

if __name__ == "__main__":
    H.main()
