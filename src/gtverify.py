"""전체 스택 검증 — game_type을 피처로 되살린 변형 (§7.71 후속, 사용자 제안)

가설: soft label(§7.70)이 gt의 독(오염 라벨의 가짜 F 레벨)을 제거했으므로,
gt는 이제 무해를 넘어 유용할 수 있다. 단독 학습기 2×2는 "무해(0 근처)"까지
확인했다 — 전체 스택(δ축·MLP 상호작용 포함)에서 마지막으로 묻는다.

비교 기준: 배포 구성(gt 드랍)의 verify 홀드아웃 **917.0** (verify_softlabel.log).

⚠️ config.py는 건드리지 않는다 — 다른 세션이 같은 저장소에서 작업 중이라
   파일을 고치면 그쪽 실행이 조용히 오염된다. 이 프로세스 안에서만 덮어쓴다.
   (`from config import DROP_FEATURES`는 import 시점 값 복사라, config와
    verify_pipeline **양쪽의** 바인딩을 모두 덮어야 한다.)

실행:  .venv/Scripts/python.exe -u src/gtverify.py 2>&1 | tee experiments/gtverify.log
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config
config.DROP_FEATURES = []

import verify_pipeline
verify_pipeline.DROP_FEATURES = []

print("=" * 70)
print("game_type 유지 변형 (DROP_FEATURES=[] — 이 프로세스 한정)")
print(f"  기준: 배포(gt 드랍) verify 홀드아웃 917.0")
print("=" * 70)

verify_pipeline.main()
