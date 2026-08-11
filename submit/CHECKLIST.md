# submit/CHECKLIST.md

제출 직전에만 읽는다. 대회 스펙 전체는 `competition.md`.

## zip 구조

```
submit.zip
├── model/              # 가중치 + lookup parquet
├── script.py           # 평가 서버가 실행
└── requirements.txt
```

최상위에 추가 폴더가 있으면 설치 오류. 평가 서버가 `data/`와 `output/`을 자동 추가한다.

## 체크

- [ ] `script.py`가 `./data/test.csv` + `./data/sample_submission.csv`를 읽고
      `./output/submission.csv`를 생성
- [ ] `script.py`가 `features.py`를 import 하거나 파일째 포함 (재구현 아님)
- [ ] `row_id` 순서·값이 `sample_submission.csv`와 정확히 일치 (245,789행)
- [ ] 예측값이 [0, 1] 실수. 0/1 반올림 금지
- [ ] `requirements.txt`에 `lightgbm` 명시
- [ ] `requirements.txt`에서 **제외**: `pandas`, `numpy`, `scikit-learn`, `scipy`, `torch`,
      `joblib`, `transformers` — 서버 기본 설치와 버전 충돌 시 설치 오류
- [ ] `pitcher_train_totals.parquet` 등 lookup을 `model/`에 동봉
- [ ] 절편 상수가 하드코딩 또는 `model/`에 저장돼 있고 **test를 참조하지 않음**
- [ ] 5행 샘플 `test.csv`로 end-to-end 실행 성공
- [ ] 245,789행 추론이 로컬에서 10분 내 (서버: 6 vCPU / 28GB RAM / L4 22.4GiB)
- [ ] 인터넷 없이 동작 (다운로드·API 호출 없음)
- [ ] zip ≤ 10GB (해제 후 ≤ 32GB), 패키지 설치 ≤ 10분

## 로컬 리허설

```bash
cd submit && rm -rf output && mkdir -p output data
cp ../data/test.csv ../data/sample_submission.csv data/
time python script.py
head -3 output/submission.csv
python -c "
import pandas as pd
s = pd.read_csv('output/submission.csv'); e = pd.read_csv('data/sample_submission.csv')
assert (s.row_id.values == e.row_id.values).all(), 'row_id 불일치'
assert s.control_success.between(0, 1).all(), '범위 이탈'
assert s.control_success.notna().all(), 'NaN 존재'
print('OK', len(s), s.control_success.mean().round(5))
"
```

> 라이브러리 설치 에러는 설치 오류로 처리되며 **일일 제출 횟수에 반영되지 않는다.**
