# 혜진 챔피언 재현 — `hj05_r047f330.zip` (LB 1132.2409729548)

**2026-08-29 · 팀 공유용.** 이 문서만 따라 하면 다른 로컬에서 같은 zip 이 나온다.

---

## 0. 무엇이 우리 것인가 — 먼저 보고 시작할 것

`sy02_c4.zip`(신영, LB 1127.1558) 대비 **82개 파일 중 7개**만 바꿨다.

```
model/mlp_emb4_{42,7,2024,1234,31337}.npz   ← MLP emb4 재학습 + 콜드스타트 패치
predict.py                                   ← R/F 라우팅 t = −0.70 + PROB_SHIFT
model/meta.json                              ← 진실원에서 재생성
```

**나머지 75개(LGBM 20 · CatBoost 12 · general LGBM 12 · MLP noid 3 · compact 6 ·
fspec 3 · features.py 등)는 물려받은 그대로다.** 그 부분의 학습 코드는 우리에게 없다 —
`origin/wonwoo`(`t1core.py`·`t1cap.py`·`t1cat.py`·`t1gen.py`)와 신영 로컬에 있다.

> 즉 **이 문서는 "sy02_c4 → hj05" 재현이지 "0 → hj05" 재현이 아니다.**
> 전체 재현이 필요하면 그 두 사람의 학습 코드가 추가로 필요하다.

---

## 1. 환경

제출 `requirements.txt` 와 동일해야 한다. **numpy 는 2.x 여야 한다** — 1.x 는
`artifacts.joblib`(numpy 2 로 저장) 을 못 읽는다.

```bash
python3.11 -m venv .venv_sota
.venv_sota/bin/pip install \
  lightgbm==4.6.0 numpy==2.2.6 pandas==2.3.3 scipy==1.15.2 \
  joblib==1.4.2 scikit-learn==1.6.1 catboost==1.2.10
.venv_sota/bin/pip install torch          # 학습에만 필요. 추론은 numpy 순전파다
```

필요한 것: `data/train.csv` · `outputs/sy02_c4.zip` · `git fetch origin` (wonwoo 브랜치).

---

## 2. 작업 디렉터리 조립

학습 루프는 `origin/wonwoo:src/` 에 있고 **피처·인코딩은 제출 zip 것을 써야 한다.**
배포와 다른 `features.py` 를 쓰면 78피처가 달라져 비교 자체가 무의미해진다.

```bash
mkdir -p /tmp/z && (cd /tmp/z && unzip -oq outputs/sy02_c4.zip)
bash tools/setup_work.sh /tmp/work /tmp/z            # origin/wonwoo src + zip 의 3개 덮어쓰기
```

---

## 3. MLP `emb4` 재학습 (`pitcher_id` 임베딩 8차원)

```bash
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=4 \
.venv_sota/bin/python -u tools/train_mlp_dim.py \
  --work /tmp/work --zip-dir /tmp/z --dim 8 --out /tmp/npz_dim8
```

시드 5개, 각 60~70초. **다음 세 값이 아래와 같아야 레시피가 맞은 것이다.**

```
학습 행        1,475,092        (soft label 복제 후 1,546,291)
피처           78
soft_h         2019 0.092962360777 · 2020 0.109738306713
               2021 0.242586155671 · 2022 0.296954143731    ← 12자리 일치
보조 라벨 복원  99.9%
```

⚠️ **인코더를 재적합하지 않는다.** `bundle["mlp_enc"]` 를 그대로 쓴다 — 재적합하면
범주 인덱스가 어긋나 배포 임베딩과 행이 안 맞는다. 바꾸는 것은 `emb_dim["pitcher_id"]`
하나뿐이다.

⚠️ **`meta.json` 을 믿지 말 것.** `emb_dim.pitcher_id` 가 4 로 적혀 있는데 실제 npz 는
8/12 다 (T1 계보 빌드가 meta 를 복사해 와서 낡았다). 값은 `artifacts.joblib` 과 npz 로
확인한다.

---

## 4. 멤버 교체 + 레벨 재보정

```bash
.venv_sota/bin/python tools/build_mlp_zip.py \
  --src outputs/sy02_c4.zip --npz-dir /tmp/npz_dim8 --out /tmp/hj00_dim8_control.zip
```

T1 계보는 `logit_delta` 가 동결이라 성분이 바뀌면 레벨이 깨진다. 2024 전 행을 프록시로
평균 이동을 재고 `PROB_SHIFT` 를 패치한다. **변경 격리 6/82 (npz 5 + predict.py)** 확인.

---

## 5. 콜드스타트 패치 (`index 0` ← 학습 임베딩 평균)

```bash
.venv_sota/bin/python tools/build_coldstart_zip.py \
  --src /tmp/hj00_dim8_control.zip --out /tmp/hj03_dim8_cold.zip
```

임베딩 표는 투수 792명 + **미지값 1칸**인데, LUT 를 학습 데이터로 적합했으므로
`index 0` 은 **학습 중 한 행도 받지 않는다** — 초기 난수에 머문다. 그런데 매 시즌 행의
14~22% 가 그 전에 없던 투수다. `emb0[0] = emb0[1:].mean(0)` 로 그 난수를 지운다.

레벨 재보정에 **마스킹 프록시**를 쓴다 — 2024 프록시에는 미지 투수가 0명이라
(학습에 2024 가 들어갔다) 투수 74명의 `pitcher_id` 를 LUT 밖 값으로 바꿔 실제 test 의
미지 비율(20.61%)을 흉내낸 뒤 잰다. **`알려진 행 Δp = +0`** 이 나와야 한다.

---

## 6. R/F 라우팅 `t = −0.70`

```bash
.venv_sota/bin/python tools/rebase_routing.py \
  --src /tmp/hj03_dim8_cold.zip --t -0.70 --out /tmp/hj05_pre.zip
```

```
ray:  R = 0.10 + t·0.075     F = 0.40 + t·0.10
t = −0.70  →  R 0.047500 / F 0.330000
```

---

## 7. `meta.json` 재생성 — **반드시 마지막에**

```bash
.venv_sota/bin/python tools/fix_meta.py \
  --src /tmp/hj05_pre.zip --out outputs/hj05_r047f330.zip
.venv_sota/bin/python tools/fix_meta.py --src outputs/hj05_r047f330.zip   # 0/20 확인
```

⚠️ `predict.py` 를 건드리는 변경 뒤에 meta 를 만들어야 한다. 순서를 뒤집으면
`routing`·`prob_affine` 이 다시 어긋나 재감사가 2/20 이 된다 (실제로 밟았다).

---

## 8. 검증

```
meta 재감사        0/20
변경 격리          sy02_c4 대비 7개 (npz 5 + predict.py + meta.json)
행 독립성          청크·셔플·단일행 ~4e−09  (패치 없는 원본도 같은 값. BLAS 배치 효과)
                   같은 입력 2회는 0.000e+00
추론 시간          245,789행 54.9초 (한도 600초의 9.2%)
SHA-256            46570746cfea575e0108e9cb91254a81585556b5db159ea9013565f2a6635dd5
```

⚠️ **MLP 는 torch 라 비트 동일이 안 나온다.** 우리 dim8 재학습이 신영 출하본과
`corr 0.998943`, `Δstd 2.46e−03` 이었다 (원우 `t1full.py` 기준은 0.999). 플랫폼·스레드가
다르면 SHA 가 달라진다 — **점수가 아니라 재현도(corr)로 판정할 것.**

---

## 9. 도구 목록

| 도구 | 하는 일 |
| --- | --- |
| `tools/setup_work.sh` | 학습 작업 디렉터리 조립 |
| `tools/train_mlp_dim.py` | MLP `emb4` 재학습 (`--dim` · `--unk-dropout` · `--only-seeds` · `--holdout-season`) |
| `tools/build_mlp_zip.py` | 멤버 교체 + 레벨 재보정 |
| `tools/build_coldstart_zip.py` | `index 0` 패치 + 마스킹 프록시 재보정 |
| `tools/rebase_routing.py` | R/F 라우팅 재적용 |
| `tools/fix_meta.py` | `meta.json` 감사(`--src`)·재작성(`--out`) |

전부 **변경 격리를 검사하고 SHA-256 을 남긴다.**
