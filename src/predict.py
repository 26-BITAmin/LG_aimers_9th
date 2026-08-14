"""추론 로직 — script.py가 이 함수를 호출한다.

이 파일과 features.py는 제출 zip 루트에 그대로 복사된다 (build_submit.py 참조).
따라서 config / data / validation 등 학습 전용 모듈에 의존하지 않는다.
"""
import os

import joblib
import lightgbm as lgb
import numpy as np

from features import build_features, logit_adjust
from mlp_infer import MLPPredictor

CLIP_EPS = 1e-6


class Predictor:
    def __init__(self, model_dir="./model"):
        bundle = joblib.load(os.path.join(model_dir, "artifacts.joblib"))
        self.art = bundle["artifacts"]
        self.cols = bundle["feature_cols"]
        self.delta = float(bundle["logit_delta"])
        # 사후 로짓 보정 축은 self.art["ctx"] 안에 β까지 함께 들어 있다.
        # (번들과 아티팩트 두 곳에 β를 두면 어긋난다 — 한 곳으로 모았다)
        self.ctx_hits = {}

        # MLP 멤버 — torch가 아니라 npz + numpy 순전파다 (mlp_infer.py 참조)
        self.alpha = float(bundle.get("mlp_alpha") or 0.0)
        self.mlps = []
        if self.alpha > 0:
            enc = bundle["mlp_enc"]
            for tag in bundle.get("mlp_members") or []:
                with np.load(os.path.join(model_dir, f"mlp_{tag}.npz")) as z:
                    self.mlps.append(MLPPredictor({k: z[k] for k in z.files}, enc))
            if not self.mlps:
                self.alpha = 0.0
        # 멤버별 가중 (config.MLP_VARIANT_WEIGHT에서 나온다). 없으면 균등 —
        # 옛 아티팩트를 그대로 읽을 수 있어야 한다.
        self.mlp_w = bundle.get("mlp_weights") or None
        if self.mlp_w is not None:
            self.mlp_w = [float(v) for v in self.mlp_w]
            if len(self.mlp_w) != len(self.mlps):
                raise ValueError(f"mlp_weights({len(self.mlp_w)}) != "
                                 f"멤버 수({len(self.mlps)})")
        # model_file= 이 아니라 model_str= 로 읽는다.
        # LightGBM의 C 파일 API는 Windows에서 경로를 ANSI 코드페이지로 해석하므로
        # 경로에 비ASCII 문자가 있으면 조용히 깨진다. 파일 I/O는 파이썬에 맡긴다.
        self.boosters = []
        for tag in bundle.get("members") or [f"{s}" for s in bundle.get("seeds", [])]:
            with open(os.path.join(model_dir, f"lgbm_{tag}.txt"), encoding="utf-8") as f:
                self.boosters.append(lgb.Booster(model_str=f.read()))

    def predict(self, test_df):
        """test_df의 각 행을 **독립적으로** 예측한다.

        여기에는 test_df 전체에 대한 groupby / value_counts / rolling / fit이
        전혀 없다. 행 사이를 잇는 값은 전부 학습 때 동결된 self.art / self.delta다.
        """
        if len(test_df) == 0:
            return np.zeros(0)

        X = build_features(test_df, self.art)
        if self.art.get("use_player_ids"):
            for c in ("pitcher_id", "batter_id"):
                X[c] = test_df[c].to_numpy(dtype=np.float32)

        missing = [c for c in self.cols if c not in X.columns]
        if missing:
            raise ValueError(f"피처 누락: {missing}")
        # Trackman 테이블 조회가 통째로 빗나갔는지 진단용으로 노출한다.
        # (평가 서버 test의 season 값이 예상과 다르면 전부 결측이 되고,
        #  에러 없이 성능만 떨어진다 — 로그에 남겨 즉시 알아채기 위함)
        self.tm_match_rate = (float(X["tm_matched"].mean())
                              if "tm_matched" in X.columns else None)

        X = X[self.cols]     # 학습 때 쓴 컬럼만, 학습 때 순서로. drop_features는 여기서 걸러진다

        # 시드 앙상블은 **로짓 공간에서 평균**한다.
        # 확률 평균은 앙상블 출력을 0.5 쪽으로 수축시켜 캘리브레이션을 흐트러뜨린다.
        z = np.zeros(len(X))
        for bst in self.boosters:
            z += bst.predict(X, raw_score=True)
        z /= len(self.boosters)

        # MLP 블렌드도 로짓 공간에서. 각 MLP는 자기 행만 보고 계산한다
        # (표준화 상수 · 범주 LUT · 결측 대체값이 전부 학습 때 동결돼 있다).
        if self.alpha > 0:
            zm = np.zeros(len(X))
            if self.mlp_w is None:
                for m in self.mlps:
                    zm += m.logit(X)
                zm /= len(self.mlps)
            else:
                # 가중은 학습 시점에 정규화돼 있다 (합 = 1). 여기서 또 나누지 않는다.
                for m, wt in zip(self.mlps, self.mlp_w):
                    zm += wt * m.logit(X)
            z = (1.0 - self.alpha) * z + self.alpha * zm

        # 사후 로짓 보정 (투수 × 상황 δ) — **블렌드 이후, prior shift 이전**.
        # 이 순서는 검증에서 잰 순서와 같아야 한다 (2 fold에서 β 스윕한 그 지점).
        # 순서를 바꾸면 solve_delta가 다른 로짓을 보게 되어 목표 prior를 못 맞춘다.
        # train.py · verify_pipeline.py가 **같은 함수**를 부른다.
        z_adj, self.ctx_hits = logit_adjust(test_df, self.art)
        z = z + z_adj

        # prior shift — 학습 때 고정된 스칼라(base+delta)의 행 단위 적용
        p = 1.0 / (1.0 + np.exp(-(z + self.delta)))
        return np.clip(p, CLIP_EPS, 1 - CLIP_EPS)


def predict_test(test_df, model_dir="./model"):
    return Predictor(model_dir).predict(test_df)
