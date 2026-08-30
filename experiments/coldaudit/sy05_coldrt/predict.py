"""추론 로직 — script.py가 이 함수를 호출한다.

이 파일과 features.py는 제출 zip 루트에 그대로 복사된다 (build_submit.py 참조).
따라서 config / data / validation 등 학습 전용 모듈에 의존하지 않는다.
"""
import json
import os
import pickle

import joblib
import lightgbm as lgb
import numpy as np

from features import build_features, logit_adjust
from mlp_infer import MLPPredictor
from team_reference.features import build_features as build_friend_features
from team_reference.mlp_infer import MLPPredictor as FriendMLPPredictor

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

        # User-only complementary layers, evaluated before Hyejin's post-hoc deltas.
        self.friend_art = joblib.load(os.path.join(model_dir, "routing_artifacts.pkl"))
        self.compact_views = {}
        for view in ("baseline_noid", "compact_context"):
            with open(os.path.join(model_dir, f"{view}_encoder.pkl"), "rb") as f:
                enc = pickle.load(f)
            members = []
            for seed in (42, 7, 2024):
                with np.load(os.path.join(model_dir, f"{view}_seed{seed}.npz")) as w:
                    members.append(FriendMLPPredictor({k: w[k] for k in w.files}, enc))
            self.compact_views[view] = members
        with open(os.path.join(model_dir, "general_lgb_encoder.pkl"), "rb") as f:
            self.general_enc = pickle.load(f)
        self.general = []
        for variant in ("base", "wide", "extra", "lowff"):
            for seed in (42, 7, 2024):
                self.general.append(lgb.Booster(
                    model_file=os.path.join(model_dir, f"general_lgb_{variant}_{seed}.txt")))
        self.user_fspec = [lgb.Booster(
            model_file=os.path.join(model_dir, f"user_fspec_{seed}.txt"))
            for seed in (42, 7, 2024)]
        self.user_fspec_features = joblib.load(
            os.path.join(model_dir, "user_fspec_features.joblib"))

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
        # F 전용 전문가 (§7.101) — game_type=='F' 행만 다시 예측해 로짓에서 섞는다.
        # 옛 아티팩트에는 없는 키이므로 get()으로 읽고, 없으면 α=0이 되어 꺼진다.
        # CatBoost 이식 — 멤버 목록은 model/cat_members.json 이 진실이다.
        self.model_dir = model_dir
        cm = os.path.join(model_dir, "cat_members.json")
        if os.path.exists(cm):
            with open(cm, encoding="utf-8") as f:
                d = json.load(f)
            self.cat_alpha, self.cat_members = float(d["alpha"]), list(d["members"])
        else:
            self.cat_alpha, self.cat_members = 0.0, []
        self.cats = []

        self.fspec_alpha = float(bundle.get("fspec_alpha") or 0.0)
        self.fspec = []

        self.boosters = []
        for tag in bundle.get("members") or [f"{s}" for s in bundle.get("seeds", [])]:
            with open(os.path.join(model_dir, f"lgbm_{tag}.txt"), encoding="utf-8") as f:
                self.boosters.append(lgb.Booster(model_str=f.read()))
        if self.fspec_alpha > 0:
            for tag in bundle.get("fspec_members") or []:
                with open(os.path.join(model_dir, f"lgbm_{tag}.txt"),
                          encoding="utf-8") as f:
                    self.fspec.append(lgb.Booster(model_str=f.read()))
            if not self.fspec:
                self.fspec_alpha = 0.0

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

        # Compact-context paired delta.  The baseline-noid subtraction makes this
        # an isolated view replacement rather than another full MLP vote.
        friend = build_friend_features(test_df, self.friend_art)
        for col in test_df.columns:
            if col not in friend.columns:
                friend[col] = test_df[col].to_numpy()
        friend["reverse_x_success"] = (
            test_df["asof_pitcher_reverse_rate"].to_numpy(dtype=np.float32)
            * test_df["asof_pitcher_success_rate"].to_numpy(dtype=np.float32)
        )
        paired = {}
        for view, models in self.compact_views.items():
            paired[view] = np.mean([m.logit(friend) for m in models], axis=0)
        z += 0.75 * 0.05 * (paired["compact_context"] - paired["baseline_noid"])

        # R routing: Hyejin+Compact 90% + the user's general LGBM 10%.
        is_f = test_df["game_type"].to_numpy().astype(str) == "F"
        is_r = ~is_f
        gf = friend[self.general_enc["features"]].copy()
        for col, mapping in self.general_enc["mappings"].items():
            gf[col] = (gf[col].astype("string").fillna("__MISSING__")
                       .map(mapping).fillna(-1).astype(np.int32))
        gf = gf.replace([np.inf, -np.inf], np.nan).fillna(-999.0).astype(np.float32)
        zg = np.mean([m.predict(gf, raw_score=True) for m in self.general], axis=0)
        z[is_r] = 0.93375 * z[is_r] + 0.06625 * zg[is_r]

        # F routing: Hyejin+Compact 60% + the user's clean-F specialist 40%.
        zf = np.mean([m.predict(friend[self.user_fspec_features], raw_score=True)
                      for m in self.user_fspec], axis=0)
        z[is_f] = 0.645 * z[is_f] + 0.355 * zf[is_f]

        # ── F 전용 전문가 블렌드 (§7.101) ─────────────────────
        # LGBM+MLP **이후**, δ 보정·prior shift **이전**. 이 지점이 α를 고른
        # 검증(fspec_verify.py)과 같아야 한다.
        # game_type은 각 행이 스스로 들고 있는 값이라 행 간 정보가 아니다 —
        # 여기서 하는 건 라우팅이지 test 통계 계산이 아니다 (CLAUDE.md §3.2).
        # ⚠️ game_type은 피처로는 여전히 드랍돼 있다 (§7.71). 학습기가 F−R 레벨차를
        #    표현하게 두면 시즌마다 뒤집혀 독이 된다. 여기서는 **어느 모델에 물을지**만 가른다.
        self.fspec_rate = 0.0
        if self.fspec_alpha > 0 and "game_type" in test_df.columns:
            is_f = test_df["game_type"].to_numpy().astype(str) == "F"
            self.fspec_rate = float(is_f.mean())
            if is_f.any():
                Xf = X.iloc[np.flatnonzero(is_f)]
                zf = np.zeros(len(Xf))
                for bst in self.fspec:
                    zf += bst.predict(Xf, raw_score=True)
                zf /= len(self.fspec)
                z[is_f] = ((1.0 - self.fspec_alpha) * z[is_f]
                           + self.fspec_alpha * zf)

        # ── CatBoost 12멤버 블렌드 (우리 스택 §7.142 이식) ──────────
        # fspec **이후**, δ 보정 **이전**. 전 행에 섞는다 (라우팅이 아니다).
        # 각 멤버는 자기 행만 본다 — test 행 간 정보는 쓰지 않는다.
        if self.cat_alpha > 0:
            from catboost import CatBoostClassifier, Pool
            if not self.cats:
                for tag in self.cat_members:
                    m = CatBoostClassifier()
                    m.load_model(os.path.join(self.model_dir, f"cat_{tag}.cbm"))
                    self.cats.append(m)
            pool = Pool(X)
            zc = np.zeros(len(X))
            for m in self.cats:
                zc += m.predict(pool, prediction_type="RawFormulaVal")
            zc /= len(self.cats)
            z = (1.0 - self.cat_alpha) * z + self.cat_alpha * zc

        # 사후 로짓 보정 (투수 × 상황 δ) — **블렌드 이후, prior shift 이전**.
        # 이 순서는 검증에서 잰 순서와 같아야 한다 (2 fold에서 β 스윕한 그 지점).
        # 순서를 바꾸면 solve_delta가 다른 로짓을 보게 되어 목표 prior를 못 맞춘다.
        # train.py · verify_pipeline.py가 **같은 함수**를 부른다.
        z_adj, self.ctx_hits = logit_adjust(test_df, self.art)
        z = z + z_adj

        # prior shift — 학습 때 고정된 스칼라(base+delta)의 행 단위 적용
        p = 1.0 / (1.0 + np.exp(-(z + self.delta)))
        PROB_CENTER = 0.4801
        PROB_SCALE = 1.05
        PROB_SHIFT = 0.005554188
        p = PROB_CENTER + PROB_SCALE * (p - PROB_CENTER) + PROB_SHIFT
        return np.clip(p, CLIP_EPS, 1 - CLIP_EPS)


def predict_test(test_df, model_dir="./model"):
    return Predictor(model_dir).predict(test_df)
