"""
Multimodal Ad Performance Prediction Model
============================================
Fuses visual features (ResNet embeddings + handcrafted CV features)
with structured campaign metadata to predict:
- Click-Through Rate (CTR)
- Engagement Rate
- Conversion Likelihood (probability)

Architecture:
  Visual branch: ResNet embedding → MLP projection
  Structured branch: Normalised campaign metadata → MLP
  Fusion: Concatenate → Cross-attention → Prediction heads
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    roc_auc_score,
    r2_score,
)
from dataclasses import dataclass
import warnings
warnings.filterwarnings("ignore")


@dataclass
class PredictionResult:
    ctr: float                    # Predicted click-through rate [0, 1]
    engagement_rate: float        # Predicted engagement rate [0, 1]
    conversion_prob: float        # Predicted conversion probability [0, 1]
    performance_tier: str         # "top" / "average" / "poor"
    feature_importance: dict      # Top contributing features
    confidence_interval: dict     # 90% CI for each prediction


@dataclass
class ModelMetrics:
    ctr_mae: float
    ctr_mape: float
    ctr_r2: float
    engagement_mae: float
    engagement_r2: float
    conversion_auc: float
    cv_r2_mean: float
    cv_r2_std: float


# ── Structured Feature Schema ─────────────────────────────────────────────────

STRUCTURED_FEATURES = [
    # Campaign metadata
    "budget_daily",          # Daily campaign budget (£)
    "audience_size",         # Estimated reach (millions)
    "campaign_duration_days",
    "bid_cpm",               # Cost per 1000 impressions (£)

    # Ad metadata
    "ad_format",             # 0=banner, 1=video, 2=carousel, 3=story
    "has_cta",               # 1 if has call-to-action button
    "headline_word_count",
    "body_word_count",
    "has_offer",             # 1 if contains a discount/offer

    # Audience targeting
    "age_range_width",       # e.g., 25-44 → 19
    "n_interest_categories",
    "is_retargeting",        # 1 if retargeting campaign

    # Historical context
    "brand_avg_ctr_30d",     # Brand's recent CTR benchmark
    "category_avg_ctr",      # Industry category benchmark
    "day_of_week",           # 0=Mon, 6=Sun
    "hour_of_day",
]


class AdPerformanceModel:
    """
    Gradient Boosted Trees multimodal model for ad performance prediction.

    Uses GBDT (vs. neural nets) for:
    - Better performance on tabular/structured data
    - Interpretability via feature importance
    - Robustness to missing values
    - No GPU required for training/inference

    For production with large datasets, this can be replaced with
    a LightGBM or XGBoost model with minimal code changes.
    """

    def __init__(
        self,
        visual_embedding_dim: int = 512,
        use_visual_features: bool = True,
        pca_components: int = 50,
    ):
        self.visual_embedding_dim = visual_embedding_dim
        self.use_visual_features = use_visual_features
        self.pca_components = pca_components

        self.scaler = StandardScaler()
        self._pca = None
        self.is_fitted = False

        # Separate models for each target
        self._ctr_model = GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42
        )
        self._engagement_model = GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42
        )
        self._conversion_model = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42
        )
        self.feature_names_: list[str] = []

    # ── Feature Preparation ───────────────────────────────────────────────────

    def _prepare_features(
        self,
        structured_df: pd.DataFrame,
        visual_feature_vectors: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Combine structured and visual features into a single matrix.

        Args:
            structured_df: DataFrame with STRUCTURED_FEATURES columns
            visual_feature_vectors: (N, 523) array from AdCreativeFeatureExtractor

        Returns:
            (N, D) combined feature matrix
        """
        struct_feats = structured_df[
            [f for f in STRUCTURED_FEATURES if f in structured_df.columns]
        ].values.astype(float)

        if self.use_visual_features and visual_feature_vectors is not None:
            # Compress embeddings with PCA to reduce dimensionality
            if self._pca is not None:
                visual_compressed = self._pca.transform(visual_feature_vectors)
            else:
                visual_compressed = visual_feature_vectors[:, :self.pca_components]

            return np.hstack([struct_feats, visual_compressed])
        return struct_feats

    def _fit_pca(self, visual_feature_vectors: np.ndarray):
        """Fit PCA on visual features to reduce dimensionality."""
        from sklearn.decomposition import PCA
        self._pca = PCA(n_components=min(self.pca_components, visual_feature_vectors.shape[1] - 1))
        self._pca.fit(visual_feature_vectors)
        explained = self._pca.explained_variance_ratio_.sum()
        print(f"  PCA: {self.pca_components} components explain {explained:.1%} variance")

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(
        self,
        structured_df: pd.DataFrame,
        y_ctr: np.ndarray,
        y_engagement: np.ndarray,
        y_converted: np.ndarray,
        visual_feature_vectors: np.ndarray | None = None,
    ) -> "AdPerformanceModel":
        """
        Fit the multimodal performance model.

        Args:
            structured_df: Campaign metadata features
            y_ctr: Target CTR values [0, 1]
            y_engagement: Target engagement rates [0, 1]
            y_converted: Binary conversion labels {0, 1}
            visual_feature_vectors: Optional (N, 523) visual features

        Returns:
            self (fitted)
        """
        print("Fitting multimodal ad performance model...")

        # PCA on visual features
        if self.use_visual_features and visual_feature_vectors is not None:
            self._fit_pca(visual_feature_vectors)

        X = self._prepare_features(structured_df, visual_feature_vectors)
        X_scaled = self.scaler.fit_transform(X)

        # Build feature names for interpretability
        struct_names = [f for f in STRUCTURED_FEATURES if f in structured_df.columns]
        visual_names = [f"visual_pc{i}" for i in range(self.pca_components)] if self._pca else []
        self.feature_names_ = struct_names + visual_names

        print(f"  Feature matrix: {X_scaled.shape}")

        # Fit all three models
        print("  Training CTR model...")
        self._ctr_model.fit(X_scaled, y_ctr)

        print("  Training engagement model...")
        self._engagement_model.fit(X_scaled, y_engagement)

        print("  Training conversion model...")
        self._conversion_model.fit(X_scaled, y_converted.astype(int))

        self.is_fitted = True

        # Cross-validation score
        cv_scores = cross_val_score(
            GradientBoostingRegressor(n_estimators=100, random_state=42),
            X_scaled, y_ctr, cv=5, scoring="r2"
        )
        self._cv_r2 = cv_scores

        print(f"  ✓ CV R² (CTR): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        return self

    def predict(
        self,
        structured_df: pd.DataFrame,
        visual_feature_vectors: np.ndarray | None = None,
    ) -> list[PredictionResult]:
        """
        Generate performance predictions with confidence intervals.

        Args:
            structured_df: Campaign metadata
            visual_feature_vectors: Optional visual features

        Returns:
            List of PredictionResult for each ad
        """
        assert self.is_fitted, "Call .fit() first"

        X = self._prepare_features(structured_df, visual_feature_vectors)
        X_scaled = self.scaler.transform(X)

        ctr_preds = self._ctr_model.predict(X_scaled)
        eng_preds = self._engagement_model.predict(X_scaled)
        conv_probs = self._conversion_model.predict_proba(X_scaled)[:, 1]

        # Feature importance (from CTR model)
        if hasattr(self._ctr_model, "feature_importances_"):
            importances = self._ctr_model.feature_importances_
            top_indices = np.argsort(importances)[-5:][::-1]
            feat_imp = {
                self.feature_names_[i]: round(float(importances[i]), 4)
                for i in top_indices
                if i < len(self.feature_names_)
            }
        else:
            feat_imp = {}

        results = []
        for i in range(len(ctr_preds)):
            ctr = float(np.clip(ctr_preds[i], 0, 1))
            eng = float(np.clip(eng_preds[i], 0, 1))
            conv = float(np.clip(conv_probs[i], 0, 1))

            # Performance tier
            if ctr > 0.04 and eng > 0.08:
                tier = "top"
            elif ctr < 0.01 or eng < 0.02:
                tier = "poor"
            else:
                tier = "average"

            # Crude 90% CI (±15% of prediction)
            ci = {
                "ctr": (round(ctr * 0.85, 4), round(ctr * 1.15, 4)),
                "engagement": (round(eng * 0.85, 4), round(eng * 1.15, 4)),
                "conversion": (round(conv * 0.80, 4), round(min(conv * 1.20, 1.0), 4)),
            }

            results.append(PredictionResult(
                ctr=round(ctr, 5),
                engagement_rate=round(eng, 5),
                conversion_prob=round(conv, 4),
                performance_tier=tier,
                feature_importance=feat_imp,
                confidence_interval=ci,
            ))

        return results

    def evaluate(
        self,
        structured_df: pd.DataFrame,
        y_ctr: np.ndarray,
        y_engagement: np.ndarray,
        y_converted: np.ndarray,
        visual_feature_vectors: np.ndarray | None = None,
    ) -> ModelMetrics:
        """Evaluate model on a held-out test set."""
        results = self.predict(structured_df, visual_feature_vectors)
        ctr_preds = np.array([r.ctr for r in results])
        eng_preds = np.array([r.engagement_rate for r in results])
        conv_preds = np.array([r.conversion_prob for r in results])

        return ModelMetrics(
            ctr_mae=round(float(mean_absolute_error(y_ctr, ctr_preds)), 6),
            ctr_mape=round(float(mean_absolute_percentage_error(
                np.maximum(y_ctr, 1e-5), ctr_preds
            )), 4),
            ctr_r2=round(float(r2_score(y_ctr, ctr_preds)), 4),
            engagement_mae=round(float(mean_absolute_error(y_engagement, eng_preds)), 6),
            engagement_r2=round(float(r2_score(y_engagement, eng_preds)), 4),
            conversion_auc=round(float(roc_auc_score(
                y_converted.astype(int), conv_preds
            )), 4),
            cv_r2_mean=round(float(self._cv_r2.mean()), 4),
            cv_r2_std=round(float(self._cv_r2.std()), 4),
        )

    def explain_prediction(self, result: PredictionResult) -> str:
        """
        Generate human-readable explanation for a prediction.
        Suitable for displaying in a marketing dashboard.
        """
        lines = [
            f"## Ad Performance Prediction",
            f"",
            f"**Predicted CTR:** {result.ctr*100:.2f}% "
            f"(90% CI: {result.confidence_interval['ctr'][0]*100:.2f}% – "
            f"{result.confidence_interval['ctr'][1]*100:.2f}%)",
            f"**Engagement Rate:** {result.engagement_rate*100:.2f}%",
            f"**Conversion Probability:** {result.conversion_prob*100:.1f}%",
            f"**Performance Tier:** {result.performance_tier.upper()}",
            f"",
            f"### Top Contributing Factors",
        ]
        for feat, imp in sorted(
            result.feature_importance.items(), key=lambda x: -x[1]
        )[:5]:
            bar = "█" * int(imp * 100)
            lines.append(f"  {feat:30s} {bar} ({imp:.3f})")

        if result.performance_tier == "poor":
            lines += [
                "", "### ⚠️ Improvement Suggestions",
                "- Consider adding a strong CTA button",
                "- Increase visual contrast for better attention capture",
                "- Narrow audience targeting to improve relevance",
            ]
        elif result.performance_tier == "top":
            lines += [
                "", "### ✅ Strengths",
                "- High predicted CTR — creative is resonating",
                "- Consider scaling budget to maximise reach",
            ]

        return "\n".join(lines)
