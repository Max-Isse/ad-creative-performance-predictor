"""
Unit tests for Ad Creative Performance Predictor
"""
import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visual_features import (
    ColourAnalyser,
    CompositionAnalyser,
    VisualBackbone,
    AdCreativeFeatureExtractor,
    VisualFeatures,
)
from src.performance_model import AdPerformanceModel, PredictionResult
from src.data_generator import generate_ad_dataset, generate_synthetic_ad_image


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_image():
    return generate_synthetic_ad_image(width=300, height=250, seed=42)


@pytest.fixture
def ad_dataset():
    return generate_ad_dataset(n_ads=200, seed=42)


@pytest.fixture
def fitted_model(ad_dataset):
    df, vis, ctr, eng, conv = ad_dataset
    train_df = df.iloc[:160]
    model = AdPerformanceModel(
        visual_embedding_dim=50,
        use_visual_features=True,
        pca_components=10,
    )
    model.fit(train_df, ctr[:160], eng[:160], conv[:160], vis[:160])
    return model, ad_dataset


# ── Visual Feature Tests ──────────────────────────────────────────────────────

class TestColourAnalyser:
    def test_output_keys(self, sample_image):
        analyser = ColourAnalyser()
        result = analyser.analyse(sample_image)
        expected = ["dominant_colours", "colour_temperature", "brightness",
                    "saturation", "contrast", "colour_harmony"]
        for key in expected:
            assert key in result, f"Missing key: {key}"

    def test_brightness_range(self, sample_image):
        analyser = ColourAnalyser()
        result = analyser.analyse(sample_image)
        assert 0 <= result["brightness"] <= 1

    def test_saturation_range(self, sample_image):
        analyser = ColourAnalyser()
        result = analyser.analyse(sample_image)
        assert 0 <= result["saturation"] <= 1

    def test_temperature_valid(self, sample_image):
        analyser = ColourAnalyser()
        result = analyser.analyse(sample_image)
        assert result["colour_temperature"] in ("warm", "cool", "neutral")

    def test_dominant_colours_count(self, sample_image):
        analyser = ColourAnalyser()
        result = analyser.analyse(sample_image)
        assert len(result["dominant_colours"]) == 5


class TestCompositionAnalyser:
    def test_output_keys(self, sample_image):
        analyser = CompositionAnalyser()
        result = analyser.analyse(sample_image)
        expected = ["visual_complexity", "text_area_ratio", "face_count",
                    "person_present", "rule_of_thirds_score"]
        for key in expected:
            assert key in result

    def test_complexity_range(self, sample_image):
        analyser = CompositionAnalyser()
        result = analyser.analyse(sample_image)
        assert 0 <= result["visual_complexity"] <= 1

    def test_text_ratio_range(self, sample_image):
        analyser = CompositionAnalyser()
        result = analyser.analyse(sample_image)
        assert 0 <= result["text_area_ratio"] <= 1

    def test_face_count_non_negative(self, sample_image):
        analyser = CompositionAnalyser()
        result = analyser.analyse(sample_image)
        assert result["face_count"] >= 0


class TestVisualBackbone:
    def test_embedding_shape(self, sample_image):
        backbone = VisualBackbone()
        backbone.load()  # will use fallback
        embedding = backbone.embed(sample_image)
        assert embedding.shape == (512,)

    def test_embedding_normalised(self, sample_image):
        backbone = VisualBackbone()
        backbone.load()
        embedding = backbone.embed(sample_image)
        norm = np.linalg.norm(embedding)
        assert 0.9 < norm < 1.1 or norm < 1e-8  # L2 normalised or zero

    def test_category_valid(self, sample_image):
        backbone = VisualBackbone()
        backbone.load()
        embedding = backbone.embed(sample_image)
        cat = backbone.predict_category(embedding)
        assert cat in ("lifestyle", "product", "text-heavy", "abstract")


class TestAdCreativeFeatureExtractor:
    def test_extract_returns_visual_features(self, sample_image):
        extractor = AdCreativeFeatureExtractor(load_backbone=True)
        features = extractor.extract(sample_image)
        assert isinstance(features, VisualFeatures)

    def test_attention_score_range(self, sample_image):
        extractor = AdCreativeFeatureExtractor(load_backbone=True)
        features = extractor.extract(sample_image)
        assert 0 <= features.attention_score <= 1

    def test_brand_safety_range(self, sample_image):
        extractor = AdCreativeFeatureExtractor(load_backbone=True)
        features = extractor.extract(sample_image)
        assert 0 <= features.brand_safety_score <= 1

    def test_feature_vector_shape(self, sample_image):
        extractor = AdCreativeFeatureExtractor(load_backbone=True)
        features = extractor.extract(sample_image)
        vec = extractor.to_feature_vector(features)
        assert vec.shape == (524,)  # 12 scalar + 512 embedding


# ── Performance Model Tests ───────────────────────────────────────────────────

class TestAdPerformanceModel:
    def test_fit_runs(self, ad_dataset):
        df, vis, ctr, eng, conv = ad_dataset
        model = AdPerformanceModel(pca_components=5)
        model.fit(df[:100], ctr[:100], eng[:100], conv[:100], vis[:100])
        assert model.is_fitted

    def test_predict_returns_results(self, fitted_model):
        model, (df, vis, ctr, eng, conv) = fitted_model
        results = model.predict(df[160:170], vis[160:170])
        assert len(results) == 10
        assert all(isinstance(r, PredictionResult) for r in results)

    def test_ctr_in_range(self, fitted_model):
        model, (df, vis, ctr, eng, conv) = fitted_model
        results = model.predict(df[160:], vis[160:])
        for r in results:
            assert 0 <= r.ctr <= 1, f"CTR out of range: {r.ctr}"

    def test_conversion_prob_in_range(self, fitted_model):
        model, (df, vis, ctr, eng, conv) = fitted_model
        results = model.predict(df[160:], vis[160:])
        for r in results:
            assert 0 <= r.conversion_prob <= 1

    def test_performance_tier_valid(self, fitted_model):
        model, (df, vis, ctr, eng, conv) = fitted_model
        results = model.predict(df[160:], vis[160:])
        for r in results:
            assert r.performance_tier in ("top", "average", "poor")

    def test_evaluate_returns_metrics(self, fitted_model):
        model, (df, vis, ctr, eng, conv) = fitted_model
        metrics = model.evaluate(df[160:], ctr[160:], eng[160:], conv[160:], vis[160:])
        assert metrics.ctr_r2 > -1  # not infinitely bad
        assert 0 <= metrics.conversion_auc <= 1

    def test_explain_prediction(self, fitted_model):
        model, (df, vis, ctr, eng, conv) = fitted_model
        result = model.predict(df[:1], vis[:1])[0]
        explanation = model.explain_prediction(result)
        assert "CTR" in explanation
        assert "Engagement" in explanation

    def test_no_visual_features_works(self, ad_dataset):
        df, _, ctr, eng, conv = ad_dataset
        model = AdPerformanceModel(use_visual_features=False)
        model.fit(df[:100], ctr[:100], eng[:100], conv[:100])
        results = model.predict(df[100:110])
        assert len(results) == 10
