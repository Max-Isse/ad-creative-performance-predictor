"""
Synthetic Ad Creative Dataset Generator
Produces realistic campaign metadata + simulated visual features + performance labels.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_ad_dataset(
    n_ads: int = 1000,
    seed: int = 42,
    output_path: str | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic ad performance dataset.

    Returns:
        structured_df: Campaign metadata features
        visual_vectors: (N, 50) simulated visual feature vectors (PCA-compressed)
        y_ctr: Click-through rates
        y_engagement: Engagement rates
        y_converted: Binary conversion labels
    """
    rng = np.random.default_rng(seed)

    # ── Structured Campaign Features ─────────────────────────────────────────
    ad_format = rng.integers(0, 4, n_ads)  # 0=banner, 1=video, 2=carousel, 3=story
    has_cta = rng.integers(0, 2, n_ads)
    has_offer = rng.integers(0, 2, n_ads)
    is_retargeting = rng.integers(0, 2, n_ads)

    structured_df = pd.DataFrame({
        "budget_daily": rng.uniform(50, 5000, n_ads),
        "audience_size": rng.uniform(0.1, 10, n_ads),
        "campaign_duration_days": rng.integers(1, 90, n_ads),
        "bid_cpm": rng.uniform(1, 25, n_ads),
        "ad_format": ad_format,
        "has_cta": has_cta,
        "headline_word_count": rng.integers(3, 15, n_ads),
        "body_word_count": rng.integers(5, 50, n_ads),
        "has_offer": has_offer,
        "age_range_width": rng.integers(5, 40, n_ads),
        "n_interest_categories": rng.integers(1, 10, n_ads),
        "is_retargeting": is_retargeting,
        "brand_avg_ctr_30d": rng.uniform(0.005, 0.08, n_ads),
        "category_avg_ctr": rng.uniform(0.01, 0.05, n_ads),
        "day_of_week": rng.integers(0, 7, n_ads),
        "hour_of_day": rng.integers(0, 24, n_ads),
    })

    # ── Simulated Visual Features (PCA-compressed) ────────────────────────────
    # In production: extracted by AdCreativeFeatureExtractor
    visual_vectors = rng.standard_normal((n_ads, 50)).astype(np.float32)

    # ── Performance Labels (ground truth with realistic signal) ──────────────
    # CTR is influenced by: CTA, offer, retargeting, visual quality, format
    base_ctr = 0.02
    ctr_signal = (
        base_ctr
        + has_cta * 0.012
        + has_offer * 0.008
        + is_retargeting * 0.015
        + (ad_format == 2) * 0.005   # carousel boost
        + (ad_format == 3) * 0.007   # story boost
        + structured_df["brand_avg_ctr_30d"].values * 0.5
        + visual_vectors[:, 0] * 0.003  # visual quality proxy
    )
    y_ctr = np.clip(ctr_signal + rng.normal(0, 0.005, n_ads), 0.001, 0.25)

    # Engagement follows similar pattern
    y_engagement = np.clip(
        y_ctr * 2.5 + rng.normal(0, 0.01, n_ads), 0.001, 0.6
    )

    # Conversion: binary, correlated with CTR + retargeting + offer
    conv_prob = np.clip(
        0.05
        + y_ctr * 3
        + has_offer * 0.1
        + is_retargeting * 0.15
        + rng.normal(0, 0.05, n_ads),
        0, 1,
    )
    y_converted = rng.binomial(1, conv_prob).astype(float)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        structured_df.to_csv(output_path, index=False)
        np.save(output_path.replace(".csv", "_visual.npy"), visual_vectors)
        np.save(output_path.replace(".csv", "_ctr.npy"), y_ctr)
        print(f"✓ Saved {n_ads} ads to {output_path}")

    return structured_df, visual_vectors, y_ctr, y_engagement, y_converted


def generate_synthetic_ad_image(
    width: int = 300,
    height: int = 250,
    seed: int | None = None,
) -> np.ndarray:
    """
    Generate a synthetic ad creative image (numpy array).
    Used for visual feature extraction demos.

    Returns:
        H×W×3 RGB numpy array (uint8)
    """
    rng = np.random.default_rng(seed)
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Background colour
    bg_colour = rng.integers(50, 255, 3)
    image[:, :] = bg_colour

    # Simulated text block
    text_y = height // 4
    text_h = height // 3
    text_colour = (255 - bg_colour).clip(0, 255)
    image[text_y:text_y+text_h, 20:width-20] = text_colour

    # Simulated product/graphic region
    graphic_start = height // 2 + 10
    graphic_colour = rng.integers(100, 255, 3)
    image[graphic_start:height-20, 20:width//2] = graphic_colour

    # Add some noise
    noise = rng.integers(-15, 15, image.shape, dtype=np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return image


if __name__ == "__main__":
    df, vis, ctr, eng, conv = generate_ad_dataset(
        output_path="data/sample/ad_data.csv"
    )
    print(f"CTR stats: mean={ctr.mean():.3f}, std={ctr.std():.3f}")
    print(f"Conversion rate: {conv.mean():.1%}")
    print(df.describe())
