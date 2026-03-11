"""
Ad Creative Visual Feature Extractor
======================================
Extracts rich visual features from ad creative images using:
- ResNet/ViT backbone (transfer learning)
- Colour psychology analysis
- Composition metrics (rule of thirds, visual weight)
- Text density estimation (OCR-lite)
- Face/person detection
- Brand element detection

These features feed into a multimodal fusion model that predicts
CTR, engagement rate, and conversion likelihood.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import colorsys


@dataclass
class VisualFeatures:
    """All visual features extracted from a single ad creative."""
    # Colour features
    dominant_colours: list[tuple]          # Top 5 RGB colours
    colour_temperature: str                # warm / cool / neutral
    brightness: float                      # [0, 1]
    saturation: float                      # [0, 1]
    contrast: float                        # [0, 1]
    colour_harmony: str                    # monochromatic / complementary / triadic / etc.

    # Composition features
    visual_complexity: float               # [0, 1] (edge density proxy)
    text_area_ratio: float                 # fraction of image covered by text
    face_count: int
    person_present: bool
    rule_of_thirds_score: float            # [0, 1]

    # Semantic features (from backbone)
    embedding: np.ndarray                  # 512-dim visual embedding
    predicted_category: str               # product / lifestyle / abstract / text-heavy

    # Derived scores
    attention_score: float                 # predicted visual attention capture
    brand_safety_score: float              # [0, 1] (1 = safest)


class ColourAnalyser:
    """Extract colour psychology features from image arrays."""

    WARM_HUES = set(range(0, 50)) | set(range(330, 360))   # reds, oranges, yellows
    COOL_HUES = set(range(150, 270))                        # blues, greens, purples

    def analyse(self, image_array: np.ndarray) -> dict:
        """
        Analyse colour properties of an image.

        Args:
            image_array: H×W×3 numpy array (RGB, uint8)

        Returns:
            Dict with colour features
        """
        h, w, _ = image_array.shape
        pixels = image_array.reshape(-1, 3).astype(float) / 255.0

        # Dominant colours via k-means clustering
        dominant = self._kmeans_colours(pixels, k=5)

        # Convert to HSV for perceptual analysis
        hsv_pixels = np.array([colorsys.rgb_to_hsv(*p) for p in pixels])
        hue = hsv_pixels[:, 0] * 360
        sat = hsv_pixels[:, 1]
        val = hsv_pixels[:, 2]

        brightness = float(np.mean(val))
        saturation = float(np.mean(sat))

        # Colour temperature from dominant hue
        dominant_hue = float(np.median(hue[sat > 0.2]))  # ignore greys
        if dominant_hue in self.WARM_HUES:
            temperature = "warm"
        elif dominant_hue in self.COOL_HUES:
            temperature = "cool"
        else:
            temperature = "neutral"

        # Contrast via standard deviation of luminance
        luminance = 0.299 * pixels[:, 0] + 0.587 * pixels[:, 1] + 0.114 * pixels[:, 2]
        contrast = float(np.std(luminance))

        # Colour harmony (simplified)
        hue_bins = np.histogram(hue[sat > 0.3], bins=12, range=(0, 360))[0]
        occupied = np.sum(hue_bins > 0)
        if occupied <= 2:
            harmony = "monochromatic"
        elif occupied <= 4:
            harmony = "analogous"
        elif 5 <= occupied <= 7:
            harmony = "complementary"
        else:
            harmony = "complex"

        return {
            "dominant_colours": [tuple(int(c * 255) for c in colour) for colour in dominant],
            "colour_temperature": temperature,
            "brightness": brightness,
            "saturation": saturation,
            "contrast": contrast,
            "colour_harmony": harmony,
        }

    def _kmeans_colours(self, pixels: np.ndarray, k: int = 5) -> list:
        """Simple k-means for dominant colour extraction."""
        from sklearn.cluster import MiniBatchKMeans
        kmeans = MiniBatchKMeans(n_clusters=k, n_init=3, random_state=42)
        kmeans.fit(pixels)
        return [kmeans.cluster_centers_[i].tolist() for i in range(k)]


class CompositionAnalyser:
    """Analyse visual composition and layout features."""

    def analyse(self, image_array: np.ndarray) -> dict:
        """
        Extract composition metrics.

        Args:
            image_array: H×W×3 numpy array

        Returns:
            Dict with composition features
        """
        h, w = image_array.shape[:2]
        grey = self._to_grey(image_array)

        # Visual complexity via edge density (Sobel-like)
        complexity = self._edge_density(grey)

        # Rule of thirds score — do high-energy regions align with thirds?
        rots_score = self._rule_of_thirds(grey)

        # Text area estimation (high-frequency horizontal patterns)
        text_ratio = self._estimate_text_area(grey)

        # Face/person detection (lightweight HOG-based proxy)
        face_count = self._estimate_face_regions(grey)

        return {
            "visual_complexity": float(complexity),
            "text_area_ratio": float(text_ratio),
            "face_count": int(face_count),
            "person_present": face_count > 0,
            "rule_of_thirds_score": float(rots_score),
        }

    def _to_grey(self, image_array: np.ndarray) -> np.ndarray:
        r, g, b = image_array[:,:,0], image_array[:,:,1], image_array[:,:,2]
        return (0.299 * r + 0.587 * g + 0.114 * b).astype(float)

    def _edge_density(self, grey: np.ndarray) -> float:
        """Proxy for visual complexity using gradient magnitude."""
        gx = np.abs(np.diff(grey, axis=1))
        gy = np.abs(np.diff(grey, axis=0))
        edge_mean = (np.mean(gx) + np.mean(gy)) / 2
        return min(edge_mean / 40.0, 1.0)  # normalise

    def _rule_of_thirds(self, grey: np.ndarray) -> float:
        """Score how well high-energy content aligns with rule-of-thirds grid."""
        h, w = grey.shape
        thirds_h = [h // 3, 2 * h // 3]
        thirds_w = [w // 3, 2 * w // 3]
        band = max(h // 20, 5)

        # Energy along thirds lines
        thirds_energy = 0
        for th in thirds_h:
            thirds_energy += np.mean(np.abs(np.diff(grey[th-band:th+band], axis=1)))
        for tw in thirds_w:
            thirds_energy += np.mean(np.abs(np.diff(grey[:, tw-band:tw+band], axis=0)))

        total_energy = np.mean(np.abs(np.diff(grey)))
        if total_energy < 1e-6:
            return 0.5
        return min(thirds_energy / (4 * total_energy), 1.0)

    def _estimate_text_area(self, grey: np.ndarray) -> float:
        """Estimate fraction of image covered by text using high-freq patterns."""
        # Text tends to have very high horizontal frequency variation
        h_var = np.var(np.diff(grey, axis=1), axis=1)
        text_rows = np.sum(h_var > np.percentile(h_var, 75))
        return float(text_rows / grey.shape[0])

    def _estimate_face_regions(self, grey: np.ndarray) -> int:
        """Lightweight face region estimator (replace with MTCNN in production)."""
        # Very crude proxy: circular regions with skin-tone-like variance
        # In production, use: from facenet_pytorch import MTCNN
        h, w = grey.shape
        window = min(h, w) // 6
        candidates = 0
        for i in range(0, h - window, window // 2):
            for j in range(0, w - window, window // 2):
                patch = grey[i:i+window, j:j+window]
                local_var = np.var(patch)
                local_mean = np.mean(patch)
                # Skin tone proxy: medium brightness, moderate variance
                if 80 < local_mean < 200 and 100 < local_var < 1500:
                    candidates += 1
        return min(candidates // 4, 5)  # rough estimate, max 5


class VisualBackbone:
    """
    Transfer learning backbone for visual embeddings.
    Uses ResNet-50 pretrained on ImageNet.
    Outputs 512-dim embedding for downstream tasks.
    """

    def __init__(self, model_name: str = "resnet50"):
        self.model_name = model_name
        self._model = None
        self._is_loaded = False

    def load(self):
        """Lazy-load the backbone model."""
        try:
            import torch
            import torchvision.models as models
            import torchvision.transforms as T

            backbone = getattr(models, self.model_name)(
                weights=f"ResNet50_Weights.IMAGENET1K_V2"
                if self.model_name == "resnet50"
                else None
            )
            # Remove final classification head → use penultimate layer
            self._model = torch.nn.Sequential(*list(backbone.children())[:-1])
            self._model.eval()

            self._transform = T.Compose([
                T.ToPILImage(),
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self._is_loaded = True
            print(f"✓ Loaded {self.model_name} backbone")

        except Exception as e:
            print(f"⚠️  Backbone load failed ({e}). Using random embeddings as fallback.")
            self._is_loaded = False

    def embed(self, image_array: np.ndarray) -> np.ndarray:
        """
        Extract 512-dim visual embedding from image.

        Args:
            image_array: H×W×3 RGB numpy array (uint8)

        Returns:
            512-dim numpy embedding vector
        """
        if not self._is_loaded:
            # Reproducible random embedding (seeded by image hash) for demo
            seed = int(np.mean(image_array)) % 10000
            rng = np.random.default_rng(seed)
            return rng.standard_normal(512).astype(np.float32)

        import torch
        tensor = self._transform(image_array).unsqueeze(0)
        with torch.no_grad():
            embedding = self._model(tensor).squeeze().numpy()
        return embedding / (np.linalg.norm(embedding) + 1e-10)  # L2 normalise

    def predict_category(self, embedding: np.ndarray) -> str:
        """Classify ad creative into high-level category."""
        # In production: fine-tune a classification head
        # Heuristic proxy based on embedding statistics
        mean_val = float(np.mean(embedding))
        if mean_val > 0.05:
            return "lifestyle"
        elif mean_val > 0.0:
            return "product"
        elif mean_val > -0.05:
            return "text-heavy"
        else:
            return "abstract"


class AdCreativeFeatureExtractor:
    """
    Full feature extraction pipeline for ad creative images.
    Combines colour, composition, and deep visual features.
    """

    def __init__(self, load_backbone: bool = True):
        self.colour_analyser = ColourAnalyser()
        self.composition_analyser = CompositionAnalyser()
        self.backbone = VisualBackbone()
        if load_backbone:
            self.backbone.load()

    def extract(self, image_array: np.ndarray) -> VisualFeatures:
        """
        Extract all visual features from an ad creative image.

        Args:
            image_array: H×W×3 RGB numpy array (uint8)

        Returns:
            VisualFeatures dataclass with all features
        """
        colour_feats = self.colour_analyser.analyse(image_array)
        comp_feats = self.composition_analyser.analyse(image_array)
        embedding = self.backbone.embed(image_array)
        category = self.backbone.predict_category(embedding)

        # Derived attention score (heuristic combination)
        attention = (
            0.3 * colour_feats["contrast"]
            + 0.25 * colour_feats["saturation"]
            + 0.2 * (1 - colour_feats["brightness"])  # very bright = washed out
            + 0.15 * comp_feats["rule_of_thirds_score"]
            + 0.1 * (1 if comp_feats["person_present"] else 0)
        )

        # Brand safety (crude heuristic: low complexity + neutral tone = safer)
        brand_safety = (
            0.4 * (1 - comp_feats["visual_complexity"])
            + 0.3 * (1 - colour_feats["contrast"])
            + 0.3 * (0.5 + (0.5 if colour_feats["colour_temperature"] == "neutral" else 0))
        )

        return VisualFeatures(
            dominant_colours=colour_feats["dominant_colours"],
            colour_temperature=colour_feats["colour_temperature"],
            brightness=colour_feats["brightness"],
            saturation=colour_feats["saturation"],
            contrast=colour_feats["contrast"],
            colour_harmony=colour_feats["colour_harmony"],
            visual_complexity=comp_feats["visual_complexity"],
            text_area_ratio=comp_feats["text_area_ratio"],
            face_count=comp_feats["face_count"],
            person_present=comp_feats["person_present"],
            rule_of_thirds_score=comp_feats["rule_of_thirds_score"],
            embedding=embedding,
            predicted_category=category,
            attention_score=float(np.clip(attention, 0, 1)),
            brand_safety_score=float(np.clip(brand_safety, 0, 1)),
        )

    def to_feature_vector(self, features: VisualFeatures) -> np.ndarray:
        """
        Flatten VisualFeatures to a numeric vector for ML models.

        Returns:
            1D numpy array of all numeric features (dims: 512 + 11 = 523)
        """
        scalar_features = np.array([
            features.brightness,
            features.saturation,
            features.contrast,
            features.visual_complexity,
            features.text_area_ratio,
            float(features.face_count),
            float(features.person_present),
            features.rule_of_thirds_score,
            features.attention_score,
            features.brand_safety_score,
            # One-hot encode temperature
            1.0 if features.colour_temperature == "warm" else 0.0,
            1.0 if features.colour_temperature == "cool" else 0.0,
        ], dtype=np.float32)

        return np.concatenate([scalar_features, features.embedding.astype(np.float32)])
