"""
Detection module for Parkinson's patient state detection.

Uses a trained RandomForestClassifier loaded from parkinsons_model_acc_gyro.pkl.
Falls back to a rule-based detector if the model file is unavailable.
"""

import logging
import os
import numpy as np
import joblib
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, List
from dataclasses import dataclass

from config import settings
from schemas import PatientState

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Container for detection output."""
    state: PatientState
    confidence: float
    explanation: str
    debug_info: Dict[str, Any]  # For development/debugging


class BaseDetector(ABC):
    """Abstract base class for state detectors."""

    @abstractmethod
    def detect(self, sensor_stats: Dict[str, float]) -> DetectionResult:
        """
        Detect patient state from sensor statistics.

        Args:
            sensor_stats: Dictionary of computed sensor statistics

        Returns:
            DetectionResult with state, confidence, and explanation
        """
        pass

    @property
    @abstractmethod
    def is_ready(self) -> bool:
        """Check if detector is ready for detection."""
        pass


class RuleBasedDetector(BaseDetector):
    """
    Placeholder rule-based detector for development.

    This detector uses simplified clinical heuristics based on
    Parkinson's disease motor symptoms research:

    - OFF state: Typically shows increased tremor (4-7Hz oscillations),
      reduced overall movement, and more irregular motion patterns
    - ON state: More controlled movement, lower tremor amplitude,
      smoother motion transitions

    These rules are ILLUSTRATIVE and should NOT be used for clinical
    decisions. Replace with validated ML model for production.
    """

    def __init__(self):
        self._ready = True

    @property
    def is_ready(self) -> bool:
        return self._ready

    def detect(self, sensor_stats: Dict[str, float]) -> DetectionResult:
        """
        Apply rule-based heuristics for state detection.

        The logic considers:
        1. Movement intensity - very low may indicate OFF bradykinesia
        2. Tremor indicators - high ZCR in tremor range suggests OFF
        3. Movement variability - erratic patterns may indicate OFF
        """
        debug_info = {}
        scores = {'ON': 0.0, 'OFF': 0.0, 'UNKNOWN': 0.0}
        explanations = []

        # Extract key features
        movement_intensity = sensor_stats.get('movement_intensity', 0.5)
        avg_zcr = sensor_stats.get('avg_zcr', 0)
        std_magnitude = sensor_stats.get('std_magnitude', 0)
        mean_activity = sensor_stats.get('mean_activity', 0)

        debug_info['movement_intensity'] = movement_intensity
        debug_info['avg_zcr'] = avg_zcr
        debug_info['std_magnitude'] = std_magnitude

        # Rule 1: Movement Intensity Analysis
        # Very low movement may indicate OFF-state bradykinesia
        if movement_intensity < settings.low_activity_threshold:
            scores['OFF'] += 0.3
            scores['UNKNOWN'] += 0.1
            explanations.append("Low movement intensity detected (possible bradykinesia)")
            debug_info['intensity_contribution'] = 'OFF'
        elif movement_intensity > 0.7:
            scores['ON'] += 0.2
            explanations.append("Normal movement intensity observed")
            debug_info['intensity_contribution'] = 'ON'
        else:
            scores['ON'] += 0.1
            scores['OFF'] += 0.1
            debug_info['intensity_contribution'] = 'neutral'

        # Rule 2: Tremor Analysis (using zero-crossing rate as proxy)
        # PD tremor typically 4-7 Hz, which corresponds to 8-14 zero-crossings/sec
        tremor_zcr_min = settings.tremor_frequency_min * 2  # ~8 ZC/s
        tremor_zcr_max = settings.tremor_frequency_max * 2  # ~14 ZC/s

        if tremor_zcr_min <= avg_zcr <= tremor_zcr_max:
            # ZCR in typical tremor range
            tremor_score = 0.35
            scores['OFF'] += tremor_score
            explanations.append("Oscillatory patterns in tremor frequency range detected")
            debug_info['tremor_detected'] = True
        elif avg_zcr > tremor_zcr_max:
            # High frequency - might be intentional movement or noise
            scores['ON'] += 0.15
            scores['UNKNOWN'] += 0.1
            explanations.append("High-frequency motion detected (likely intentional movement)")
            debug_info['tremor_detected'] = False
        else:
            # Low ZCR - smooth movements
            scores['ON'] += 0.25
            explanations.append("Smooth movement patterns detected")
            debug_info['tremor_detected'] = False

        # Rule 3: Movement Variability
        # OFF state often shows more erratic, less controlled movements
        variability_threshold = 0.5  # Normalized threshold
        normalized_variability = min(std_magnitude / 2.0, 1.0)

        if normalized_variability > variability_threshold:
            scores['OFF'] += 0.2
            explanations.append("High movement variability observed")
            debug_info['high_variability'] = True
        else:
            scores['ON'] += 0.2
            explanations.append("Consistent movement patterns")
            debug_info['high_variability'] = False

        # Rule 4: Data Quality Check
        # If data seems unreliable, increase UNKNOWN score
        sample_count = sensor_stats.get('sample_count', 0)
        if sample_count < settings.min_samples * 1.2:
            scores['UNKNOWN'] += 0.15
            explanations.append("Limited data available for confident assessment")

        # Normalize scores to get confidence-like values
        total_score = sum(scores.values())
        if total_score > 0:
            for key in scores:
                scores[key] /= total_score

        debug_info['normalized_scores'] = scores.copy()

        # Determine final state
        max_state = max(scores, key=scores.get)
        max_confidence = scores[max_state]

        # Apply confidence threshold
        if max_confidence < settings.confidence_threshold:
            final_state = PatientState.UNKNOWN
            final_confidence = max_confidence
            explanations.append(
                f"Confidence ({max_confidence:.0%}) below threshold - returning UNKNOWN"
            )
        else:
            final_state = PatientState(max_state)
            final_confidence = max_confidence

        # Build explanation string
        explanation = self._build_explanation(final_state, explanations, sensor_stats)

        return DetectionResult(
            state=final_state,
            confidence=round(final_confidence, 3),
            explanation=explanation,
            debug_info=debug_info
        )

    def _build_explanation(
        self,
        state: PatientState,
        factors: list,
        stats: dict
    ) -> str:
        """Build human-readable explanation of the detection."""

        state_descriptions = {
            PatientState.ON: "Movement patterns indicate controlled motor function consistent with medication effectiveness (ON state).",
            PatientState.OFF: "Movement patterns suggest reduced medication effectiveness with possible motor symptoms (OFF state).",
            PatientState.UNKNOWN: "Insufficient evidence for confident state determination."
        }

        base = state_descriptions[state]

        # Add key observations
        key_observations = [f for f in factors if f][:3]  # Limit to top 3
        if key_observations:
            observations_str = " ".join(key_observations)
            return f"{base} Key observations: {observations_str}"

        return base


class MLDetector(BaseDetector):
    """
    ML detector using a trained RandomForestClassifier from a .pkl bundle.

    Supports acc-only (103 features) and acc+gyro (206 features) models.
    The feature vector is: raw window features + baseline z-score deviations.
    """

    CLASS_MAP = {0: PatientState.OFF, 1: PatientState.ON}

    def __init__(self, model_path: str):
        self._ready = False
        self.model = None
        self.scaler = None
        self.feature_names: List[str] = []
        self.baseline_stats: Dict = {}
        self.config: Dict = {}
        self._n_raw_features = 0

        try:
            # Handle legacy pickle files that reference __main__.SystemConfig
            import __main__
            if not hasattr(__main__, 'SystemConfig'):
                class _SystemConfig:
                    pass
                __main__.SystemConfig = _SystemConfig

            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.baseline_stats = model_data.get('baseline_stats', {})
            raw_config = model_data.get('config', {})
            self.config = raw_config if isinstance(raw_config, dict) else vars(raw_config) if hasattr(raw_config, '__dict__') else {}

            # Determine how many raw features vs z-score features
            zscore_names = [f for f in self.feature_names if f.endswith('_zscore')]
            self._n_raw_features = len(self.feature_names) - len(zscore_names)

            self._ready = True
            logger.info(
                f"ML model loaded: {type(self.model).__name__}, "
                f"{self.model.n_features_in_} features ({self._n_raw_features} raw + {len(zscore_names)} zscore), "
                f"classes={list(self.model.classes_)}"
            )
        except Exception as e:
            logger.error(f"Failed to load ML model from {model_path}: {e}")

    @property
    def is_ready(self) -> bool:
        return self._ready

    def _get_baseline(self) -> Dict:
        """Return baseline stats dict.

        Handles two formats:
          - Flat: {feature_name: {mean, std, median}}
          - Nested by patient: {patient_id: {stats: {feature_name: ...}}}
        """
        if not self.baseline_stats:
            return {}
        first_key = next(iter(self.baseline_stats))
        first_val = self.baseline_stats[first_key]
        # Flat format: value is {mean, std, median}
        if isinstance(first_val, dict) and 'mean' in first_val:
            return self.baseline_stats
        # Nested format: value is {stats: {...}}
        if isinstance(first_val, dict) and 'stats' in first_val:
            return first_val['stats']
        return self.baseline_stats

    def _build_feature_vector(self, raw_features: List[float]) -> np.ndarray:
        """Build full feature vector: raw features + baseline z-score deviations."""
        baseline = self._get_baseline()
        raw_names = self.feature_names[:self._n_raw_features]
        zscore_features = []

        for i, fname in enumerate(raw_names):
            if fname not in baseline:
                zscore_features.append(0.0)
                continue
            b_std = baseline[fname].get('std', 0)
            b_mean = baseline[fname].get('mean', 0)
            if b_std == 0 or b_std != b_std:
                zscore_features.append(0.0)
            else:
                zscore_features.append((raw_features[i] - b_mean) / b_std)

        full = list(raw_features) + zscore_features
        return np.array(full, dtype=np.float64).reshape(1, -1)

    def detect(self, sensor_stats: Dict[str, float]) -> DetectionResult:
        """Run ML model detection on sensor data."""
        from utils import extract_window_features

        ax = sensor_stats['_raw_ax']
        ay = sensor_stats['_raw_ay']
        az = sensor_stats['_raw_az']
        gx = sensor_stats.get('_raw_gx')
        gy = sensor_stats.get('_raw_gy')
        gz = sensor_stats.get('_raw_gz')

        # If model needs gyro but CSV had none, use zeros
        has_gyro_features = any(f.startswith('gyro_') for f in self.feature_names)
        if has_gyro_features and gx is None:
            gx = np.zeros_like(ax)
            gy = np.zeros_like(ay)
            gz = np.zeros_like(az)
            logger.warning("Model expects gyro data but CSV has none — using zeros")

        sr = int(self.config.get('sampling_rate', settings.expected_sample_rate_hz))
        tremor_low = self.config.get('tremor_band_low', 4.0)
        tremor_high = self.config.get('tremor_band_high', 6.0)
        voluntary_high = self.config.get('voluntary_band_high', 3.0)
        window_dur = self.config.get('window_duration', 1.0)
        window_overlap = self.config.get('window_overlap', 0.5)
        window_samples = int(sr * window_dur)
        step_samples = int(window_samples * (1 - window_overlap))

        n = len(ax)
        all_window_features = []
        start = 0
        while start + window_samples <= n:
            end = start + window_samples
            wf = extract_window_features(
                ax[start:end], ay[start:end], az[start:end],
                gx[start:end], gy[start:end], gz[start:end],
                sampling_rate=sr,
                tremor_low=tremor_low, tremor_high=tremor_high,
                voluntary_high=voluntary_high,
            )
            all_window_features.append(wf)
            start += step_samples

        if not all_window_features:
            all_window_features.append(
                extract_window_features(
                    ax, ay, az, gx, gy, gz,
                    sampling_rate=sr,
                    tremor_low=tremor_low, tremor_high=tremor_high,
                    voluntary_high=voluntary_high,
                )
            )

        raw_features = np.mean(all_window_features, axis=0).tolist()
        feature_vec = self._build_feature_vector(raw_features)

        scaled = self.scaler.transform(feature_vec)
        detection = int(self.model.predict(scaled)[0])
        proba = self.model.predict_proba(scaled)[0]

        state = self.CLASS_MAP.get(detection, PatientState.UNKNOWN)
        # Confidence = probability of detected class
        det_idx = list(self.model.classes_).index(detection)
        confidence = float(proba[det_idx])

        explanation = (
            f"ML model detected {state.value} state with {confidence:.0%} confidence "
            f"based on {len(all_window_features)} analysis window(s)."
        )

        return DetectionResult(
            state=state,
            confidence=round(confidence, 3),
            explanation=explanation,
            debug_info={
                'model_class': detection,
                'probabilities': {int(c): round(float(p), 3) for c, p in zip(self.model.classes_, proba)},
                'n_windows': len(all_window_features),
                'n_features': feature_vec.shape[1],
            }
        )


# Factory function for detector instantiation
_detector_instance: BaseDetector = None


def get_detector() -> BaseDetector:
    """
    Factory function to get the active detector.

    Tries to load the ML model from settings.model_path.
    Falls back to RuleBasedDetector if the model file is missing or fails to load.
    """
    global _detector_instance

    if _detector_instance is None:
        model_path = os.path.join(os.path.dirname(__file__), settings.model_path)
        if os.path.exists(model_path):
            ml = MLDetector(model_path)
            if ml.is_ready:
                _detector_instance = ml
                logger.info("Using ML detector")
            else:
                logger.warning("ML model failed to load, falling back to rule-based detector")
                _detector_instance = RuleBasedDetector()
        else:
            logger.warning(f"Model file not found at {model_path}, using rule-based detector")
            _detector_instance = RuleBasedDetector()

    return _detector_instance


def detect_patient_state(sensor_stats: Dict[str, float]) -> DetectionResult:
    """
    Convenience function for detecting patient state.

    Args:
        sensor_stats: Dictionary of sensor statistics from utils.compute_sensor_statistics()

    Returns:
        DetectionResult with state, confidence, and explanation
    """
    detector = get_detector()
    return detector.detect(sensor_stats)
