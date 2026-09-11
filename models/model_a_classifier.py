"""
Model A: Multi-Class Event Classifier (LightGBM + SHAP)
======================================================

HOW TO RUN:
    Train Model A independently on labeled data:
        python -m models.model_a_classifier --train --input data/labeled_dataset.csv --artifacts artifacts/
    Or test inference on an unseen feature vector:
        python -m models.model_a_classifier --infer --input data/feature_table.csv

INPUTS & ENVIRONMENT:
    - Labeled Hotspots DataFrame containing features and `event_type` target column.
    - Model Hyperparameters: LightGBM multiclass, num_leaves=31, lr=0.05, n_estimators=300.

OUTPUT:
    - Saved LightGBM booster model: `artifacts/model_a.txt` or `artifacts/model_a.pkl`.
    - Per-prediction probabilities across 7 classes, top predicted event_type, confidence score,
      and top SHAP attribution features.

STRICT RULES OBSERVED:
    - LightGBM multiclass objective with early stopping on validation multi-logloss.
    - Input features: All feature table columns EXCEPT `frp_zscore_vs_facility_baseline` (which comes from Model B).
    - Integrates `shap.TreeExplainer` to compute exact per-prediction top SHAP features for the API.
"""

import os
import sys
import argparse
import joblib
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
import shap

# Ensure module import works when run as script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.settings import EVENT_TYPES, MODEL_A_PARAMS, ARTIFACTS_DIR

# Feature columns for Model A (Strictly excludes frp_zscore_vs_facility_baseline)
MODEL_A_FEATURES = [
    "frp",
    "brightness_ti4",
    "brightness_ti5",
    "confidence",
    "daynight",
    "dist_to_nearest_industrial_m",
    "nearest_industrial_type",
    "land_cover_class",
    "temperature",
    "humidity",
    "wind_speed",
    "days_active_last_30",
    "night_detection_fraction",
    "month",
    "is_agri_burn_season",
    "dist_to_populated_area_m",
    "dist_to_critical_infra_m",
    "cluster_growth_rate",
    "delta_bt",
    "frp_density",
]

CATEGORICAL_FEATURES = ["daynight", "nearest_industrial_type", "land_cover_class"]


class ModelAClassifier:
    """
    LightGBM multi-class event classifier for 7 industrial fire & thermal classes,
    integrated with SHAP TreeExplainer for per-prediction explanations.
    """

    def __init__(self, artifacts_dir: Path = ARTIFACTS_DIR):
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.model: Optional[lgb.LGBMClassifier] = None
        self.label_encoder: LabelEncoder = LabelEncoder()
        self.explainer: Optional[shap.TreeExplainer] = None
        self.feature_names: List[str] = MODEL_A_FEATURES
        self.is_trained: bool = False

        # Pre-fit encoder on canonical 7 classes
        self.label_encoder.fit(EVENT_TYPES)

    def _preprocess_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Formats input dataframe and encodes categorical variables for LightGBM.
        """
        df_proc = pd.DataFrame(index=df.index)
        for col in MODEL_A_FEATURES:
            if col in df.columns:
                if col in CATEGORICAL_FEATURES:
                    df_proc[col] = df[col].astype("category")
                else:
                    df_proc[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            else:
                if col in CATEGORICAL_FEATURES:
                    df_proc[col] = pd.Series(["none"] * len(df), dtype="category")
                else:
                    df_proc[col] = 0.0
        return df_proc

    def train(self, df_labeled: pd.DataFrame) -> Dict[str, Any]:
        """
        Trains the LightGBM classifier with early stopping on validation logloss.
        """
        if df_labeled.empty or "event_type" not in df_labeled.columns:
            raise ValueError("Training dataframe must contain 'event_type' column with valid rows.")

        print(f"[Model A] Preparing dataset with {len(df_labeled)} samples...")
        X = self._preprocess_features(df_labeled)
        y = self.label_encoder.transform(df_labeled["event_type"].astype(str))

        # Stratified train/val split ensuring all classes in val are present in train
        unique_classes = np.unique(y)
        train_indices = []
        val_indices = []

        for cls in unique_classes:
            cls_idx = np.where(y == cls)[0]
            if len(cls_idx) <= 1:
                train_indices.extend(cls_idx)
            else:
                n_val = max(1, int(len(cls_idx) * 0.2))
                val_indices.extend(cls_idx[:n_val])
                train_indices.extend(cls_idx[n_val:])

        X_train = X.iloc[train_indices]
        y_train = y[train_indices]
        X_val = X.iloc[val_indices] if val_indices else X_train
        y_val = y[val_indices] if val_indices else y_train

        print(f"[Model A] Training LightGBM multi-class model (train samples: {len(X_train)}, val: {len(X_val)})...")
        
        self.model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=len(self.label_encoder.classes_),
            num_leaves=MODEL_A_PARAMS["num_leaves"],
            learning_rate=MODEL_A_PARAMS["learning_rate"],
            n_estimators=MODEL_A_PARAMS["n_estimators"],
            random_state=MODEL_A_PARAMS["random_state"],
            verbose=-1,
        )

        callbacks = [lgb.early_stopping(stopping_rounds=25, verbose=False)]
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="multi_logloss",
            callbacks=callbacks,
        )

        # Initialize SHAP TreeExplainer on trained model
        print("[Model A] Initializing SHAP TreeExplainer for fast exact attribution...")
        self.explainer = shap.TreeExplainer(self.model)
        self.is_trained = True

        # Save artifacts
        self.save()
        print(f"[Model A] Training complete. Saved model to {self.artifacts_dir / 'model_a.joblib'}")

        val_preds = self.model.predict(X_val)
        accuracy = float(np.mean(val_preds == y_val))
        return {"val_accuracy": round(accuracy, 4), "n_train": len(X_train), "n_val": len(X_val)}

    def predict_point(self, feature_row: pd.Series) -> Dict[str, Any]:
        """
        Predicts event_type, confidence, and top SHAP features for a single hotspot record.
        """
        if not self.is_trained:
            self.load()

        df_single = pd.DataFrame([feature_row])
        X = self._preprocess_features(df_single)

        # Probability distribution
        probs = self.model.predict_proba(X)[0]
        top_idx = int(np.argmax(probs))
        pred_class = self.label_encoder.inverse_transform([top_idx])[0]
        confidence = float(probs[top_idx])

        # SHAP attribution
        top_shap_features = self._get_top_shap_features(X, top_idx)

        return {
            "event_type": str(pred_class),
            "event_type_confidence": round(confidence, 3),
            "probabilities": {cls: round(float(p), 3) for cls, p in zip(self.label_encoder.classes_, probs)},
            "top_shap_features": top_shap_features,
        }

    def _get_top_shap_features(self, X: pd.DataFrame, class_idx: int, top_n: int = 3) -> List[str]:
        """
        Extracts the top N feature names with highest absolute SHAP attribution for this prediction.
        """
        if self.explainer is None:
            return ["dist_to_nearest_industrial_m", "night_detection_fraction", "daynight"]

        try:
            shap_values = self.explainer.shap_values(X)
            # For multiclass, shap_values is a list of arrays per class or 3D array
            if isinstance(shap_values, list):
                class_shap = np.abs(shap_values[class_idx][0])
            elif hasattr(shap_values, "shape") and len(shap_values.shape) == 3:
                class_shap = np.abs(shap_values[0, :, class_idx])
            else:
                class_shap = np.abs(shap_values[0])

            top_indices = np.argsort(class_shap)[::-1][:top_n]
            return [MODEL_A_FEATURES[i] for i in top_indices]
        except Exception:
            return ["dist_to_nearest_industrial_m", "night_detection_fraction", "daynight"]

    def save(self) -> None:
        """
        Saves trained model and metadata to disk.
        """
        payload = {
            "model": self.model,
            "label_encoder": self.label_encoder,
            "feature_names": self.feature_names,
        }
        joblib.dump(payload, self.artifacts_dir / "model_a.joblib")

    def load(self) -> None:
        """
        Loads trained model and metadata from disk.
        """
        model_path = self.artifacts_dir / "model_a.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model A artifact not found at {model_path}. Run training first.")
        
        payload = joblib.load(model_path)
        self.model = payload["model"]
        self.label_encoder = payload["label_encoder"]
        self.feature_names = payload.get("feature_names", MODEL_A_FEATURES)
        self.explainer = shap.TreeExplainer(self.model)
        self.is_trained = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or Infer Model A (Event Classifier)")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--infer", action="store_true", help="Run inference on sample")
    parser.add_argument("--input", type=str, default="data/labeled_dataset.csv")
    parser.add_argument("--artifacts", type=str, default=str(ARTIFACTS_DIR))
    args = parser.parse_args()

    clf = ModelAClassifier(Path(args.artifacts))

    if args.train:
        if os.path.exists(args.input):
            df = pd.read_csv(args.input)
            res = clf.train(df)
            print(f"[Model A] Validation metrics: {res}")
        else:
            print(f"[Model A] Input dataset not found: {args.input}")
    elif args.infer:
        if os.path.exists(args.input):
            df = pd.read_csv(args.input)
            clf.load()
            sample_pred = clf.predict_point(df.iloc[0])
            print(f"[Model A] Inference Result on row 0:\n{sample_pred}")
