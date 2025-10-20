"""
Multi-Model Stacked Ensemble

Stacking ensemble with multiple ML algorithms for robust predictions.
Based on Two Sigma/Kaggle winning approaches.

Architecture:
- Level 1: 5 diverse base models (XGBoost, LightGBM, RF, LogReg, SVM)
- Level 2: Meta-learner trained on out-of-fold predictions
- Out-of-fold methodology prevents overfitting
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import KFold
import warnings

# Optional imports for gradient boosting
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    warnings.warn("XGBoost not available - using RandomForest substitute")

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    warnings.warn("LightGBM not available - using RandomForest substitute")


class StackedMLEnsemble:
    """
    Stacked ensemble with multiple base models and meta-learner.

    Uses out-of-fold predictions to avoid overfitting.
    Based on Two Sigma/Kaggle-winning approaches.

    Features:
    - 5 diverse base models with different learning paradigms
    - Out-of-fold predictions for meta-training
    - Probability-based stacking for richer information
    - Model weight attribution
    - Robust to individual model failures

    Example:
        >>> ensemble = StackedMLEnsemble(n_folds=5)
        >>> ensemble.fit(X_train, y_train)
        >>> predictions = ensemble.predict(X_test)
        >>> probabilities = ensemble.predict_proba(X_test)
        >>> weights = ensemble.get_model_weights()
    """

    def __init__(
        self,
        n_folds: int = 5,
        use_probabilities: bool = True,
        random_state: int = 42
    ):
        """
        Initialize stacked ensemble.

        Args:
            n_folds: Number of folds for out-of-fold predictions (default: 5)
            use_probabilities: Use probabilities vs hard predictions (default: True)
            random_state: Random seed for reproducibility (default: 42)
        """
        self.n_folds = n_folds
        self.use_probabilities = use_probabilities
        self.random_state = random_state

        # Level 1: Base models
        self.base_models = self._initialize_base_models()

        # Level 2: Meta-learner
        self.meta_learner = LogisticRegression(
            max_iter=1000,
            random_state=random_state,
            solver='lbfgs'
        )

        self.is_fitted = False
        self.n_classes_ = None

    def _initialize_base_models(self) -> Dict[str, Any]:
        """Initialize base model library."""
        models = {}

        # Model 1: XGBoost (gradient boosting - tree-based)
        if HAS_XGBOOST:
            models['xgboost'] = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                eval_metric='logloss',
                use_label_encoder=False
            )
        else:
            models['xgboost_substitute'] = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state
            )

        # Model 2: LightGBM (gradient boosting - leaf-wise)
        if HAS_LIGHTGBM:
            models['lightgbm'] = LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                verbose=-1,
                force_col_wise=True
            )
        else:
            models['lightgbm_substitute'] = RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                random_state=self.random_state + 1
            )

        # Model 3: Random Forest (bagging - tree ensemble)
        models['random_forest'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=self.random_state
        )

        # Model 4: Logistic Regression (linear - probabilistic)
        models['logistic'] = LogisticRegression(
            max_iter=1000,
            random_state=self.random_state,
            solver='lbfgs'
        )

        # Model 5: SVM (kernel - margin-based)
        models['svm'] = SVC(
            probability=True,
            random_state=self.random_state,
            kernel='rbf',
            C=1.0
        )

        return models

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        verbose: bool = True
    ):
        """
        Fit stacked ensemble using out-of-fold predictions.

        Args:
            X: Feature matrix
            y: Target labels
            verbose: Print progress information
        """
        if verbose:
            print("=" * 80)
            print("TRAINING STACKED ML ENSEMBLE")
            print("=" * 80)

        # Store number of classes
        self.n_classes_ = len(np.unique(y))

        # 1. Generate out-of-fold predictions from base models
        if verbose:
            print(f"\n[1/3] Generating out-of-fold predictions ({self.n_folds} folds)...")
        oof_predictions = self._generate_oof_predictions(X, y, verbose)

        # 2. Train meta-learner on OOF predictions
        if verbose:
            print("\n[2/3] Training meta-learner...")
        self.meta_learner.fit(oof_predictions, y)

        # 3. Retrain base models on full data
        if verbose:
            print("\n[3/3] Retraining base models on full data...")
        for name, model in self.base_models.items():
            if verbose:
                print(f"  • Training {name}...")
            model.fit(X, y)

        self.is_fitted = True

        if verbose:
            print("\n" + "=" * 80)
            print("✅ STACKED ENSEMBLE TRAINED SUCCESSFULLY")
            print("=" * 80)

            # Show model weights
            weights = self.get_model_weights()
            print("\n📊 MODEL WEIGHTS:")
            for model_name, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
                print(f"  {model_name:20s}: {weight:6.2%}")

    def _generate_oof_predictions(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Generate out-of-fold predictions for all base models.

        Returns:
            DataFrame with OOF predictions for each base model
        """
        n_samples = len(X)
        n_models = len(self.base_models)

        # Initialize OOF prediction arrays
        if self.use_probabilities:
            oof_preds = np.zeros((n_samples, n_models * self.n_classes_))
        else:
            oof_preds = np.zeros((n_samples, n_models))

        # K-Fold cross-validation
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        for model_idx, (name, model) in enumerate(self.base_models.items()):
            if verbose:
                print(f"\n  {name}:")

            for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
                if verbose:
                    print(f"    Fold {fold + 1}/{self.n_folds}...", end=' ')

                # Split data
                X_train_fold = X.iloc[train_idx]
                y_train_fold = y.iloc[train_idx]
                X_val_fold = X.iloc[val_idx]

                # Clone and train model on fold
                model_clone = self._clone_model(model)
                model_clone.fit(X_train_fold, y_train_fold)

                # Predict on validation fold
                if self.use_probabilities:
                    preds = model_clone.predict_proba(X_val_fold)
                    start_col = model_idx * self.n_classes_
                    end_col = start_col + self.n_classes_
                    oof_preds[val_idx, start_col:end_col] = preds
                else:
                    preds = model_clone.predict(X_val_fold)
                    oof_preds[val_idx, model_idx] = preds

                if verbose:
                    print("✓")

        # Convert to DataFrame with meaningful column names
        if self.use_probabilities:
            columns = []
            for model_name in self.base_models.keys():
                for class_idx in range(self.n_classes_):
                    columns.append(f"{model_name}_class{class_idx}")
        else:
            columns = list(self.base_models.keys())

        return pd.DataFrame(oof_preds, columns=columns, index=X.index)

    def _clone_model(self, model: Any) -> Any:
        """
        Clone a model with same parameters.

        Args:
            model: Model to clone

        Returns:
            New model instance with same parameters
        """
        return model.__class__(**model.get_params())

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict using stacked ensemble.

        Args:
            X: Feature matrix

        Returns:
            Array of predictions
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")

        # 1. Get predictions from base models
        base_predictions = self._get_base_predictions(X)

        # 2. Meta-learner predicts on base model outputs
        final_predictions = self.meta_learner.predict(base_predictions)

        return final_predictions

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities using stacked ensemble.

        Args:
            X: Feature matrix

        Returns:
            Array of probability predictions
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")

        # 1. Get predictions from base models
        base_predictions = self._get_base_predictions(X)

        # 2. Meta-learner predicts probabilities
        final_probabilities = self.meta_learner.predict_proba(base_predictions)

        return final_probabilities

    def _get_base_predictions(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Get predictions from all base models.

        Args:
            X: Feature matrix

        Returns:
            DataFrame with predictions from each base model
        """
        n_samples = len(X)
        n_models = len(self.base_models)

        if self.use_probabilities:
            predictions = np.zeros((n_samples, n_models * self.n_classes_))
        else:
            predictions = np.zeros((n_samples, n_models))

        for idx, (name, model) in enumerate(self.base_models.items()):
            if self.use_probabilities:
                preds = model.predict_proba(X)
                start_col = idx * self.n_classes_
                end_col = start_col + self.n_classes_
                predictions[:, start_col:end_col] = preds
            else:
                preds = model.predict(X)
                predictions[:, idx] = preds

        # Convert to DataFrame with column names matching OOF predictions
        if self.use_probabilities:
            columns = []
            for model_name in self.base_models.keys():
                for class_idx in range(self.n_classes_):
                    columns.append(f"{model_name}_class{class_idx}")
        else:
            columns = list(self.base_models.keys())

        return pd.DataFrame(predictions, columns=columns, index=X.index)

    def get_model_weights(self) -> Dict[str, float]:
        """
        Get relative importance of each base model.

        Based on meta-learner coefficients.

        Returns:
            Dictionary mapping model name to weight
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before getting weights")

        # For logistic regression, coefficients indicate importance
        if hasattr(self.meta_learner, 'coef_'):
            coefs = np.abs(self.meta_learner.coef_).mean(axis=0)

            # Group by base model
            n_models = len(self.base_models)

            model_weights = {}
            for idx, name in enumerate(self.base_models.keys()):
                if self.use_probabilities:
                    start_idx = idx * self.n_classes_
                    end_idx = start_idx + self.n_classes_
                    weight = coefs[start_idx:end_idx].mean()
                else:
                    weight = coefs[idx]

                model_weights[name] = float(weight)

            # Normalize to sum to 1
            total = sum(model_weights.values())
            if total > 0:
                model_weights = {k: v/total for k, v in model_weights.items()}

            return model_weights

        # Fallback: equal weights
        return {name: 1/len(self.base_models) for name in self.base_models.keys()}

    def get_model_count(self) -> int:
        """Get number of base models."""
        return len(self.base_models)

    def get_model_names(self) -> List[str]:
        """Get names of all base models."""
        return list(self.base_models.keys())
