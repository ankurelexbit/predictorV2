"""
Feature Selection Pipeline - Phase 7

Optimizes feature set through:
1. Correlation analysis (remove redundant features)
2. Feature importance (remove low-impact features)
3. Recursive feature elimination (find optimal subset)

Expected impact: -0.005 to -0.010 log loss improvement
Target: Reduce from ~296 to ~200 features
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import logging
import joblib
from xgboost import XGBClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureSelector:
    """Intelligent feature selection to optimize model performance."""
    
    def __init__(
        self,
        correlation_threshold: float = 0.95,
        importance_threshold: float = 0.001,
        target_features: int = 200
    ):
        """
        Initialize feature selector.
        
        Args:
            correlation_threshold: Remove features with correlation > this
            importance_threshold: Remove features with importance < this
            target_features: Target number of features after selection
        """
        self.correlation_threshold = correlation_threshold
        self.importance_threshold = importance_threshold
        self.target_features = target_features
        
        self.selected_features = None
        self.feature_importance = None
        self.correlation_matrix = None
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        method: str = 'all'
    ) -> List[str]:
        """
        Perform feature selection.
        
        Args:
            X: Feature matrix
            y: Target variable
            method: 'correlation', 'importance', 'rfe', or 'all'
        
        Returns:
            List of selected feature names
        """
        logger.info(f"Starting feature selection from {len(X.columns)} features")
        logger.info(f"Target: {self.target_features} features")
        
        selected = list(X.columns)
        
        if method in ['correlation', 'all']:
            selected = self._remove_correlated_features(X[selected])
            logger.info(f"After correlation filter: {len(selected)} features")
        
        if method in ['importance', 'all']:
            selected = self._remove_low_importance_features(
                X[selected], y, selected
            )
            logger.info(f"After importance filter: {len(selected)} features")
        
        if method in ['rfe', 'all']:
            if len(selected) > self.target_features:
                selected = self._recursive_feature_elimination(
                    X[selected], y, selected
                )
                logger.info(f"After RFE: {len(selected)} features")
        
        self.selected_features = selected
        logger.info(f"✅ Final feature count: {len(selected)}")
        
        return selected
    
    def _remove_correlated_features(
        self,
        X: pd.DataFrame
    ) -> List[str]:
        """
        Remove highly correlated features.
        
        Strategy: For each pair with correlation > threshold,
        keep the feature that appears first (arbitrary but consistent).
        
        Args:
            X: Feature matrix
        
        Returns:
            List of features to keep
        """
        logger.info("Analyzing feature correlations...")
        
        # Calculate correlation matrix
        corr_matrix = X.corr().abs()
        self.correlation_matrix = corr_matrix
        
        # Get upper triangle (avoid duplicates)
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features with correlation > threshold
        to_drop = [
            column for column in upper.columns
            if any(upper[column] > self.correlation_threshold)
        ]
        
        logger.info(f"Removing {len(to_drop)} highly correlated features")
        
        # Return features to keep
        return [f for f in X.columns if f not in to_drop]
    
    def _remove_low_importance_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        features: List[str]
    ) -> List[str]:
        """
        Remove features with low importance.
        
        Args:
            X: Feature matrix
            y: Target variable
            features: Current feature list
        
        Returns:
            List of important features
        """
        logger.info("Training model to assess feature importance...")
        
        # Train XGBoost model
        model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        
        model.fit(X, y)
        
        # Get feature importance
        importance = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.feature_importance = importance
        
        # Remove low-importance features
        important_features = importance[
            importance['importance'] > self.importance_threshold
        ]['feature'].tolist()
        
        logger.info(
            f"Removing {len(features) - len(important_features)} "
            f"low-importance features"
        )
        
        return important_features
    
    def _recursive_feature_elimination(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        features: List[str]
    ) -> List[str]:
        """
        Use RFE to find optimal feature subset.
        
        Args:
            X: Feature matrix
            y: Target variable
            features: Current feature list
        
        Returns:
            Optimal feature subset
        """
        logger.info("Performing recursive feature elimination...")
        
        # Use RFECV with cross-validation
        estimator = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        
        # Calculate step size (remove 10% at a time)
        step = max(1, int(len(features) * 0.1))
        
        selector = RFECV(
            estimator=estimator,
            step=step,
            cv=StratifiedKFold(5),
            scoring='neg_log_loss',
            n_jobs=-1,
            verbose=1
        )
        
        selector.fit(X, y)
        
        # Get selected features
        selected = [f for f, selected in zip(features, selector.support_) if selected]
        
        logger.info(f"RFE selected {len(selected)} optimal features")
        logger.info(f"Optimal CV score: {selector.cv_results_['mean_test_score'].max():.4f}")
        
        return selected
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform feature matrix to selected features.
        
        Args:
            X: Feature matrix
        
        Returns:
            Transformed feature matrix
        """
        if self.selected_features is None:
            raise ValueError("Must call fit() before transform()")
        
        return X[self.selected_features]
    
    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        method: str = 'all'
    ) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            X: Feature matrix
            y: Target variable
            method: Selection method
        
        Returns:
            Transformed feature matrix
        """
        self.fit(X, y, method)
        return self.transform(X)
    
    def save(self, filepath: str):
        """Save selected features to file."""
        if self.selected_features is None:
            raise ValueError("No features selected yet")
        
        data = {
            'selected_features': self.selected_features,
            'feature_importance': self.feature_importance.to_dict() if self.feature_importance is not None else None,
            'correlation_threshold': self.correlation_threshold,
            'importance_threshold': self.importance_threshold
        }
        
        joblib.dump(data, filepath)
        logger.info(f"Saved feature selection to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'FeatureSelector':
        """Load feature selector from file."""
        data = joblib.load(filepath)
        
        selector = cls(
            correlation_threshold=data['correlation_threshold'],
            importance_threshold=data['importance_threshold']
        )
        
        selector.selected_features = data['selected_features']
        if data['feature_importance'] is not None:
            selector.feature_importance = pd.DataFrame(data['feature_importance'])
        
        logger.info(f"Loaded feature selector with {len(selector.selected_features)} features")
        
        return selector
    
    def get_feature_importance_report(self) -> pd.DataFrame:
        """Get feature importance report."""
        if self.feature_importance is None:
            raise ValueError("No feature importance available")
        
        return self.feature_importance.copy()
    
    def get_correlation_report(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get top correlated feature pairs.
        
        Args:
            top_n: Number of top pairs to return
        
        Returns:
            DataFrame with correlated pairs
        """
        if self.correlation_matrix is None:
            raise ValueError("No correlation matrix available")
        
        # Get upper triangle
        upper = self.correlation_matrix.where(
            np.triu(np.ones(self.correlation_matrix.shape), k=1).astype(bool)
        )
        
        # Find top correlations
        correlations = []
        for col in upper.columns:
            for idx in upper.index:
                if pd.notna(upper.loc[idx, col]):
                    correlations.append({
                        'feature_1': idx,
                        'feature_2': col,
                        'correlation': upper.loc[idx, col]
                    })
        
        df = pd.DataFrame(correlations).sort_values('correlation', ascending=False)
        
        return df.head(top_n)


def main():
    """Example usage of feature selector."""
    import sys
    from pathlib import Path
    
    # Add parent to path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    # Load training data
    logger.info("Loading training data...")
    data_path = Path(__file__).parent.parent / 'data' / 'processed' / 'training_data.csv'
    
    if not data_path.exists():
        logger.error(f"Training data not found: {data_path}")
        logger.info("Please run regenerate_training_data.py first")
        return
    
    df = pd.read_csv(data_path)
    
    # Separate features and target
    target_col = 'result'
    feature_cols = [c for c in df.columns if c not in [target_col, 'fixture_id', 'date']]
    
    X = df[feature_cols]
    y = df[target_col]
    
    logger.info(f"Loaded {len(df)} samples with {len(feature_cols)} features")
    
    # Perform feature selection
    selector = FeatureSelector(
        correlation_threshold=0.95,
        importance_threshold=0.001,
        target_features=200
    )
    
    X_selected = selector.fit_transform(X, y, method='all')
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Feature Selection Complete")
    logger.info(f"{'='*60}")
    logger.info(f"Original features: {len(feature_cols)}")
    logger.info(f"Selected features: {len(selector.selected_features)}")
    logger.info(f"Reduction: {len(feature_cols) - len(selector.selected_features)} features")
    
    # Save selected features
    output_path = Path(__file__).parent.parent / 'models' / 'feature_selector.pkl'
    output_path.parent.mkdir(exist_ok=True)
    selector.save(output_path)
    
    # Save feature importance report
    importance_report = selector.get_feature_importance_report()
    report_path = Path(__file__).parent.parent / 'models' / 'feature_importance.csv'
    importance_report.to_csv(report_path, index=False)
    logger.info(f"Saved feature importance to {report_path}")
    
    # Show top 20 features
    logger.info(f"\n{'='*60}")
    logger.info("Top 20 Most Important Features:")
    logger.info(f"{'='*60}")
    for idx, row in importance_report.head(20).iterrows():
        logger.info(f"{row['feature']:40s} {row['importance']:.6f}")


if __name__ == '__main__':
    main()
