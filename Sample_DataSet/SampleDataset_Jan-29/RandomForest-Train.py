"""
aCare Parkinson's Disease OFF State Detection using Random Forest
==================================================================

This module implements a high-performance Random Forest classifier for detecting
ON/OFF states in Parkinson's disease patients using wearable sensor data.

Based on research by Aich et al. (2020) achieving 96.72% accuracy.
DOI: 10.3390/diagnostics10060421

Author: Mohammad - aCare Development Team
Date: January 2026
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             roc_auc_score, roc_curve)
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import os
warnings.filterwarnings('ignore')

class ParkinsonsOFFDetector:
    """
    Main class for training and evaluating Random Forest models for Parkinson's OFF detection.
    
    The model distinguishes between:
    - ON state (label=1): Patient has good motor function, medication is effective
    - OFF state (label=0): Patient has reduced motor function, medication has worn off
    
    Attributes:
        model (RandomForestClassifier): The trained Random Forest model
        scaler (StandardScaler): Feature normalization scaler
        feature_names (list): Names of features used in training
        feature_importance (dict): Importance scores for each feature
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the OFF detector with optimized Random Forest parameters.
        
        Parameters based on research achieving 96.72% accuracy:
        - n_estimators=500: Number of decision trees
        - max_depth=8: Maximum tree depth to prevent overfitting
        - min_samples_split=8: Minimum samples to split node
        - min_samples_leaf=10: Minimum samples in leaf node
        
        Args:
            random_state (int): Random seed for reproducibility
        """
        # Random Forest with optimized hyperparameters from research
        self.model = RandomForestClassifier(
            n_estimators=500,           # 500 trees for robust predictions
            criterion='gini',           # Gini impurity for splits
            max_depth=8,                # Limit depth to prevent overfitting
            min_samples_split=8,        # Minimum samples to split
            min_samples_leaf=10,        # Minimum samples in leaf
            max_features='sqrt',        # Use sqrt(n_features) for each split
            bootstrap=True,             # Bootstrap sampling
            n_jobs=-1,                  # Use all CPU cores
            random_state=random_state,
            verbose=0,
            class_weight='balanced'     # Handle class imbalance
        )
        
        # Scaler for feature normalization
        self.scaler = StandardScaler()
        
        # Storage for model metadata
        self.feature_names = None
        self.feature_importance = None
        self.training_history = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': [],
            'roc_auc': []
        }
    
    def train(self, X_train, y_train, feature_names=None):
        """
        Train the Random Forest model on labeled data.
        
        Args:
            X_train (np.array): Training features, shape (n_samples, n_features)
            y_train (np.array): Training labels, shape (n_samples,)
                               0 = OFF state, 1 = ON state
            feature_names (list): Optional list of feature names
        
        Returns:
            dict: Training metrics (accuracy, precision, recall, f1_score)
        """
        print("=" * 80)
        print("TRAINING RANDOM FOREST MODEL FOR PARKINSON'S OFF DETECTION")
        print("=" * 80)
        
        # Store feature names
        if feature_names is not None:
            self.feature_names = feature_names
        else:
            self.feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
        
        # Normalize features
        print(f"\n[1/5] Normalizing {X_train.shape[1]} features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train Random Forest
        print(f"[2/5] Training Random Forest with {self.model.n_estimators} trees...")
        print(f"      Training samples: {X_train.shape[0]}")
        print(f"      ON state samples: {np.sum(y_train == 1)}")
        print(f"      OFF state samples: {np.sum(y_train == 0)}")
        
        self.model.fit(X_train_scaled, y_train)
        
        # Calculate training predictions
        print("[3/5] Evaluating training performance...")
        y_train_pred = self.model.predict(X_train_scaled)
        y_train_proba = self.model.predict_proba(X_train_scaled)[:, 1]
        
        # Calculate metrics
        train_metrics = {
            'accuracy': accuracy_score(y_train, y_train_pred),
            'precision': precision_score(y_train, y_train_pred, average='weighted'),
            'recall': recall_score(y_train, y_train_pred, average='weighted'),
            'f1_score': f1_score(y_train, y_train_pred, average='weighted'),
            'roc_auc': roc_auc_score(y_train, y_train_proba)
        }
        
        # Store feature importance
        print("[4/5] Calculating feature importance...")
        self.feature_importance = dict(zip(
            self.feature_names,
            self.model.feature_importances_
        ))
        
        # Sort features by importance
        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        print("[5/5] Training completed!\n")
        print("=" * 80)
        print("TRAINING RESULTS")
        print("=" * 80)
        print(f"Accuracy:  {train_metrics['accuracy']:.4f} ({train_metrics['accuracy']*100:.2f}%)")
        print(f"Precision: {train_metrics['precision']:.4f} ({train_metrics['precision']*100:.2f}%)")
        print(f"Recall:    {train_metrics['recall']:.4f} ({train_metrics['recall']*100:.2f}%)")
        print(f"F1-Score:  {train_metrics['f1_score']:.4f} ({train_metrics['f1_score']*100:.2f}%)")
        print(f"ROC-AUC:   {train_metrics['roc_auc']:.4f}")
        
        print("\n" + "=" * 80)
        print("TOP 10 MOST IMPORTANT FEATURES")
        print("=" * 80)
        for i, (feature, importance) in enumerate(sorted_features[:10], 1):
            print(f"{i:2d}. {feature:40s} : {importance:.6f} ({importance*100:.2f}%)")
        
        return train_metrics
    
    def evaluate(self, X_test, y_test, output_dir='../results'):
        """
        Evaluate the model on test data and generate comprehensive metrics.
        
        Args:
            X_test (np.array): Test features
            y_test (np.array): Test labels
            output_dir (str): Directory to save evaluation results
        
        Returns:
            dict: Test metrics and predictions
        """
        print("\n" + "=" * 80)
        print("EVALUATING ON TEST DATA")
        print("=" * 80)

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Normalize test features
        X_test_scaled = self.scaler.transform(X_test)
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        y_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        test_metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, 
                                                          target_names=['OFF', 'ON'])
        }
        
        print("\nTest Set Performance:")
        print("=" * 80)
        print(f"Accuracy:  {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
        print(f"Precision: {test_metrics['precision']:.4f} ({test_metrics['precision']*100:.2f}%)")
        print(f"Recall:    {test_metrics['recall']:.4f} ({test_metrics['recall']*100:.2f}%)")
        print(f"F1-Score:  {test_metrics['f1_score']:.4f} ({test_metrics['f1_score']*100:.2f}%)")
        print(f"ROC-AUC:   {test_metrics['roc_auc']:.4f}")
        
        print("\nDetailed Classification Report:")
        print("=" * 80)
        print(test_metrics['classification_report'])
        
        # Generate visualizations
        self._plot_confusion_matrix(test_metrics['confusion_matrix'], output_dir)
        self._plot_roc_curve(y_test, y_proba, test_metrics['roc_auc'], output_dir)
        self._plot_feature_importance(output_dir)
        
        return test_metrics
    
    def cross_validate(self, X, y, cv_folds=5):
        """
        Perform k-fold cross-validation to assess model generalization.
        
        Args:
            X (np.array): Full feature dataset
            y (np.array): Full labels
            cv_folds (int): Number of cross-validation folds
        
        Returns:
            dict: Cross-validation results
        """
        print("\n" + "=" * 80)
        print(f"PERFORMING {cv_folds}-FOLD CROSS-VALIDATION")
        print("=" * 80)
        
        X_scaled = self.scaler.fit_transform(X)
        
        # Calculate cross-validation scores
        cv_accuracy = cross_val_score(self.model, X_scaled, y, cv=cv_folds, 
                                     scoring='accuracy', n_jobs=-1)
        cv_precision = cross_val_score(self.model, X_scaled, y, cv=cv_folds, 
                                      scoring='precision_weighted', n_jobs=-1)
        cv_recall = cross_val_score(self.model, X_scaled, y, cv=cv_folds, 
                                   scoring='recall_weighted', n_jobs=-1)
        cv_f1 = cross_val_score(self.model, X_scaled, y, cv=cv_folds, 
                               scoring='f1_weighted', n_jobs=-1)
        
        cv_results = {
            'accuracy_mean': cv_accuracy.mean(),
            'accuracy_std': cv_accuracy.std(),
            'precision_mean': cv_precision.mean(),
            'precision_std': cv_precision.std(),
            'recall_mean': cv_recall.mean(),
            'recall_std': cv_recall.std(),
            'f1_mean': cv_f1.mean(),
            'f1_std': cv_f1.std()
        }
        
        print(f"\nCross-Validation Results ({cv_folds} folds):")
        print("=" * 80)
        print(f"Accuracy:  {cv_results['accuracy_mean']:.4f} ± {cv_results['accuracy_std']:.4f}")
        print(f"Precision: {cv_results['precision_mean']:.4f} ± {cv_results['precision_std']:.4f}")
        print(f"Recall:    {cv_results['recall_mean']:.4f} ± {cv_results['recall_std']:.4f}")
        print(f"F1-Score:  {cv_results['f1_mean']:.4f} ± {cv_results['f1_std']:.4f}")
        
        return cv_results
    
    def predict(self, X):
        """
        Predict ON/OFF state for new data.
        
        Args:
            X (np.array): Features for prediction
        
        Returns:
            tuple: (predictions, probabilities)
                  predictions: 0=OFF, 1=ON
                  probabilities: probability of ON state
        """
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        return predictions, probabilities
    
    def save_model(self, filepath='../models/random_forest_model.pkl'):
        """
        Save the trained model and scaler to disk.

        Args:
            filepath (str): Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        joblib.dump(model_data, filepath)
        print(f"\nModel saved successfully to: {filepath}")
    
    def load_model(self, filepath):
        """
        Load a previously trained model from disk.
        
        Args:
            filepath (str): Path to the saved model
        """
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.feature_importance = model_data['feature_importance']
        print(f"Model loaded successfully from: {filepath}")
        print(f"Model trained on: {model_data['timestamp']}")
    
    def _plot_confusion_matrix(self, cm, output_dir):
        """Generate and save confusion matrix visualization."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['OFF', 'ON'], 
                   yticklabels=['OFF', 'ON'])
        plt.title('Confusion Matrix - OFF State Detection', fontsize=14, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved to: {output_dir}/confusion_matrix.png")
    
    def _plot_roc_curve(self, y_true, y_proba, roc_auc, output_dir):
        """Generate and save ROC curve visualization."""
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - OFF State Detection', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"ROC curve saved to: {output_dir}/roc_curve.png")
    
    def _plot_feature_importance(self, output_dir, top_n=20):
        """Generate and save feature importance visualization."""
        # Sort features by importance
        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]

        features, importances = zip(*sorted_features)

        plt.figure(figsize=(10, 8))
        plt.barh(range(len(features)), importances, color='steelblue')
        plt.yticks(range(len(features)), features)
        plt.xlabel('Feature Importance Score')
        plt.title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Feature importance plot saved to: {output_dir}/feature_importance.png")

    def check_overfitting(self, X_train, y_train, X_test, y_test, output_dir='../results'):
        """
        Detect overfitting by comparing train vs test performance and plotting learning curves.

        Overfitting indicators:
        - Train accuracy >> Test accuracy (gap > 5-10%)
        - Learning curve shows divergence
        - Very high train accuracy (>98%) with lower test accuracy

        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            output_dir: Directory to save learning curve plot

        Returns:
            dict: Overfitting analysis results
        """
        print("\n" + "=" * 80)
        print("OVERFITTING ANALYSIS")
        print("=" * 80)

        os.makedirs(output_dir, exist_ok=True)

        # Scale data
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Get train and test accuracy
        train_acc = self.model.score(X_train_scaled, y_train)
        test_acc = self.model.score(X_test_scaled, y_test)
        gap = train_acc - test_acc

        print(f"\nTrain Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
        print(f"Test Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
        print(f"Gap:            {gap:.4f} ({gap*100:.2f}%)")

        # Determine overfitting status
        if gap > 0.10:
            status = "SEVERE OVERFITTING"
            recommendation = "Reduce max_depth, increase min_samples_leaf, or get more data"
        elif gap > 0.05:
            status = "MODERATE OVERFITTING"
            recommendation = "Consider regularization or more training data"
        elif gap > 0.02:
            status = "MILD OVERFITTING"
            recommendation = "Acceptable for most applications"
        else:
            status = "GOOD FIT"
            recommendation = "Model generalizes well"

        print(f"\nStatus: {status}")
        print(f"Recommendation: {recommendation}")

        # Plot learning curve
        print("\nGenerating learning curve...")
        X_all = np.vstack([X_train, X_test])
        y_all = np.hstack([y_train, y_test])
        X_all_scaled = self.scaler.fit_transform(X_all)

        train_sizes, train_scores, test_scores = learning_curve(
            self.model, X_all_scaled, y_all,
            train_sizes=np.linspace(0.1, 1.0, 10),
            cv=5, n_jobs=-1, scoring='accuracy'
        )

        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        test_mean = test_scores.mean(axis=1)
        test_std = test_scores.std(axis=1)

        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                        alpha=0.1, color='blue')
        plt.plot(train_sizes, test_mean, 'o-', color='orange', label='Cross-validation score')
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std,
                        alpha=0.1, color='orange')

        plt.xlabel('Training Set Size')
        plt.ylabel('Accuracy Score')
        plt.title('Learning Curve - Overfitting Detection', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)

        # Add gap annotation
        plt.axhline(y=train_mean[-1], color='blue', linestyle='--', alpha=0.5)
        plt.axhline(y=test_mean[-1], color='orange', linestyle='--', alpha=0.5)
        plt.annotate(f'Gap: {gap*100:.1f}%', xy=(train_sizes[-1], (train_mean[-1]+test_mean[-1])/2),
                    fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/learning_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Learning curve saved to: {output_dir}/learning_curve.png")

        # Check for other overfitting signs
        print("\n" + "-" * 40)
        print("Additional Checks:")
        print("-" * 40)

        if train_acc > 0.98:
            print(f"[WARNING] Train accuracy ({train_acc:.2%}) is suspiciously high")
            print("          This may indicate the model is memorizing training data")

        if test_acc < 0.70:
            print(f"[WARNING] Test accuracy ({test_acc:.2%}) is low")
            print("          Model may not generalize well to new data")

        # Check if learning curve is converging
        if test_mean[-1] - test_mean[-3] < 0.01:
            print("[INFO] Learning curve has plateaued - more data may not help significantly")
        else:
            print("[INFO] Model may benefit from more training data")

        return {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'gap': gap,
            'status': status,
            'recommendation': recommendation
        }


def main():
    """
    Main function demonstrating model training and evaluation.
    This is a demo showing how to use the ParkinsonsOFFDetector class.
    """
    print("""
    =======================================================================
    aCare - Parkinson's Disease OFF State Detection System
    Random Forest Classifier Training Pipeline
    =======================================================================
    """)
    
    # Load your extracted features here
    # This is a placeholder - replace with actual feature loading
    print("Loading feature data...")
    print("NOTE: Please provide extracted features from sensor data")
    print("      Expected format: CSV with features and 'label' column")
    print("      Label: 0 = OFF state, 1 = ON state\n")
    
    # Example of how to use the detector with real data:
    # df = pd.read_csv('../data/extracted_features.csv')
    # X = df.drop('label', axis=1).values
    # y = df['label'].values
    # feature_names = df.drop('label', axis=1).columns.tolist()
    
    # For demonstration, create REALISTIC synthetic data
    # Real-world Parkinson's data has significant class overlap
    print("Creating realistic synthetic data for demonstration...")
    print("(Real sensor data has class overlap - not perfectly separable)\n")
    np.random.seed(42)
    n_samples = 1000
    n_features = 32

    # Realistic simulation: classes have OVERLAP (harder to classify)
    # ON state: slightly higher mean, some noise
    X_on = np.random.randn(n_samples//2, n_features) * 1.2 + 0.3

    # OFF state: slightly lower mean, more variance (tremor/dyskinesia)
    X_off = np.random.randn(n_samples//2, n_features) * 1.5 - 0.3

    # Add noise to make it more realistic (some features are uninformative)
    noise_features = n_features // 3  # ~33% of features are noise
    X_on[:, -noise_features:] = np.random.randn(n_samples//2, noise_features)
    X_off[:, -noise_features:] = np.random.randn(n_samples//2, noise_features)

    X = np.vstack([X_on, X_off])
    y = np.hstack([np.ones(n_samples//2), np.zeros(n_samples//2)])

    # Shuffle data
    shuffle_idx = np.random.permutation(len(y))
    X = X[shuffle_idx]
    y = y[shuffle_idx]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Initialize detector
    detector = ParkinsonsOFFDetector(random_state=42)

    # Train model
    train_metrics = detector.train(X_train, y_train)

    # Evaluate on test set
    test_metrics = detector.evaluate(X_test, y_test)

    # CHECK FOR OVERFITTING - Important!
    overfit_results = detector.check_overfitting(X_train, y_train, X_test, y_test)

    # Perform cross-validation
    cv_results = detector.cross_validate(X, y, cv_folds=5)

    # Save model
    detector.save_model()

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print("\nOverfitting Summary:")
    print(f"  Status: {overfit_results['status']}")
    print(f"  Train-Test Gap: {overfit_results['gap']*100:.2f}%")
    print(f"  Recommendation: {overfit_results['recommendation']}")
    print("\nNext steps:")
    print("1. Review learning_curve.png to check for overfitting")
    print("2. Review the confusion matrix and ROC curve in ../results/")
    print("3. Check feature importance to understand which features matter most")
    print("4. If overfitting detected, tune hyperparameters or get more data")
    print("5. Use the saved model for real-time prediction in production")


if __name__ == "__main__":
    main()