
DATASET_PATH = "C:/Users/yaswa/OneDrive/Desktop/GENZ internship/Predictive_Maintenance.csv"
  # Put NONE is no dataset available

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve, f1_score, accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import os
import warnings
warnings.filterwarnings('ignore')

class FraudDetectionModel:
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {}
        self.best_model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.data = None
        
    def load_data(self, filepath=None):
        """Load dataset from file or prompt user for options"""
        if filepath is None:
            return self._prompt_for_data_source()
        
        try:
            if not os.path.exists(filepath):
                print(f"Error: File '{filepath}' not found.")
                return self._prompt_for_data_source()
            
            # Try to load the dataset
            if filepath.endswith('.csv'):
                self.data = pd.read_csv(filepath)
            elif filepath.endswith(('.xlsx', '.xls')):
                self.data = pd.read_excel(filepath)
            else:
                print(f"Error: Unsupported file format. Please provide a CSV or Excel file.")
                return self._prompt_for_data_source()
            
            # Validate dataset structure
            if not self._validate_dataset():
                print("Dataset validation failed.")
                return self._prompt_for_data_source()
            
            print(f"Dataset loaded successfully from '{filepath}'")
            print(f"Shape: {self.data.shape}")
            return self.data
            
        except Exception as e:
            print(f"Error loading data from '{filepath}': {str(e)}")
            return self._prompt_for_data_source()
    
    def _prompt_for_data_source(self):
        """Prompt user to choose data source"""
        print("\n" + "="*60)
        print("NO DATASET PROVIDED OR LOADING FAILED")
        print("="*60)
        print("Options:")
        print("1. Provide a different dataset file path")
        print("2. Create synthetic data for demonstration")
        print("3. Exit")
        
        while True:
            try:
                choice = input("\nPlease select an option (1-3): ").strip()
                
                if choice == '1':
                    filepath = input("Enter the path to your dataset file (CSV/Excel): ").strip()
                    if filepath:
                        return self.load_data(filepath)
                    else:
                        print("No file path provided.")
                        continue
                        
                elif choice == '2':
                    confirm = input("Create synthetic data for demonstration? (y/n): ").strip().lower()
                    if confirm in ['y', 'yes']:
                        return self._create_synthetic_data()
                    else:
                        print("Synthetic data creation cancelled.")
                        continue
                        
                elif choice == '3':
                    print("Exiting...")
                    return None
                    
                else:
                    print("Invalid choice. Please select 1, 2, or 3.")
                    continue
                    
            except KeyboardInterrupt:
                print("\nOperation cancelled by user.")
                return None
            except Exception as e:
                print(f"Error: {str(e)}")
                continue
    
    def _validate_dataset(self):
        """Validate if the dataset has the required structure"""
        try:
            if self.data is None or self.data.empty:
                print("Error: Dataset is empty.")
                return False
            
            # Check if 'Class' column exists (target variable)
            if 'Class' not in self.data.columns:
                print("Error: Dataset must contain a 'Class' column for fraud labels.")
                print("Available columns:", list(self.data.columns))
                
                # Ask user if they want to specify the target column
                target_col = input("Enter the name of the target/label column (or press Enter to skip): ").strip()
                if target_col and target_col in self.data.columns:
                    self.data = self.data.rename(columns={target_col: 'Class'})
                    print(f"Renamed '{target_col}' to 'Class' column.")
                else:
                    return False
            
            # Check if Class column has binary values
            unique_classes = self.data['Class'].unique()
            if len(unique_classes) != 2:
                print(f"Error: Class column must have exactly 2 unique values. Found: {unique_classes}")
                return False
            
            # Ensure binary encoding (0 and 1)
            if set(unique_classes) != {0, 1}:
                print(f"Converting class labels to binary (0, 1). Original values: {unique_classes}")
                # Map to 0 and 1
                unique_sorted = sorted(unique_classes)
                label_map = {unique_sorted[0]: 0, unique_sorted[1]: 1}
                self.data['Class'] = self.data['Class'].map(label_map)
                print(f"Mapped {unique_sorted[0]} -> 0, {unique_sorted[1]} -> 1")
            
            # Check for minimum number of samples
            if len(self.data) < 100:
                print("Warning: Dataset has very few samples. Results may not be reliable.")
            
            # Check for missing values
            missing_values = self.data.isnull().sum().sum()
            if missing_values > 0:
                print(f"Warning: Dataset contains {missing_values} missing values.")
                print("Missing values will be handled during preprocessing.")
            
            print(f"Dataset validation successful!")
            print(f"Features: {len(self.data.columns) - 1}")
            print(f"Samples: {len(self.data)}")
            print(f"Class distribution: {dict(self.data['Class'].value_counts())}")
            
            return True
            
        except Exception as e:
            print(f"Error during dataset validation: {str(e)}")
            return False
    
    def _create_synthetic_data(self):
        """Create synthetic credit card fraud dataset"""
        print("\nCreating synthetic fraud detection dataset...")
        
        try:
            np.random.seed(42)
            n_samples = 10000
            n_features = 28
            
            # Create sample data similar to credit card dataset
            normal_data = np.random.normal(0, 1, (int(n_samples * 0.998), n_features))
            fraud_data = np.random.normal(2, 1.5, (int(n_samples * 0.002), n_features))
            
            X_sample = np.vstack([normal_data, fraud_data])
            y_sample = np.hstack([np.zeros(len(normal_data)), np.ones(len(fraud_data))])
            
            # Add Time and Amount features
            time_feature = np.random.uniform(0, 172800, n_samples)
            amount_feature = np.random.lognormal(3, 1.5, n_samples)
            
            # Create feature names
            feature_names = [f'V{i}' for i in range(1, n_features-1)] + ['Time', 'Amount']
            
            # Create DataFrame
            self.data = pd.DataFrame(X_sample, columns=feature_names)
            self.data['Time'] = time_feature
            self.data['Amount'] = amount_feature
            self.data['Class'] = y_sample
            
            print("Synthetic dataset created successfully!")
            print(f"Shape: {self.data.shape}")
            print(f"Fraud percentage: {(self.data['Class'].sum() / len(self.data)) * 100:.2f}%")
            
            return self.data
            
        except Exception as e:
            print(f"Error creating synthetic data: {str(e)}")
            return None
    
    def explore_data(self):
        """Perform exploratory data analysis"""
        if self.data is None:
            print("No data available for exploration.")
            return
        
        print("Dataset Information:")
        print(self.data.info())
        print("\nDataset Description:")
        print(self.data.describe())
        print("\nClass Distribution:")
        print(self.data['Class'].value_counts())
        print(f"Fraud Percentage: {(self.data['Class'].sum() / len(self.data)) * 100:.2f}%")
        
        # Check for missing values
        missing_count = self.data.isnull().sum().sum()
        print(f"\nMissing Values: {missing_count}")
        
        # Handle missing values if present
        if missing_count > 0:
            print("Handling missing values by forward filling...")
            self.data = self.data.fillna(method='ffill').fillna(method='bfill')
        
        # Visualizations
        plt.figure(figsize=(15, 10))
        
        # Class distribution
        plt.subplot(2, 3, 1)
        self.data['Class'].value_counts().plot(kind='bar')
        plt.title('Class Distribution')
        plt.xlabel('Class (0: Normal, 1: Fraud)')
        plt.ylabel('Count')
        
        # Try to plot Amount distribution if it exists
        if 'Amount' in self.data.columns:
            plt.subplot(2, 3, 2)
            plt.hist(self.data[self.data['Class'] == 0]['Amount'], bins=50, alpha=0.7, label='Normal')
            plt.hist(self.data[self.data['Class'] == 1]['Amount'], bins=50, alpha=0.7, label='Fraud')
            plt.xlabel('Amount')
            plt.ylabel('Frequency')
            plt.title('Amount Distribution')
            plt.legend()
            plt.yscale('log')
        
        # Try to plot Time distribution if it exists
        if 'Time' in self.data.columns:
            plt.subplot(2, 3, 3)
            plt.hist(self.data[self.data['Class'] == 0]['Time'], bins=50, alpha=0.7, label='Normal')
            plt.hist(self.data[self.data['Class'] == 1]['Time'], bins=50, alpha=0.7, label='Fraud')
            plt.xlabel('Time')
            plt.ylabel('Frequency')
            plt.title('Time Distribution')
            plt.legend()
        
        # Correlation heatmap for selected features
        plt.subplot(2, 3, 4)
        # Select numeric columns for correlation
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 10:
            numeric_cols = numeric_cols[:10]  # Limit to first 10 numeric columns
        
        if len(numeric_cols) > 1:
            corr_matrix = self.data[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
            plt.title('Correlation Matrix')
        
        # Box plot for Amount if exists
        if 'Amount' in self.data.columns:
            plt.subplot(2, 3, 5)
            self.data.boxplot(column='Amount', by='Class', ax=plt.gca())
            plt.title('Amount by Class')
            plt.suptitle('')
        
        # Feature correlation with target
        plt.subplot(2, 3, 6)
        numeric_features = [col for col in numeric_cols if col != 'Class'][:5]  # Top 5 numeric features
        if numeric_features:
            fraud_corr = self.data[numeric_features + ['Class']].corr()['Class'].abs().sort_values(ascending=False)[:-1]
            if not fraud_corr.empty:
                fraud_corr.plot(kind='bar')
                plt.title('Feature Correlation with Target')
                plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def preprocess_data(self, use_sampling=None):
        """Preprocess the data"""
        if self.data is None:
            print("No data available for preprocessing.")
            return
        
        # Separate features and target
        X = self.data.drop('Class', axis=1)
        y = self.data['Class']
        
        # Handle non-numeric columns
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) < len(X.columns):
            print(f"Warning: Non-numeric columns detected. Using only numeric columns.")
            print(f"Numeric columns: {len(numeric_columns)}, Total columns: {len(X.columns)}")
            X = X[numeric_columns]
        
        # Scale the data
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Apply sampling techniques if specified
        if use_sampling == 'smote':
            smote = SMOTE(random_state=42)
            self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
            print("Applied SMOTE oversampling")
        elif use_sampling == 'undersample':
            undersampler = RandomUnderSampler(random_state=42)
            self.X_train, self.y_train = undersampler.fit_resample(self.X_train, self.y_train)
            print("Applied Random undersampling")
        
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Test set shape: {self.X_test.shape}")
        print(f"Training set fraud ratio: {self.y_train.sum() / len(self.y_train):.4f}")
        
    def train_models(self):
        """Train multiple models"""
        if self.X_train is None:
            print("Data not preprocessed. Please run preprocess_data() first.")
            return
        
        # Define models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'SVM': SVC(probability=True, random_state=42),
            'Isolation Forest': IsolationForest(contamination=0.1, random_state=42)
        }
        
        self.models = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            if name == 'Isolation Forest':
                # Isolation Forest works differently
                model.fit(self.X_train)
                # Convert anomaly scores to binary predictions
                train_pred = model.decision_function(self.X_train)
                threshold = np.percentile(train_pred, 10)  # Assume 10% are anomalies
                self.models[name] = (model, threshold)
            else:
                model.fit(self.X_train, self.y_train)
                self.models[name] = model
            
            print(f"{name} training completed")
    
    def evaluate_models(self):
        """Evaluate all trained models"""
        if not self.models:
            print("No trained models available. Please run train_models() first.")
            return None
        
        results = {}
        
        plt.figure(figsize=(15, 10))
        
        for i, (name, model_info) in enumerate(self.models.items(), 1):
            print(f"\nEvaluating {name}:")
            print("-" * 50)
            
            if name == 'Isolation Forest':
                model, threshold = model_info
                y_pred_scores = model.decision_function(self.X_test)
                y_pred = (y_pred_scores < threshold).astype(int)
                y_pred_proba = None
            else:
                model = model_info
                y_pred = model.predict(self.X_test)
                y_pred_proba = model.predict_proba(self.X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            
            print(f"Accuracy: {accuracy:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print("\nClassification Report:")
            print(classification_report(self.y_test, y_pred))
            
            # Store results
            results[name] = {
                'accuracy': accuracy,
                'f1_score': f1,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            # Plot confusion matrix
            plt.subplot(2, 4, i)
            cm = confusion_matrix(self.y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'{name}\nConfusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            
            # Plot ROC curve if probabilities available
            if y_pred_proba is not None:
                plt.subplot(2, 4, i + 4)
                fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
                auc = roc_auc_score(self.y_test, y_pred_proba)
                plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'{name}\nROC Curve')
                plt.legend()
                
                results[name]['auc'] = auc
                print(f"AUC Score: {auc:.4f}")
        
        plt.tight_layout()
        plt.show()
        
        # Find best model based on F1 score
        best_model_name = max(results.keys(), key=lambda x: results[x]['f1_score'])
        self.best_model = (best_model_name, self.models[best_model_name])
        
        print(f"\nBest Model: {best_model_name}")
        print(f"Best F1 Score: {results[best_model_name]['f1_score']:.4f}")
        
        return results
    
    def hyperparameter_tuning(self, model_name='Random Forest'):
        """Perform hyperparameter tuning for specified model"""
        if self.X_train is None:
            print("Data not preprocessed. Please run preprocess_data() first.")
            return
        
        print(f"Performing hyperparameter tuning for {model_name}...")
        
        if model_name == 'Random Forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            model = RandomForestClassifier(random_state=42, n_jobs=-1)
        elif model_name == 'Logistic Regression':
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            }
            model = LogisticRegression(random_state=42, max_iter=1000)
        else:
            print(f"Hyperparameter tuning not implemented for {model_name}")
            return
        
        # Use smaller sample for faster tuning
        sample_size = min(10000, len(self.X_train))
        X_sample = self.X_train.sample(n=sample_size, random_state=42)
        y_sample = self.y_train[X_sample.index]
        
        grid_search = GridSearchCV(
            model, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_sample, y_sample)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV F1 score: {grid_search.best_score_:.4f}")
        
        # Train final model with best parameters
        best_model = grid_search.best_estimator_
        best_model.fit(self.X_train, self.y_train)
        self.models[f'{model_name} (Tuned)'] = best_model
        
        return grid_search.best_estimator_
    
    def detect_fraud(self, transaction_data):
        """Detect fraud in new transaction data"""
        if self.best_model is None:
            print("No trained model available. Please train models first.")
            return None
        
        model_name, model_info = self.best_model
        
        # Preprocess the transaction data
        transaction_scaled = self.scaler.transform(transaction_data)
        
        if model_name == 'Isolation Forest':
            model, threshold = model_info
            scores = model.decision_function(transaction_scaled)
            predictions = (scores < threshold).astype(int)
            probabilities = None
        else:
            model = model_info
            predictions = model.predict(transaction_scaled)
            probabilities = model.predict_proba(transaction_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
        
        results = {
            'predictions': predictions,
            'probabilities': probabilities,
            'model_used': model_name
        }
        
        return results
    
    def feature_importance_analysis(self):
        """Analyze feature importance"""
        if 'Random Forest' not in self.models:
            print("Random Forest model not available for feature importance analysis")
            return
        
        rf_model = self.models['Random Forest']
        feature_importance = rf_model.feature_importances_
        feature_names = self.X_train.columns
        
        # Create feature importance dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        plt.subplot(1, 2, 1)
        top_features = importance_df.head(15)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.title('Top 15 Feature Importances')
        plt.gca().invert_yaxis()
        
        # Plot cumulative importance
        plt.subplot(1, 2, 2)
        cumulative_importance = np.cumsum(importance_df['importance'])
        plt.plot(range(1, len(cumulative_importance) + 1), cumulative_importance)
        plt.xlabel('Number of Features')
        plt.ylabel('Cumulative Importance')
        plt.title('Cumulative Feature Importance')
        plt.axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        print("Top 10 Most Important Features:")
        print(importance_df.head(10))
        
        return importance_df

# Example usage
def main():
    print("="*80)
    print("FRAUD DETECTION MODEL - STARTING")
    print("="*80)
    
    # Initialize the fraud detection model
    fraud_detector = FraudDetectionModel()
    
    # Load data using the configured path
    if DATASET_PATH is not None:
        print(f"Attempting to load dataset from: {DATASET_PATH}")
        data = fraud_detector.load_data(DATASET_PATH)
    else:
        print("No dataset path configured. Starting interactive mode...")
        data = fraud_detector.load_data()
    
    if data is None:
        print("No data loaded. Exiting...")
        return
    
    print("\n" + "="*80)
    print("DATA EXPLORATION")
    print("="*80)
    
    # Explore the data
    fraud_detector.explore_data()
    
    print("\n" + "="*80)
    print("DATA PREPROCESSING")
    print("="*80)
    
    # Preprocess data
    fraud_detector.preprocess_data(use_sampling='smote')
    
    print("\n" + "="*80)
    print("MODEL TRAINING")
    print("="*80)
    
    # Train models
    fraud_detector.train_models()
    
    print("\n" + "="*80)
    print("MODEL EVALUATION")
    print("="*80)
    
    # Evaluate models
    results = fraud_detector.evaluate_models()
    
    if results:
        print("\n" + "="*80)
        print("HYPERPARAMETER TUNING")
        print("="*80)
        
        # Perform hyperparameter tuning
        fraud_detector.hyperparameter_tuning('Random Forest')
        
        print("\n" + "="*80)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*80)
        
        # Feature importance analysis
        fraud_detector.feature_importance_analysis()
        
        print("\n" + "="*80)
        print("FRAUD DETECTION TEST")
        print("="*80)
        
        # Example fraud detection on new data
        print("\nTesting fraud detection on sample transactions:")
        new_transactions = fraud_detector.data.drop('Class', axis=1).sample(min(5, len(fraud_detector.data)))
        fraud_results = fraud_detector.detect_fraud(new_transactions)
        
        if fraud_results:
            predictions = fraud_results['predictions']
            probabilities = fraud_results['probabilities']
            
            # Handle the case where probabilities might be None or an array
            if probabilities is None:
                probabilities = [None] * len(predictions)
            
            for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                status = "FRAUD" if pred == 1 else "NORMAL"
                prob_str = f" (Probability: {prob:.3f})" if prob is not None else ""
                print(f"Transaction {i+1}: {status}{prob_str}")
        
        print("\n" + "="*80)
        print("FRAUD DETECTION MODEL - COMPLETED SUCCESSFULLY!")
        print("="*80)

if __name__ == "__main__":
    main()