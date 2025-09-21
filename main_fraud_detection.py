"""
main_fraud_detection.py - Complete Credit Card Fraud Detection System
Advanced ML models with multiple algorithms and comprehensive analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                           roc_curve, precision_recall_curve, average_precision_score,
                           accuracy_score, precision_score, recall_score, f1_score)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import time
import warnings
warnings.filterwarnings('ignore')

# Import our dataset handler
try:
    from fraud_dataset import FraudDatasetHandler, quick_analysis
    print(" Dataset handler imported successfully!")
except ImportError:
    print(" Error: Make sure fraud_dataset.py is in the same folder!")
    exit(1)

class CreditCardFraudDetector:
    def __init__(self, dataset_path="creditcard.csv"):
        """Initialize the fraud detection system"""
        self.dataset_path = dataset_path
        self.data_handler = FraudDatasetHandler(dataset_path)
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_score = 0
        
        print(" Credit Card Fraud Detection System Initialized!")
        print(f" Dataset: {dataset_path}")
    
    def load_and_prepare_data(self, balance_method='smote', scale_method='standard'):
        """Load and prepare the dataset"""
        print("\n Loading and preparing dataset...")
        
        # Load dataset
        if not self.data_handler.load_dataset():
            return False
        
        # Explore dataset
        stats = self.data_handler.explore_dataset()
        
        # Prepare features
        if not self.data_handler.prepare_features(scale_method=scale_method):
            return False
        
        # Split dataset
        if not self.data_handler.split_dataset():
            return False
        
        # Handle imbalance
        if not self.data_handler.handle_imbalance(method=balance_method):
            return False
        
        print(" Dataset loaded and prepared successfully!")
        return True
    
    def initialize_models(self):
        """Initialize all ML models"""
        print("\n Initializing ML models...")
        
        self.models = {
            'Logistic Regression ': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest ': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting ': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'XGBoost ': XGBClassifier(random_state=42, eval_metric='logloss'),
            'SVM ': SVC(probability=True, random_state=42),
            'Neural Network ': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500),
            'Decision Tree ': DecisionTreeClassifier(random_state=42),
            'Naive Bayes ': GaussianNB(),
            'K-Nearest Neighbors 👥': KNeighborsClassifier(n_neighbors=5)
        }
        
        print(f" {len(self.models)} models initialized")
        return True
    
    def train_and_evaluate_models(self):
        """Train and evaluate all models"""
        print("\n Training and evaluating models...")
        print("=" * 60)
        
        X_train = self.data_handler.X_train
        X_test = self.data_handler.X_test
        y_train = self.data_handler.y_train
        y_test = self.data_handler.y_test
        
        for name, model in self.models.items():
            print(f"\n Training {name}...")
            start_time = time.time()
            
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                roc_auc = roc_auc_score(y_test, y_pred_proba)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
                
                training_time = time.time() - start_time
                
                # Store results
                self.results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'roc_auc': roc_auc,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'training_time': training_time,
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba
                }
                
                # Track best model
                if roc_auc > self.best_score:
                    self.best_score = roc_auc
                    self.best_model = name
                
                print(f"    Accuracy: {accuracy:.4f} | ROC-AUC: {roc_auc:.4f} | Time: {training_time:.2f}s")
                
            except Exception as e:
                print(f"    Error training {name}: {e}")
                continue
        
        print(f"\n Best Model: {self.best_model} (ROC-AUC: {self.best_score:.4f})")
        return True
    
    def create_ensemble_model(self):
        """Create an ensemble of top 3 models"""
        print("\n🎭 Creating ensemble model...")
        
        # Sort models by ROC-AUC score
        sorted_models = sorted(self.results.items(), key=lambda x: x[1]['roc_auc'], reverse=True)
        top_3_models = sorted_models[:3]
        
        print("🏅 Top 3 models for ensemble:")
        for i, (name, results) in enumerate(top_3_models, 1):
            print(f"   {i}. {name}: ROC-AUC {results['roc_auc']:.4f}")
        
        # Create ensemble
        estimators = [(name.split()[0], results['model']) for name, results in top_3_models]
        ensemble = VotingClassifier(estimators=estimators, voting='soft')
        
        # Train ensemble
        X_train = self.data_handler.X_train
        X_test = self.data_handler.X_test
        y_train = self.data_handler.y_train
        y_test = self.data_handler.y_test
        
        ensemble.fit(X_train, y_train)
        y_pred_ensemble = ensemble.predict(X_test)
        y_pred_proba_ensemble = ensemble.predict_proba(X_test)[:, 1]
        
        # Evaluate ensemble
        ensemble_metrics = {
            'model': ensemble,
            'accuracy': accuracy_score(y_test, y_pred_ensemble),
            'precision': precision_score(y_test, y_pred_ensemble),
            'recall': recall_score(y_test, y_pred_ensemble),
            'f1_score': f1_score(y_test, y_pred_ensemble),
            'roc_auc': roc_auc_score(y_test, y_pred_proba_ensemble),
            'y_pred': y_pred_ensemble,
            'y_pred_proba': y_pred_proba_ensemble
        }
        
        self.results['Ensemble Model 🎭'] = ensemble_metrics
        
        print(f" Ensemble ROC-AUC: {ensemble_metrics['roc_auc']:.4f}")
        
        if ensemble_metrics['roc_auc'] > self.best_score:
            self.best_model = 'Ensemble Model 🎭'
            self.best_score = ensemble_metrics['roc_auc']
            print(" Ensemble is the new best model!")
        
        return ensemble
    
    def create_comprehensive_visualizations(self):
        """Create comprehensive visualizations"""
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle(' Credit Card Fraud Detection - Model Analysis', fontsize=20, fontweight='bold')
        
        # 1. Model Performance Comparison
        ax1 = plt.subplot(3, 3, 1)
        models = list(self.results.keys())
        roc_scores = [self.results[model]['roc_auc'] for model in models]
        
        bars = ax1.barh(models, roc_scores, color=plt.cm.viridis(np.linspace(0, 1, len(models))))
        ax1.set_xlabel('ROC-AUC Score')
        ax1.set_title(' Model Performance (ROC-AUC)', fontweight='bold')
        ax1.set_xlim(0, 1)
        
        # Add score labels
        for bar, score in zip(bars, roc_scores):
            ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{score:.3f}', va='center', fontweight='bold')
        
        # 2. Precision-Recall Comparison
        ax2 = plt.subplot(3, 3, 2)
        precision_scores = [self.results[model]['precision'] for model in models]
        recall_scores = [self.results[model]['recall'] for model in models]
        
        scatter = ax2.scatter(recall_scores, precision_scores, 
                            c=roc_scores, cmap='viridis', s=100, alpha=0.8)
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title(' Precision vs Recall', fontweight='bold')
        
        # Add model names as annotations
        for i, model in enumerate(models):
            ax2.annotate(model.split()[0], (recall_scores[i], precision_scores[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.colorbar(scatter, ax=ax2, label='ROC-AUC')
        
        # 3. Training Time Comparison
        ax3 = plt.subplot(3, 3, 3)
        training_times = [self.results[model].get('training_time', 0) for model in models]
        
        bars = ax3.bar(range(len(models)), training_times, color='skyblue', alpha=0.8)
        ax3.set_xlabel('Models')
        ax3.set_ylabel('Training Time (seconds)')
        ax3.set_title(' Training Time Comparison', fontweight='bold')
        ax3.set_xticks(range(len(models)))
        ax3.set_xticklabels([m.split()[0] for m in models], rotation=45)
        
        # Add time labels
        for bar, time_val in zip(bars, training_times):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{time_val:.1f}s', ha='center', va='bottom', fontweight='bold')
        
        # 4. ROC Curves for top 3 models
        ax4 = plt.subplot(3, 3, 4)
        y_test = self.data_handler.y_test
        
        # Get top 3 models by ROC-AUC
        sorted_models = sorted(self.results.items(), key=lambda x: x[1]['roc_auc'], reverse=True)
        colors = ['red', 'blue', 'green']
        
        for i, (name, results) in enumerate(sorted_models[:3]):
            fpr, tpr, _ = roc_curve(y_test, results['y_pred_proba'])
            ax4.plot(fpr, tpr, color=colors[i], linewidth=2, 
                    label=f'{name.split()[0]} (AUC: {results["roc_auc"]:.3f})')
        
        ax4.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax4.set_xlabel('False Positive Rate')
        ax4.set_ylabel('True Positive Rate')
        ax4.set_title(' ROC Curves (Top 3 Models)', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Confusion Matrix for Best Model
        ax5 = plt.subplot(3, 3, 5)
        best_results = self.results[self.best_model]
        cm = confusion_matrix(y_test, best_results['y_pred'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax5)
        ax5.set_xlabel('Predicted')
        ax5.set_ylabel('Actual')
        ax5.set_title(f'🎯 Confusion Matrix\n{self.best_model.split()[0]}', fontweight='bold')
        
        # 6. Feature Importance (if available)
        ax6 = plt.subplot(3, 3, 6)
        best_model = best_results['model']
        
        if hasattr(best_model, 'feature_importances_'):
            feature_names = self.data_handler.feature_names
            importances = best_model.feature_importances_
            
            # Get top 10 features
            indices = np.argsort(importances)[-10:]
            
            ax6.barh(range(len(indices)), importances[indices], color='purple', alpha=0.8)
            ax6.set_yticks(range(len(indices)))
            ax6.set_yticklabels([feature_names[i] for i in indices])
            ax6.set_xlabel('Feature Importance')
            ax6.set_title(f' Top 10 Feature Importance\n{self.best_model.split()[0]}', fontweight='bold')
        else:
            ax6.text(0.5, 0.5, 'Feature importance\nnot available\nfor this model', 
                    ha='center', va='center', transform=ax6.transAxes, fontsize=12)
            ax6.set_title(' Feature Importance', fontweight='bold')
        
        # 7. Precision-Recall Curve for Best Model
        ax7 = plt.subplot(3, 3, 7)
        precision, recall, _ = precision_recall_curve(y_test, best_results['y_pred_proba'])
        avg_precision = average_precision_score(y_test, best_results['y_pred_proba'])
        
        ax7.plot(recall, precision, color='red', linewidth=2, 
                label=f'AP Score: {avg_precision:.3f}')
        ax7.set_xlabel('Recall')
        ax7.set_ylabel('Precision')
        ax7.set_title(f' Precision-Recall Curve\n{self.best_model.split()[0]}', fontweight='bold')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # 8. Model Metrics Comparison
        ax8 = plt.subplot(3, 3, 8)
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        # Get top 5 models for comparison
        top_models = sorted_models[:5]
        model_names = [name.split()[0] for name, _ in top_models]
        
        x = np.arange(len(metrics))
        width = 0.15
        
        for i, (name, results) in enumerate(top_models):
            values = [results[metric] for metric in metrics]
            ax8.bar(x + i*width, values, width, label=name.split()[0], alpha=0.8)
        
        ax8.set_xlabel('Metrics')
        ax8.set_ylabel('Score')
        ax8.set_title(' Model Metrics Comparison', fontweight='bold')
        ax8.set_xticks(x + width * 2)
        ax8.set_xticklabels(metrics)
        ax8.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax8.set_ylim(0, 1)
        
        # 9. Cross-Validation Scores
        ax9 = plt.subplot(3, 3, 9)
        cv_means = [self.results[model].get('cv_mean', 0) for model in models]
        cv_stds = [self.results[model].get('cv_std', 0) for model in models]
        
        bars = ax9.bar(range(len(models)), cv_means, yerr=cv_stds, 
                      capsize=5, color='orange', alpha=0.8)
        ax9.set_xlabel('Models')
        ax9.set_ylabel('CV ROC-AUC Score')
        ax9.set_title(' Cross-Validation Scores', fontweight='bold')
        ax9.set_xticks(range(len(models)))
        ax9.set_xticklabels([m.split()[0] for m in models], rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def generate_detailed_report(self):
        """Generate a comprehensive analysis report"""
        print("\n" + "="*80)
        print(" CREDIT CARD FRAUD DETECTION - COMPREHENSIVE REPORT")
        print("="*80)
        
        # Dataset Summary
        dataset_summary = self.data_handler.get_dataset_summary()
        print(f"\n DATASET SUMMARY:")
        print("-" * 50)
        print(f"Total Transactions: {dataset_summary['total_transactions']:,}")
        print(f"Features: {dataset_summary['total_features']}")
        print(f"Fraud Cases: {dataset_summary['fraud_transactions']:,} ({dataset_summary['fraud_percentage']:.2f}%)")
        print(f"Normal Cases: {dataset_summary['normal_transactions']:,}")
        print(f"Imbalance Ratio: {dataset_summary['imbalance_ratio']:.1f}:1")
        print(f"Amount Range: ${dataset_summary['amount_range'][0]:.2f} - ${dataset_summary['amount_range'][1]:,.2f}")
        
        # Model Performance Summary
        print(f"\n MODEL PERFORMANCE RANKING:")
        print("-" * 50)
        
        sorted_results = sorted(self.results.items(), key=lambda x: x[1]['roc_auc'], reverse=True)
        
        for i, (name, results) in enumerate(sorted_results, 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}️⃣"
            print(f"{emoji} {i}. {name}")
            print(f"    ROC-AUC: {results['roc_auc']:.4f} | F1: {results['f1_score']:.4f} | "
                  f"Precision: {results['precision']:.4f} | Recall: {results['recall']:.4f}")
            
            if 'cv_mean' in results:
                print(f"    CV Score: {results['cv_mean']:.4f} (±{results['cv_std']:.4f})")
            print()
        
        # Best Model Analysis
        best_results = self.results[self.best_model]
        print(f" BEST MODEL ANALYSIS: {self.best_model}")
        print("-" * 50)
        print(f"ROC-AUC Score: {best_results['roc_auc']:.4f}")
        print(f"Accuracy: {best_results['accuracy']:.4f}")
        print(f"Precision: {best_results['precision']:.4f}")
        print(f"Recall: {best_results['recall']:.4f}")
        print(f"F1-Score: {best_results['f1_score']:.4f}")
        
        if 'training_time' in best_results:
            print(f"Training Time: {best_results['training_time']:.2f} seconds")
        
        # Confusion Matrix Analysis
        y_test = self.data_handler.y_test
        cm = confusion_matrix(y_test, best_results['y_pred'])
        tn, fp, fn, tp = cm.ravel()
        
        print(f"\n CONFUSION MATRIX ANALYSIS:")
        print("-" * 50)
        print(f"True Negatives (Correct Normal): {tn:,}")
        print(f"False Positives (False Alarms): {fp:,}")
        print(f"False Negatives (Missed Fraud): {fn:,}")
        print(f"True Positives (Caught Fraud): {tp:,}")
        print(f"\nFalse Positive Rate: {fp/(fp+tn):.4f}")
        print(f"False Negative Rate: {fn/(fn+tp):.4f}")
        
        # Business Impact Analysis
        print(f"\n BUSINESS IMPACT ANALYSIS:")
        print("-" * 50)
        
        # Assume average fraud amount and costs
        avg_fraud_amount = 100  # Average fraud transaction amount
        investigation_cost = 10  # Cost to investigate each flagged transaction
        
        potential_savings = tp * avg_fraud_amount
        investigation_costs = (tp + fp) * investigation_cost
        net_benefit = potential_savings - investigation_costs
        
        print(f"Fraud Cases Detected: {tp} out of {tp + fn} ({tp/(tp+fn)*100:.1f}%)")
        print(f"Potential Fraud Prevented: ${potential_savings:,.2f}")
        print(f"Investigation Costs: ${investigation_costs:,.2f}")
        print(f"Net Benefit: ${net_benefit:,.2f}")
        
        # Recommendations
        print(f"\n RECOMMENDATIONS:")
        print("-" * 50)
        
        if best_results['precision'] < 0.8:
            print(" Consider tuning model to reduce false positives")
        if best_results['recall'] < 0.8:
            print(" Consider tuning model to catch more fraud cases")
        if best_results['roc_auc'] > 0.95:
            print(" Excellent performance - ready for production")
        elif best_results['roc_auc'] > 0.90:
            print(" Good performance - consider fine-tuning")
        else:
            print(" Consider feature engineering or trying other algorithms")
        
        print("\n DEPLOYMENT RECOMMENDATIONS:")
        print("-" * 50)
        print("1. Implement real-time scoring for transactions")
        print("2. Set up monitoring for model drift")
        print("3. Create feedback loop for model improvement")
        print("4. Establish threshold tuning based on business needs")
        print("5. Regular model retraining schedule")
        
        return {
            'best_model': self.best_model,
            'best_score': self.best_score,
            'model_results': self.results,
            'dataset_summary': dataset_summary
        }
    
    def predict_new_transaction(self, transaction_data):
        """Predict if a new transaction is fraudulent"""
        if self.best_model is None:
            print(" No trained model available. Please train models first.")
            return None
        
        # Get best model
        best_model = self.results[self.best_model]['model']
        
        # Make prediction
        if isinstance(transaction_data, dict):
            # Convert dict to DataFrame
            transaction_df = pd.DataFrame([transaction_data])
        else:
            transaction_df = pd.DataFrame(transaction_data)
        
        # Scale features if scaler is available
        if self.data_handler.scaler:
            transaction_scaled = self.data_handler.scaler.transform(transaction_df)
        else:
            transaction_scaled = transaction_df
        
        # Predict
        prediction = best_model.predict(transaction_scaled)[0]
        probability = best_model.predict_proba(transaction_scaled)[0][1] if hasattr(best_model, 'predict_proba') else None
        
        result = {
            'is_fraud': bool(prediction),
            'fraud_probability': probability,
            'risk_level': 'HIGH' if probability and probability > 0.8 else 'MEDIUM' if probability and probability > 0.5 else 'LOW'
        }
        
        return result
    
    def run_complete_analysis(self):
        """Run the complete fraud detection analysis"""
        print(" Starting Complete Credit Card Fraud Detection Analysis...")
        
        # Step 1: Load and prepare data
        if not self.load_and_prepare_data():
            print(" Failed to load and prepare data")
            return False
        
        # Step 2: Initialize models
        self.initialize_models()
        
        # Step 3: Train and evaluate models
        self.train_and_evaluate_models()
        
        # Step 4: Create ensemble model
        self.create_ensemble_model()
        
        # Step 5: Create visualizations
        print("\n📊 Generating comprehensive visualizations...")
        self.create_visualizations()
        
        # Step 6: Generate detailed report
        report = self.generate_detailed_report()
        
        print(f"\n Analysis Complete!")
        print(f" Best Model: {self.best_model}")
        print(f" Best ROC-AUC Score: {self.best_score:.4f}")
        
        return report

# Interactive Menu System
def interactive_menu():
    """Interactive menu for fraud detection system"""
    detector = CreditCardFraudDetector()
    
    while True:
        print("\n" + "="*60)
        print(" CREDIT CARD FRAUD DETECTION SYSTEM")
        print("="*60)
        print("Choose an option:")
        print("1.  Complete Analysis (Load → Train → Evaluate → Report)")
        print("2.  Quick Dataset Overview")
        print("3.  Train Models Only")
        print("4.  Create Visualizations")
        print("5.  Generate Report")
        print("6.  Test Single Transaction")
        print("7.  Custom Analysis Settings")
        print("8.  Export Results")
        print("9.  Exit")
        
        try:
            choice = input("\nEnter your choice (1-9): ").strip()
            
            if choice == '1':
                print("\n🚀 Running Complete Analysis...")
                detector.run_complete_analysis()
                
            elif choice == '2':
                print("\n Quick Dataset Overview...")
                handler, stats = quick_analysis(detector.dataset_path)
                
            elif choice == '3':
                print("\n Training Models...")
                if detector.load_and_prepare_data():
                    detector.initialize_models()
                    detector.train_and_evaluate_models()
                    detector.create_ensemble_model()
                
            elif choice == '4':
                if detector.results:
                    detector.create_comprehensive_visualizations()
                else:
                    print(" No models trained yet. Please run option 1 or 3 first.")
                
            elif choice == '5':
                if detector.results:
                    detector.generate_detailed_report()
                else:
                    print(" No models trained yet. Please run option 1 or 3 first.")
                
            elif choice == '6':
                if detector.best_model:
                    print("\n Testing Single Transaction...")
                    print("Enter transaction features (press Enter to use sample):")
                    
                    # Sample transaction
                    sample = {f'V{i}': np.random.randn() for i in range(1, 29)}
                    sample['Time'] = 3600
                    sample['Amount'] = 100.0
                    
                    result = detector.predict_new_transaction(sample)
                    print(f"\nPrediction Result:")
                    print(f"Is Fraud: {result['is_fraud']}")
                    print(f"Fraud Probability: {result['fraud_probability']:.4f}")
                    print(f"Risk Level: {result['risk_level']}")
                else:
                    print(" No trained model available. Please run option 1 or 3 first.")
                
            elif choice == '7':
                print("\n Custom Analysis Settings:")
                print("1. Balance Method: SMOTE, undersample, smote_tomek, none")
                print("2. Scaling Method: standard, robust, none")
                
                balance = input("Balance method (default: smote): ").strip() or 'smote'
                scaling = input("Scaling method (default: standard): ").strip() or 'standard'
                
                detector = CreditCardFraudDetector()
                if detector.load_and_prepare_data(balance_method=balance, scale_method=scaling):
                    detector.initialize_models()
                    detector.train_and_evaluate_models()
                    detector.create_ensemble_model()
                
            elif choice == '8':
                if detector.data_handler and detector.data_handler.X_train is not None:
                    detector.data_handler.export_processed_data()
                else:
                    print(" No processed data available. Please run analysis first.")
                
            elif choice == '9':
                print("\n Thank you for using Credit Card Fraud Detection System!")
                print(" Stay vigilant against fraud!")
                break
                
            else:
                print(" Invalid choice. Please enter 1-9.")
                
        except KeyboardInterrupt:
            print("\n\n Program interrupted. Goodbye!")
            break
        except Exception as e:
            print(f" Error: {e}")
            print("Please try again.")

# Main execution
if __name__ == "__main__":
    print(" Welcome to Credit Card Fraud Detection System!")
    
    # Check if dataset exists
    import os
    if not os.path.exists("creditcard.csv"):
        print(" creditcard.csv not found in current directory!")
        print(" Please download the dataset and place it in the same folder")
        exit(1)
    
    # Option to run direct analysis or interactive menu
    print("\nChoose mode:")
    print("1.  Direct Complete Analysis")
    print("2.  Interactive Menu")
    
    try:
        mode = input("\nEnter choice (1 or 2): ").strip()
        
        if mode == '1':
            detector = CreditCardFraudDetector()
            detector.run_complete_analysis()
        elif mode == '2':
            interactive_menu()
        else:
            print("Running interactive menu by default...")
            interactive_menu()
            
    except KeyboardInterrupt:
        print("\n Program interrupted. Goodbye!")
    except Exception as e:
        print(f" Error: {e}")
        print("Running interactive menu...")
        interactive_menu()