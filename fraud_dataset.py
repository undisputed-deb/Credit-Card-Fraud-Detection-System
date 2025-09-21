"""
fraud_dataset.py - Credit Card Fraud Detection Dataset Handler
Handles the creditcard.csv file and prepares data for ML models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
import warnings
warnings.filterwarnings('ignore')

class FraudDatasetHandler:
    def __init__(self, csv_file_path="creditcard.csv"):
        """Initialize the fraud detection dataset handler"""
        self.csv_file_path = csv_file_path
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.feature_names = None
        
        print("🏦 Credit Card Fraud Detection Dataset Handler Initialized!")
        print(f"📁 Dataset path: {csv_file_path}")
    
    def load_dataset(self):
        """Load the credit card dataset"""
        try:
            print("\n Loading credit card dataset...")
            self.df = pd.read_csv(self.csv_file_path)
            
            print(f" Dataset loaded successfully!")
            print(f" Dataset shape: {self.df.shape}")
            print(f" Columns: {list(self.df.columns)}")
            
            return True
            
        except FileNotFoundError:
            print(f" Error: File '{self.csv_file_path}' not found!")
            print(" Make sure the creditcard.csv file is in the same directory")
            return False
        except Exception as e:
            print(f" Error loading dataset: {e}")
            return False
    
    def explore_dataset(self):
        """Perform exploratory data analysis"""
        if self.df is None:
            print(" Dataset not loaded. Please run load_dataset() first.")
            return
        
        print("\n DATASET EXPLORATION:")
        print("=" * 50)
        
        # Basic info
        print(f" Dataset Shape: {self.df.shape}")
        print(f" Memory Usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Missing values
        missing_values = self.df.isnull().sum()
        print(f"\n Missing Values:")
        if missing_values.sum() == 0:
            print("    No missing values found!")
        else:
            print(missing_values[missing_values > 0])
        
        # Target distribution
        fraud_stats = self.df['Class'].value_counts()
        fraud_percentage = (fraud_stats[1] / len(self.df)) * 100
        
        print(f"\n Target Distribution:")
        print(f"   Normal transactions (Class 0): {fraud_stats[0]:,} ({100-fraud_percentage:.2f}%)")
        print(f"   Fraud transactions (Class 1):  {fraud_stats[1]:,} ({fraud_percentage:.2f}%)")
        print(f"    Imbalance Ratio: {fraud_stats[0]/fraud_stats[1]:.1f}:1")
        
        # Amount statistics
        print(f"\n Transaction Amount Statistics:")
        print(f"   Mean: ${self.df['Amount'].mean():.2f}")
        print(f"   Median: ${self.df['Amount'].median():.2f}")
        print(f"   Max: ${self.df['Amount'].max():.2f}")
        print(f"   Min: ${self.df['Amount'].min():.2f}")
        
        # Time statistics
        print(f"\n Time Statistics:")
        print(f"   Duration: {self.df['Time'].max() / 3600:.1f} hours")
        print(f"   Time range: {self.df['Time'].min()} - {self.df['Time'].max()} seconds")
        
        return {
            'shape': self.df.shape,
            'fraud_percentage': fraud_percentage,
            'imbalance_ratio': fraud_stats[0]/fraud_stats[1],
            'amount_stats': self.df['Amount'].describe()
        }
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        if self.df is None:
            print(" Dataset not loaded. Please run load_dataset() first.")
            return
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(' Credit Card Fraud Detection - Dataset Analysis', fontsize=18, fontweight='bold')
        
        # 1. Class Distribution
        ax1 = axes[0, 0]
        fraud_counts = self.df['Class'].value_counts()
        colors = ['#2E8B57', '#DC143C']  # Green for normal, Red for fraud
        
        bars = ax1.bar(['Normal', 'Fraud'], fraud_counts.values, color=colors, alpha=0.8)
        ax1.set_title(' Transaction Class Distribution', fontweight='bold')
        ax1.set_ylabel('Number of Transactions')
        
        # Add percentage labels
        total = len(self.df)
        for bar, count in zip(bars, fraud_counts.values):
            percentage = (count / total) * 100
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
                    f'{count:,}\n({percentage:.2f}%)', ha='center', va='bottom', fontweight='bold')
        
        # 2. Amount Distribution by Class
        ax2 = axes[0, 1]
        normal_amounts = self.df[self.df['Class'] == 0]['Amount']
        fraud_amounts = self.df[self.df['Class'] == 1]['Amount']
        
        ax2.hist(normal_amounts, bins=50, alpha=0.7, label='Normal', color='#2E8B57', density=True)
        ax2.hist(fraud_amounts, bins=50, alpha=0.7, label='Fraud', color='#DC143C', density=True)
        ax2.set_title('💰 Amount Distribution by Class', fontweight='bold')
        ax2.set_xlabel('Transaction Amount')
        ax2.set_ylabel('Density')
        ax2.legend()
        ax2.set_xlim(0, 1000)  # Focus on smaller amounts
        
        # 3. Time Distribution
        ax3 = axes[0, 2]
        time_hours = self.df['Time'] / 3600  # Convert to hours
        ax3.hist(time_hours, bins=24, alpha=0.7, color='#4682B4', edgecolor='black')
        ax3.set_title(' Transaction Time Distribution', fontweight='bold')
        ax3.set_xlabel('Time (Hours)')
        ax3.set_ylabel('Number of Transactions')
        
        # 4. Correlation Heatmap (subset of features)
        ax4 = axes[1, 0]
        # Select a subset of V features for visualization
        corr_features = ['V1', 'V2', 'V3', 'V4', 'V5', 'Amount', 'Class']
        corr_matrix = self.df[corr_features].corr()
        
        sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, 
                   square=True, ax=ax4, fmt='.2f')
        ax4.set_title(' Feature Correlation Matrix', fontweight='bold')
        
        # 5. Amount by Class (Box Plot)
        ax5 = axes[1, 1]
        box_data = [normal_amounts[normal_amounts <= 200], fraud_amounts[fraud_amounts <= 200]]
        box_plot = ax5.boxplot(box_data, labels=['Normal', 'Fraud'], patch_artist=True)
        
        # Color the boxes
        box_plot['boxes'][0].set_facecolor('#2E8B57')
        box_plot['boxes'][1].set_facecolor('#DC143C')
        
        ax5.set_title(' Amount Distribution (Box Plot)', fontweight='bold')
        ax5.set_ylabel('Transaction Amount')
        ax5.set_ylim(0, 200)
        
        # 6. Feature Importance Preview (V1-V10)
        ax6 = axes[1, 2]
        v_features = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10']
        correlations = [abs(self.df[feature].corr(self.df['Class'])) for feature in v_features]
        
        bars = ax6.bar(v_features, correlations, color='#9370DB', alpha=0.8)
        ax6.set_title(' Feature-Target Correlations (V1-V10)', fontweight='bold')
        ax6.set_ylabel('Absolute Correlation with Class')
        ax6.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def prepare_features(self, scale_method='standard'):
        """Prepare features for machine learning"""
        if self.df is None:
            print(" Dataset not loaded. Please run load_dataset() first.")
            return False
        
        print(f"\n🔧 Preparing features for machine learning...")
        
        # Separate features and target
        self.X = self.df.drop(['Class'], axis=1)
        self.y = self.df['Class']
        self.feature_names = self.X.columns.tolist()
        
        print(f" Features shape: {self.X.shape}")
        print(f"🎯 Target shape: {self.y.shape}")
        
        # Scale features
        if scale_method == 'standard':
            self.scaler = StandardScaler()
        elif scale_method == 'robust':
            self.scaler = RobustScaler()
        else:
            print("⚠️ No scaling applied")
            return True
        
        self.X = pd.DataFrame(
            self.scaler.fit_transform(self.X),
            columns=self.feature_names
        )
        
        print(f" Features scaled using {scale_method} scaler")
        return True
    
    def split_dataset(self, test_size=0.2, random_state=42):
        """Split dataset into training and testing sets"""
        if self.X is None or self.y is None:
            print("❌ Features not prepared. Please run prepare_features() first.")
            return False
        
        print(f"\n Splitting dataset...")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, 
            stratify=self.y  # Maintain class distribution
        )
        
        print(f" Training set: {self.X_train.shape}")
        print(f" Testing set: {self.X_test.shape}")
        print(f"🎯 Training target distribution:")
        print(f"   Normal: {(self.y_train == 0).sum():,}")
        print(f"   Fraud: {(self.y_train == 1).sum():,}")
        
        return True
    
    def handle_imbalance(self, method='smote'):
        """Handle class imbalance using various techniques"""
        if self.X_train is None:
            print("❌ Dataset not split. Please run split_dataset() first.")
            return False
        
        print(f"\n Handling class imbalance using {method.upper()}...")
        
        original_shape = self.X_train.shape
        original_fraud = (self.y_train == 1).sum()
        original_normal = (self.y_train == 0).sum()
        
        if method == 'smote':
            # SMOTE oversampling
            smote = SMOTE(random_state=42)
            self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
            
        elif method == 'undersample':
            # Random undersampling
            undersampler = RandomUnderSampler(random_state=42)
            self.X_train, self.y_train = undersampler.fit_resample(self.X_train, self.y_train)
            
        elif method == 'smote_tomek':
            # SMOTE + Tomek links
            smote_tomek = SMOTETomek(random_state=42)
            self.X_train, self.y_train = smote_tomek.fit_resample(self.X_train, self.y_train)
            
        elif method == 'none':
            print("   No resampling applied")
            return True
        else:
            print(f" Unknown method: {method}")
            return False
        
        new_fraud = (self.y_train == 1).sum()
        new_normal = (self.y_train == 0).sum()
        
        print(f" Original: Normal {original_normal:,}, Fraud {original_fraud:,}")
        print(f" After {method}: Normal {new_normal:,}, Fraud {new_fraud:,}")
        print(f" New training set shape: {self.X_train.shape}")
        
        return True
    
    def get_dataset_summary(self):
        """Get a comprehensive dataset summary"""
        if self.df is None:
            return None
        
        summary = {
            'total_transactions': len(self.df),
            'total_features': len(self.df.columns) - 1,  # Excluding target
            'fraud_transactions': (self.df['Class'] == 1).sum(),
            'normal_transactions': (self.df['Class'] == 0).sum(),
            'fraud_percentage': ((self.df['Class'] == 1).sum() / len(self.df)) * 100,
            'imbalance_ratio': (self.df['Class'] == 0).sum() / (self.df['Class'] == 1).sum(),
            'amount_range': (self.df['Amount'].min(), self.df['Amount'].max()),
            'time_range_hours': self.df['Time'].max() / 3600,
            'missing_values': self.df.isnull().sum().sum(),
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024**2
        }
        
        return summary
    
    def export_processed_data(self, output_dir="processed_data"):
        """Export processed data for model training"""
        import os
        
        if self.X_train is None:
            print(" Data not processed. Please run the complete pipeline first.")
            return False
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Save training and test sets
            pd.DataFrame(self.X_train).to_csv(f"{output_dir}/X_train.csv", index=False)
            pd.DataFrame(self.X_test).to_csv(f"{output_dir}/X_test.csv", index=False)
            pd.Series(self.y_train).to_csv(f"{output_dir}/y_train.csv", index=False)
            pd.Series(self.y_test).to_csv(f"{output_dir}/y_test.csv", index=False)
            
            # Save feature names
            with open(f"{output_dir}/feature_names.txt", 'w') as f:
                f.write('\n'.join(self.feature_names))
            
            print(f"Processed data exported to '{output_dir}/' directory")
            return True
            
        except Exception as e:
            print(f" Error exporting data: {e}")
            return False

# Quick dataset analysis function
def quick_analysis(csv_file_path="creditcard.csv"):
    """Perform quick analysis of the credit card dataset"""
    handler = FraudDatasetHandler(csv_file_path)
    
    if not handler.load_dataset():
        return None
    
    # Basic exploration
    stats = handler.explore_dataset()
    
    # Create visualizations
    handler.create_visualizations()
    
    return handler, stats

# Export main components for easy import
if __name__ == "__main__":
    print(" Credit Card Fraud Detection Dataset Handler")
    print(" Make sure 'creditcard.csv' is in the same directory")
    print(" Run quick_analysis() to start exploring your data!")
    
    # Example usage
    print("\n Example usage:")
    print("handler, stats = quick_analysis('creditcard.csv')")
    print("handler.prepare_features()")
    print("handler.split_dataset()")
    print("handler.handle_imbalance('smote')")