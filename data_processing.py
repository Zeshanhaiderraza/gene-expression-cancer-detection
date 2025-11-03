"""
Data Processing Module for Gene Expression Analysis
Author: Zeshan Haider Raza
Description: Utilities for loading, cleaning, and preprocessing gene expression data
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class GeneExpressionProcessor:
    """
    A class for processing gene expression data for cancer detection.
    
    Attributes:
        data (pd.DataFrame): The loaded dataset
        target_column (str): Name of the target variable column
        scaler: Fitted scaler object for normalization
    """
    
    def __init__(self, target_column='target'):
        """
        Initialize the GeneExpressionProcessor.
        
        Args:
            target_column (str): Name of the target variable column
        """
        self.data = None
        self.target_column = target_column
        self.scaler = None
        
    def load_data(self, filepath, **kwargs):
        """
        Load gene expression data from a file.
        
        Args:
            filepath (str): Path to the data file
            **kwargs: Additional arguments to pass to pd.read_csv()
            
        Returns:
            pd.DataFrame: Loaded dataset
        """
        try:
            self.data = pd.read_csv(filepath, **kwargs)
            print(f"✓ Data loaded successfully!")
            print(f"  Shape: {self.data.shape}")
            print(f"  Columns: {self.data.shape[1]}")
            print(f"  Samples: {self.data.shape[0]}")
            return self.data
        except Exception as e:
            print(f"✗ Error loading data: {str(e)}")
            return None
    
    def get_basic_info(self):
        """
        Display basic information about the dataset.
        """
        if self.data is None:
            print("No data loaded. Please load data first.")
            return
        
        print("\n" + "="*50)
        print("DATASET INFORMATION")
        print("="*50)
        print(f"\nShape: {self.data.shape}")
        print(f"\nData Types:\n{self.data.dtypes.value_counts()}")
        print(f"\nMissing Values:\n{self.data.isnull().sum().sum()} total")
        
        if self.target_column in self.data.columns:
            print(f"\nTarget Variable Distribution:")
            print(self.data[self.target_column].value_counts())
            print(f"\nClass Balance:")
            print(self.data[self.target_column].value_counts(normalize=True) * 100)
    
    def check_missing_values(self):
        """
        Check for missing values in the dataset.
        
        Returns:
            pd.Series: Missing value counts per column
        """
        if self.data is None:
            print("No data loaded.")
            return None
        
        missing = self.data.isnull().sum()
        missing_percent = (missing / len(self.data)) * 100
        
        missing_df = pd.DataFrame({
            'Missing_Count': missing,
            'Percentage': missing_percent
        })
        
        missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values(
            by='Missing_Count', ascending=False
        )
        
        if len(missing_df) > 0:
            print(f"\n⚠ Found {len(missing_df)} columns with missing values:")
            print(missing_df)
        else:
            print("\n✓ No missing values found!")
        
        return missing_df
    
    def handle_missing_values(self, strategy='mean', threshold=0.5):
        """
        Handle missing values in the dataset.
        
        Args:
            strategy (str): Strategy for handling missing values 
                          ('mean', 'median', 'mode', 'drop')
            threshold (float): If strategy is 'drop', drop columns with 
                             missing values above this threshold
            
        Returns:
            pd.DataFrame: Dataset with missing values handled
        """
        if self.data is None:
            print("No data loaded.")
            return None
        
        initial_shape = self.data.shape
        
        if strategy == 'drop':
            # Drop columns with missing values above threshold
            missing_percent = self.data.isnull().sum() / len(self.data)
            cols_to_drop = missing_percent[missing_percent > threshold].index
            self.data = self.data.drop(columns=cols_to_drop)
            print(f"✓ Dropped {len(cols_to_drop)} columns with >{threshold*100}% missing values")
        
        elif strategy in ['mean', 'median']:
            # Impute numeric columns
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if self.data[col].isnull().sum() > 0:
                    if strategy == 'mean':
                        self.data[col].fillna(self.data[col].mean(), inplace=True)
                    else:
                        self.data[col].fillna(self.data[col].median(), inplace=True)
        
        elif strategy == 'mode':
            # Impute with mode (for categorical)
            for col in self.data.columns:
                if self.data[col].isnull().sum() > 0:
                    self.data[col].fillna(self.data[col].mode()[0], inplace=True)
        
        print(f"✓ Missing value handling complete")
        print(f"  Before: {initial_shape}, After: {self.data.shape}")
        
        return self.data
    
    def remove_outliers(self, method='iqr', threshold=1.5):
        """
        Remove outliers from numeric columns.
        
        Args:
            method (str): Method for outlier detection ('iqr' or 'zscore')
            threshold (float): Threshold for outlier detection
            
        Returns:
            pd.DataFrame: Dataset with outliers removed
        """
        if self.data is None:
            print("No data loaded.")
            return None
        
        initial_len = len(self.data)
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != self.target_column]
        
        if method == 'iqr':
            for col in numeric_cols:
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                self.data = self.data[
                    (self.data[col] >= lower_bound) & 
                    (self.data[col] <= upper_bound)
                ]
        
        elif method == 'zscore':
            from scipy import stats
            z_scores = np.abs(stats.zscore(self.data[numeric_cols]))
            self.data = self.data[(z_scores < threshold).all(axis=1)]
        
        removed = initial_len - len(self.data)
        print(f"✓ Removed {removed} outliers ({removed/initial_len*100:.2f}%)")
        
        return self.data
    
    def normalize_features(self, method='standard', exclude_target=True):
        """
        Normalize numeric features.
        
        Args:
            method (str): Normalization method ('standard' or 'minmax')
            exclude_target (bool): Whether to exclude target column
            
        Returns:
            pd.DataFrame: Dataset with normalized features
        """
        if self.data is None:
            print("No data loaded.")
            return None
        
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        if exclude_target and self.target_column in numeric_cols:
            numeric_cols.remove(self.target_column)
        
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            print(f"Unknown normalization method: {method}")
            return None
        
        self.data[numeric_cols] = self.scaler.fit_transform(self.data[numeric_cols])
        
        print(f"✓ Features normalized using {method} scaling")
        return self.data
    
    def split_data(self, test_size=0.2, val_size=0.1, random_state=42, stratify=True):
        """
        Split data into train, validation, and test sets.
        
        Args:
            test_size (float): Proportion of data for test set
            val_size (float): Proportion of data for validation set
            random_state (int): Random seed for reproducibility
            stratify (bool): Whether to stratify split by target variable
            
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        if self.data is None:
            print("No data loaded.")
            return None
        
        if self.target_column not in self.data.columns:
            print(f"Target column '{self.target_column}' not found.")
            return None
        
        # Separate features and target
        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]
        
        # First split: train+val vs test
        stratify_param = y if stratify else None
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=stratify_param
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        stratify_param = y_temp if stratify else None
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=stratify_param
        )
        
        print("\n✓ Data split completed:")
        print(f"  Training set:   {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
        print(f"  Validation set: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
        print(f"  Test set:       {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save_processed_data(self, output_dir='data/processed'):
        """
        Save processed data to CSV files.
        
        Args:
            output_dir (str): Directory to save processed data
        """
        if self.data is None:
            print("No data to save.")
            return
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        filepath = os.path.join(output_dir, 'preprocessed_data.csv')
        self.data.to_csv(filepath, index=False)
        
        print(f"✓ Processed data saved to: {filepath}")


# Example usage
if __name__ == "__main__":
    # Initialize processor
    processor = GeneExpressionProcessor(target_column='target')
    
    # Load data
    data = processor.load_data('data/raw/gene_expression.csv')
    
    # Get basic information
    processor.get_basic_info()
    
    # Check for missing values
    processor.check_missing_values()
    
    # Handle missing values
    processor.handle_missing_values(strategy='mean')
    
    # Remove outliers
    processor.remove_outliers(method='iqr', threshold=1.5)
    
    # Normalize features
    processor.normalize_features(method='standard')
    
    # Split data
    splits = processor.split_data(test_size=0.2, val_size=0.1)
    
    # Save processed data
    processor.save_processed_data()
    
    print("\n✓ Data preprocessing pipeline completed successfully!")
