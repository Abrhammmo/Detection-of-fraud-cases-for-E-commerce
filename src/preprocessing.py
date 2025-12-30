# src/preprocessing.py
import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler


def _validate_required_columns(df, required_cols, function_name="preprocess_fraud"):
    """
    Validate that required columns exist in the dataframe.
    
    Args:
        df: Input dataframe
        required_cols: List of required column names
        function_name: Name of the calling function for error messages
    
    Raises:
        ValueError: If any required columns are missing
    """
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"{function_name}: Missing required columns: {missing_cols}. "
            f"Available columns: {df.columns.tolist()}"
        )


def _convert_to_numeric(df, col, function_name="preprocess_fraud"):
    """
    Convert a column to numeric type, handling errors gracefully.
    
    Args:
        df: Input dataframe
        col: Column name to convert
        function_name: Name of the calling function for error messages
    
    Returns:
        Series with numeric values, with non-convertible values set to NaN
    """
    if col not in df.columns:
        return None
    
    try:
        # Try direct conversion first
        if pd.api.types.is_numeric_dtype(df[col]):
            return df[col]
        
        # Attempt conversion, coercing errors to NaN
        converted = pd.to_numeric(df[col], errors='coerce')
        
        # Check if conversion resulted in many NaNs
        nan_count = converted.isna().sum()
        if nan_count > 0:
            warnings.warn(
                f"{function_name}: Column '{col}' had {nan_count} non-numeric values "
                f"converted to NaN. Original dtype: {df[col].dtype}"
            )
        
        return converted
    except Exception as e:
        warnings.warn(
            f"{function_name}: Failed to convert column '{col}' to numeric: {str(e)}. "
            f"Original dtype: {df[col].dtype}"
        )
        return df[col]


def _convert_to_datetime(df, col, function_name="preprocess_fraud"):
    """
    Convert a column to datetime type, handling errors gracefully.
    
    Args:
        df: Input dataframe
        col: Column name to convert
        function_name: Name of the calling function for error messages
    
    Returns:
        Series with datetime values, with non-convertible values set to NaT
    """
    if col not in df.columns:
        return None
    
    try:
        # Try conversion with error handling
        converted = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
        
        # Check if conversion resulted in many NaTs
        nat_count = converted.isna().sum()
        if nat_count > 0:
            warnings.warn(
                f"{function_name}: Column '{col}' had {nat_count} non-datetime values "
                f"converted to NaT. Original dtype: {df[col].dtype}"
            )
        
        return converted
    except Exception as e:
        warnings.warn(
            f"{function_name}: Failed to convert column '{col}' to datetime: {str(e)}. "
            f"Original dtype: {df[col].dtype}"
        )
        return df[col]


def preprocess_fraud(df):
    """
    Preprocess fraud detection dataset with validation and error handling.
    
    Required columns: purchase_value, age
    Optional columns: device_id, country, ip_address, purchase_time, signup_time
    
    Args:
        df: Input dataframe with fraud transaction data
    
    Returns:
        Preprocessed dataframe with encoded features and scaled numeric columns
    
    Raises:
        ValueError: If required columns are missing
    """
    # Validate required columns
    required_cols = ['purchase_value', 'age']
    _validate_required_columns(df, required_cols, "preprocess_fraud")
    
    # Create a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Convert required numeric columns, handling type mismatches gracefully
    for col in required_cols:
        converted = _convert_to_numeric(df, col, "preprocess_fraud")
        if converted is not None:
            df[col] = converted
            # Fill NaN values with median for required columns
            if df[col].isna().any():
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                warnings.warn(
                    f"preprocess_fraud: Filled NaN values in required column '{col}' "
                    f"with median: {median_val}"
                )
    
    # Handle optional categorical columns with frequency encoding
    if 'device_id' in df.columns:
        try:
            # Convert to string if needed for frequency encoding
            if not pd.api.types.is_object_dtype(df['device_id']):
                df['device_id'] = df['device_id'].astype(str)
            
            freq_enc = df['device_id'].value_counts(normalize=True)
            df['device_id_freq'] = df['device_id'].map(freq_enc).fillna(0)
            df.drop(columns=['device_id'], inplace=True)
        except Exception as e:
            warnings.warn(f"preprocess_fraud: Failed to encode 'device_id': {str(e)}")
    
    if 'country' in df.columns:
        try:
            # Convert to string if needed for frequency encoding
            if not pd.api.types.is_object_dtype(df['country']):
                df['country'] = df['country'].astype(str)
            
            freq_enc = df['country'].value_counts(normalize=True)
            df['country_freq'] = df['country'].map(freq_enc).fillna(0)
            df.drop(columns=['country'], inplace=True)
        except Exception as e:
            warnings.warn(f"preprocess_fraud: Failed to encode 'country': {str(e)}")
    
    # Drop ip_address if it exists (not used in modeling)
    if 'ip_address' in df.columns:
        df.drop(columns=['ip_address'], inplace=True)
    
    # Handle time features if both time columns exist
    numeric_cols = ['purchase_value', 'age']
    
    if 'purchase_time' in df.columns and 'signup_time' in df.columns:
        try:
            # Convert to datetime with error handling
            df['purchase_time'] = _convert_to_datetime(df, 'purchase_time', "preprocess_fraud")
            df['signup_time'] = _convert_to_datetime(df, 'signup_time', "preprocess_fraud")
            
            # Only create time features if datetime conversion was successful
            if df['purchase_time'].notna().any() and df['signup_time'].notna().any():
                df['hour_of_day'] = df['purchase_time'].dt.hour.fillna(0)
                df['day_of_week'] = df['purchase_time'].dt.dayofweek.fillna(0)
                
                # Calculate time since signup, handling NaT values
                time_diff = (df['purchase_time'] - df['signup_time']).dt.total_seconds()
                df['time_since_signup'] = time_diff.fillna(time_diff.median())
                
                numeric_cols.append('time_since_signup')
            else:
                warnings.warn(
                    "preprocess_fraud: Time columns could not be converted to datetime. "
                    "Skipping time-based feature engineering."
                )
        except Exception as e:
            warnings.warn(
                f"preprocess_fraud: Failed to process time features: {str(e)}. "
                "Continuing without time-based features."
            )
    
    # Ensure all numeric columns exist and are numeric before scaling
    available_numeric_cols = [
        col for col in numeric_cols 
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
    ]
    
    if not available_numeric_cols:
        raise ValueError(
            f"preprocess_fraud: No valid numeric columns found for scaling. "
            f"Expected: {numeric_cols}, Available: {df.columns.tolist()}"
        )
    
    # Check for infinite values before scaling
    for col in available_numeric_cols:
        if np.isinf(df[col]).any():
            warnings.warn(
                f"preprocess_fraud: Column '{col}' contains infinite values. "
                "Replacing with NaN and filling with median."
            )
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            df[col].fillna(df[col].median(), inplace=True)
    
    # Scale numeric features
    try:
        scaler = StandardScaler()
        df[available_numeric_cols] = scaler.fit_transform(df[available_numeric_cols])
    except Exception as e:
        raise ValueError(
            f"preprocess_fraud: Failed to scale numeric features: {str(e)}. "
            f"Columns: {available_numeric_cols}"
        )
    
    return df


def preprocess_creditcard(df):
    """
    Preprocess credit card transaction dataset with validation and error handling.
    
    Required columns: Time, Amount
    Optional columns: V1-V28 (PCA features)
    
    Args:
        df: Input dataframe with credit card transaction data
    
    Returns:
        Preprocessed dataframe with scaled numeric columns
    
    Raises:
        ValueError: If required columns are missing
    """
    # Validate required columns
    required_cols = ['Time', 'Amount']
    _validate_required_columns(df, required_cols, "preprocess_creditcard")
    
    # Create a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Convert required numeric columns, handling type mismatches gracefully
    numeric_cols = []
    for col in required_cols:
        converted = _convert_to_numeric(df, col, "preprocess_creditcard")
        if converted is not None:
            df[col] = converted
            numeric_cols.append(col)
            # Fill NaN values with median for required columns
            if df[col].isna().any():
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                warnings.warn(
                    f"preprocess_creditcard: Filled NaN values in required column '{col}' "
                    f"with median: {median_val}"
                )
    
    if not numeric_cols:
        raise ValueError(
            f"preprocess_creditcard: No valid numeric columns found for scaling. "
            f"Expected: {required_cols}, Available: {df.columns.tolist()}"
        )
    
    # Check for infinite values before scaling
    for col in numeric_cols:
        if np.isinf(df[col]).any():
            warnings.warn(
                f"preprocess_creditcard: Column '{col}' contains infinite values. "
                "Replacing with NaN and filling with median."
            )
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            df[col].fillna(df[col].median(), inplace=True)
    
    # Scale numeric features
    try:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    except Exception as e:
        raise ValueError(
            f"preprocess_creditcard: Failed to scale numeric features: {str(e)}. "
            f"Columns: {numeric_cols}"
        )
    
    # PCA columns V1-V28 are kept as-is (if they exist)
    return df

def separate_features_target(df, target_col):
    """
    Separate features and target variable from dataframe.
    
    Args:
        df: Input dataframe
        target_col: Name of the target column
    
    Returns:
        Tuple of (X, y) where X is features dataframe and y is target series
    
    Raises:
        ValueError: If target column is missing
    """
    if target_col not in df.columns:
        raise ValueError(
            f"separate_features_target: Target column '{target_col}' not found. "
            f"Available columns: {df.columns.tolist()}"
        )
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Validate that target column has values
    if y.isna().all():
        raise ValueError(
            f"separate_features_target: Target column '{target_col}' contains only NaN values"
        )
    
    return X, y
