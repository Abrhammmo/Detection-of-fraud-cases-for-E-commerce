# src/preprocessing.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def preprocess_fraud(df):
    print("in preprocess_fraud, columns:", df.columns.tolist())
    # Example: Encode categorical
    if 'device_id' in df.columns:
        freq_enc = df['device_id'].value_counts(normalize=True)
        df['device_id_freq'] = df['device_id'].map(freq_enc)
        df.drop(columns=['device_id'], inplace=True)
        
    # Encode country if exists
    if 'country' in df.columns:
        freq_enc = df['country'].value_counts(normalize=True)
        df['country_freq'] = df['country'].map(freq_enc)
        df.drop(columns=['country'], inplace=True)
    
    # Example: Time features - only if columns exist
    if 'purchase_time' in df.columns and 'signup_time' in df.columns:
        df['purchase_time'] = pd.to_datetime(df['purchase_time'])
        df['signup_time'] = pd.to_datetime(df['signup_time'])
        df['hour_of_day'] = df['purchase_time'].dt.hour
        df['day_of_week'] = df['purchase_time'].dt.dayofweek
        df['time_since_signup'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds()
        numeric_cols = ['purchase_value', 'age', 'time_since_signup']
    else:
        numeric_cols = ['purchase_value', 'age']
    
    # Scale numeric features
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return df

def preprocess_creditcard(df):
    # Scale numeric columns
    numeric_cols = ['Time', 'Amount']
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    # PCA columns V1-V28 are kept as-is
    return df

def separate_features_target(df, target_col):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y
