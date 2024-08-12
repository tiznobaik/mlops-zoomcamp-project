import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import logging

def preprocess_data(df):
    """
    Preprocess the input DataFrame by performing data cleaning and feature engineering.

    - Drops unnecessary columns.
    - Converts the 'Gender' column to a binary format.
    - Sets up preprocessing pipelines for numeric and categorical features.

    Args:
    - df (pd.DataFrame): The input DataFrame containing the raw data.

    Returns:
    - df (pd.DataFrame): The DataFrame after dropping unnecessary columns and modifying the 'Gender' column.
    - preprocessor (ColumnTransformer): The preprocessor object to be used for transforming the data.
    """
    # Drop unnecessary columns
    columns_to_drop = ['RowNumber', 'CustomerId', 'Surname']
    df.drop(columns=columns_to_drop, inplace=True)

    # Convert Gender to binary (0 and 1)
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

    # Define numeric and categorical features
    numeric_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
    categorical_features = ['Geography']

    # Create a pipeline for numeric features (scaling)
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    # Create a pipeline for categorical features (one-hot encoding)
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine the numeric and categorical transformers into a single ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return df, preprocessor

def split_data(df):
    """
    Split the data into training and validation sets.

    Args:
    - df (pd.DataFrame): The preprocessed DataFrame.

    Returns:
    - X_train (pd.DataFrame): The training features.
    - X_valid (pd.DataFrame): The validation features.
    - y_train (pd.Series): The training target.
    - y_valid (pd.Series): The validation target.
    """
    X = df.drop('Exited', axis=1)
    y = df['Exited']
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_valid, y_train, y_valid

def transform_data(X_train, X_valid, preprocessor):
    """
    Transform the training and validation data using the preprocessor.

    Args:
    - X_train (pd.DataFrame): The training features.
    - X_valid (pd.DataFrame): The validation features.
    - preprocessor (ColumnTransformer): The preprocessor to apply transformations.

    Returns:
    - X_train_preprocessed (np.array): The transformed training features.
    - X_valid_preprocessed (np.array): The transformed validation features.
    """
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_valid_preprocessed = preprocessor.transform(X_valid)
    return X_train_preprocessed, X_valid_preprocessed

if __name__ == "__main__":
    # Load configuration and data
    from loading_data import load_config, load_data
    config = load_config()
    data_path = config['data']['file_path']
    project_id = config['project']['project_id']
    df = load_data(data_path, project_id)
    
    # Preprocess the data
    df, preprocessor = preprocess_data(df)
    
    # Split the data into training and validation sets
    X_train, X_valid, y_train, y_valid = split_data(df)
    
    # Transform the training and validation data
    X_train_preprocessed, X_valid_preprocessed = transform_data(X_train, X_valid, preprocessor)
    
    # Indicate that data preprocessing is complete
    print("Data preprocessed successfully.")