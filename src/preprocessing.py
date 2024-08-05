import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import logging

def preprocess_data(df):
    # Drop unnecessary columns
    columns_to_drop = ['RowNumber', 'CustomerId', 'Surname']
    df.drop(columns=columns_to_drop, inplace=True)

    # Convert Gender to binary (0 and 1)
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

    numeric_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
    categorical_features = ['Geography']

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return df, preprocessor

def split_data(df):
    X = df.drop('Exited', axis=1)
    y = df['Exited']
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_valid, y_train, y_valid

def transform_data(X_train, X_valid, preprocessor):
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_valid_preprocessed = preprocessor.transform(X_valid)
    return X_train_preprocessed, X_valid_preprocessed

if __name__ == "__main__":
    from loading_data import load_config, load_data
    config = load_config()
    data_path = config['data']['file_path']
    project_id = config['project']['project_id']
    df = load_data(data_path, project_id)
    df, preprocessor = preprocess_data(df)
    X_train, X_valid, y_train, y_valid = split_data(df)
    X_train_preprocessed, X_valid_preprocessed = transform_data(X_train, X_valid, preprocessor)
    
    # logging.info("Data preprocessed successfully.")
    print("Data preprocessed successfully.")