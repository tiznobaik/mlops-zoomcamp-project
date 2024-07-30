#!/usr/bin/env python3

import argparse
from google.cloud import aiplatform as ai
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


# Configuration
PROJECT_ID = 'vertextutorial-429522'
BUCKET_URI = 'gs://mlops_churn_bucket'
REGION = 'us-central1'

# Initialize Vertex AI
ai.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET_URI)


def load_and_split_data(file_path):
    """Load, clean, and split the data into train, validation, and test sets."""
    # Load the dataset
    df = pd.read_csv(file_path)

    # Drop unnecessary columns
    columns_to_drop = ['RowNumber', 'CustomerId', 'Surname']
    df.drop(columns=columns_to_drop, inplace=True)

    # Convert Gender to binary (0 and 1)
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

    # Define features and target
    X = df.drop(columns=['Exited'])
    y = df['Exited']

    # Split the data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    # Verify the sizes
    print(f"Training set size: {X_train.shape[0]} records")
    print(f"Validation set size: {X_valid.shape[0]} records")
    print(f"Test set size: {X_test.shape[0]} records")

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def transform_data(X_train, X_valid, X_test):
    """Create a preprocessing pipeline and apply it to the data."""
    # Identify numeric, binary, and categorical columns
    numeric_features = ['Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
    binary_features = ['HasCrCard', 'IsActiveMember', 'Gender']
    categorical_features = ['Geography']
    binning_features = ['CreditScore']

    # Create the preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('bin', 'passthrough', binary_features),  # Pass through binary features without change
            ('cat', OneHotEncoder(drop='first'), categorical_features),  # One-hot encode categorical features
            ('credit_bin', KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform'), binning_features)  # Binning for CreditScore
        ], remainder='passthrough')  # Ensure passthrough for any remaining columns

    # Create the full pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor)
    ])

    # Fit and transform the training data
    X_train_preprocessed = pipeline.fit_transform(X_train)

    # Transform the validation and test data
    X_valid_preprocessed = pipeline.transform(X_valid)
    X_test_preprocessed = pipeline.transform(X_test)

    # Print the shape of the transformed training data
    print(f"Transformed training data shape: {X_train_preprocessed.shape}")

    # Generate feature names for all features including one-hot encoded features
    onehot_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features).tolist()
    preprocessor_feature_names = (
        preprocessor.named_transformers_['num'].get_feature_names_out(numeric_features).tolist() +
        binary_features +
        onehot_feature_names +
        binning_features  # Single binned CreditScore column
    )

    # Ensure the number of feature names matches the number of columns in the transformed data
    assert len(preprocessor_feature_names) == X_train_preprocessed.shape[1], "Number of feature names does not match the number of columns in the transformed data."

    # Convert the preprocessed data back to DataFrame for better readability
    X_train_preprocessed_df = pd.DataFrame(X_train_preprocessed, columns=preprocessor_feature_names)
    X_valid_preprocessed_df = pd.DataFrame(X_valid_preprocessed, columns=preprocessor_feature_names)
    X_test_preprocessed_df = pd.DataFrame(X_test_preprocessed, columns=preprocessor_feature_names)

    return X_train_preprocessed_df, X_valid_preprocessed_df, X_test_preprocessed_df

def train_and_evaluate_model(X_train, y_train, X_valid, y_valid):
    """Train a model and evaluate it."""
    # Train a RandomForest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions on the validation set
    y_valid_pred = model.predict(X_valid)

    # Calculate metrics
    accuracy = accuracy_score(y_valid, y_valid_pred)
    precision = precision_score(y_valid, y_valid_pred)
    recall = recall_score(y_valid, y_valid_pred)
    f1 = f1_score(y_valid, y_valid_pred)

    return model, accuracy, precision, recall, f1



def main(args):
    # Create an experiment if it doesn't already exist
    experiment_name = 'gcp-experiment'
    experiment = ai.Experiment(experiment_name=experiment_name, project=PROJECT_ID, location=REGION)
    
    # Start an experiment run
    with ai.ExperimentRun(experiment=experiment, run_name='gcp-run-01') as run:
        # Load and preprocess data
        X_train, X_valid, X_test, y_train, y_valid, y_test = load_and_split_data(args.data_path)
        X_train_preprocessed, X_valid_preprocessed, X_test_preprocessed, feature_names = transform_data(X_train, X_valid, X_test)

        # Train and evaluate model
        model, accuracy, precision, recall, f1 = train_and_evaluate_model(X_train_preprocessed, y_train, X_valid_preprocessed, y_valid)

        # Log metrics to Vertex AI
        run.log_metrics(accuracy=accuracy, precision=precision, recall=recall, f1_score=f1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run ML experiment with Vertex AI')
    parser.add_argument('--data-path', type=str, required=True, help='Path to the CSV data file')
    args = parser.parse_args()
    main(args)