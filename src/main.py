import argparse
from data_preprocessing import load_and_split_data, transform_data
from model_training import train_and_tune_model, evaluate_model

def main(args):
    # Load, clean, and split data
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_and_split_data(args.data_path)
    
    # Create and apply preprocessing pipeline
    X_train_preprocessed_df, X_valid_preprocessed_df, X_test_preprocessed_df = transform_data(X_train, X_valid, X_test)
    
    # Train and tune model
    best_model, best_params = train_and_tune_model(X_train_preprocessed_df, y_train)
    
    # Evaluate model
    evaluate_model(best_model, X_valid_preprocessed_df, y_valid, X_test_preprocessed_df, y_test)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a model.")
    parser.add_argument('--data-path', type=str, required=True, help="Path to the dataset CSV file.")
    args = parser.parse_args()

    main(args)