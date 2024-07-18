from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm

def train_and_tune_model(X_train, y_train):
    """Train and perform hyperparameter tuning using GridSearchCV."""
    # Define the parameter grid for XGBoost
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }

    # Initialize the XGBoost model
    xgb_model = XGBClassifier(eval_metric='logloss')

    # Set up GridSearchCV
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5, scoring='f1', n_jobs=-1)

    # Fit GridSearchCV on the training data with progress bar
    for _ in tqdm(range(1), desc="Training and hyperparameter tuning"):
        grid_search.fit(X_train, y_train)

    # Get the best model and parameters
    best_xgb = grid_search.best_estimator_
    best_params = grid_search.best_params_

    print("Best parameters found by GridSearchCV:")
    print(best_params)

    return best_xgb, best_params

def evaluate_model(model, X_valid, y_valid, X_test, y_test):
    """Evaluate the model on the validation and test sets."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
    
    # Evaluate the model on the validation set with progress bar
    for _ in tqdm(range(1), desc="Evaluating on validation set"):
        y_valid_pred = model.predict(X_valid)

    print("Validation Set Performance:")
    print(f"Accuracy: {accuracy_score(y_valid, y_valid_pred):.4f}")
    print(f"Precision: {precision_score(y_valid, y_valid_pred):.4f}")
    print(f"Recall: {recall_score(y_valid, y_valid_pred):.4f}")
    print(f"F1 Score: {f1_score(y_valid, y_valid_pred):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_valid, y_valid_pred))
    print("Classification Report:")
    print(classification_report(y_valid, y_valid_pred))

    # Evaluate the model on the test set with progress bar
    for _ in tqdm(range(1), desc="Evaluating on test set"):
        y_test_pred = model.predict(X_test)

    print("Test Set Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_test_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_test_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_test_pred):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_test_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_test_pred))