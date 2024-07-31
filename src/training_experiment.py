import os
import datetime
import joblib
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from google.cloud import storage
from loading_data import load_data, load_config
from preprocessing import preprocess_data, split_data, transform_data
from google.cloud import aiplatform as ai
import certifi


os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['GRPC_VERBOSITY'] = 'ERROR'

def train_and_evaluate_model(model, X_train, y_train, X_valid, y_valid):
    model.fit(X_train, y_train)
    y_valid_pred = model.predict(X_valid)
    accuracy = accuracy_score(y_valid, y_valid_pred)
    precision = precision_score(y_valid, y_valid_pred)
    recall = recall_score(y_valid, y_valid_pred)
    f1 = f1_score(y_valid, y_valid_pred)
    
    return model, accuracy, precision, recall, f1

def upload_artifact_to_gcs(local_path, bucket_uri, artifacts_folder):
    client = storage.Client()
    bucket = client.bucket(bucket_uri.replace('gs://', ''))
    blob = bucket.blob(f'{artifacts_folder}/{os.path.basename(local_path)}')
    blob.upload_from_filename(local_path)
    return f'{bucket_uri}/{artifacts_folder}/{os.path.basename(local_path)}'

def run_experiment(config, X_train, y_train, X_valid, y_valid):
    scorer = make_scorer(f1_score)
    model_type = config['model_type']
    
    if model_type == 'xgboost':
        param_grid = config['xgboost']['param_grid']
        model = XGBClassifier(eval_metric='logloss')
    elif model_type == 'random-forest':
        param_grid = config['random-forest']['param_grid']
        model = RandomForestClassifier()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    print(f"Starting {model_type} grid search...")
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scorer, cv=3, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(f"{model_type} grid search completed.")

    best_run_name = None
    best_run_metrics = None

    print(f"Starting {model_type} experiment runs...")
    with tqdm(total=len(grid_search.cv_results_['params'])) as pbar:
        for i, (params, mean_score, std_score) in enumerate(zip(grid_search.cv_results_['params'], grid_search.cv_results_['mean_test_score'], grid_search.cv_results_['std_test_score'])):
            run_name = f'{model_type}-run-{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}-{i}'
            print(f"Starting run: {run_name}")
            with ai.start_run(run_name) as run:
                model = grid_search.estimator.set_params(**params)
                model, accuracy, precision, recall, f1 = train_and_evaluate_model(model, X_train, y_train, X_valid, y_valid)
                run.log_params(params)
                metrics = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'mean_test_score': mean_score,
                    'std_test_score': std_score
                }
                run.log_metrics(metrics)
                print(f"Completed run: {run_name} with F1 score: {f1}")
                if best_run_name is None or f1 > best_run_metrics['f1_score']:
                    best_run_name = run_name
                    best_run_metrics = metrics
            pbar.update(1)

    print(f"{model_type} experiment runs completed.")

    print(f"Saving the best {model_type} model artifact...")
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    run_name = f'best-{model_type}-run-{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}'
    with ai.start_run(run_name) as run:
        model, accuracy, precision, recall, f1 = train_and_evaluate_model(best_model, X_train, y_train, X_valid, y_valid)
        run.log_params(best_params)
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        run.log_metrics(metrics)
        artifact_path = f'{run_name}.joblib'
        joblib.dump(model, artifact_path)
        artifact_gcs_path = upload_artifact_to_gcs(artifact_path, config['bucket_uri'], config['artifact_folder'])
        run.log_params({'model_artifact': artifact_gcs_path})
        os.remove(artifact_path)
    print(f"Best {model_type} model artifact saved.")

def main():
    config = load_config()

    # Initialize the AI Platform with the experiment name
    ai.init(experiment=config['experiment_name'])

    df = load_data(config['data_path'], config['project_id'])
    df, preprocessor = preprocess_data(df)
    X_train, X_valid, y_train, y_valid = split_data(df)
    X_train_preprocessed, X_valid_preprocessed = transform_data(X_train, X_valid, preprocessor)

    run_experiment(config, X_train_preprocessed, y_train, X_valid_preprocessed, y_valid)

if __name__ == "__main__":
    main()