from google.cloud import aiplatform
from google.cloud import storage
from loading_data import load_config

def upload_model_to_vertex_ai(model_display_name, artifact_uri, project, region):
    """
    Uploads a trained model to Google Cloud's Vertex AI for serving.

    Args:
    - model_display_name (str): The display name of the model in Vertex AI.
    - artifact_uri (str): The URI of the model artifact in Google Cloud Storage.
    - project (str): The Google Cloud project ID.
    - region (str): The Google Cloud region where Vertex AI is located.

    Returns:
    - model: The uploaded model object in Vertex AI.
    """
    # Initialize the AI Platform with the specified project and region
    aiplatform.init(project=project, location=region)

    # Upload the model to Vertex AI
    model = aiplatform.Model.upload(
        display_name=model_display_name,
        artifact_uri=artifact_uri,
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.0-24:latest",
    )

    # Wait for the model upload to complete
    model.wait()
    return model

def list_model_files(bucket_name, prefix):
    """
    Lists all model files in a specified Google Cloud Storage bucket with a given prefix.

    Args:
    - bucket_name (str): The name of the Google Cloud Storage bucket.
    - prefix (str): The prefix (folder path) to filter the files.

    Returns:
    - model_files (list): A list of model file names ending with '.joblib'.
    """
    storage_client = storage.Client()
    blobs = storage_client.list_blobs(bucket_name, prefix=prefix)
    model_files = [blob.name for blob in blobs if blob.name.endswith('.joblib')]
    return model_files

def check_file_exists(bucket_name, file_path):
    """
    Checks if a specific file exists in a Google Cloud Storage bucket.

    Args:
    - bucket_name (str): The name of the Google Cloud Storage bucket.
    - file_path (str): The path to the file in the bucket.

    Returns:
    - exists (bool): True if the file exists, False otherwise.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_path)
    exists = blob.exists()
    print(f"Checking if file exists at {file_path}: {exists}")
    return exists

if __name__ == "__main__":
    # Load the configuration from the YAML file
    config = load_config("config/config.yaml")
    project = config['project']['project_id']
    region = config['region']
    bucket_name = config['bucket_uri'].replace('gs://', '')
    artifact_folder = config['artifact_folder']
    
    # List all model files in the specified bucket and folder
    print(f"Listing files in bucket: {bucket_name} with prefix: {artifact_folder}")
    model_files = list_model_files(bucket_name, artifact_folder)
    print(f"Found model files: {model_files}")
    
    # Check and upload the Random Forest model if it exists
    rf_model_file = next((file for file in model_files if 'random-forest' in file), None)
    xgb_model_file = next((file for file in model_files if 'xgboost' in file), None)
    
    if rf_model_file:
        rf_model_display_name = "best-random-forest-model"
        rf_model_uri = f"gs://{bucket_name}/{rf_model_file}"
        if check_file_exists(bucket_name, rf_model_file):
            try:
                print(f"Uploading Random Forest model from: {rf_model_uri}")
                rf_model = upload_model_to_vertex_ai(rf_model_display_name, rf_model_uri, project, region)
                print(f"Random Forest Model ID: {rf_model.resource_name}")
            except Exception as e:
                print(f"Failed to upload Random Forest model: {e}")
        else:
            print(f"Random Forest model file does not exist at: {rf_model_uri}")
    else:
        print("Random Forest model file not found.")
    
    # Check and upload the XGBoost model if it exists
    if xgb_model_file:
        xgb_model_display_name = "best-xgboost-model"
        xgb_model_uri = f"gs://{bucket_name}/{xgb_model_file}"
        if check_file_exists(bucket_name, xgb_model_file):
            try:
                print(f"Uploading XGBoost model from: {xgb_model_uri}")
                xgb_model = upload_model_to_vertex_ai(xgb_model_display_name, xgb_model_uri, project, region)
                print(f"XGBoost Model ID: {xgb_model.resource_name}")
            except Exception as e:
                print(f"Failed to upload XGBoost model: {e}")
        else:
            print(f"XGBoost model file does not exist at: {xgb_model_uri}")
    else:
        print("XGBoost model file not found.")