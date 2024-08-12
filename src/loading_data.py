import pandas as pd
import yaml
import logging
import gcsfs

# Set up logging to output information about the execution
logging.basicConfig(level=logging.INFO)

def load_config(config_path='config/config.yaml'):
    """
    Load configuration settings from a YAML file.

    Args:
    - config_path (str): Path to the YAML configuration file.

    Returns:
    - config (dict): Parsed configuration settings.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_data(file_path, project_id):
    """
    Load data from Google Cloud Storage (GCS) into a pandas DataFrame.

    Args:
    - file_path (str): Path to the data file in GCS.
    - project_id (str): Google Cloud project ID.

    Returns:
    - df (pd.DataFrame): Loaded data in a pandas DataFrame.
    """
    logging.info(f"Loading data from {file_path}")
    try:
        # Initialize the GCS filesystem object using the project ID
        fs = gcsfs.GCSFileSystem(project=project_id)
        
        # Open and read the CSV file from GCS
        with fs.open(file_path) as f:
            df = pd.read_csv(f)
        
        logging.info("Data loaded successfully")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

if __name__ == "__main__":
    # Load configuration settings
    config = load_config()
    
    # Extract file path and project ID from the configuration
    data_path = config['data']['file_path']
    project_id = config['project']['project_id']
    
    # Load the data from GCS
    df = load_data(data_path, project_id)
    
    # Confirm successful data loading
    print("Data loaded successfully.")