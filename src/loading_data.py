# loading_data.py
import pandas as pd
import yaml
import logging
import gcsfs

logging.basicConfig(level=logging.INFO)


def load_config(config_path='config/config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_data(file_path, project_id):
    logging.info(f"Loading data from {file_path}")
    try:
        fs = gcsfs.GCSFileSystem(project=project_id)
        with fs.open(file_path) as f:
            df = pd.read_csv(f)
        logging.info("Data loaded successfully")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

if __name__ == "__main__":
    config = load_config()
    data_path = config['data']['file_path']
    project_id = config['project']['project_id']
    df = load_data(data_path, project_id)
    print("Data loaded successfully.")