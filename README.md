# Bank Customer Churn Prediction

In this project, I aimed to develop a robust machine learning pipeline to predict bank customer churn using the dataset provided on Kaggle. The dataset includes customer demographic information, account details, and behavioral patterns, which are crucial for identifying the likelihood of a customer leaving the bank. The primary objective was to build an accurate prediction model while establishing an MLOps workflow that ensures seamless deployment, monitoring, and retraining of the model. The project was implemented on Google Cloud Platform (GCP), with Python scripts executed from a local machine. Initial Exploratory Data Analysis (EDA) and experimental tracking were conducted using notebooks. This project showcases the end-to-end process, from data preprocessing and model training to implementing data drift monitoring and CI/CD practices in a cloud-based environment.

## Summary of Python Scripts:

### loading_data.py

This script is responsible for loading the dataset from Google Cloud Storage (GCS). The key components of the script include:

	1.	Configuration Loading: It loads configuration settings from a YAML file located at config/config.yaml.
	2.	Data Loading: The script uses the gcsfs library to access Google Cloud Storage and load the data file specified in the configuration. The data is read into a Pandas DataFrame.
	3.	Logging: It includes logging to track the progress of the data loading process, providing information on whether the data was loaded successfully or if an error occurred.

The script can be executed as a standalone program, and upon execution, it loads the configuration, retrieves the data from GCS, and confirms successful data loading.


### preprocessing.py

This script handles the preprocessing of the dataset before feeding it into a machine learning model. The key functions and processes include:

	1.	Data Cleaning and Feature Engineering:
	•	Unnecessary columns such as RowNumber, CustomerId, and Surname are dropped from the dataset.
	•	The Gender column is converted to a binary format (1 for Male and 0 for Female).
	2.	Feature Transformation:
	•	The script defines two types of transformers:
	•	Numeric Transformer: Applies standard scaling to numeric features like CreditScore, Age, Tenure, Balance, NumOfProducts, and EstimatedSalary.
	•	Categorical Transformer: Applies one-hot encoding to categorical features like Geography.
	3.	Data Splitting:
	•	The dataset is split into training and validation sets using train_test_split, with a test size of 20% and a fixed random state for reproducibility.
	4.	Data Transformation:
	•	The script transforms the training and validation data using the defined preprocessor, which combines the numeric and categorical transformers.
	5.	Main Execution:
	•	The script can be run as a standalone program. When executed, it loads the configuration and data, preprocesses the data, splits it into training and validation sets, and applies the necessary transformations.


### training_experiment.py

This script is responsible for training and evaluating machine learning models, conducting hyperparameter tuning, and managing the experimental process. The key components include:

	1.	Model Training and Evaluation:
	•	The train_and_evaluate_model function trains the model on the training set and evaluates it on the validation set. It calculates key metrics such as accuracy, precision, recall, and F1-score.
	2.	Grid Search for Hyperparameter Tuning:
	•	The run_experiment function is designed to conduct a grid search for hyperparameter tuning. Depending on the specified model type (either XGBoost or Random Forest), it sets up a parameter grid and performs the grid search using cross-validation to find the best hyperparameters.
	3.	Artifact Upload to Google Cloud Storage (GCS):
	•	The upload_artifact_to_gcs function handles the uploading of local artifacts (e.g., trained models) to a specified Google Cloud Storage bucket.
	4.	Integration with Google Cloud AI Platform:
	•	The script includes functionality to integrate with Google Cloud AI Platform for model deployment, though this part is not fully expanded in the snippet provided.
	5.	Main Execution:
	•	The script can be executed to load the configuration, preprocess the data, split it into training and validation sets, and run the training and evaluation process with hyperparameter tuning. The best model is selected based on the F1-score.


### deploy_model.py

This script is responsible for deploying the trained machine learning models to Google Cloud’s Vertex AI for serving. The key functions and processes include:

	1.	Model Upload to Vertex AI:
	•	The upload_model_to_vertex_ai function handles the process of uploading a trained model to Google Cloud’s Vertex AI. It initializes the AI Platform with the specified project and region and uploads the model artifact, specifying the container image for serving the model.
	2.	Listing Model Artifacts in GCS:
	•	The list_model_files function lists all model files stored in a specified Google Cloud Storage bucket. It filters out files ending with .joblib, which are assumed to be the trained models.
	3.	File Existence Check:
	•	The check_file_exists function checks whether a specific model file exists in the Google Cloud Storage bucket. It prints the result and returns a boolean indicating the file’s presence.
	4.	Main Execution:
	•	When run, the script loads the configuration settings, retrieves the list of model files in the GCS bucket, and identifies the best model files (Random Forest or XGBoost). It then uploads the selected model to Vertex AI if the file exists.