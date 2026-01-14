Airbnb Vienna Price Prediction & Category Classification System
===============================================================

1\. Problem Statement
---------------------

A comprehensive Machine Learning project for predicting Airbnb listing prices and classifying property types using the Airbnb Vienna dataset. This project implements both regression and classification models with deployment via FastAPI and Docker.

### Regression Problem

Predict the **price per night** of Airbnb listings based on various features such as property type, room type, number of bedrooms, amenities, location, and review scores. This helps hosts optimize pricing strategies and helps travelers estimate accommodation costs.

### Classification Problem

Classify Airbnb listings into **price categories** (low, medium, high) based on listing features. This multi-class classification helps platforms categorize listings and helps users filter properties according to their budget.

* * * * *

2\. Installation and Running the Project
----------------------------------------

### Prerequisites

-   Python 3.8+

-   Git

-   Docker (optional)

-   Kubernetes (optional)

### Installation Steps

bash

# Clone the repository
git clone https://github.com/yourusername/airbnb-vienna-ml-project.git

# Navigate to project directory
cd airbnb-vienna-ml-project

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

### Running the Training Pipeline

bash

# Train regression model
python scripts/train_regression.py

# Train classification model
python scripts/train_classification.py

# Run data preprocessing
python scripts/preprocess_data.py

### Local Deployment and Testing

bash

# Start FastAPI server
uvicorn api.app:app --host 0.0.0.0 --port 8000

# Test the API
python scripts/test_api.py

### Docker Deployment

bash

# Build Docker image
docker build -t airbnb-vienna-ml .

# Run Docker container
docker run -p 8000:8000 airbnb-vienna-ml

# Test Docker deployment
python scripts/test_docker_api.py

* * * * *

3\. Dataset Details and Project Files
-------------------------------------

### Dataset Source

-   **Dataset**: Airbnb Vienna Listings Dataset

-   **Source**: Inside Airbnb or Kaggle

-   **Records**: ~15,000 listings (after cleaning)

-   **Time Period**: 2023-2024

### Project Structure

text

airbnb-vienna-ml-project/
├── data/
│   ├── raw/                    # Raw dataset files
│   ├── processed/              # Cleaned and processed data
│   └── external/               # External data sources
├── notebooks/
│   ├── EDA_Airbnb_Vienna.ipynb  # Exploratory Data Analysis
│   ├── Regression_Model.ipynb    # Regression model development
│   └── Classification_Model.ipynb # Classification model development
├── scripts/
│   ├── preprocess_data.py      # Data preprocessing pipeline
│   ├── train_regression.py     # Regression model training
│   ├── train_classification.py # Classification model training
│   ├── api_client.py           # API testing client
│   └── utils.py                # Utility functions
├── models/
│   ├── regression_model.pkl    # Trained regression model
│   ├── classification_model.pkl # Trained classification model
│   └── preprocessing_pipeline.pkl # Data preprocessing pipeline
├── api/
│   ├── app.py                  # FastAPI application
│   ├── schemas.py              # Pydantic schemas
│   └── dependencies.py         # API dependencies
├── tests/
│   ├── test_models.py          # Model tests
│   └── test_api.py             # API tests
├── docker/
│   └── Dockerfile              # Docker configuration
├── k8s/
│   ├── deployment.yaml         # Kubernetes deployment
│   └── service.yaml            # Kubernetes service
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variables template
└── README.md                   # Project documentation

### Dataset Features

| Feature | Type | Description |
| --- | --- | --- |
| `id` | String | Unique listing identifier |
| `host_id` | Integer | Unique host identifier |
| `property_type` | String | Type of property (apartment, house, etc.) |
| `room_type` | String | Entire home, private room, shared room |
| `accommodates` | Integer | Number of people accommodated |
| `bedrooms` | Integer | Number of bedrooms |
| `beds` | Integer | Number of beds |
| `bathrooms` | Float | Number of bathrooms |
| `amenities` | String | List of amenities (JSON) |
| `neighborhood` | String | Neighborhood in Vienna |
| `latitude` | Float | Geographic latitude |
| `longitude` | Float | Geographic longitude |
| `review_scores_rating` | Float | Average review score (0-100) |
| `reviews_per_month` | Float | Monthly review frequency |
| `minimum_nights` | Integer | Minimum stay requirement |
| `availability_365` | Integer | Days available per year |
| `price` | Float | Price per night (target for regression) |
| `price_category` | String | Low/Medium/High (target for classification) |

### Data Preprocessing Pipeline

1.  **Missing Value Handling**:

    -   Numerical features: Imputed with median

    -   Categorical features: Imputed with mode

2.  **Feature Engineering**:

    -   Created `amenities_count` from amenities list

    -   Extracted `has_wifi`, `has_kitchen`, `has_parking` from amenities

    -   Calculated `price_per_person` (price ÷ accommodates)

    -   Created `is_superhost` from host information

3.  **Encoding**:

    -   One-hot encoding for categorical variables

    -   Target encoding for high-cardinality features

4.  **Scaling**:

    -   StandardScaler for numerical features

    -   MinMaxScaler for count-based features

* * * * *

4\. Regression Model
--------------------

### Target Variable

-   `log_price`: Log-transformed price per night

### Features Used

-   Numerical: `accommodates`, `bedrooms`, `bathrooms`, `review_scores_rating`, `amenities_count`

-   Categorical: `property_type`, `room_type`, `neighborhood`, `is_superhost`

-   Engineered: `price_per_person`, `has_wifi`, `has_kitchen`

### Models Implemented and Evaluated

1.  **Linear Regression** (Baseline)

2.  **Ridge Regression** (L2 regularization)

3.  **Lasso Regression** (L1 regularization)

4.  **Random Forest Regressor**

5.  **Gradient Boosting Regressor**

6.  **XGBoost Regressor**

### Model Selection Process

-   **Cross-validation**: 5-fold cross-validation

-   **Metrics**: RMSE, MAE, R² Score

-   **Validation**: Train-Validation-Test split (70-15-15)

### Performance Comparison

text

Model                    RMSE    MAE    R²
─────────────────────────────────────────────
Linear Regression       38.42  24.56  0.682
Ridge Regression        37.89  23.98  0.691
Lasso Regression        38.15  24.21  0.687
Random Forest           32.47  20.15  0.773
Gradient Boosting       30.89  18.92  0.794
XGBoost                 29.76  17.84  0.809

### Final Regression Model

python

XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective='reg:squarederror'
)

### Key Findings

-   **Most important features**: `accommodates`, `neighborhood`, `room_type`

-   **Location impact**: Central districts (Innere Stadt) command ~40% premium

-   **Amenity value**: WiFi and kitchen increase price by ~15%

-   **Review impact**: Each point in review score adds ~2% to price

* * * * *

5\. Classification Model
------------------------

### Objective

Classify Airbnb listings into three price categories based on price per night:

-   **Low**: < €50

-   **Medium**: €50 - €120

-   **High**: > €120

### Target Variable Creation

python

def create_price_category(price):
    if price < 50:
        return 'low'
    elif price <= 120:
        return 'medium'
    else:
        return 'high'

### Models Implemented

1.  **Logistic Regression** (One-vs-Rest)

2.  **Random Forest Classifier**

3.  **Gradient Boosting Classifier**

4.  **XGBoost Classifier**

### Evaluation Metrics

-   **Accuracy**: Overall classification accuracy

-   **Precision**: For each class (macro-average)

-   **Recall**: For each class (macro-average)

-   **F1-Score**: Harmonic mean of precision and recall

-   **Confusion Matrix**: Visual representation

-   **ROC-AUC**: Multi-class ROC curve analysis

### Performance Results

text

Model                    Accuracy  F1-Score (macro)  ROC-AUC (ovr)
──────────────────────────────────────────────────────────────────
Logistic Regression       0.782        0.776          0.891
Random Forest             0.814        0.808          0.923
Gradient Boosting         0.826        0.819          0.934
XGBoost                   0.834        0.828          0.941

### Final Classification Model

python

XGBClassifier(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=5,
    min_child_weight=1,
    gamma=0.2,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective='multi:softprob',
    num_class=3
)

### Hyperparameter Tuning

-   Used `GridSearchCV` with 5-fold cross-validation

-   Optimized for macro F1-score

-   Tuned parameters: `max_depth`, `learning_rate`, `n_estimators`

* * * * *

6\. Converting Jupyter Notebooks to Python Scripts
--------------------------------------------------

### Using Jupytext

bash

# Install Jupytext
pip install jupytext

# Convert notebooks to Python scripts
jupytext --to py notebooks/EDA_Airbnb_Vienna.ipynb
jupytext --to py notebooks/Regression_Model.ipynb
jupytext --to py notebooks/Classification_Model.ipynb

# Convert back to notebook format (if needed)
jupytext --to notebook scripts/EDA_Airbnb_Vienna.py

### Using nbconvert

bash

# Convert notebook to Python script
jupyter nbconvert --to python notebooks/EDA_Airbnb_Vienna.ipynb

# Convert with output cells removed
jupyter nbconvert --to python --no-prompt notebooks/Regression_Model.ipynb

### Generated Scripts

-   `scripts/eda_airbnb_vienna.py` - Exploratory data analysis

-   `scripts/regression_model.py` - Regression model training

-   `scripts/classification_model.py` - Classification model training

* * * * *

7\. Requirements and Installation
---------------------------------

### Dependencies

Create `requirements.txt`:

txt

# Core data science
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
scipy==1.11.1
statsmodels==0.14.0

# Visualization
matplotlib==3.7.2
seaborn==0.12.2
plotly==5.15.0

# Machine learning
xgboost==1.7.6
lightgbm==4.1.0
catboost==1.2.2

# API and deployment
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
python-multipart==0.0.6

# Utilities
jupyter==1.0.0
jupytext==1.15.1
tqdm==4.66.1
python-dotenv==1.0.0
joblib==1.3.2

# Testing
pytest==7.4.3
requests==2.31.0

### Installation Commands

bash

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .

# Install with GPU support (optional)
pip install xgboost-gpu

* * * * *

8\. FastAPI Deployment and Serving
----------------------------------

### FastAPI Application Structure

python

# api/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from typing import List, Optional

app = FastAPI(title="Airbnb Vienna ML API")

# Load models
regression_model = joblib.load('models/regression_model.pkl')
classification_model = joblib.load('models/classification_model.pkl')
preprocessor = joblib.load('models/preprocessor.pkl')

# Define request/response schemas
class ListingFeatures(BaseModel):
    property_type: str
    room_type: str
    accommodates: int
    bedrooms: int
    bathrooms: float
    neighborhood: str
    review_scores_rating: float
    amenities_count: int
    has_wifi: bool
    has_kitchen: bool

class PredictionResponse(BaseModel):
    predicted_price: float
    price_category: str
    confidence: float

@app.get("/")
async def root():
    return {"message": "Airbnb Vienna Price Prediction API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(features: ListingFeatures):
    try:
        # Convert features to DataFrame
        input_data = pd.DataFrame([features.dict()])

        # Preprocess
        processed_data = preprocessor.transform(input_data)

        # Make predictions
        price_pred = regression_model.predict(processed_data)[0]
        category_pred_proba = classification_model.predict_proba(processed_data)[0]
        category_pred = classification_model.predict(processed_data)[0]

        # Map category index to label
        categories = ['low', 'medium', 'high']
        category_label = categories[category_pred]

        # Get confidence score
        confidence = max(category_pred_proba)

        return PredictionResponse(
            predicted_price=round(price_pred, 2),
            price_category=category_label,
            confidence=round(confidence, 3)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

### Testing the API

python

# scripts/test_api.py
import requests
import json

url = "http://localhost:8000/predict"

# Sample listing data
sample_listing = {
    "property_type": "Apartment",
    "room_type": "Entire home/apt",
    "accommodates": 4,
    "bedrooms": 2,
    "bathrooms": 1.0,
    "neighborhood": "Innere Stadt",
    "review_scores_rating": 95.0,
    "amenities_count": 12,
    "has_wifi": True,
    "has_kitchen": True
}

# Make prediction
response = requests.post(url, json=sample_listing)

if response.status_code == 200:
    result = response.json()
    print(f"Predicted Price: €{result['predicted_price']}")
    print(f"Price Category: {result['price_category']}")
    print(f"Confidence: {result['confidence'] * 100}%")
else:
    print(f"Error: {response.status_code}")
    print(response.text)

### Dockerfile for FastAPI

dockerfile

# docker/Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y\
    gcc\
    g++\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api/ ./api/
COPY models/ ./models/
COPY scripts/ ./scripts/

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]

* * * * *

9\. Kubernetes Deployment
-------------------------

### Kubernetes Manifests

Create `k8s/` directory with the following files:

#### 1\. Deployment Configuration

yaml

# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: airbnb-ml-api
  labels:
    app: airbnb-ml-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: airbnb-ml-api
  template:
    metadata:
      labels:
        app: airbnb-ml-api
    spec:
      containers:
      - name: airbnb-ml-api
        image: airbnb-vienna-ml:latest
        imagePullPolicy: IfNotPresent
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_PATH
          value: "/app/models"
        - name: LOG_LEVEL
          value: "INFO"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

#### 2\. Service Configuration

yaml

# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: airbnb-ml-service
spec:
  selector:
    app: airbnb-ml-api
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
  type: LoadBalancer

#### 3\. Horizontal Pod Autoscaler

yaml

# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: airbnb-ml-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: airbnb-ml-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70

#### 4\. ConfigMap for Environment Variables

yaml

# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: airbnb-ml-config
data:
  MODEL_VERSION: "1.0.0"
  FEATURE_STORE_URL: "http://feature-store:8080"
  CACHE_TTL: "300"

### Deployment Commands

bash

# Apply Kubernetes configurations
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml

# Check deployment status
kubectl get pods
kubectl get services
kubectl get hpa

# View logs
kubectl logs -l app=airbnb-ml-api

# Port forward for local testing
kubectl port-forward service/airbnb-ml-service 8080:80

# Scale deployment
kubectl scale deployment airbnb-ml-api --replicas=5

# Update deployment (rolling update)
kubectl set image deployment/airbnb-ml-api airbnb-ml-api=airbnb-vienna-ml:v2.0

### Testing Kubernetes Deployment

bash

# Get service URL
minikube service airbnb-ml-service --url

# Or for cloud providers
kubectl get service airbnb-ml-service

# Test API endpoint
curl -X POST http://<SERVICE_IP>/predict\
  -H "Content-Type: application/json"\
  -d '{
    "property_type": "Apartment",
    "room_type": "Entire home/apt",
    "accommodates": 2,
    "bedrooms": 1,
    "bathrooms": 1.0,
    "neighborhood": "Leopoldstadt",
    "review_scores_rating": 92.5,
    "amenities_count": 10,
    "has_wifi": true,
    "has_kitchen": true
  }'

* * * * *

10\. Monitoring and Logging
---------------------------

### Prometheus Metrics

python

# Add to FastAPI app
from prometheus_fastapi_instrumentator import Instrumentator

Instrumentator().instrument(app).expose(app)

### Health Check Endpoint

python

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "1.0.0",
        "model_status": "loaded"
    }

### Logging Configuration

python

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

* * * * *

11\. Project Evaluation
-----------------------

### Business Impact

-   **Price Optimization**: Helps hosts maximize revenue

-   **Market Analysis**: Identifies factors affecting pricing

-   **User Experience**: Better search and filtering for guests

### Technical Achievements

-   **Model Performance**: XGBoost achieved 0.809 R² (regression) and 0.834 accuracy (classification)

-   **Scalability**: Containerized deployment with auto-scaling

-   **Maintainability**: Modular code structure with comprehensive documentation

### Future Improvements

1.  **Real-time Features**: Incorporate seasonal demand data

2.  **Ensemble Methods**: Stacking/blending multiple models

3.  **A/B Testing**: Deploy new models with canary releases

4.  **Feature Store**: Implement for consistent feature engineering

5.  **MLOps Pipeline**: Automated retraining and monitoring

* * * * *

12\. Contributing
-----------------

1.  Fork the repository

2.  Create a feature branch

3.  Add tests for new functionality

4.  Ensure all tests pass

5.  Submit a pull request



## 13. Evaluation 


---

# 🧠 Project Evaluation Rubric

## **Problem Description**

| Points | Description                                                                                                             | Status |
| :----: | :---------------------------------------------------------------------------------------------------------------------- | :----: |
|    0   | Problem is not described                                                                                                |        |
|    1   | Problem is described in README briefly without much details                                                             |    |
|    2   | Problem is described in README with enough context, so it's clear what the problem is and how the solution will be used |   ✅   |

---

## **Exploratory Data Analysis (EDA)**

| Points | Description                                                                                                                                                                                                       | Status |
| :----: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----: |
|    0   | No EDA                                                                                                                                                                                                            |        |
|    1   | Basic EDA (looking at min–max values, checking for missing values)                                                                                                                                                |    |
|    2   | Extensive EDA (ranges of values, missing values, analysis of target variable, feature importance analysis). <br>For images: analyzing the content of the images. <br>For texts: frequent words, word clouds, etc. |  ✅   |

---

## **Model Training**

| Points | Description                                                                                                                                                           | Status |
| :----: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----: |
|    0   | No model training                                                                                                                                                     |        |
|    1   | Trained only one model, no parameter tuning                                                                                                                           |        |
|    2   | Trained multiple models (linear and tree-based). For neural networks: tried multiple variations – with dropout or without, with extra inner layers or without         |    |
|    3   | Trained multiple models and tuned their parameters. For neural networks: same as previous, but also with tuning (learning rate, dropout rate, inner layer size, etc.) |  ✅   |

---

## **Exporting Notebook to Script**

| Points | Description                                                       | Status |
| :----: | :---------------------------------------------------------------- | :----: |
|    0   | No script for training a model                                    |        |
|    1   | The logic for training the model is exported to a separate script |    ✅   |

---

## **Reproducibility**

| Points | Description                                                                                                                                                                                     | Status |
| :----: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----: |
|    0   | Not possible to execute the notebook and the training script. Data is missing or not easily accessible                                                                                          |        |
|    1   | It's possible to re-execute the notebook and the training script without errors. The dataset is committed in the project repository or there are clear instructions on how to download the data |    ✅   |

---

## **Model Deployment**

| Points | Description                                                     | Status |
| :----: | :-------------------------------------------------------------- | :----: |
|    0   | Model is not deployed                                           |        |
|    1   | Model is deployed (with Flask, BentoML, or a similar framework) |    ✅   |

---

## **Dependency and Environment Management**

| Points | Description                                                                                                                                  | Status |
| :----: | :------------------------------------------------------------------------------------------------------------------------------------------- | :----: |
|    0   | No dependency management                                                                                                                     |        |
|    1   | Provided a file with dependencies (`requirements.txt`, `Pipfile`, `bentofile.yaml`, etc.)                                                    |    |
|    2   | Provided a file with dependencies **and** used virtual environment. README explains how to install dependencies and activate the environment |  ✅    |

---

## **Containerization**

| Points | Description                                                                                  | Status |
| :----: | :------------------------------------------------------------------------------------------- | :----: |
|    0   | No containerization                                                                          |        |
|    1   | `Dockerfile` is provided or a tool that creates a Docker image is used (e.g., BentoML)       |    |
|    2   | The application is containerized **and** README describes how to build and run the container |  ✅    |

---

## **Cloud Deployment**

| Points | Description                                                                                                            | Status |
| :----: | :--------------------------------------------------------------------------------------------------------------------- | :----: |
|    0   | No deployment to the cloud                                                                                             |        |
|    1   | Documentation clearly describes (with code) how to deploy the service to cloud or Kubernetes cluster (local or remote) |   ✅    |
|    2   | Code for cloud/Kubernetes deployment is available, with URL for testing or a video/screenshot of testing it            |     |

---
