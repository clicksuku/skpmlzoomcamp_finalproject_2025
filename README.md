🏡 Airbnb Vienna -- Price Prediction & Listing Classification System
===================================================================

1\. Problem Statement
---------------------

The short-term rental market is highly competitive, and pricing or positioning a listing incorrectly can significantly impact occupancy and revenue. Hosts and property managers often struggle to determine:

-   What is a **fair nightly price** for a listing?

-   Which listings are likely to be **high-performing (premium)** versus **standard**?

This project applies **Machine Learning techniques** on the **Airbnb Vienna public dataset** to solve two core problems:

### Regression Problem

Predict the **nightly price** of an Airbnb listing based on location, room characteristics, availability, host attributes, and review metrics.

### Classification Problem

Classify listings into **Premium vs Non-Premium** categories based on pricing, amenities, and demand-related features.\
This enables better market segmentation and pricing strategy insights.

The project follows an **end-to-end ML lifecycle**: data analysis → modeling → evaluation → deployment using **FastAPI, Docker, and Kubernetes**.

* * * * *

2\. Installation and Running the Project
----------------------------------------

### Clone the Repository

`git clone https://github.com/<your-username>/airbnb-vienna-capstone.git cd airbnb-vienna-capstone `

### Create Virtual Environment

`python3 -m venv mlenv source mlenv/bin/activate `

### Install Dependencies

`pip install -r requirements.txt `

* * * * *

3\. Dataset and Project Details
-------------------------------

### Dataset Source

-   **Dataset**: Airbnb Vienna Open Dataset

-   **Source**: Inside Airbnb (Publicly available)

-   **City**: Vienna, Austria

### Key Dataset Files

-   `listings.csv` -- Detailed listing-level information

-   `reviews.csv` -- Guest review data

-   `calendar.csv` -- Availability and pricing over time

### Key Features Used

| Feature | Description |
| --- | --- |
| `latitude`, `longitude` | Geographical location |
| `room_type` | Entire home / Private room / Shared room |
| `accommodates` | Number of guests |
| `bedrooms`, `beds`, `bathrooms` | Property attributes |
| `number_of_reviews` | Engagement indicator |
| `review_scores_rating` | Guest satisfaction |
| `availability_365` | Demand proxy |
| `host_is_superhost` | Host quality indicator |
| `price` | Target variable (Regression) |

### Data Preprocessing

-   Currency normalization and price cleaning

-   Outlier removal using IQR

-   One-hot encoding for categorical features

-   Missing value imputation

-   Feature scaling for linear models

* * * * *

4\. Regression Model -- Price Prediction
---------------------------------------

### Objective

Predict the **nightly price** of Airbnb listings in Vienna.

### Target Variable

-   `price` (log-transformed for stability)

### Models Implemented

1.  **Linear Regression** (Baseline)

2.  **Random Forest Regressor**

3.  **Gradient Boosting Regressor**

### Evaluation Metrics

-   **RMSE** -- Root Mean Squared Error

-   **MAE** -- Mean Absolute Error

-   **R² Score**

### Model Selection

Gradient Boosting Regressor performed best due to:

-   Non-linear relationship handling

-   Robustness to outliers

-   Superior RMSE and R² scores

* * * * *

5\. Classification Model -- Premium vs Non-Premium Listings
----------------------------------------------------------

### Objective

Classify listings into **Premium** or **Non-Premium** categories.

### Target Definition

`premium = 1  if price > median_price else  0  `

### Models Implemented

1.  **Logistic Regression**

2.  **Decision Tree Classifier**

### Evaluation Metrics

-   **F1 Score**

-   **ROC-AUC**

-   **Confusion Matrix**

-   **Precision & Recall**

### Observations

-   Logistic Regression offered better generalization

-   Decision Trees captured non-linear splits but overfit without pruning

* * * * *

6\. Converting Jupyter Notebooks with Jupytext
----------------------------------------------

To ensure reproducibility and clean pipelines, notebooks were converted to scripts using **Jupytext**.

### Install Jupytext

`pip install jupytext `

### Convert Notebooks

`jupytext --to py notebooks/airbnb_regression.ipynb
jupytext --to py notebooks/airbnb_classification.ipynb `

### Generated Scripts

-   `airbnb_regression.py`

-   `airbnb_classification.py`

* * * * *

7\. Requirements and Installation
---------------------------------

### `requirements.txt`

`pandas
numpy
scikit-learn
matplotlib
seaborn
jupytext
fastapi
uvicorn
pydantic
joblib
requests `

* * * * *

8\. FastAPI Deployment and Serving
----------------------------------

### Run API Locally

`uvicorn api.app:app --host 0.0.0.0 --port 8000 `

### Available Endpoints

| Endpoint | Description |
| --- | --- |
| `/predict_price` | Predict nightly price |
| `/predict_premium` | Predict premium probability |
| `/health` | Health check |

### Example Request

`curl -X POST http://localhost:8000/predict_price\
-H "Content-Type: application/json"\
-d '{"accommodates":4,"bedrooms":2,"availability_365":180}'  `

* * * * *

9\. Docker Deployment
---------------------

### Build Docker Image

`docker build -t airbnb-vienna:latest . `

### Run Container

`docker run -p 8000:8000 airbnb-vienna:latest `

* * * * *

10\. Kubectl Deployment (Local Kubernetes)
------------------------------------------

### Apply Manifests

`kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml `

### Verify

`kubectl get pods
kubectl get svc `

### Access Service (Minikube)

`minikube service airbnb-vienna-svc --url `

* * * * *

11\. Project Highlights
-----------------------

-   End-to-end ML lifecycle

-   Public dataset with real-world complexity

-   Multiple models with comparative evaluation

-   Reproducible pipelines

-   Production-ready deployment (FastAPI + Docker + Kubernetes)

* * * * *

12\. Future Improvements
------------------------

-   Incorporate **calendar-level demand modeling**

-   Add **spatial clustering** (HDBSCAN)

-   Time-series price forecasting

-   Image-based room quality scoring using CNNs
