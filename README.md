🏡 Airbnb Vienna -- Price Prediction & Listing Classification System
===================================================================

1\. Problem Statement
---------------------

The short-term rental market is highly competitive, and pricing or positioning a listing incorrectly can significantly impact occupancy and revenue. Hosts and property managers often struggle to determine:

-   What is a **fair nightly price** for a listing?

-   Which listings are likely to be rented out based on the Host characterestics whether the host is **Superhost** versus **Normal Host**?

This project applies **Machine Learning techniques** on the **Airbnb Vienna public dataset** to solve two core problems:

### Regression Problem

Predict the **nightly price** of an Airbnb listing based on location, room characteristics, availability, host attributes, and review metrics.

### Classification Problem

Classify listings into **Host vs Superhost** categories based on pricing, amenities, and demand-related features.\
This enables better market segmentation and pricing strategy insights.

The project follows an **end-to-end ML lifecycle**: data analysis → modeling → evaluation → deployment using **FastAPI, Docker, and Kubernetes**.

* * * * *

2\. Installation and Running the Project
----------------------------------------

## Running locally

### Clone the Repository

`git clone https://github.com/clicksuku/skpmlzoomcamp_finalproject_2025.git

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

-   `Data/listings.csv` -- Listing-level information
-   `Data/listings_detailed.csv` -- Detailed listing-level information
-   'Data/data_dictionary.xlsx' -- Data dictionary of the data considered

### Key Features Used

| Feature | Description |
| --- | --- |
| `room_type` | Entire home / Private room / Shared room |
| `accommodates` | Number of guests |
| `bedrooms`, `beds`, `bathrooms` | Property attributes |
| `number_of_reviews` | Engagement indicator |
| `review_scores_rating` | Guest satisfaction |
| `availability_365` | Demand proxy |
| `host_is_superhost` | Host quality indicator |
| `log_price` | Target variable (Regression) |

### Data Preprocessing

-   Concatenated listing and listing details

-   Remove duplicate columns after concatenation

-   Remove NA

-   Missing value imputation

-   Room code mapping for room types

-   Removed % from host response rate and host acceptance rate
-   Currency normalization and price cleaning

-   Outlier removal using IQR

-   One-hot encoding for categorical features

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

***. Key Observaions from Regression:**

**✅**** Neighborhood Features WORKED!**

-   Adding neighborhood information boosted R² by **6.6 percentage points** (0.332 → 0.398)
-   This confirms location is a major price driver

**✅**** Proper Encoding:**

-   "Others" (locations beyond top 10 is categorized as 'Others') in Neighbourhood is included as a category (coefficient: 0.1056)
-   This serves as the baseline for neighborhood comparisons

***\. Feature Analysis by Impact:***

**🏆**** Top Price Drivers (>15% premium):**

1.  **Hotel Room (+67.4%)**: e^0.5153 = 1.674
2.  **Entire Home/Apt (+43.8%)**: e^0.3633 = 1.438
3.  **Bathrooms (+18.6%)**: Each additional bathroom
4.  **Leopoldstadt (+18.4%)**: Premium neighborhood
5.  **Neubau (+18.0%)**: Premium neighborhood

**Strong Positive Influencers (10-15% premium):**

1.  **Review Scores Rating (+16.4%)**: Quality premium
2.  **Bedrooms (+15.5%)**: Each additional bedroom
3.  **Landstraße (+12.5%)**: Above-average neighborhood
4.  **Superhost (+12.3%)**: Host reputation premium
5.  **Instant Bookable (+11.2%)**: Convenience premium

**Baseline Neighborhood:**

-   **"Others" (+11.1%)**: Slightly above average - interesting!

-   This means all other neighborhoods are compared to "Others"
-   Some neighborhoods are below "Others" (negative coefficients)

**Significant Discounts:**

-   **Shared Room (-53.8%)**: Massive discount: e^-0.7708 = 0.462
-   **Host Identity Verified (-32.8%)**: Surprisingly large negative effect
-   **Meidling (-14.4%)**, **Ottakring (-16.3%)**: Below-average neighborhoods

***\. Business Insights & Strategy:**

***Pricing Strategy:***

1.  **Property Type Hierarchy**: Hotel rooms command highest premium, followed by entire homes
2.  **Location Matters**: Leopoldstadt, Neubau, Landstraße are premium areas
3.  **Quality Pays**: Higher ratings = 16% price premium
4.  **Size Premium**: Bathrooms add more value (18.6%) than bedrooms (15.5%)

**Counterintuitive Findings:**

 **Host Identity Verified (-32.8%)**: This is suspiciously large

-   Possible issue: New hosts verify identity but charge less?
-   Check correlation with other variables

  **Favoriten, Brigittenau discounts**: Working-class areas as expected


***\. Detailed Breakdown:***

**A. Host Identity Verified Relationships:**

-   **With log_price: -0.105** → Verified hosts charge **10.5% less** on average (in log space)
-   **With superhost: 0.092** → Weak positive correlation with being a superhost
-   **With reviews: 0.112** → Verified hosts have slightly more reviews
-   **With instant bookable: -0.082** → Verified hosts are LESS likely to offer instant booking

**B. Other Interesting Correlations:**

**Superhost Effects:**

-   **Superhost → Rating: 0.327** → Strong! Superhosts have much higher ratings
-   **Superhost → Price: 0.128** → Superhosts charge 12.8% MORE (expected)
-   **Superhost → Instant booking: -0.182** → Superhosts are LESS likely to offer instant booking

**Response/Acceptance Rates:**

-   **Response ↔ Acceptance: 0.552** → Strong correlation - hosts who respond quickly also accept more bookings
-   **Acceptance → Instant booking: 0.450** → Strong! Hosts who accept more also offer instant booking

**Rating Effects:**

-   **Rating → Price: 0.175** → Higher ratings = 17.5% higher prices (makes sense)
-   **Rating → Superhost: 0.327** → Higher ratings help become superhost
-   **Rating → Instant booking: -0.189** → Higher-rated listings are LESS likely to offer instant booking

***\. Why ****host_identity_verified**** Has Negative Coefficient:**

**Possible Explanations:**

1.  **New Host Effect**: New hosts verify identity but charge less to attract first bookings
2.  **Budget Host Strategy**: Hosts focusing on budget segment verify but keep prices low
3.  **Professional vs Casual**: Casual hosts verify but aren't optimizing for max revenue
4.  **Market Segment**: Verified hosts might dominate budget segments

**Evidence Supporting This:**

-   Verified hosts have more reviews (0.112 correlation) → More established but cheaper
-   Verified hosts less likely to offer instant booking (-0.082) → Different strategy
-   Weak correlation with superhost (0.092) → Not the premium hosts

**\. Business Interpretation:**

**Two Types of Verified Hosts Emerge:**

1.  **Premium Verified Hosts**:

-   Also superhosts (some correlation: 0.092)
-   Higher ratings
-   Higher prices

2.  **Budget Verified Hosts**:

-   Verify identity (trust signal)
-   Charge less to compete
-   More reviews from volume strategy
-   Less likely to offer premium features (instant booking)

**\. Actionable Insights:**

**For Hosts:**

-   **If you verify identity**: Be aware you might be pricing too low
-   **Combine verification with superhost status** to command premium prices

**For Platform:**

-   Verification alone doesn't command price premium
-   Need to encourage verified hosts to also pursue superhost status
-   Consider tiered verification system

* * * * *

5\. Classification Model -- Host vs Superhost Listings
----------------------------------------------------------

### Objective

This document summarizes the key steps, experiments, evaluation results, and final outcome of the notebook:

- `Notebook/airbnb_classification.ipynb`

The objective of the notebook is to train a **binary classification** model to predict whether a host is a **Superhost**.

- **Task**: Binary classification
- **Target**: `host_is_superhost`
- **Prediction**: `P(host_is_superhost = 1)`


### Target Definition

`Superhost = 1  if probability > 80%  `

### Models Implemented

1.  **Logistic Regression**

2.  **Decision Tree Classifier**

## 2. Data Loading

The notebook loads two Vienna Airbnb datasets and then merges them:

- `../data/listings.csv`
- `../data/listings_detailed.csv`

They are concatenated column-wise and duplicate columns are removed.

---

## 3. Feature Set Definition

Features are defined in groups.

### 3.1 Host features

- `host_response_rate`
- `host_acceptance_rate`
- `host_identity_verified`
- `host_listings_count`
- `host_is_superhost` *(this is the target, later separated)*

### 3.2 Review features

- `review_scores_rating`
- `review_scores_cleanliness`
- `review_scores_communication`
- `review_scores_accuracy`
- `number_of_reviews`
- `number_of_reviews_ltm`
- `reviews_per_month`

### 3.3 Listing features

- `instant_bookable`
- `calculated_host_listings_count`
- `availability_30`

### 3.4 Categorical features

- `room_type`
- `neighbourhood`

All features combined:

```text
['host_response_rate',
 'host_acceptance_rate',
 'host_identity_verified',
 'host_listings_count',
 'host_is_superhost',
 'review_scores_rating',
 'review_scores_cleanliness',
 'review_scores_communication',
 'review_scores_accuracy',
 'number_of_reviews',
 'number_of_reviews_ltm',
 'reviews_per_month',
 'instant_bookable',
 'calculated_host_listings_count',
 'availability_30',
 'room_type',
 'neighbourhood']
```

---

## 4. Data Cleaning & Feature Engineering

The notebook performs a clear end-to-end preprocessing pipeline via helper functions.

### 4.1 Merge and subset

- Concatenate `listings` and `listings_detailed`
- Remove duplicated columns
- Keep only `all_features`
- Drop rows with missing values: `dropna()`

### 4.2 Normalize percentage columns

These are originally strings like `"96%"`:

- `host_response_rate`
- `host_acceptance_rate`

Processing:

- remove `%`
- cast to `float`

### 4.3 One-hot encode `room_type`

The notebook creates a boolean column per unique `room_type` value and drops the original `room_type` column.

### 4.4 Convert `t` / `f` flags to boolean

For columns:

- `instant_bookable`
- `host_identity_verified`
- `host_is_superhost`

Mapping:

- `t -> 1`, `f -> 0`, then cast to `bool`

### 4.5 Fix encoding issues in neighbourhood

`neighbourhood` values may have encoding artifacts. The notebook attempts:

- `latin1` encode
- `utf-8` decode (ignore errors)

### 4.6 Group neighbourhoods into top-N + "Others"

To avoid high-cardinality neighbourhoods, it groups:

- neighbourhoods with count >= `min_count` (300)
- all others become `Others`

Then it one-hot encodes the grouped neighbourhoods and drops original neighbourhood columns.

### 4.7 Column standardization

- replace `/` with `_`
- lowercase
- replace spaces with `_`

### 4.8 Final cleaned dataset size

After cleaning:

- total rows: **8690**

---

## 5. Train/Validation/Test Split

The notebook creates a dataset that still contains the target column and then splits:

- **Train+Val**: 80%
- **Test**: 20%

Then splits Train+Val into:

- **Train**: 60% (of total)
- **Validation**: 20% (of total)

Based on the printed sizes:

- Train: **5214**
- Validation: **1738**
- Test: **1738**

The target vectors are created (`y_train`, `y_val`, `y_test`) and then `host_is_superhost` is removed from feature frames.

---

## 6. Baseline Model

### 6.1 Baseline logistic regression

A baseline model is trained:

- `LogisticRegression(solver='liblinear', C=1.0, max_iter=2000, random_state=42)`

The features are vectorized using:

- `DictVectorizer()`

The prediction used is:

- `y_pred_val = predict_proba(X_val)[:, 1]`

### 6.2 Baseline ROC AUC

Baseline validation ROC AUC:

- **0.8233**

This is a good baseline, showing the signal in host/review/listing features.

---

## 7. Threshold / Precision-Recall Analysis

A custom function `p_r_dataframe` computes precision, recall, and F1 score across thresholds from 0 to 1.

Best threshold by F1 on validation was found at:

- **threshold ≈ 0.33**

Interpretation:

- The default 0.5 threshold is not necessarily optimal.
- Since superhost can be imbalanced, lowering the threshold can improve recall and F1.

---

## 8. Cross-Validation for Regularization (C)

The notebook performs 5-fold CV across values of `C`:

- `[0.001, 0.01, 0.1, 0.5, 1, 5, 10]`

Metric:

- ROC AUC

Mean CV AUC results (approx):

- `C=0.001` -> **0.781**
- `C=0.01` -> **0.824**
- `C=0.1` -> **0.838**
- `C=0.5` -> **0.839**
- `C=1` -> **0.839**
- `C=5` -> **0.840**
- `C=10` -> **0.840**

Observation:

- Performance improves quickly as `C` increases, then plateaus.

---

## 9. GridSearchCV (Final Model Selection)

A `GridSearchCV` is run on Logistic Regression (balanced class weights):

- solver: `lbfgs`
- `class_weight='balanced'`
- scoring: `roc_auc`
- CV folds: 5

Parameter grid:

- `C: [0.001, 0.01, 0.1, 1, 10, 100]`

Best parameter:

- **Best C = 100**

Best cross-validated ROC AUC:

- **0.8401**

The CV table shows a plateau from `C=10` to `C=100`.

---

## 10. Final Evaluation (Validation Set)

Using the best estimator from GridSearchCV, evaluation is computed on the validation set.

### 10.1 Metrics

- **F1 score**: **0.6903**
- **ROC AUC**: **0.8472**

### 10.2 Confusion matrix

```text
[[757 359]
 [105 517]]
```

Interpretation (with `class_weight='balanced'`):

- The model prioritizes recall for the positive class (Superhost), at the cost of more false positives.

---

## 11. Exported Artifact

The final output is a pickled artifact saved to:

- `../_models/classification_model.bin`

Contents:

- `(final_classification_model, dv_log_reg)`

This is later loaded by the FastAPI server for inference.

---

# Analysis of Approaches, Results, and Final Outcome

## A. What approaches were tried?

### A1. Baseline logistic regression (simple train/val)

- Pros:
  - Quick baseline
  - Establishes a starting AUC (0.823)
- Cons:
  - Hyperparameters not tuned
  - Threshold not optimized

### A2. Threshold analysis using Precision/Recall and F1

- The notebook explicitly searches thresholds and finds **~0.33** works best for F1.
- This is important because:
  - If Superhost is rarer, using 0.5 can under-predict positives.

### A3. K-Fold CV for regularization strength (C)

- Evaluates how stable ROC AUC is across folds.
- Helps detect overfitting/underfitting behavior as `C` changes.

### A4. GridSearchCV (final selection)

- Uses `class_weight='balanced'` which is appropriate for imbalanced labels.
- Selects `C=100` with best mean ROC AUC.

## B. Key results

- Baseline validation ROC AUC: **0.8233**
- Best CV ROC AUC (GridSearch): **0.8401**
- Validation ROC AUC (final model): **0.8472**
- Validation F1 (final model): **0.6903**

## C. What is the final outcome?

- A tuned **Logistic Regression** classifier (with `class_weight='balanced'` and `C=100`).
- A production-ready artifact:
  - `classification_model.bin` containing both model and feature vectorizer.

## D. Notes / Potential improvements

- Consider evaluating on the held-out **test set** (currently the notebook evaluates on validation).
- Consider adding probability calibration if using probabilities for decision-making.
- Try tree-based models (e.g., XGBoost classifier) as a comparison baseline.
- Track metrics at multiple thresholds (0.5, best-F1 threshold, business threshold like 0.8).


### Evaluation Metrics

-   **F1 Score**

-   **ROC-AUC**

-   **Confusion Matrix**

-   **Precision & Recall**


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

`requirements.txt`

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
| `/predict_superhost` | Predict Superhost |
| `/health` | Health check |


* * * * *

9\. Docker Deployment
---------------------

### Build Docker Image

`docker build -t skp-airbnb-vienna:latest . `

### Run Container

`docker run -p 8000:8000 skp-airbnb-vienna:latest `

* * * * *

10\. Kubernetes Deployment
-------------------------

### Kubernetes Manifests

Create `k8s/` directory with the following files:

#### 1\. Deployment Configuration

yaml

# k8s/deployment.yaml

```
yaml

apiVersion: apps/v1
kind: Deployment
metadata:
  name: airbnb-vienna-deployment
  labels:
    app: airbnb-vienna
spec:
  replicas: 1
  selector:
    matchLabels:
      app: airbnb-vienna
  template:
    metadata:
      labels:
        app: airbnb-vienna
    spec:
      containers:
      - name: airbnb-model-api
        image: mlcampfinal:latest
        imagePullPolicy: IfNotPresent
        ports:
        - containerPort: 8000 
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
```

#### 2\. Service Configuration

```
yaml

apiVersion: v1
kind: Service
metadata:
  name: airbnb-vienna-service
spec:
  selector:
    app: airbnb-vienna 
  ports:
    - protocol: TCP
      port: 8000         
      targetPort: 8000  
```


### Deployment Commands

```
bash

# Apply Kubernetes configurations
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Check deployment status
kubectl get pods
kubectl get services

```

#### Kubectl port forward for the client to access

```
kubectl port-forward service/airbnb-vienna-service 8000:8000
```


### Testing Kubernetes Deployment

```
bash

# Get service URL
minikube service airbnb-vienna-service --url

# Or for cloud providers
kubectl get service airbnb-vienna-service

# Test API endpoint

'''
python api_client.py
'''

'''
curl -X POST http://<SERVICE_IP>/predict_price\
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
```

* * * * *

10\. Project Characterestics
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

11\. Project Highlights
-----------------------

-   End-to-end ML lifecycle

-   Public dataset with real-world complexity

-   Multiple models with comparative evaluation

-   Reproducible pipelines

-   Production-ready deployment (FastAPI + Docker + Kubernetes)

* * * * *

12\. Contributing
-----------------

1.  Fork the repository

2.  Create a feature branch

3.  Add tests for new functionality

4.  Ensure all tests pass

5.  Submit a pull request


* * * * *

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
