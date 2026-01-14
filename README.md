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

5\. Classification Model -- Premium vs Non-Premium Listings
----------------------------------------------------------

### Objective

Classify listings into **Superhost** or **Normal Host** categories.

### Target Definition

`Superhost = 1  if probability > 80%  `

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
| `/predict_superhost` | Predict Superhost |
| `/health` | Health check |

### Example Request

`curl -X POST http://localhost:8000/predict_price\
-H "Content-Type: application/json"\
-d '{"accommodates":4,"bedrooms":2,"availability_365":180}'  `

* * * * *

9\. Docker Deployment
---------------------

### Build Docker Image

`docker build -t skp-airbnb-vienna:latest . `

### Run Container

`docker run -p 8000:8000 skp-airbnb-vienna:latest `

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
