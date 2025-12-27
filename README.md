# Oral Cancer Risk Prediction Using Machine Learning

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset Description](#dataset-description)
- [Data Preprocessing](#data-preprocessing)
- [Feature Engineering](#feature-engineering)
- [Model Development](#model-development)
  - [Baseline Models](#baseline-models)
  - [Cross-Validation](#cross-validation)
  - [Hyperparameter Tuning](#hyperparameter-tuning)
  - [Ensemble Modeling](#ensemble-modeling)
- [Model Evaluation](#model-evaluation)
- [Feature Importance](#feature-importance)
- [Model Export](#model-export)
- [API Development](#api-development)
  - [Flask Application](#flask-application)
  - [Swagger / OpenAPI Documentation](#swagger--openapi-documentation)
  - [Batch Prediction & Metadata](#batch-prediction--metadata)
- [Deployment](#deployment)
  - [Dockerization](#dockerization)
  - [Docker Compose](#docker-compose)
  - [CI/CD via GitHub Actions](#cicd-via-github-actions)
- [Disclaimer](#disclaimer)
- [Future Work](#future-work)

---

## Project Overview

Oral cancer is a serious health concern with significant mortality rates if detected late. Early prediction based on pre-diagnosis features such as lifestyle, symptoms, and risk factors can support timely medical interventions.

This project aims to develop a machine learning pipeline for predicting oral cancer risk, with the following objectives:

- Preprocess patient data and handle imbalances.
- Train, evaluate, and compare multiple machine learning models.
- Provide model explainability and feature importance insights.
- Deploy the model as a production-ready API with batch prediction and Swagger documentation.

**🔗 Live API Documentation:** [http://oral-cancer.duckdns.org/docs](http://oral-cancer.duckdns.org/docs)

---

## Dataset Description

The dataset contains **84,922 patient records** with **17 variables**, including:

| Feature              | Description                                    |
|----------------------|------------------------------------------------|
| `age`                | Patient age (years)                            |
| `gender`             | 0 = Female, 1 = Male                           |
| `tobacco`            | Tobacco usage (binary)                         |
| `alcohol`            | Alcohol consumption (binary)                   |
| `hpv`                | HPV infection status (binary)                  |
| `betel_quid`         | Betel quid chewing (binary)                    |
| `sun_exposure`       | Exposure to sunlight (binary)                  |
| `oral_hygiene`       | Oral hygiene status (binary)                   |
| `diet_quality`       | Dietary habits (0 = Poor, 1 = Good)            |
| `family_history`     | Family history of oral cancer (binary)         |
| `immune_compromised` | Immunocompromised status (binary)              |
| `oral_lesions`       | Presence of oral lesions (binary)              |
| `bleeding`           | Oral bleeding (binary)                         |
| `swallowing`         | Difficulty swallowing (binary)                 |
| `mouth_patches`      | Mouth patches (binary)                         |
| `diagnosis`          | Oral cancer diagnosis (target)                 |
| `region`             | Patient region (categorical)                   |

> **Note:** Clinical features such as tumor size, cancer stage, treatment type, and survival rate were excluded to prevent data leakage, as these are not available prior to diagnosis.

---

## Data Preprocessing

### Encoding Categorical Features
- `gender`, `region`, and `diet_quality` were encoded numerically.
- For `region`, only the unique countries present in the dataset were used to avoid unseen categories in deployment.

### Train-Test Split
- Dataset split into **80% training** and **20% testing**.
- Stratification was applied to ensure balanced class distribution.

### Handling Class Imbalance
- Applied **SMOTE (Synthetic Minority Oversampling Technique)** to the training set to generate synthetic samples of the minority class.

### Scaling Numerical Features
- Features with large ranges, such as `age`, were scaled using **StandardScaler** to improve model convergence and interpretability.

---

## Feature Engineering

- Only **pre-diagnosis features** were retained.
- No additional feature creation was required as the dataset contained well-defined risk factors and symptoms.
- Correlation analysis was attempted, but categorical features required encoding first.

---

## Model Development

### Baseline Models

- **Logistic Regression (LR)**
- **Random Forest (RF)**
- **XGBoost (XGB)**

### Cross-Validation

**Cross-Validation F1 Scores:**

| Model               | Mean CV F1 |
|---------------------|------------|
| Logistic Regression | 0.495      |
| Random Forest       | 0.496      |
| XGBoost             | 0.491      |

**Observation:** Random Forest showed the highest mean CV F1 score and is computationally efficient compared to parameter-tuned XGBoost.

### Hyperparameter Tuning

- Parameter tuning was attempted but computationally expensive for the full dataset.
- Considering performance gain vs. time cost, default Random Forest parameters were retained for deployment.

### Ensemble Modeling

- A **Voting Classifier** combining LR, RF, and XGB was implemented.
- Soft voting (probabilities) was used due to class imbalance.
- **Ensemble F1 Score:** 0.491

**Observation:** Random Forest alone slightly outperformed the ensemble and is simpler for deployment.

---

## Model Evaluation

**Logistic Regression Confusion Matrix Example:**

|          | Predicted 0 | Predicted 1 |
|----------|-------------|-------------|
| Actual 0 | 4356        | 4159        |
| Actual 1 | 4391        | 4079        |

F1, precision, and recall were approximately **0.50**, indicating a balanced model performance given the dataset.

---

## Feature Importance

Using **Random Forest feature importances**, top predictors:

| Feature        | Importance |
|----------------|------------|
| `age`          | 0.303      |
| `region`       | 0.114      |
| `diet_quality` | 0.070      |
| `oral_lesions` | 0.044      |
| `bleeding`     | 0.043      |

> **Note:** SHAP was explored for explainability but removed due to execution time constraints on the full dataset.

---

## Model Export

Model serialized using **Pickle**:
```python
import pickle

with open("random_forest_model.pkl", "wb") as f:
    pickle.dump(rf_model, f)
```

Ready to be loaded in API for predictions.

---

## API Development

### Flask Application

- REST API with `/predict` endpoint for single prediction.
- Supports probabilities and JSON inputs.

**🔗 Interactive API Documentation:** [http://oral-cancer.duckdns.org/docs](http://oral-cancer.duckdns.org/docs)

**Example input:**
```json
{
  "age": 36,
  "gender": 0,
  "tobacco": 1,
  "alcohol": 1,
  "hpv": 1,
  "betel_quid": 0,
  "sun_exposure": 0,
  "oral_hygiene": 1,
  "diet_quality": 0,
  "family_history": 0,
  "immune_compromised": 0,
  "oral_lesions": 0,
  "bleeding": 0,
  "swallowing": 0,
  "mouth_patches": 0,
  "region": 0
}
```

**Output:**
```json
{
  "prediction": 0,
  "probability": [[0.81, 0.19]]
}
```

### Swagger / OpenAPI Documentation

- Exposes interactive documentation at [http://oral-cancer.duckdns.org/docs](http://oral-cancer.duckdns.org/docs)
- Users can send requests directly via UI.
- Metadata and disclaimer included.

### Batch Prediction & Metadata

- `/predict/batch` endpoint allows multiple records prediction.
- Returns predictions, probabilities, and input metadata.

---

## Deployment

### Dockerization

- Flask API containerized with **Gunicorn** (production-ready).
- Port exposed: **5555**.

**Dockerfile highlights:**
```dockerfile
EXPOSE 5555
CMD ["gunicorn", "--bind", "0.0.0.0:5555", "app:app"]
```

### Docker Compose

Pull image from Docker Hub and deploy easily:
```yaml
services:
  oral-cancer-api:
    image: <DOCKER_USERNAME>/oral-cancer-api:latest
    ports:
      - "5555:5555"
```

### CI/CD via GitHub Actions

- Workflow triggers on push to `main`.
- Builds Docker image and pushes to Docker Hub using secrets:
```yaml
username: ${{ secrets.DOCKER_USERNAME }}
password: ${{ secrets.DOCKER_PASSWORD }}
```

---

## Disclaimer

⚠️ **This API provides risk prediction based on pre-diagnosis features. It is not a substitute for professional medical advice. Clinical diagnosis should be performed by qualified healthcare providers.**

---

## Future Work

- Improve model performance via hyperparameter optimization.
- Explore multi-modal inputs (images, genetic data).
- Optimize SHAP explainability for production use.
- Deploy on cloud platforms for high availability.

---