# Amazon-ML-Prediction-Model

---

## 📘 README.md — Amazon ML Prediction Model (XGBoost + Optuna + SHAP)

---

### 🧠 Project Title

**Netflix/Amazon Titles Popularity Prediction using XGBoost, Optuna, and SHAP Explainability**

---

### 📌 Objective

The goal of this project is to predict the popularity (e.g., IMDb score or weighted rating) of Netflix/Amazon titles based on metadata such as genre, cast, director, release year, and votes. This is achieved by training a regression model using XGBoost, optimizing it with Optuna, and interpreting it using SHAP.

---

### 🗂️ Datasets

This project uses  **two main CSV files** :

1. **`titles.csv`**

   Contains details about movies/TV shows:

   * `id`: Unique content ID
   * `title`: Name of the title
   * `release_year`: Year of release
   * `age_certification`, `genres`, `production_countries`, etc.
   * `imdb_score`, `tmdb_score`, `runtime`, etc.
2. **`credits.csv`**

   Contains cast and crew details:

   * `id`: Matches `titles.csv`
   * `person_id`, `name`, `character`, `role`

These two are merged using the `id` column for model training.

---

### 📊 Exploratory Data Analysis (EDA)

* Dataset dimensions and null values inspected
* Distribution plots for scores, runtime, and release years
* Correlation heatmap between numerical variables
* Frequency plots for genres and certifications
* Merging and filtering of crew data to isolate director and writer roles

---

### 🧹 Data Preprocessing

* Merging `credits.csv` with `titles.csv` using `id`
* Extracting key crew roles (e.g., director, writer)
* Converting multi-label fields like genres and countries
* Label encoding and one-hot encoding for categorical variables
* Handling missing values
* Train-test split (80/20)

---

### 🎯 Target Variable

* **`imdb_score`** is used as the **target** for regression.

---

### 🤖 Model: XGBoost Regressor

* Trained using `xgb.train()` for greater control
* Evaluation metric: **RMSE**
* Feature importance explored post-training

---

### 🧪 Hyperparameter Tuning with Optuna

* Search space includes:
  * `n_estimators`
  * `max_depth`
  * `learning_rate`
  * `subsample`
  * `colsample_bytree`
  * `lambda`, `alpha`
* Early stopping enabled (20 rounds)
* 50 trials for optimization

---

### 📈 SHAP Explainability

* SHAP’s `TreeExplainer` used on the XGBoost Booster
* Generates:
  * Beeswarm plot to show feature impact
  * Bar plot to rank feature importance

---

### 🏁 Results

* Best model identified via Optuna with minimized MSE
* Key influential features (from SHAP):
  * Genre
  * Director
  * Release Year
  * Runtime
  * Cast presence
* Insightful visualizations for explainable AI

---

### 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Amazon-ML-Prediction-Model.git
   cd Amazon-ML-Prediction-Model
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch the notebook:
   ```bash
   jupyter notebook amazon-ml-project.ipynb
   ```
4. Make sure both `titles.csv` and `credits.csv` are in the working directory.

---

### 📦 Requirements

```txt
xgboost>=2.0
optuna
shap
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

---

### 🧠 Insights

This project combines state-of-the-art ML tuning with interpretable modeling, offering practical insights into what makes a show/movie successful on streaming platforms like Netflix or Amazon.

---

### 📌 Acknowledgements

* **Optuna** for efficient hyperparameter tuning
* **XGBoost** for high-performance modeling
* **SHAP** for model transparency
* Data sourced from Kaggle's "Netflix Movies and TV Shows" and IMDb

---
