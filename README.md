# 📊 Customer Churn Prediction

## 📌 Overview

This project aims to predict customer churn using machine learning and deep learning techniques. Customer churn refers to customers who stop using a company's services. By analyzing customer data, this project builds a predictive model to identify customers likely to churn.

---

## 🎯 Objective

* Analyze customer data and identify churn patterns
* Handle missing values and preprocess data
* Train and evaluate machine learning / deep learning models
* Build a prediction system for real-time use

---

## 📁 Project Structure

```bash
Customer-Churn-Prediction/
│
├── logs/                              # Logs generated during model training
├── 10-Handling-Missing-values(1).ipynb # Data preprocessing notebook
├── experiments.ipynb                  # Model experiments
├── prediction.ipynb                   # Prediction testing notebook
├── Churn_Modelling.csv                # Dataset
├── app.py                             # Application script (for deployment)
├── model.h5                           # Trained deep learning model
├── scaler.pkl                         # Feature scaler
├── label_encoder_gender.pkl           # Label encoder
├── onehot_encoder_geo.pkl             # One-hot encoder
├── requirements.txt                   # Dependencies
├── .gitignore                         # Ignored files
```

---

## 📊 Dataset

The dataset (`Churn_Modelling.csv`) contains customer information such as:

* Credit Score
* Geography
* Gender
* Age
* Tenure
* Balance
* Number of Products
* Estimated Salary
* Churn (Target Variable)

---

## ⚙️ Technologies Used

* Python
* Pandas, NumPy
* Scikit-learn
* TensorFlow / Keras
* Matplotlib, Seaborn
* Jupyter Notebook

---

## 🔍 Workflow

1. **Data Preprocessing**

   * Handling missing values
   * Encoding categorical variables
   * Feature scaling

2. **Model Training**

   * Experiments performed in `experiments.ipynb`
   * Deep learning model saved as `model.h5`

3. **Model Saving**

   * Scaler and encoders saved using `.pkl` files

4. **Prediction**

   * Predictions tested in `prediction.ipynb`
   * `app.py` used for running prediction system

---

## 🤖 Model Details

* Deep Learning model built using TensorFlow/Keras
* Input features processed using:

  * Label Encoding
  * One-Hot Encoding
  * Standard Scaling

---

## 📈 Evaluation

Model performance evaluated using:

* Accuracy
* Loss function
* Validation performance

---

## 🚀 How to Run the Project

### 1. Clone Repository

```bash
git clone https://github.com/abhijeet3295/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Run Jupyter Notebooks

```bash
jupyter notebook
```

### 4. Run Application

```bash
python app.py
```

---

## 📌 Key Features

* End-to-end ML pipeline
* Data preprocessing + feature engineering
* Deep learning model
* Model deployment-ready structure

---

## 🔮 Future Improvements

* Hyperparameter tuning
* Use advanced models (XGBoost, LightGBM)
* Deploy using Streamlit or Flask UI
* Add API support

---

## 👨‍💻 Author

**Abhijeet**

---

## ⭐ Note

This project demonstrates a complete machine learning workflow from data preprocessing to model deployment.
