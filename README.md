# Housing Price Prediction – Machine Learning Model 🏠📈

## Overview
This project builds a complete **machine learning pipeline** to predict housing prices using structured tabular data.  
It covers data preprocessing, model training, evaluation, and inference using **scikit-learn**.

---

## 📁 Project Structure

- `01_Main.py`  
  Entry point script to train the model and run predictions  

- `02_CompleteModel.py`  
  Complete ML pipeline including preprocessing and model logic  

- `housing.csv`  
  Housing dataset used for training and evaluation  

- `README.md`  
  Project documentation  

---

## 🛠️ Tools & Technologies
- Python  
- Pandas  
- NumPy  
- Scikit-learn  

---

## ⚙️ Model & Pipeline
- Stratified train–test split  
- Data preprocessing:
  - Missing value imputation  
  - Feature scaling  
  - One-hot encoding  
- Model used:
  - **Random Forest Regressor**  
- Pipeline saved for reuse and inference  

---

## 🎯 Key Features
- End-to-end ML workflow  
- Reusable preprocessing + model pipeline  
- Automatic model loading if already trained  
- Predictions saved to `output.csv`  

---

## ▶️ How to Run

1. Clone the repository  
2. Install required dependencies  
   ```bash
   pip install pandas numpy scikit-learn
