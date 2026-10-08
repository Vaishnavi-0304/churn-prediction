# Customer Churn Prediction (Intermediate ML Mini-Project)

## Files
- `Customer_Churn_Prediction.ipynb` – full workflow: problem, dataset, preprocessing, EDA, feature selection, 3 models, evaluation, comparison, tuning, saving the model
- `app.py` – Streamlit web interface for predictions
- `requirements.txt` – libraries

## How to run (about 10 minutes)
1. `pip install -r requirements.txt`
2. Put `WA_Fn-UseC_-Telco-Customer-Churn.csv` (Kaggle: "Telco Customer Churn") next to the notebook.
   If the file is missing, the notebook downloads it automatically from IBM's public GitHub (needs internet).
3. Open the notebook in Jupyter / VS Code / Colab and use **Run All** (the tuning cell takes a few minutes).
   This creates `churn_model.joblib` and `model_comparison.csv`.
4. Run the web app: `streamlit run app.py`

## Before you submit
- Add your names / roll numbers in the first cell.
- Read Section 8 (why performance differs) and adjust the wording to your actual result table.
- Take screenshots of the notebook outputs and the Streamlit app for your report/PPT.
