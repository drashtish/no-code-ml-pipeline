# No-Code ML Pipeline Builder
Interactive web app to train and evaluate classification models without writing code.


This project is a web-based, no-code machine learning pipeline builder designed to help users train and evaluate ML models without writing any code.

The application follows a step-by-step workflow where users can:
- Upload a dataset
- Apply preprocessing
- Perform train-test split
- Select and train a model
- View evaluation results

## Features
- Supports CSV and Excel datasets
- Standardization and normalization options
- Automatic handling of missing values
- Train-test split with configurable ratio
- Model selection (Logistic Regression, Decision Tree)
- Model evaluation with accuracy and confusion matrix

## Tech Stack
- Python
- Streamlit
- Scikit-learn
- Pandas, NumPy
- Matplotlib, Seaborn

## How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
