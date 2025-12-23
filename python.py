# No-Code ML Pipeline Builder 
# Feature-rich, UI-polished, error-safe Streamlit application

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas.api.types as pdt
import joblib
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# PAGE CONFIG

st.set_page_config(
    page_title="No-Code ML Pipeline Builder",
    layout="wide"
)

# SESSION STATE INITIALIZATION

def init_state():
    defaults = {
        "step": 1,
        "df": None,
        "df_clean": None,
        "df_processed": None,
        "data_split": None,
        "task_type": None,
        "model": None,
        "metrics": None
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# SIDEBAR – NAVIGATION & PROGRESS
st.sidebar.title("Pipeline Navigator")
steps = {
    1: "Upload Data",
    2: "EDA",
    3: "Data Cleaning",
    4: "Preprocessing",
    5: "Train-Test Split",
    6: "Model Training",
    7: "Evaluation & Export"
}

st.sidebar.progress((st.session_state.step - 1) / (len(steps) - 1))

for i, name in steps.items():
    if st.sidebar.button(f"{i}. {name}", disabled=(i > st.session_state.step)):
        st.session_state.step = i

if st.sidebar.button(" Reset Pipeline"):
    st.session_state.clear()
    st.rerun()

# UTILITY FUNCTIONS
def encode_features(X: pd.DataFrame) -> pd.DataFrame:
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns
    if len(cat_cols) > 0:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    return X


# TITLE
st.title("No-Code Machine Learning Pipeline Builder")
st.caption("A complete, guided ML workflow – designed for screening evaluation")


# STEP 1: DATA UPLOAD
if st.session_state.step == 1:
    st.header("Step 1: Upload Dataset")
    file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

    if file:
        try:
            df = pd.read_csv(file) if file.name.endswith("csv") else pd.read_excel(file)
            st.session_state.df = df
            st.success("Dataset uploaded successfully")

            c1, c2, c3 = st.columns(3)
            c1.metric("Rows", df.shape[0])
            c2.metric("Columns", df.shape[1])
            c3.metric("Missing Values", int(df.isnull().sum().sum()))

            st.dataframe(df.head(), use_container_width=True)

            if st.button("Proceed to EDA "):
                st.session_state.step = 2
        except Exception:
            st.error("Invalid file format or corrupted file.")
            st.stop()


# STEP 2: EDA

if st.session_state.step == 2:
    st.header("Step 2: Exploratory Data Analysis")
    df = st.session_state.df

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    c1, c2, c3 = st.columns(3)
    c1.metric("Numeric Features", len(num_cols))
    c2.metric("Categorical Features", len(cat_cols))
    c3.metric("Total Features", df.shape[1])

    col = st.selectbox("Select column for distribution", df.columns)
    fig, ax = plt.subplots()
    if col in num_cols:
        sns.histplot(df[col], kde=True, ax=ax)
    else:
        df[col].value_counts().plot(kind="bar", ax=ax)
    st.pyplot(fig)

    if len(num_cols) >= 2:
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(df[num_cols].corr(), cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    if st.button("Proceed to Data Cleaning "):
        st.session_state.step = 3


# STEP 3: DATA CLEANING

if st.session_state.step == 3:
    st.header("Step 3: Data Cleaning")
    df = st.session_state.df.copy()

    st.subheader("Missing Value Handling")

    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if missing.empty:
        st.success(" No missing values present in the dataset.")
    else:
        st.warning(" Missing values detected.")
        
        missing_df = (
            missing
            .rename("Missing Count")
            .to_frame()
            .assign(Percentage=lambda x: (x["Missing Count"] / len(df) * 100).round(2))
        )
        
        st.dataframe(missing_df, use_container_width=True)

        fill_numeric = st.checkbox(
            "Fill numeric missing values with mean",
            help="Only numeric columns will be imputed"
        )

        if fill_numeric:
            num_cols = df.select_dtypes(include=np.number).columns
            for col in num_cols:
                df[col] = df[col].fillna(df[col].mean())

            st.session_state.df_clean = df
            st.success(" Missing values in numeric columns have been filled.")




    st.subheader("Duplicate Rows")

    dup_count = df.duplicated().sum()

    if dup_count == 0:
        st.success("No duplicate rows found.")
    else:
        st.warning(f" {dup_count} duplicate rows detected.")
        if st.button("Remove Duplicates"):
            df.drop_duplicates(inplace=True)
            st.success("Duplicate rows removed.")


    st.subheader("Drop Columns")
    drop_cols = st.multiselect("Select columns to drop", df.columns)
    st.caption("Tip: Drop ID columns, names, or leakage features.")

    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)
        st.success("Columns dropped")

    st.session_state.df_clean = df
    if st.button("Proceed to Preprocessing "):
        st.session_state.step = 4


# STEP 4: PREPROCESSING

if st.session_state.step == 4:
    st.header("Step 4: Feature Preprocessing")
    df = st.session_state.df_clean.copy()

    method = st.radio("Scaling Method", ["None", "StandardScaler", "MinMaxScaler"])

    if st.button("Apply Scaling"):
        if method != "None":
            num_cols = df.select_dtypes(include=np.number).columns
            scaler = StandardScaler() if method == "StandardScaler" else MinMaxScaler()
            df[num_cols] = scaler.fit_transform(df[num_cols])
        st.session_state.df_processed = df

        st.success("Preprocessing complete")
    
    if st.button("Proceed to Train-Test Split "):
        st.session_state.step = 5
        st.rerun()


# STEP 5: SPLIT & TASK DETECTION

if st.session_state.step == 5:
    st.header("Step 5: Train-Test Split")
    df = st.session_state.df_processed

    target = st.selectbox("Select target column", df.columns)
    test_size = st.slider("Test size", 0.2, 0.4, 0.25)

    y = df[target]
    task_type = "Regression" if pdt.is_numeric_dtype(y) and y.nunique() > 20 else "Classification"
    st.info(f"Detected Task Type: {task_type}")
    st.session_state.task_type = task_type

    if st.button("Split Data"):
        try:
            X = df.drop(columns=[target])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            st.session_state.data_split = (X_train, X_test, y_train, y_test)

            st.success("Data split successful")
        except Exception:
            st.error("Failed to split data")
            st.stop()
    if st.session_state.data_split is not None:
        if st.button("Proceed to Model Training "):
            st.session_state.step = 6
            st.rerun()


# STEP 6: MODEL TRAINING

if st.session_state.step == 6:
    st.header("Step 6: Model Training")
    X_train, X_test, y_train, y_test = st.session_state.data_split

    model_name = st.selectbox(
        "Choose model",
        ["Logistic Regression", "Decision Tree"] if st.session_state.task_type == "Classification"
        else ["Linear Regression", "Decision Tree Regressor"]
    )

    if st.button("Train Model"):
        try:
            X_train_enc = encode_features(X_train)
            X_test_enc = encode_features(X_test)
            X_train_enc, X_test_enc = X_train_enc.align(X_test_enc, join="left", axis=1, fill_value=0)

            if st.session_state.task_type == "Classification":
                model = LogisticRegression(max_iter=1000) if model_name == "Logistic Regression" else DecisionTreeClassifier()
                model.fit(X_train_enc, y_train)
            else:
                model = LinearRegression() if model_name == "Linear Regression" else DecisionTreeRegressor()
                model.fit(X_train_enc, y_train)

            st.session_state.model = model

            st.success("Model trained successfully")
        except Exception:
            st.error("Model training failed")
            st.stop()

    if st.session_state.model is not None:
            if st.button("Proceed to Evaluation & Export "):
                st.session_state.step = 7
                st.rerun()

# STEP 7: EVALUATION & EXPORT
if st.session_state.step == 7:
    st.header("Step 7: Evaluation & Export")
    model = st.session_state.model
    X_train, X_test, y_train, y_test = st.session_state.data_split

    X_test_enc = encode_features(X_test)
    X_train_enc = encode_features(X_train)
    X_train_enc, X_test_enc = X_train_enc.align(X_test_enc, join="left", axis=1, fill_value=0)

    preds = model.predict(X_test_enc)

    if st.session_state.task_type == "Classification":
        acc = accuracy_score(y_test, preds)
        bal = balanced_accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")

        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{acc:.3f}")
        c2.metric("Balanced Acc", f"{bal:.3f}")
        c3.metric("F1 Score", f"{f1:.3f}")

        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)

        with st.expander("Classification Report"):
            st.text(classification_report(y_test, preds))

    else:
        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", f"{mae:.3f}")
        c2.metric("MSE", f"{mse:.3f}")
        c3.metric("R²", f"{r2:.3f}")

    st.subheader("Export Model")
    if st.button("Download Model"):
        filename = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
        joblib.dump(model, filename)
        with open(filename, "rb") as f:
            st.download_button("Download", f, file_name=filename)

    st.success("Pipeline completed successfully ")
