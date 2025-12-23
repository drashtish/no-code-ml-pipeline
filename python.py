# No-Code ML Pipeline Builder
# Handles EDA, Cleaning, Encoding, Task Detection, Training, Evaluation

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas.api.types as pdt

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
        "data_uploaded": False,
        "eda_done": False,
        "cleaning_done": False,
        "preprocessed": False,
        "split_done": False,
        "task_type": None,
        "df": None,
        "df_clean": None,
        "df_processed": None,
        "data_split": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# SIDEBAR – PIPELINE STATUS & RESET
st.sidebar.header("Pipeline Progress")
st.sidebar.checkbox("1. Dataset Uploaded", value=st.session_state.data_uploaded, disabled=True)
st.sidebar.checkbox("2. Data Explored", value=st.session_state.eda_done, disabled=True)
st.sidebar.checkbox("3. Data Cleaned", value=st.session_state.cleaning_done, disabled=True)
st.sidebar.checkbox("4. Preprocessing Applied", value=st.session_state.preprocessed, disabled=True)
st.sidebar.checkbox("5. Data Split", value=st.session_state.split_done, disabled=True)

if st.sidebar.button("🔄 Reset Pipeline"):
    st.session_state.clear()
    st.rerun()


# UTILITY: SAFE ENCODING
def encode_categorical_features(X: pd.DataFrame) -> pd.DataFrame:
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns
    if len(cat_cols) > 0:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    return X


# TITLE
st.title("No-Code Machine Learning Pipeline Builder")
st.caption("End-to-end ML workflow with full error handling")


# STEP 1: UPLOAD DATA
st.header("Step 1: Upload Dataset")
file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

if file:
    try:
        df = pd.read_csv(file) if file.name.endswith("csv") else pd.read_excel(file)
        st.session_state.df = df
        st.session_state.data_uploaded = True

        st.success("Dataset uploaded successfully")
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", df.shape[0])
        c2.metric("Columns", df.shape[1])
        c3.metric("Missing Values", int(df.isnull().sum().sum()))

        with st.expander("Preview Dataset"):
            st.dataframe(df.head(), use_container_width=True)
    except Exception:
        st.error("Failed to load dataset. Please upload a valid CSV or Excel file.")
        st.stop()


# STEP 2: EDA
if st.session_state.data_uploaded:
    st.header("Step 2: Explore Data")
    df = st.session_state.df

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    c1, c2, c3 = st.columns(3)
    c1.metric("Numeric Columns", len(num_cols))
    c2.metric("Categorical Columns", len(cat_cols))
    c3.metric("Total Features", df.shape[1])

    selected_col = st.selectbox("Select column for distribution", df.columns)
    fig, ax = plt.subplots()
    if selected_col in num_cols:
        sns.histplot(df[selected_col], kde=True, ax=ax)
    else:
        df[selected_col].value_counts().plot(kind="bar", ax=ax)
    st.pyplot(fig)

    if len(num_cols) >= 2:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(df[num_cols].corr(), cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    st.session_state.eda_done = True


# STEP 3: DATA CLEANING
if st.session_state.eda_done:
    st.header("Step 3: Clean Data")
    df = st.session_state.df.copy()

    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if not missing.empty:
        st.dataframe(missing.rename("Missing Count"))
        if st.checkbox("Fill numeric missing values with mean"):
            for col in df.select_dtypes(include=np.number):
                df[col].fillna(df[col].mean(), inplace=True)
            st.success("Filled numeric missing values")

    dup_count = df.duplicated().sum()
    if dup_count > 0:
        st.warning(f"{dup_count} duplicate rows found")
        if st.button("Remove duplicates"):
            df.drop_duplicates(inplace=True)
            st.success("Duplicates removed")

    cols_to_drop = st.multiselect("Select columns to drop", df.columns)
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)
        st.success(f"Dropped columns: {cols_to_drop}")

    st.session_state.df_clean = df
    st.session_state.cleaning_done = True


# STEP 4: PREPROCESSING (FEATURES ONLY)
if st.session_state.cleaning_done:
    st.header("Step 4: Preprocessing")
    df = st.session_state.df_clean.copy()

    method = st.radio("Select preprocessing method", ["None", "Standardization", "Normalization"])

    if st.button("Apply Preprocessing"):
        if method != "None":
            num_cols = df.select_dtypes(include=np.number).columns
            df[num_cols] = StandardScaler().fit_transform(df[num_cols]) if method == "Standardization" else MinMaxScaler().fit_transform(df[num_cols])

        st.session_state.df_processed = df
        st.session_state.preprocessed = True
        st.success("Preprocessing applied")


# STEP 5: TRAIN–TEST SPLIT & TASK DETECTION
if st.session_state.preprocessed:
    st.header("Step 5: Train–Test Split")
    df = st.session_state.df_processed

    target = st.selectbox("Select target column", df.columns)
    test_size = st.slider("Test size", 0.2, 0.4, 0.25, 0.05)

    if target:
        y = df[target]
        if pdt.is_numeric_dtype(y) and y.nunique() > 20:
            task_type = "Regression"
        else:
            task_type = "Classification"
        st.info(f"Detected ML Task: **{task_type}**")
        st.session_state.task_type = task_type

    if st.button("Split Data"):
        try:
            X = df.drop(columns=[target])
            y = df[target]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            st.session_state.data_split = (X_train, X_test, y_train, y_test)
            st.session_state.split_done = True
            st.success("Data split completed")
        except Exception:
            st.error("Data split failed. Check dataset size and target column.")
            st.stop()


# STEP 6: MODEL TRAINING & EVALUATION
if st.session_state.split_done:
    st.header("Step 6: Model Training & Evaluation")

    X_train, X_test, y_train, y_test = st.session_state.data_split
    task_type = st.session_state.task_type

    if X_train.select_dtypes(include=["object"]).shape[1] > 0:
        st.warning("Categorical features detected and will be encoded automatically.")

    model_name = st.selectbox(
        "Choose model",
        ["Logistic Regression", "Decision Tree"] if task_type == "Classification"
        else ["Linear Regression", "Decision Tree Regressor"]
    )

    if st.button("Train Model"):
        try:
            X_train_enc = encode_categorical_features(X_train)
            X_test_enc = encode_categorical_features(X_test)
            X_train_enc, X_test_enc = X_train_enc.align(X_test_enc, join="left", axis=1, fill_value=0)

            if task_type == "Classification":
                model = LogisticRegression(max_iter=1000) if model_name == "Logistic Regression" else DecisionTreeClassifier()
                model.fit(X_train_enc, y_train)
                preds = model.predict(X_test_enc)

                acc = accuracy_score(y_test, preds)
                bal_acc = balanced_accuracy_score(y_test, preds)
                macro_f1 = f1_score(y_test, preds, average="macro")
                weighted_f1 = f1_score(y_test, preds, average="weighted")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Accuracy", f"{acc:.3f}")
                c2.metric("Balanced Accuracy", f"{bal_acc:.3f}")
                c3.metric("Macro F1", f"{macro_f1:.3f}")
                c4.metric("Weighted F1", f"{weighted_f1:.3f}")

                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt="d", cmap="Blues", ax=axes[0])
                sns.heatmap(confusion_matrix(y_test, preds, normalize="true"), annot=True, fmt=".2f", cmap="Blues", ax=axes[1])
                st.pyplot(fig)

                with st.expander("Classification Report"):
                    st.text(classification_report(y_test, preds))

            else:
                model = LinearRegression() if model_name == "Linear Regression" else DecisionTreeRegressor()
                model.fit(X_train_enc, y_train)
                preds = model.predict(X_test_enc)

                mae = mean_absolute_error(y_test, preds)
                mse = mean_squared_error(y_test, preds)
                r2 = r2_score(y_test, preds)

                c1, c2, c3 = st.columns(3)
                c1.metric("MAE", f"{mae:.3f}")
                c2.metric("MSE", f"{mse:.3f}")
                c3.metric("R² Score", f"{r2:.3f}")

            st.success("Pipeline executed successfully")

        except Exception:
            st.error("Model training failed due to incompatible data.")
            st.info("Please review data cleaning and feature types.")
            st.stop()

