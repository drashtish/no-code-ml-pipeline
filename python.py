# No-Code ML Pipeline Builder (Stable, Screening-Ready Version)
# Focus: UX clarity, robust state handling, zero flicker

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------
# PAGE CONFIG
# ----------------------
st.set_page_config(
    page_title="No-Code ML Pipeline Builder",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------
# SESSION STATE INIT (CRITICAL)
# ----------------------
def init_state():
    defaults = {
        "data_uploaded": False,
        "preprocessed": False,
        "split_done": False,
        "df": None,
        "df_processed": None,
        "data_split": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ----------------------
# SIDEBAR – PIPELINE STATUS & RESET
# ----------------------
st.sidebar.header("Pipeline Progress")


st.sidebar.checkbox("Dataset uploaded", value=st.session_state.data_uploaded, disabled=True)
st.sidebar.checkbox("Preprocessing applied", value=st.session_state.preprocessed, disabled=True)
st.sidebar.checkbox("Data split", value=st.session_state.split_done, disabled=True)

if st.sidebar.button("🔄 Reset Pipeline"):
    st.session_state.clear()
    st.rerun()

# ----------------------
# MAIN TITLE
# ----------------------
st.title("No-Code Machine Learning Pipeline Builder")
st.caption("Design, train, and evaluate ML models through a guided visual workflow")

# ======================
# STEP 1: DATASET UPLOAD
# ======================
st.header("Step 1: Upload Dataset")

file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

if file:
    try:
        df = pd.read_csv(file) if file.name.endswith("csv") else pd.read_excel(file)
        st.session_state.df = df
        st.session_state.data_uploaded = True

        st.success("Dataset loaded successfully")

        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", df.shape[0])
        c2.metric("Columns", df.shape[1])
        c3.metric("Missing values", df.isnull().sum().sum())

        with st.expander("Preview dataset"):
            st.dataframe(df.head(10))

    except Exception:
        st.error("Unable to read file. Please upload a valid CSV or Excel file.")

# ======================
# STEP 2: PREPROCESSING
# ======================
if st.session_state.data_uploaded:
    st.header("Step 2: Data Preprocessing")

    df = st.session_state.df.copy()
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if not numeric_cols:
        st.warning("No numeric columns available for preprocessing")
    else:
        preprocess = st.selectbox(
            "Choose preprocessing method",
            ["None", "Standardization", "Normalization"],
            key="preprocess_method"
        )

        selected_cols = st.multiselect(
            "Select numeric features",
            numeric_cols,
            default=numeric_cols,
            key="selected_numeric_cols"
        )

        handle_missing = st.checkbox(
            "Handle missing values (mean imputation)",
            key="handle_missing"
        )

        if st.button("Apply Preprocessing"):
            if handle_missing:
                df[selected_cols] = df[selected_cols].fillna(df[selected_cols].mean())

            if preprocess == "Standardization":
                df[selected_cols] = StandardScaler().fit_transform(df[selected_cols])
            elif preprocess == "Normalization":
                df[selected_cols] = MinMaxScaler().fit_transform(df[selected_cols])

            st.session_state.df_processed = df
            st.session_state.preprocessed = True
            st.success("Preprocessing completed successfully")

# ======================
# STEP 3: TRAIN–TEST SPLIT
# ======================
if st.session_state.preprocessed:
    st.header("Step 3: Train–Test Split")

    df = st.session_state.df_processed

    target = st.selectbox(
        "Select target variable",
        df.columns,
        key="target_col"
    )

    test_ratio = st.slider(
        "Test set size",
        0.2, 0.4, 0.25, 0.05,
        key="test_ratio"
    )

    if st.button("Split Data"):
        X = df.drop(columns=[target])
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_ratio, random_state=42
        )

        st.session_state.data_split = (X_train, X_test, y_train, y_test)
        st.session_state.split_done = True

        st.success("Data split completed")
        st.write(f"Training samples: {X_train.shape[0]}")
        st.write(f"Testing samples: {X_test.shape[0]}")

# ======================
# STEP 4: MODEL TRAINING
# ======================
if st.session_state.split_done:
    st.header("Step 4: Model Training & Evaluation")

    X_train, X_test, y_train, y_test = st.session_state.data_split

    model_name = st.selectbox(
        "Select model",
        ["Logistic Regression", "Decision Tree"],
        key="model_choice"
    )

    if st.button("Train Model"):
        model = (
            LogisticRegression(max_iter=1000)
            if model_name == "Logistic Regression"
            else DecisionTreeClassifier()
        )

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        from sklearn.metrics import (
            accuracy_score,
            balanced_accuracy_score,
            f1_score
        )

        acc = accuracy_score(y_test, preds)
        bal_acc = balanced_accuracy_score(y_test, preds)
        macro_f1 = f1_score(y_test, preds, average="macro")
        weighted_f1 = f1_score(y_test, preds, average="weighted")

        cm = confusion_matrix(y_test, preds)

        st.success("Model trained successfully")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{acc:.3f}")
        m2.metric("Balanced Accuracy", f"{bal_acc:.3f}")
        m3.metric("Macro F1", f"{macro_f1:.3f}")
        m4.metric("Weighted F1", f"{weighted_f1:.3f}")


        c1, c2 = st.columns(2)
        with c1:
            from sklearn.metrics import ConfusionMatrixDisplay

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            # Raw counts
            sns.heatmap(
                confusion_matrix(y_test, preds),
                annot=True, fmt="d", cmap="Blues", ax=axes[0]
            )
            axes[0].set_title("Confusion Matrix (Counts)")

            # Normalized
            sns.heatmap(
                confusion_matrix(y_test, preds, normalize="true"),
                annot=True, fmt=".2f", cmap="Blues", ax=axes[1]
            )
            axes[1].set_title("Confusion Matrix (Normalized)")

            st.pyplot(fig)


            st.subheader("Class Distribution in Test Set")

            fig, ax = plt.subplots()
            y_test.value_counts().plot(kind="bar", ax=ax)
            ax.set_xlabel("Class")
            ax.set_ylabel("Samples")
            st.pyplot(fig)


        with c2:
            st.text("Classification Report")
            st.text(classification_report(y_test, preds))

        st.info("Pipeline executed end-to-end successfully")

