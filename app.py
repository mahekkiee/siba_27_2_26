import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report,
    mean_squared_error,
    r2_score
)

st.title("Flexible ML Model Trainer")

uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # STEP 1 — Select Model Type First
    problem_type = st.radio(
        "Select Problem Type",
        ["Classification", "Regression"]
    )

    # Automatically filter valid targets
    if problem_type == "Classification":
        possible_targets = [
            col for col in df.columns
            if df[col].dtype == "object" or df[col].nunique() < 15
        ]
    else:
        possible_targets = [
            col for col in df.columns
            if df[col].dtype != "object"
        ]

    if not possible_targets:
        st.error("No valid target columns available for selected model type.")
        st.stop()

    target_column = st.selectbox("Select Target Column", possible_targets)

    feature_columns = st.multiselect(
        "Select Feature Columns",
        [col for col in df.columns if col != target_column]
    )

    test_size = st.slider("Select Test Size (%)", 10, 50, 30) / 100

    if target_column and feature_columns:

        X = df[feature_columns]
        y = df[target_column]

        X = pd.get_dummies(X)

        if st.button("Train Model"):

            if problem_type == "Classification":

                # Encode target
                if y.dtype == "object":
                    le = LabelEncoder()
                    y = le.fit_transform(y)
                    class_labels = le.classes_
                else:
                    class_labels = sorted(np.unique(y))

                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=test_size,
                    random_state=42,
                    stratify=y
                )

                model = GaussianNB()
                model.fit(X_train, y_train)

                y_test_pred = model.predict(X_test)

                st.subheader("Test Accuracy")
                st.write(round(accuracy_score(y_test, y_test_pred), 4))

                cm = confusion_matrix(y_test, y_test_pred)

                st.subheader("Confusion Matrix")

                fig, ax = plt.subplots()
                im = ax.imshow(cm, cmap="Blues")

                ax.set_xticks(np.arange(len(class_labels)))
                ax.set_yticks(np.arange(len(class_labels)))
                ax.set_xticklabels(class_labels)
                ax.set_yticklabels(class_labels)

                for i in range(len(class_labels)):
                    for j in range(len(class_labels)):
                        ax.text(j, i, cm[i, j],
                                ha="center", va="center", color="black")

                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                fig.colorbar(im)

                st.pyplot(fig)

                st.subheader("Classification Report")
                st.text(classification_report(y_test, y_test_pred))

            else:

                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=test_size,
                    random_state=42
                )

                model = LinearRegression()
                model.fit(X_train, y_train)

                y_test_pred = model.predict(X_test)

                st.subheader("Regression Metrics")
                st.write("R²:", round(r2_score(y_test, y_test_pred), 4))
                st.write("MSE:", round(mean_squared_error(y_test, y_test_pred), 4))
