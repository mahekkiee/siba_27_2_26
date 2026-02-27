import streamlit as st
import pandas as pd
import numpy as np
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

    target_column = st.selectbox("Select Target Column", df.columns)

    feature_columns = st.multiselect(
        "Select Feature Columns",
        [col for col in df.columns if col != target_column]
    )

    test_size = st.slider("Select Test Size (%)", 10, 50, 30) / 100

    if target_column and feature_columns:

        X = df[feature_columns]
        y = df[target_column]

        # Convert categorical features
        X = pd.get_dummies(X)

        # Detect problem type
        if y.dtype == "object" or y.nunique() < 10:
            problem_type = "Classification"
        else:
            problem_type = "Regression"

        st.write("Detected Problem Type:", problem_type)

        if st.button("Train Model"):

            if problem_type == "Classification":

                # Encode target if needed
                if y.dtype == "object":
                    le = LabelEncoder()
                    y = le.fit_transform(y)
                    class_labels = le.classes_
                else:
                    class_labels = sorted(y.unique())

                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=test_size,
                    random_state=42,
                    stratify=y
                )

                model = GaussianNB()
                model.fit(X_train, y_train)

                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                st.subheader("Accuracy")
                st.write("Train Accuracy:", round(accuracy_score(y_train, y_train_pred), 4))
                st.write("Test Accuracy:", round(accuracy_score(y_test, y_test_pred), 4))

                st.subheader("Confusion Matrix - Train")
                train_cm = confusion_matrix(y_train, y_train_pred)
                st.write(pd.DataFrame(train_cm, index=class_labels, columns=class_labels))

                st.subheader("Confusion Matrix - Test")
                test_cm = confusion_matrix(y_test, y_test_pred)
                st.write(pd.DataFrame(test_cm, index=class_labels, columns=class_labels))

                st.subheader("Classification Report")
                st.text(classification_report(y_test, y_test_pred))

            else:  # Regression

                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=test_size,
                    random_state=42
                )

                model = LinearRegression()
                model.fit(X_train, y_train)

                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                st.subheader("Regression Metrics")
                st.write("Train R²:", round(r2_score(y_train, y_train_pred), 4))
                st.write("Test R²:", round(r2_score(y_test, y_test_pred), 4))
                st.write("Test MSE:", round(mean_squared_error(y_test, y_test_pred), 4))
