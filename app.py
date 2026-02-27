import streamlit as st
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

st.title("Naive Bayes Classifier (Custom Dataset)")

uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.dataframe(df.head())

    target_column = st.selectbox("Select Target Variable", df.columns)

    feature_columns = st.multiselect(
        "Select Feature Columns",
        [col for col in df.columns if col != target_column]
    )

    if target_column and feature_columns:

        X = df[feature_columns]
        y = df[target_column]

        # Ensure numeric features
        X = pd.get_dummies(X)

        # 30% training, 70% testing
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            train_size=0.30,
            random_state=42
        )

        if st.button("Train Model"):

            model = GaussianNB()
            model.fit(X_train, y_train)

            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Accuracy
            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_test_pred)

            st.subheader("Accuracy")
            st.write("Training Accuracy:", train_acc)
            st.write("Testing Accuracy:", test_acc)

            # Confusion Matrices
            st.subheader("Confusion Matrix - Training Data")
            train_cm = confusion_matrix(y_train, y_train_pred)
            st.write(train_cm)

            st.subheader("Confusion Matrix - Testing Data")
            test_cm = confusion_matrix(y_test, y_test_pred)
            st.write(test_cm)
