import streamlit as st
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

st.title("Naive Bayes Classifier with Confusion Matrix")

uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview")
    st.dataframe(df.head())

    target_column = st.selectbox("Select Target Variable", df.columns)

    feature_columns = st.multiselect(
        "Select Feature Columns",
        [col for col in df.columns if col != target_column]
    )

    if target_column and feature_columns:

        X = df[feature_columns]
        y = df[target_column]

        # Convert categorical features if any
        X = pd.get_dummies(X)

        # 30% training, 70% testing with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            train_size=0.30,
            random_state=42,
            stratify=y
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
            st.write("Training Accuracy:", round(train_acc, 4))
            st.write("Testing Accuracy:", round(test_acc, 4))

            # Ensure full class labels appear
            labels = sorted(y.unique())

            train_cm = confusion_matrix(
                y_train, y_train_pred, labels=labels
            )

            test_cm = confusion_matrix(
                y_test, y_test_pred, labels=labels
            )

            st.subheader("Confusion Matrix - Training Data")
            st.write(pd.DataFrame(train_cm, index=labels, columns=labels))

            st.subheader("Confusion Matrix - Testing Data")
            st.write(pd.DataFrame(test_cm, index=labels, columns=labels))
