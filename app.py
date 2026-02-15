import streamlit as st #AnjaliTiwari
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Titanic Survival Prediction")

st.sidebar.header("Settings")
test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
max_iter = st.sidebar.slider("Logistic Regression Max Iterations", 100, 2000, 1000, 100)

file = st.file_uploader("Upload Titanic CSV", type="csv")
if file:
    df = pd.read_csv(file)
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df.dropna(subset=['Embarked'], inplace=True)
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

    features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']
    X = df[features]
    y = df['Survived']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(max_iter=max_iter)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='Blues', ax=ax)
    ax.set_title("Titanic Survival Confusion Matrix")
    st.pyplot(fig)
