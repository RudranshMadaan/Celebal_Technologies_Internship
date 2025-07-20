import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

@st.cache_resource
def load_model():
    return joblib.load(r"C:\Users\RUDRANSH\Downloads\ml_streamlit_app\model\model.pkl")

model = load_model()
iris = load_iris()
feature_names = iris.feature_names
target_names = iris.target_names

st.title("🌸 Iris Flower Classifier")
st.write("Provide flower measurements to predict the Iris species.")

st.sidebar.header("Input Features")
def get_input():
    sepal_length = st.sidebar.slider('Sepal length (cm)', 4.0, 8.0, 5.8)
    sepal_width = st.sidebar.slider('Sepal width (cm)', 2.0, 4.5, 3.0)
    petal_length = st.sidebar.slider('Petal length (cm)', 1.0, 7.0, 4.0)
    petal_width = st.sidebar.slider('Petal width (cm)', 0.1, 2.5, 1.2)
    return pd.DataFrame([{
        'sepal length (cm)': sepal_length,
        'sepal width (cm)': sepal_width,
        'petal length (cm)': petal_length,
        'petal width (cm)': petal_width
    }])

input_df = get_input()

st.subheader("Input Features")
st.write(input_df)

prediction = model.predict(input_df)[0]
prediction_proba = model.predict_proba(input_df)

st.subheader("Prediction")
st.write(f"🌺 Predicted Iris type: **{target_names[prediction]}**")

st.subheader("Prediction Probabilities")
st.write(pd.DataFrame(prediction_proba, columns=target_names))

# Probability chart
st.subheader("Probability Bar Chart")
fig, ax = plt.subplots()
sns.barplot(x=target_names, y=prediction_proba[0], palette='viridis', ax=ax)
ax.set_ylabel("Probability")
st.pyplot(fig)