# train_model.py
import joblib
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load data
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model
joblib.dump(model, r"C:\Users\RUDRANSH\Downloads\ml_streamlit_app\model\model.pkl")
print("✅ Model trained and saved at model/model.pkl")
