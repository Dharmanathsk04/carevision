import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("diabetes_prediction_dataset.csv")
df.columns = df.columns.str.lower()

le_gender = LabelEncoder()
le_smoke = LabelEncoder()

df["gender"] = le_gender.fit_transform(df["gender"].str.lower())
df["smoking_history"] = le_smoke.fit_transform(df["smoking_history"].str.lower())

X = df.drop("diabetes", axis=1)
y = df["diabetes"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression(max_iter=1000)
model.fit(X_scaled, y)

joblib.dump(model, "diabetes_model.pkl")
joblib.dump(scaler, "diabetes_scaler.pkl")
joblib.dump(le_gender, "gender_encoder.pkl")
joblib.dump(le_smoke, "smoking_encoder.pkl")

print("✅ Diabetes model saved")
