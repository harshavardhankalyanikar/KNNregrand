import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Page config
st.set_page_config("KNN Regression with insurance dataset", layout="centered")


# Load CSS
def load_css(file):
    with open(file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css("style.css")

# Title
st.markdown(
    """
<div class="card">
<h1>KNN Regression with insurance dataset</h1>
<p>Predict <b>Charges</b> from <b>Age, Sex, BMI, Children</b> and other factors</p>
</div>
""",
    unsafe_allow_html=True,
)


# Load data
@st.cache_data
def load_data():
    return pd.read_csv("insurance.csv")


df = load_data()

# Standardize column names to lowercase
df.columns = df.columns.str.lower()

# Dataset preview
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Dataset Preview")
st.dataframe(df.head())
st.markdown("</div>", unsafe_allow_html=True)

# Encode categorical variables
df_processed = df.copy()
label_encoders = {}
for col in ['sex', 'smoker', 'region']:
    le = LabelEncoder()
    df_processed[col] = le.fit_transform(df_processed[col])
    label_encoders[col] = le

# Prepare data
X = df_processed.drop(columns=["charges"])
y = df_processed["charges"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model settings
n_neighbors = st.slider("Number of neighbors (k)", 1, 20, 5)

# Train model
model = KNeighborsRegressor(n_neighbors=n_neighbors)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Metrics
mae=mean_absolute_error(y_test, y_pred)
rmse=np.sqrt(mean_squared_error(y_test, y_pred))
r2=r2_score(y_test, y_pred)
adj_r2=1-(1-r2)*(len(y_test)-1)/(len(y_test)-X.shape[1]-1)

# Visualization (Charges vs Age)
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Insurance Charges vs Age")

fig, ax = plt.subplots()
ax.scatter(df["age"], df["charges"], alpha=0.6)

age_vals = np.linspace(df["age"].min(), df["age"].max(), 100)

# Create feature matrix with average values for other features
mean_values = df_processed.drop(columns=["charges"]).mean()
X_line = pd.DataFrame([mean_values] * len(age_vals))
X_line["age"] = age_vals

X_line_scaled = scaler.transform(X_line)
y_line = model.predict(X_line_scaled)

ax.plot(age_vals, y_line, color="red", linewidth=2, label="Trend")
ax.set_xlabel("Age")
ax.set_ylabel("Charges")
ax.legend()

st.pyplot(fig)
st.markdown("</div>", unsafe_allow_html=True)

# Performance
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Model Performance")

c1, c2 = st.columns(2)
c1.metric("MAE", f"{mae:.2f}")
c2.metric("RMSE", f"{rmse:.2f}")

c3, c4 = st.columns(2)
c3.metric("R²", f"{r2:.3f}")
c4.metric("Adjusted R²", f"{adj_r2:.3f}")

st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    f"""
<div class="card">
<h3>Model Details</h3>
<p>
<b>Model type:</b> K-Nearest Neighbors Regression<br>
<b>Neighbors (k):</b> {model.n_neighbors}<br>
<b>Algorithm:</b> {model.algorithm}<br>
<b>Distance metric:</b> {model.metric}
</p>
</div>
""",
    unsafe_allow_html=True,
)

# Prediction
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Predict Charges")

sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", float(df["age"].min()), float(df["age"].max()), 30.0)
bmi = st.slider("BMI", float(df["bmi"].min()), float(df["bmi"].max()), 25.0)
children = st.slider("Children", int(df["children"].min()), int(df["children"].max()), 0)
smoker = st.selectbox("Smoker", ["no", "yes"])
region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

# Encode inputs
sex_encoded = label_encoders["sex"].transform([sex])[0]
smoker_encoded = label_encoders["smoker"].transform([smoker])[0]
region_encoded = label_encoders["region"].transform([region])[0]

# Create input array in correct feature order
input_data = [[age, sex_encoded, bmi, children, smoker_encoded, region_encoded]]
input_scaled = scaler.transform(input_data)
pred_charges = model.predict(input_scaled)[0]

st.markdown(
    f'<div class="prediction-box">Predicted Charges: ${pred_charges:.2f}</div>',
    unsafe_allow_html=True,
)

st.markdown("</div>", unsafe_allow_html=True)