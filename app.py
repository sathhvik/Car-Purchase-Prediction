# --- Imports ---
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

# --- Load and prepare the main dataset for prediction ---
try:
    df = pd.read_csv("car_purchasing.csv", encoding='ISO-8859-1')
except Exception as e:
    st.error(f"Error loading car purchasing data: {e}")

df_model = df.drop(columns=["customer name", "customer e-mail", "country"])
X = df_model.drop(columns=["car purchase amount"])
y = df_model["car purchase amount"]

# --- Train model ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = RandomForestRegressor(random_state=42)
model.fit(X_scaled, y)

# --- Model evaluation on training data ---
y_pred = model.predict(X_scaled)

r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)

# --- Prepare the Actual vs Predicted plot ---
fig, ax = plt.subplots()
ax.scatter(y, y_pred, edgecolors=(0, 0, 0))
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
ax.set_xlabel('Actual Purchase Amount')
ax.set_ylabel('Predicted Purchase Amount')
ax.set_title('Actual vs Predicted Car Purchase Amount')

# --- Load and clean car listing dataset ---
try:
    cars_df = pd.read_csv("Sport car price.csv")
except Exception as e:
    st.error(f"Error loading car listing data: {e}")

cars_df["Price"] = cars_df["Price (in USD)"].str.replace(",", "").astype(float)
cars_df["Car Name"] = cars_df["Car Make"] + " " + cars_df["Car Model"]
cars_df["Brand"] = cars_df["Car Make"]
cars_df["Fuel Type"] = "N/A"  # No fuel type info available

# --- Streamlit UI ---
st.title("Car Purchase Amount Predictor 🚗")

# Show model performance
st.subheader("Model Evaluation Metrics")
st.write(f"**R² Score:** {r2:.4f}")
st.write(f"**Mean Squared Error:** {mse:,.2f}")

st.subheader("Actual vs Predicted Visualization")
st.pyplot(fig)

# User input
st.write("### Fill in the details below to estimate your car purchase amount:")

gender = st.selectbox("Gender", ("Male", "Female"))
gender = 0 if gender == "Male" else 1

age = st.number_input("Age", min_value=18, max_value=100, value=30)
annual_salary = st.number_input("Annual Salary ($)", min_value=10000, value=50000)
credit_card_debt = st.number_input("Credit Card Debt ($)", min_value=0, value=5000)
net_worth = st.number_input("Net Worth ($)", min_value=0, value=100000)

if st.button("Predict and Recommend Cars"):
    # Prediction
    input_features = np.array([[gender, age, annual_salary, credit_card_debt, net_worth]])
    input_scaled = scaler.transform(input_features)
    prediction = model.predict(input_scaled)
    amount = prediction[0]

    st.success(f"Estimated Car Purchase Amount: **${amount:,.2f}**")

    # Recommend cars in ±10% range
    lower_bound = amount * 0.9
    upper_bound = amount * 1.1
    recommended_cars = cars_df[(cars_df["Price"] >= lower_bound) & (cars_df["Price"] <= upper_bound)]

    st.subheader("Recommended Cars in Your Budget:")

    if not recommended_cars.empty:
        for _, row in recommended_cars.iterrows():
            st.markdown(
                f"**{row['Car Name']}** (${row['Price']:,.0f})  \n"
                f"**Brand**: {row['Brand']}  \n"
                f"**Fuel Type**: {row['Fuel Type']}"
            )
            st.markdown("---")
    else:
        st.warning("No cars found in your predicted price range.")
