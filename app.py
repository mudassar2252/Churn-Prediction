import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go

# Load the trained model, preprocessing pipeline, and label encoders
@st.cache_resource  # Use st.cache_resource for caching models and resources
def load_artifacts():
    model = joblib.load('best_model.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
    pca = joblib.load('pca.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    return model, preprocessor, pca, label_encoders

model, preprocessor, pca, label_encoders = load_artifacts()

# Title of the app
st.title("Telecom Customer Churn Prediction")

st.write("""
This app predicts whether a telecom customer will churn or not based on their details.
""")

# Sidebar for user input
st.sidebar.header("User Input Features")

def user_input_features():
    tenure = st.sidebar.slider("Tenure (months)", 1, 72, 12)
    monthly_charges = st.sidebar.slider("Monthly Charges", 18.0, 120.0, 29.85)
    total_charges = st.sidebar.slider("Total Charges", 18.0, 8500.0, 358.2)
    
    # Dynamic logic for other features based on input
    if monthly_charges > 70:
        internet_service = "Fiber optic"
    else:
        internet_service = "DSL"
    
    if tenure > 24:
        contract = "Two year"
    elif tenure > 12:
        contract = "One year"
    else:
        contract = "Month-to-month"
    
    if total_charges > 2000:
        payment_method = "Credit card (automatic)"
    else:
        payment_method = "Electronic check"
    
    # Default values for other features
    data = {
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'gender': "Female",  # Default value
        'Contract': contract,  # Dynamically set
        'PaymentMethod': payment_method,  # Dynamically set
        'SeniorCitizen': "No",  # Default value
        'Partner': "No",  # Default value
        'Dependents': "No",  # Default value
        'InternetService': internet_service  # Dynamically set
    }
    return pd.DataFrame(data, index=[0]), data

input_df, input_data = user_input_features()

# Display user input features
st.subheader("User Input Features")
st.write(input_df[['tenure', 'MonthlyCharges', 'TotalCharges']])

# Display remaining features (output)
st.subheader("Customer Details Breakdown")
st.write(input_df.drop(columns=['tenure', 'MonthlyCharges', 'TotalCharges']))

# Preprocess input data
category_mappings = {
    'gender': {'Male': 0, 'Female': 1},
    'Contract': {'Month-to-month': 0, 'One year': 1, 'Two year': 2},
    'PaymentMethod': {'Electronic check': 0, 'Mailed check': 1, 'Bank transfer (automatic)': 2, 'Credit card (automatic)': 3},
    'SeniorCitizen': {'No': 0, 'Yes': 1},
    'Partner': {'No': 0, 'Yes': 1},
    'Dependents': {'No': 0, 'Yes': 1},
    'InternetService': {'DSL': 0, 'Fiber optic': 1, 'No': 2}
}
for col, mapping in category_mappings.items():
    input_df[col] = input_df[col].map(mapping)

# Make predictions
try:
    processed_data = preprocessor.transform(input_df)
    processed_data_pca = pca.transform(processed_data)
    prediction = model.predict(processed_data_pca)
    prediction_proba = model.predict_proba(processed_data_pca)

    churn_status = "Churn" if prediction[0] == 1 else "Not Churn"
    st.subheader("Prediction")
    st.write(f"The customer is predicted to: **{churn_status}**")
    
    st.subheader("Prediction Probability")
    st.write(f"Probability of Churn: {prediction_proba[0][1]:.2f}")
    st.write(f"Probability of Not Churning: {prediction_proba[0][0]:.2f}")

    # Sunburst Chart for Customer Details
    st.subheader("Customer Details Visualization")
    sunburst_df = pd.DataFrame({
        'Category': ["Gender", "Contract", "Payment Method", "Senior Citizen", "Partner", "Dependents", "Internet Service"],
        'Value': [input_data['gender'], input_data['Contract'], input_data['PaymentMethod'], input_data['SeniorCitizen'], input_data['Partner'], input_data['Dependents'], input_data['InternetService']],
    })
    sunburst_fig = px.sunburst(
        sunburst_df, path=['Category', 'Value'],
        title="Customer Details Breakdown",
        color='Value',
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    st.plotly_chart(sunburst_fig)
    
    # Donut Chart for Churn Prediction
    donut_fig = go.Figure(data=[
        go.Pie(labels=["Not Churn", "Churn"], values=[prediction_proba[0][0], prediction_proba[0][1]],
               hole=0.5, marker=dict(colors=["#1f77b4", "#ff7f0e"]))
    ])
    donut_fig.update_layout(title="Churn Prediction Distribution")
    st.plotly_chart(donut_fig)
    
except Exception as e:
    st.error(f"An error occurred: {e}")