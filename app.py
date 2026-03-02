
import streamlit as st
import joblib
import pandas as pd

# Load the pre-trained models
try:
    minmax_scaler = joblib.load('minmax_scaler.joblib')
    xgb_model = joblib.load('xgboost_model.joblib')
    st.success("Models loaded successfully!")
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# Streamlit app title
st.title('Recompra clientes')
st.write('Introduce los detalles para predecir recompra')

# Input fields for user data
dias_ultima_compra = st.number_input('Dias despues de la ultima compra', min_value=0, value=30)
compras_ult_90d = st.number_input('compras en los ultimos 90 días', min_value=0, value=5)
total_compras_historicas = st.number_input('Total de compras históricas', min_value=0, value=100)
ticket_promedio = st.number_input('Ticket promedio', min_value=0.0, value=50.0)

# Create a DataFrame from user input
input_data = pd.DataFrame([{
    'dias_ultima_compra': dias_ultima_compra,
    'compras_ult_90d': compras_ult_90d,
    'total_compras_historicas': total_compras_historicas,
    'ticket_promedio': ticket_promedio
}])

# Preprocess the input data using the loaded MinMaxScaler
try:
    # Ensure column order matches training data if scaler was fitted on a DataFrame
    # Assuming the order was 'dias_ultima_compra', 'compras_ult_90d', 'total_compras_historicas', 'ticket_promedio'
    scaled_input = minmax_scaler.transform(input_data)
    scaled_input_df = pd.DataFrame(scaled_input, columns=input_data.columns)
    st.write('Scaled Input Data:')
    st.dataframe(scaled_input_df)
except Exception as e:
    st.error(f"Error scaling input data: {e}")
    st.stop()

# Make prediction
if st.button('Predecir recompra'):
    try:
        prediction = xgb_model.predict(scaled_input_df)
        prediction_proba = xgb_model.predict_proba(scaled_input_df)[:, 1]

        st.subheader('Prediction Result:')
        if prediction[0] == 1:
            st.error(f"The customer is likely to churn with a probability of {prediction_proba[0]:.2f}")
        else:
            st.success(f"The customer is unlikely to churn with a probability of {prediction_proba[0]:.2f}")
    except Exception as e:
        st.error(f"Error making prediction: {e}")
