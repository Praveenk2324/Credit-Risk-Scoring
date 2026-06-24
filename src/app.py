# src/app.py
import streamlit as st
import requests

st.set_page_config(page_title="Credit Risk Engine", page_icon="💳", layout="wide")

st.title("💳 Credit Risk Decision Engine")
# st.markdown("Enter the borrower's actual financial details below. The Streamlit UI sends this raw data to our Docker API, which dynamically scales it using the production `scaler.pkl` before consulting the PyTorch model.")
st.markdown("Enter the borrower's actual financial details below")
# --- 1. RAW DATA COLLECTION (Human-Readable UI) ---
st.subheader("Borrower Financial Profile")

col1, col2, col3, col4 = st.columns(4)

with col1:
    age = st.number_input("Borrower Age", min_value=18, max_value=100, value=45, step=1)
    income = st.number_input("Monthly Income ($)", min_value=0, value=6500, step=500, format="%d")
    dependents = st.number_input("Number of Dependents", min_value=0, max_value=20, value=2, step=1)

with col2:
    debt_ratio = st.number_input("Debt Ratio (%)", min_value=0.0, value=35.0, step=1.0) / 100.0
    utilization = st.number_input("Credit Utilization (%)", min_value=0.0, value=30.0, step=1.0) / 100.0
    open_lines = st.number_input("Open Credit Lines", min_value=0, value=8, step=1)

with col3:
    real_estate = st.number_input("Real Estate Loans", min_value=0, value=1, step=1)
    late_30_59 = st.number_input("Times 30-59 Days Late", min_value=0, value=0, step=1)
    late_60_89 = st.number_input("Times 60-89 Days Late", min_value=0, value=0, step=1)

with col4:
    late_90_plus = st.number_input("Times 90+ Days Late", min_value=0, value=0, step=1)
    
    # Auto-calculate the missing flags based on user input
    income_missing = 1.0 if income == 0 else 0.0
    dependents_missing = 1.0 if dependents == 0 else 0.0

st.divider()

# --- 2. API COMMUNICATION ---
if st.button("Predict Risk", type="primary", use_container_width=True):
    
    # The exact order must match the 12 features expected by the PyTorch model
    # (10 continuous features first, followed by the 2 binary missing flags)
    raw_array = [
        utilization, age, late_30_59, debt_ratio, income, 
        open_lines, late_90_plus, real_estate, late_60_89, dependents,
        income_missing, dependents_missing
    ]
    # http://16.16.196.24:8000/predict
    API_URL ="http://13.53.39.150:8000/predict"
    payload = {"features": raw_array}
    
    with st.spinner("Consulting the Docker API..."):
        try:
            # Send the RAW data to the API
            response = requests.post(API_URL, json=payload)
            response.raise_for_status() 
            
            result = response.json()
            prob = result['default_probability']
            tier = result['risk_tier']
            
            st.subheader("Prediction Results")
            
            # Display results with dynamic color coding
            if prob > 0.5:
                st.error(f"**System Decision:** {tier}")
            elif prob > 0.3:
                st.warning(f"**System Decision:** Medium Risk (Manual Review Required)")
            else:
                st.success(f"**System Decision:** {tier}")
                
            st.metric(label="Probability of Default within 2 Years", value=f"{prob * 100:.2f}%")
            
            with st.expander("View Raw Payload Sent to Docker"):
                st.write("Streamlit sent these exact unscaled values to the API:", raw_array)
            
        except requests.exceptions.ConnectionError:
            st.error("🚨 Connection Error: Could not reach the API. Is your Docker container running on port 8000?")
        except Exception as e:
            st.error(f"An error occurred: {e}")



# Step 1: Wake Up the AWS Container
# Go back to your AWS Console -> ECS -> Clusters.

# Click your credit-risk-cluster.

# Under the Services tab, click your credit-risk-service, and hit Update.

# Change Desired Tasks from 0 back to 1.

# Click Update Service.

# Step 2: Grab Your New IP Address
# Inside your cluster, click the Tasks tab.

# Wait about 30–60 seconds and hit the refresh button until the task status turns green and says RUNNING.

# Click the blue Task ID (the long string of letters/numbers).

# Look under the Configuration section and copy your brand new Public IP.
