# src/app.py
import streamlit as st
import requests

# 1. Page Configuration
st.set_page_config(page_title="Credit Risk Engine", page_icon="💳", layout="wide")

st.title("💳 Credit Risk Decision Engine")
st.markdown("This dashboard connects to our local Docker container (FastAPI) to predict default risk based on the borrower's financial profile.")

st.subheader("Borrower Financial Profile")

# 2. Define the actual feature names (MUST MATCH THE EXACT ORDER OF YOUR TRAINING DATA)
feature_names = [
    "Revolving Utilization (Unsecured Lines)",
    "Borrower Age",
    "Times 30-59 Days Past Due",
    "Debt Ratio",
    "Monthly Income",
    "Open Credit Lines & Loans",
    "Times 90+ Days Late",
    "Real Estate Loans / Mortgages",
    "Times 60-89 Days Past Due",
    "Number of Dependents",
    "Income is Missing (1=Yes, 0=No)",
    "Dependents is Missing (1=Yes, 0=No)"
]

# The default scaled values for our baseline test
default_vals = [0.5, -1.2, 0.0, 2.1, -0.5, 1.1, 0.0, -0.2, 0.0, 1.5, 1.0, 0.0]

# 3. Create an organized grid (4 columns looks better for 12 features)
cols = st.columns(4)
features = []

for i, name in enumerate(feature_names):
    with cols[i % 4]:
        # Now the UI displays the actual financial term!
        val = st.number_input(name, value=default_vals[i], step=0.1)
        features.append(val)

st.divider()

# 4. The Prediction Button
if st.button("Predict Risk", type="primary", use_container_width=True):
    API_URL = "http://127.0.0.1:8000/predict"
    payload = {"features": features}
    
    with st.spinner("Consulting the PyTorch Model..."):
        try:
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
            
        except requests.exceptions.ConnectionError:
            st.error("🚨 Connection Error: Could not reach the API. Is your Docker container running on port 8000?")
        except Exception as e:
            st.error(f"An error occurred: {e}")