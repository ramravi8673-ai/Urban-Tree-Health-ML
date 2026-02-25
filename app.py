import streamlit as st

# -----------------------------------
# Page Configuration
# -----------------------------------
st.set_page_config(
    page_title="Urban Tree AI",
    page_icon="🌳",
    layout="wide"
)

# -----------------------------------
# Main Title Section
# -----------------------------------
st.title("🌳 Urban Tree Health AI System")
st.subheader("AI-Powered Sustainability & Smart City Monitoring")

st.divider()

# -----------------------------------
# Information Cards Section
# -----------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="🔍 Prediction Engine", value="Active")

with col2:
    st.metric(label="📊 Evaluation Metrics", value="Available")

with col3:
    st.metric(label="🧠 Model Intelligence", value="Enabled")

st.divider()

# -----------------------------------
# About Section
# -----------------------------------
st.header("🌿 About This Project")

st.write("""
This application predicts urban tree health using Machine Learning models trained on tree census data.

Features included:
- Tuned Random Forest Classifier
- Logistic Regression baseline comparison
- ROC-AUC Multiclass Evaluation
- Feature Importance Analysis
- Multi-page professional dashboard
""")

st.success("Use the sidebar to navigate between Prediction and Model Insights pages.")

st.divider()

st.caption("© 2026 Urban Tree AI | Machine Learning Capstone Project")
