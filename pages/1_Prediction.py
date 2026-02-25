import streamlit as st
import joblib
import pandas as pd
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
import tempfile

st.set_page_config(page_title="Prediction", page_icon="🌿")

st.title("🌿 Tree Health Prediction")

# Load model
model = joblib.load("tree_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

st.divider()

# User Inputs
col1, col2 = st.columns(2)

with col1:
    dbh = st.number_input("Tree Diameter (DBH)", min_value=0.0)

with col2:
    stump_diam = st.number_input("Stump Diameter", min_value=0.0)

st.divider()

if st.button("🔮 Predict Health", use_container_width=True):

    input_data = pd.DataFrame(
        [[dbh, stump_diam]],
        columns=['tree_dbh', 'stump_diam']
    )

    prediction = model.predict(input_data)
    proba = model.predict_proba(input_data)

    predicted_label = label_encoder.inverse_transform(prediction)
    confidence = round(max(proba[0]) * 100, 2)

    st.success(f"🌿 Predicted Health: {predicted_label[0]}")
    st.info(f"📊 Model Confidence: {confidence}%")

    # -----------------------------
    # Generate PDF Report
    # -----------------------------
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:

        doc = SimpleDocTemplate(tmp.name)
        elements = []
        styles = getSampleStyleSheet()

        elements.append(Paragraph("<b>Urban Tree Health AI - Prediction Report</b>", styles["Title"]))
        elements.append(Spacer(1, 0.3 * inch))

        data = [
            ["Tree DBH", str(dbh)],
            ["Stump Diameter", str(stump_diam)],
            ["Predicted Health", predicted_label[0]],
            ["Model Confidence", f"{confidence}%"]
        ]

        table = Table(data, colWidths=[2.5 * inch, 2.5 * inch])
        table.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 1, colors.grey),
            ('FONTSIZE', (0,0), (-1,-1), 12),
            ('ALIGN', (1,0), (-1,-1), 'CENTER')
        ]))

        elements.append(table)
        doc.build(elements)

        with open(tmp.name, "rb") as f:
            st.download_button(
                label="📄 Download PDF Report",
                data=f,
                file_name="urban_tree_health_report.pdf",
                mime="application/pdf"
            )
