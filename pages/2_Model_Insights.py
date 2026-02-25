import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Model Insights", page_icon="📊")

st.title("📊 Model Performance Dashboard")

st.divider()

# =====================================================
# 1️⃣ Load Best Model Accuracy
# =====================================================
try:
    with open("model_accuracy.txt", "r") as f:
        accuracy = float(f.read())

    st.subheader("🎯 Tuned Random Forest Accuracy")
    st.success(f"Accuracy: {round(accuracy * 100, 2)}%")

except:
    st.error("model_accuracy.txt not found. Please run main.py.")
    st.stop()

st.divider()

# =====================================================
# 2️⃣ Confusion Matrix
# =====================================================
st.subheader("📌 Confusion Matrix")

try:
    cm = pd.read_csv("confusion_matrix.csv").values
    labels = ["Good", "Fair", "Poor"]

    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax_cm
    )

    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")

    st.pyplot(fig_cm)

except:
    st.warning("confusion_matrix.csv not found.")

st.divider()

# =====================================================
# 3️⃣ Feature Importance
# =====================================================
st.subheader("🌿 Feature Importance")

try:
    fi = pd.read_csv("feature_importance.csv")
    st.bar_chart(fi.set_index("Feature"))

except:
    st.warning("feature_importance.csv not found.")

st.divider()

# =====================================================
# 4️⃣ Model Comparison
# =====================================================
st.subheader("📈 Model Comparison")

try:
    comparison = pd.read_csv("model_comparison.csv")
    st.bar_chart(comparison.set_index("Model"))

except:
    st.warning("model_comparison.csv not found.")

st.divider()

# =====================================================
# 5️⃣ ROC Curve (Multiclass)
# =====================================================
st.subheader("📈 ROC Curve")

try:
    fig_roc, ax_roc = plt.subplots()

    class_labels = ["Good", "Fair", "Poor"]

    for i in range(3):
        df_roc = pd.read_csv(f"roc_curve_class_{i}.csv")
        ax_roc.plot(df_roc["fpr"], df_roc["tpr"], label=class_labels[i])

    ax_roc.plot([0, 1], [0, 1], linestyle="--")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend()

    st.pyplot(fig_roc)

    roc_summary = pd.read_csv("roc_summary.csv")
    st.subheader("📊 AUC Scores")

    roc_summary["Class"] = roc_summary["Class"].map({
        0: "Good",
        1: "Fair",
        2: "Poor"
    })

    st.dataframe(roc_summary)

except:
    st.warning("ROC files not found. Run main.py first.")

st.divider()

st.caption("© 2026 Urban Tree AI | Model Evaluation Dashboard")
