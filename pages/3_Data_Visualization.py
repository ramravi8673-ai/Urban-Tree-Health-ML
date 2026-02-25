import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Data Visualization", page_icon="📊")

st.title("📊 Urban Tree Data Visualization")

st.divider()

# Load Dataset
try:
    df = pd.read_csv("tree_census_processed.csv", nrows=20000)
    df = df[['tree_dbh', 'stump_diam', 'health']].dropna()
except:
    st.error("Dataset not found.")
    st.stop()

# -------------------------------------------
# 1️⃣ Health Category Distribution
# -------------------------------------------
st.subheader("🌿 Tree Health Distribution")

fig1, ax1 = plt.subplots()
sns.countplot(data=df, x='health', palette="Greens", ax=ax1)
ax1.set_xlabel("Health Category")
ax1.set_ylabel("Count")

st.pyplot(fig1)

st.divider()

# -------------------------------------------
# 2️⃣ Tree DBH Distribution
# -------------------------------------------
st.subheader("📏 Tree Diameter (DBH) Distribution")

fig2, ax2 = plt.subplots()
sns.histplot(df['tree_dbh'], bins=30, kde=True, ax=ax2)
ax2.set_xlabel("Tree DBH")

st.pyplot(fig2)

st.divider()

# -------------------------------------------
# 3️⃣ Stump Diameter Distribution
# -------------------------------------------
st.subheader("📏 Stump Diameter Distribution")

fig3, ax3 = plt.subplots()
sns.histplot(df['stump_diam'], bins=30, kde=True, ax=ax3)
ax3.set_xlabel("Stump Diameter")

st.pyplot(fig3)

st.divider()

# -------------------------------------------
# 4️⃣ DBH vs Health Boxplot
# -------------------------------------------
st.subheader("📈 DBH vs Health Relationship")

fig4, ax4 = plt.subplots()
sns.boxplot(data=df, x="health", y="tree_dbh", palette="Greens", ax=ax4)

st.pyplot(fig4)

st.caption("© 2026 Urban Tree AI | EDA Dashboard")
