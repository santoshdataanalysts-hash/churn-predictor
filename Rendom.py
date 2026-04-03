import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

st.markdown("""
<style>

/* 🔥 Full Background Gradient */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}

/* 🟣 Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a1a, #2b2b2b);
}

/* 🟢 Glass Card Effect */
.block-container {
    background: rgba(255, 255, 255, 0.08);
    border-radius: 20px;
    padding: 25px;
    backdrop-filter: blur(12px);
}

/* 🟡 Text color fix */
h1, h2, h3, h4, h5, h6, p, label {
    color: white !important;
}

/* 🔵 Button style */
.stButton>button {
    background: linear-gradient(45deg, #00c6ff, #0072ff);
    color: white;
    border-radius: 10px;
    border: none;
}

/* 🟠 Upload box */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.05);
    padding: 10px;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- SESSION ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "df_result" not in st.session_state:
    st.session_state.df_result = None


# ---------------- LOGIN ----------------
def login():
    st.title("🔐 Login")

    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")

    if st.button("Login"):
        if user == "admin" and pwd == "1234":
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Wrong credentials")


if not st.session_state.logged_in:
    login()
    st.stop()

st.sidebar.success("✅ Logged in")
if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()


# ---------------- TITLE ----------------
st.markdown("<h1 style='text-align:center; color:#00c6ff;'>🚀 AI Customer Churn Predictor</h1>", unsafe_allow_html=True)
st.success("Mst.titodel Accuracy: ~82%")


# ---------------- LOAD MODEL ----------------
try:
    model = pickle.load(open("model.pkl", "rb"))
except:
    st.error("❌ model.pkl file missing")
    st.stop()


# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"])

if uploaded_file is not None:

    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding="latin1")
        df.columns = df.columns.str.strip()

        st.write("📊 Preview", df.head())

        if df.empty:
            st.error("❌ CSV empty hai")
            st.stop()

    except Exception as e:
        st.error(f"❌ File read error: {e}")
        st.stop()

    # ---------------- FIX CHURN COLUMN ----------------
    if "Churn" not in df.columns:
        possible = [col for col in df.columns if "churn" in col.lower()]
        if possible:
            df.rename(columns={possible[0]: "Churn"}, inplace=True)
        else:
            st.error("❌ CSV must contain 'Churn' column")
            st.stop()

    # ---------------- CLEANING ----------------
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # ---------------- ENCODING ----------------
    from sklearn.preprocessing import LabelEncoder

    for col in df.columns:
        if df[col].dtype == "object" and col != "customerID":
            df[col] = LabelEncoder().fit_transform(df[col])

    # ---------------- PREDICT ----------------
    if st.button("⚡ Run Prediction"):

        X = df.drop(["Churn", "customerID"], axis=1, errors="ignore")
        X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

        preds = model.predict(X)

        df["Prediction"] = preds
        df["Prediction"] = df["Prediction"].map({
            1: "Will Leave",
            0: "Will Stay"
        })

        # Confidence
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)
            df["Confidence"] = probs.max(axis=1) * 100
        else:
            df["Confidence"] = 0

        # Save result
        st.session_state.df_result = df.copy()

        st.success("✅ Prediction Completed")


# ---------------- AFTER PREDICTION ----------------
if st.session_state.df_result is not None:

    df = st.session_state.df_result

    # ---------------- FILTER ----------------
    st.subheader("🔍 Filters")

    search = st.text_input("Search Customer ID")

    filtered_df = df.copy()

    if search and "customerID" in df.columns:
        filtered_df = filtered_df[
            filtered_df["customerID"].astype(str).str.contains(search, case=False, na=False)
        ]

    filter_option = st.selectbox("Filter Prediction", ["All", "Will Stay", "Will Leave"])

    if filter_option != "All":
        filtered_df = filtered_df[filtered_df["Prediction"] == filter_option]

    st.dataframe(filtered_df)

    # ---------------- METRICS ----------------
    st.subheader("📊 Summary")

    total = len(df)
    stay = (df["Prediction"] == "Will Stay").sum()
    leave = (df["Prediction"] == "Will Leave").sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", total)
    col2.metric("Will Stay", stay)
    col3.metric("Will Leave", leave)

    # ---------------- CHART ----------------
    st.subheader("📊 Churn Analysis")

    fig, ax = plt.subplots()
    df["Prediction"].value_counts().plot.pie(autopct="%1.1f%%", ax=ax)
    ax.axis("equal")

    st.pyplot(fig)

    # ---------------- FEATURE IMPORTANCE ----------------
    st.subheader("📊 Feature Importance")

    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        imp = model.coef_[0]
    else:
        imp = None

    if imp is not None:
        imp_df = pd.DataFrame({
            "Feature": df.drop(["Prediction"], axis=1).columns[:len(imp)],
            "Importance": imp
        }).sort_values(by="Importance", ascending=False)

        st.bar_chart(imp_df.set_index("Feature"))
    else:
        st.warning("Feature importance not available")

    # ---------------- DOWNLOAD ----------------
    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "📥 Download Results",
        csv,
        "churn_results.csv",
        "text/csv"
    )
