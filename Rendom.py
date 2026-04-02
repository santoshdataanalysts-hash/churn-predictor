import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

#background color
st.markdown("""
<style>

/* 🔴 Full App Background (gradient) */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f2027, #2c5364, #000000);
    color: white;
}

/* 🔵 Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a1a, #2b2b2b);
    color: white;
}

/* 🟢 Text color fix */
h1, h2, h3, h4, h5, h6, p, label, div {
    color: white !important;
}

/* 🟣 Buttons */
.stButton>button {
    background: linear-gradient(45deg, #ff416c, #ff4b2b);
    color: white;
    border-radius: 10px;
    border: none;
}

/* 🟡 Upload box */
[data-testid="stFileUploader"] {
    background-color: rgba(255,255,255,0.05);
    padding: 10px;
    border-radius: 10px;
}

/* 🟠 Cards / info box */
.stAlert {
    background-color: rgba(255,255,255,0.1);
    color: white;
}

/* ⚫ Remove top gap */
.block-container {
    padding-top: 2rem;
}

</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>

/* 🔥 Dataframe dark theme */
[data-testid="stDataFrame"] {
    background-color: rgba(0,0,0,0.6) !important;
    color: white !important;
    border-radius: 10px;
}

/* Table text */
[data-testid="stDataFrame"] div {
    color: white !important;
}

/* Header row */
[data-testid="stDataFrame"] thead {
    background-color: #ff416c !important;
    color: white !important;
}

/* Rows */
[data-testid="stDataFrame"] tbody tr {
    background-color: rgba(255,255,255,0.05);
}

</style>
""", unsafe_allow_html=True)

#Simple Login System

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login():
    st.title("Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "1234":
            st.session_state.logged_in = True
            st.success("Login Successful!")
            st.rerun()
        else:
            st.error("Invalid Username or Password")

#agar user login nahi hai to login page dikhana hai
if not st.session_state.logged_in:
    login()
    st.stop()
st.sidebar.success("Logged in as Admin")
if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()

st.markdown("""
<h1 style='text-align:center; color:#4CAF50;'>
AI Customer Churn Predictor
</h1>
""", unsafe_allow_html=True)
st.info("Model Accuracy: ~82%")

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Upload CSV
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    
    if "Churn" not in df.columns:
        st.error("❌ CSV must contain 'Churn' column")
        st.stop()

    st.write(df.head())

    # Cleaning
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
    df.dropna(inplace=True)

    # Encoding
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()

    for col in df.columns:
        if df[col].dtype == 'object' and col not in ["customerID"]:
            df[col] = le.fit_transform(df[col])

    #Prediction button
    if st.button("Predict"):

        X = df.drop(["Churn", "customerID"], axis=1)

        with st.spinner("Predicting..."):
            predictions = model.predict(X)

        df["Prediction"] = predictions
        df["Prediction"] = df["Prediction"].map({1: "Will Leave", 0: "Will Stay"})
        
        st.markdown("""
<div style="background:#d4edda; padding:15px; border-radius:10px; color:#155724;">
    ✅ Prediction Completed Successfully!
</div>
""", unsafe_allow_html=True)

        probs = model.predict_proba(X)

        df["Confidence"] = probs.max(axis=1) * 100

        def color_prediction(val):
            if val == "Will Leave":
                return "background-color: #ff4b2b; color: white"
            else:
                return "background-color: #00c853; color: white"
        styled_df = df[["customerID", "Prediction", "Confidence"]].style.applymap(
            color_prediction, subset=["Prediction"]
)

        st.write(styled_df)
        csv = df.to_csv(index=False).encode('utf-8')

        st.download_button(
    label="📥 Download Results",
    data=csv,
    file_name='churn_predictions.csv',
    mime='text/csv',
)
        # Display results
        st.subheader("Results")
        
        st.dataframe(df[["customerID", "Prediction", "Confidence"]])

        st.write("Total Customers:", len(df))

        st.write("Will Stay:", (df["Prediction"].str.contains("Stay")).sum())
        st.write("Will Leave:", (df["Prediction"].str.contains("Leave")).sum())
    
        col1, col2, col3 = st.columns(3)

        col1.metric("Total Customers", len(df))
        col2.metric("Will Stay", (predictions == 0).sum())
        col3.metric("Will Leave", (predictions == 1).sum())
        
        #Graph added
        st.subheader("Churn Analysis")
        st.markdown("---")
        churn_counts = df["Prediction"].value_counts()

        fig, ax = plt.subplots()


        ax.pie(
            churn_counts,
            labels=churn_counts.index,
            autopct='%1.1f%%',
            startangle=90)
        ax.axis('equal')
    
        st.pyplot(fig)
        st.markdown("---")
        # -------------------------
# 📊 Feature Importance
# -------------------------
        st.subheader("📊 Feature Importance")

        features = X.columns

        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_

        elif hasattr(model, "coef_"):
            importance = model.coef_[0]

        else:
            importance = None

        if importance is not None:
            importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": importance}).sort_values(by="Importance", ascending=False)

            st.bar_chart(importance_df.set_index("Feature"))

        else:
            st.warning("Feature importance not available")
