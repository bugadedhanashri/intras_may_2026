import streamlit as st
import pandas as pd
import joblib
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

st.sidebar.title("Credit Card Fraud Detection")
st.sidebar.write("Enter transaction details and adjust the fraud threshold.")
st.title("Credit Card Fraud Detection")
st.write("Enter the transaction details to predict whether the transaction is Fraud or Not Fraud.")

model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")
x_test = joblib.load("x_test.pkl")
y_test = joblib.load("y_test.pkl")
Time = st.number_input("Time")

V1 = st.number_input("V1")
V2 = st.number_input("V2")
V3 = st.number_input("V3")
V4 = st.number_input("V4")
V5 = st.number_input("V5")
V6 = st.number_input("V6")
V7 = st.number_input("V7")
V8 = st.number_input("V8")
V9 = st.number_input("V9")
V10 = st.number_input("V10")
V11 = st.number_input("V11")
V12 = st.number_input("V12")
V13 = st.number_input("V13")
V14 = st.number_input("V14")
V15 = st.number_input("V15")
V16 = st.number_input("V16")
V17 = st.number_input("V17")
V18 = st.number_input("V18")
V19 = st.number_input("V19")
V20 = st.number_input("V20")
V21 = st.number_input("V21")
V22 = st.number_input("V22")
V23 = st.number_input("V23")
V24 = st.number_input("V24")
V25 = st.number_input("V25")
V26 = st.number_input("V26")
V27 = st.number_input("V27")
V28 = st.number_input("V28")

Amount = st.number_input("Amount")
threshold = st.slider(
    "Fraud Threshold",
    0.0,
    1.0,
    0.5,
    0.01
)
user_input = pd.DataFrame({
    'Time': [Time],
    'V1': [V1],
    'V2': [V2],
    'V3': [V3],
    'V4': [V4],
    'V5': [V5],
    'V6': [V6],
    'V7': [V7],
    'V8': [V8],
    'V9': [V9],
    'V10': [V10],
    'V11': [V11],
    'V12': [V12],
    'V13': [V13],
    'V14': [V14],
    'V15': [V15],
    'V16': [V16],
    'V17': [V17],
    'V18': [V18],
    'V19': [V19],
    'V20': [V20],
    'V21': [V21],
    'V22': [V22],
    'V23': [V23],
    'V24': [V24],
    'V25': [V25],
    'V26': [V26],
    'V27': [V27],
    'V28': [V28],
    'Amount': [Amount]
})
user_input[['Time', 'Amount']] = scaler.transform(
    user_input[['Time', 'Amount']])
if st.button("Predict"):

    # Single Transaction Prediction
    prob = model.predict_proba(user_input)[0][1]

    prediction = 1 if prob >= threshold else 0

    st.metric("Fraud Probability", f"{prob*100:.2f}%")
    st.progress(float(prob))

    if prediction == 1:
        st.error("🚨 Fraud Transaction")
    else:
        st.success("✅ Normal Transaction")

    # Real-time Confusion Matrix

    from sklearn.metrics import confusion_matrix

    test_prob = model.predict_proba(x_test)[:, 1]
    test_pred = (test_prob >= threshold).astype(int)

    cm = confusion_matrix(y_test, test_pred)

    tn, fp, fn, tp = cm.ravel()

    business_cost = (fp * 500) + (fn * 10000)

    st.info(f"""
    False Positives (FP): {fp}
    False Negatives (FN): {fn}

    Business Cost Formula:
    - FP × ₹500
    - FN × ₹10,000
    """)

    st.metric("Estimated Business Cost", f"₹{business_cost:,}")

    cm_df = pd.DataFrame(
        cm,
        index=["Actual Normal", "Actual Fraud"],
        columns=["Predicted Normal", "Predicted Fraud"]
    )

    st.subheader("Confusion Matrix")

    st.dataframe(cm_df)