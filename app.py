import streamlit as st
import pandas as pd
import joblib

st.title("Amazon Product Rating Prediction")

# Upload file
uploaded_file = st.file_uploader("Tải lên file CSV dữ liệu sản phẩm Amazon", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Hiển thị dữ liệu
    st.subheader("Dữ liệu gốc")
    st.dataframe(df.head())

    # Load mô hình đã huấn luyện
    model = joblib.load("linear_regression_model.pkl")

    # Chọn cột đầu vào (giống như lúc training)
    features = ['discounted_price', 'actual_price', 'discount_percentage', 'rating_count', 'main_category']
    df_input = df[features]

    # Dự đoán
    predictions = model.predict(df_input)

    # Kết quả
    df['Predicted Rating'] = predictions
    st.subheader("Kết quả dự đoán")
    st.dataframe(df[['product_name', 'Predicted Rating']])
