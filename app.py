import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Amazon Product Rating Prediction", layout="centered")
st.title("📦 Amazon Product Rating Prediction")

# Load model
@st.cache_resource
def load_model():
    try:
        model = joblib.load("linear_regression_model.pkl")
        return model
    except FileNotFoundError:
        st.error("❌ Không tìm thấy file 'linear_regression_model.pkl'. Vui lòng thêm vào thư mục project.")
        return None

model = load_model()

# Upload data
st.subheader("📤 Tải file CSV sản phẩm Amazon")
uploaded_file = st.file_uploader("Chọn file CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Không thể đọc file CSV: {e}")
        st.stop()

    st.write("📊 Dữ liệu gốc:")
    st.dataframe(df.head())

    # Kiểm tra các cột cần thiết
    required_cols = ['discounted_price', 'actual_price', 'discount_percentage', 'rating_count', 'main_category']
    if not all(col in df.columns for col in required_cols):
        st.warning(f"⚠️ File của bạn thiếu các cột cần thiết: {', '.join(required_cols)}")
    elif model:
        # Dự đoán
        try:
            X = df[required_cols]
            predictions = model.predict(X)
            df['Predicted Rating'] = predictions
            st.success("✅ Dự đoán thành công!")

            st.subheader("📈 Kết quả dự đoán")
            st.dataframe(df[['product_name'] + ['Predicted Rating']] if 'product_name' in df.columns else df[['Predicted Rating']])

            # Cho phép tải xuống
            st.download_button("📥 Tải file kết quả CSV", data=df.to_csv(index=False), file_name="predicted_ratings.csv", mime="text/csv")

        except Exception as e:
            st.error(f"❌ Lỗi khi dự đoán: {e}")
else:
    st.info("⬆️ Hãy tải lên một file CSV để bắt đầu.")
