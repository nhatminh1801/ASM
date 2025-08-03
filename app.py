import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Amazon Product Rating Prediction", layout="centered")
st.title("📦 Amazon Product Rating Prediction")

@st.cache_resource
def load_model():
    try:
        return joblib.load("linear_regression_model.pkl")
    except FileNotFoundError:
        st.error("❌ Không tìm thấy file 'linear_regression_model.pkl'.")
        return None

def preprocess_data(df):
    try:
        # Clean prices
        df['discounted_price'] = df['discounted_price'].str.replace('₹', '').str.replace(',', '').astype(float)
        df['actual_price'] = df['actual_price'].str.replace('₹', '').str.replace(',', '').astype(float)
        df['discount_percentage'] = df['discount_percentage'].str.replace('%', '').astype(float) / 100

        # Convert ratings
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        df['rating_count'] = pd.to_numeric(df['rating_count'], errors='coerce')
        df['rating'].fillna(df['rating'].mean(), inplace=True)
        df['rating_count'].fillna(df['rating_count'].median(), inplace=True)

        # Fill missing
        df['about_product'].fillna('No description', inplace=True)
        df['review_content'].fillna('No review', inplace=True)
        df.dropna(subset=['product_name'], inplace=True)

        # Main category
        df['main_category'] = df['category'].str.split('|').str[0]

        # Outlier removal
        Q1 = df['discounted_price'].quantile(0.25)
        Q3 = df['discounted_price'].quantile(0.75)
        IQR = Q3 - Q1
        df = df[~((df['discounted_price'] < (Q1 - 1.5 * IQR)) | (df['discounted_price'] > (Q3 + 1.5 * IQR)))]

        # Drop duplicates
        df.drop_duplicates(subset=['product_id'], keep='first', inplace=True)

        return df

    except Exception as e:
        st.error(f"Lỗi xử lý dữ liệu: {e}")
        return None

model = load_model()

st.subheader("📤 Tải lên file Amazon CSV gốc (chưa xử lý)")
uploaded_file = st.file_uploader("Chọn file CSV", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        df = preprocess_data(df)
        if df is not None:
            st.write("✅ Dữ liệu sau xử lý:")
            st.dataframe(df.head())

            features = ['discounted_price', 'actual_price', 'discount_percentage', 'rating_count', 'main_category']
            if model and all(col in df.columns for col in features):
                df_input = df[features]
                predictions = model.predict(df_input)
                df['Predicted Rating'] = predictions

                st.success("✅ Dự đoán hoàn tất!")
                st.subheader("📈 Kết quả dự đoán")
                st.dataframe(df[['product_name', 'Predicted Rating']] if 'product_name' in df.columns else df[['Predicted Rating']])

                st.download_button("📥 Tải kết quả CSV", df.to_csv(index=False), file_name="predicted_ratings.csv", mime="text/csv")
            else:
                st.warning("⚠️ Dữ liệu đầu vào không đủ cột cần thiết để dự đoán.")
    except Exception as e:
        st.error(f"❌ Lỗi khi đọc file: {e}")
else:
    st.info("⬆️ Hãy tải lên file Amazon gốc (amazon.csv) để dự đoán.")
