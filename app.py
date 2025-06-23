
import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("should_discount_model.pkl")

st.title("📚 Tiki - Dự đoán sản phẩm có nên giảm giá không")

description = st.text_area("Mô tả sản phẩm")
category = st.text_input("Ngành hàng", "Sách thiếu nhi")
inventory_status = st.selectbox("Tình trạng kho", ["available", "out_of_stock"])
price = st.number_input("Giá (VND)", min_value=1000)
review_count = st.number_input("Số lượt đánh giá", min_value=0)
stock_qty = st.number_input("Số lượng tồn kho", min_value=0)
max_sale_qty = st.number_input("Số lượng tối đa mỗi đơn", min_value=1)

if st.button("🎯 Dự đoán"):
    input_df = pd.DataFrame([{
        "short_description": description,
        "categories_name": category,
        "inventory_status": inventory_status,
        "price": price,
        "review_count": review_count,
        "stock_item_qty": stock_qty,
        "stock_item_max_sale_qty": max_sale_qty
    }])
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.success("✅ NÊN giảm giá để kích cầu!")
    else:
        st.info("⚖️ KHÔNG cần giảm giá lúc này.")
