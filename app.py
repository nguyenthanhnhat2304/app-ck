
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline

# Sample dataset - giả lập dữ liệu sách từ Tiki
@st.cache_data
def load_data():
    data = {
        "short_description": [
            "Cuốn sách giúp trẻ phát triển tư duy logic và sáng tạo",
            "Truyện cổ tích Việt Nam hay nhất mọi thời đại",
            "Hướng dẫn đầu tư tài chính cho người mới bắt đầu",
            "Sách kỹ năng mềm dành cho sinh viên và người đi làm",
            "Tiểu thuyết trinh thám lôi cuốn với nhiều plot twist"
        ],
        "categories_name": [
            "Sách thiếu nhi", "Văn học", "Tài chính", "Kỹ năng sống", "Văn học"
        ],
        "inventory_status": [
            "available", "available", "available", "available", "available"
        ],
        "price": [50000, 80000, 120000, 95000, 110000],
        "review_count": [0, 12, 3, 0, 20],
        "stock_item_qty": [100, 30, 50, 200, 15],
        "stock_item_max_sale_qty": [10, 5, 20, 10, 2]
    }
    df = pd.DataFrame(data)
    df["should_discount"] = ((df["review_count"] < 5)).astype(int)
    return df

# Load data và huấn luyện model ngay trong app
df = load_data()
X = df.drop(columns=["should_discount"])
y = df["should_discount"]

numeric_features = ["price", "review_count", "stock_item_qty", "stock_item_max_sale_qty"]
categorical_features = ["inventory_status", "categories_name"]
text_feature = "short_description"

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("txt", TfidfVectorizer(max_features=50), text_feature)
    ]
)

pipeline = make_pipeline(preprocessor, RandomForestClassifier(class_weight="balanced", random_state=42))
pipeline.fit(X, y)

# Giao diện người dùng
st.title("📚 Tiki - Dự đoán sản phẩm có nên giảm giá không")

description = st.text_area("✏️ Mô tả sản phẩm")
category = st.text_input("📂 Ngành hàng", "Sách thiếu nhi")
inventory_status = st.selectbox("🏷️ Tình trạng kho", ["available", "out_of_stock"])
price = st.number_input("💰 Giá (VND)", min_value=1000)
review_count = st.number_input("⭐ Số lượt đánh giá", min_value=0)
stock_qty = st.number_input("📦 Số lượng tồn kho", min_value=0)
max_sale_qty = st.number_input("📈 Số lượng tối đa mỗi đơn", min_value=1)

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
    prediction = pipeline.predict(input_df)[0]
    if prediction == 1:
        st.success("✅ NÊN giảm giá để kích cầu!")
    else:
        st.info("⚖️ KHÔNG cần giảm giá lúc này.")
