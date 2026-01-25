# -----------------------------
# Shopper Spectrum Streamlit App
# -----------------------------
# Importing the required libraries
import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


# Creating the page configuration with the url title "Shopper Spectrum"
st.set_page_config(
    page_title="Shopper Spectrum",
    layout="wide"
)
# Creating the title for the page "Shopper Spectrum: Recommendation & Segmentation App"
st.title("🛒 Shopper Spectrum: Recommendation & Segmentation App")
st.markdown(
    """
    This application provides:
    - **Product recommendations** using item-based collaborative filtering  
    - **Customer segmentation** using RFM analysis and K-Means clustering
    """
)

# --------------------------------------------------
# Build Models (Cached)
# --------------------------------------------------
@st.cache_resource
def build_models():
    # Load dataset (make sure this CSV is <100MB or hosted externally)
    df = pd.read_csv("data/online_retail.csv")

    # Convert date column
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # Create total price
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    # -----------------------------
    # RFM CALCULATION
    # -----------------------------
    reference_date = df["InvoiceDate"].max()

    rfm = (
        df.groupby("CustomerID")
        .agg({
            "InvoiceDate": lambda x: (reference_date - x.max()).days,
            "InvoiceNo": "nunique",
            "TotalPrice": "sum"
        })
        .reset_index()
    )

    rfm.columns = ["CustomerID", "Recency", "Frequency", "Monetary"]

    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(
        rfm[["Recency", "Frequency", "Monetary"]]
    )

    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans.fit(rfm_scaled)

    # -----------------------------
    # PRODUCT SIMILARITY MATRIX
    # -----------------------------
    product_customer_matrix = pd.pivot_table(
        df,
        index="CustomerID",
        columns="Description",
        values="Quantity",
        aggfunc="sum",
        fill_value=0
    )

    similarity = cosine_similarity(product_customer_matrix.T)

    product_similarity_df = pd.DataFrame(
        similarity,
        index=product_customer_matrix.columns,
        columns=product_customer_matrix.columns
    )

    return kmeans, scaler, product_similarity_df


# Build models once
with st.spinner("Initializing models... Please wait ⏳"):
    kmeans_model, rfm_scaler, product_similarity_df = build_models()

st.success("Models loaded successfully ✅")


# Create a header for the product recommendation
st.header("🎯 Product Recommendation")
# User can type the desired product name which will be stored in the variable product_name
product_name = st.text_input("Enter Product Name")


# Creating the function to get the similar products with two arguments
# product_name and top_n=5 (top 5 similar products)
def get_similar_products(product_name, top_n=5):
    # if product_name is not in the product_similarity_df it will return empty list
    if product_name not in product_similarity_df.index:
        return []
    # Calculating the similarity score between the product_name with every other product
    similarity_scores = product_similarity_df[product_name]
    # Sorting the similar products in descending order (top-bottom) for top 5 products
    similar_products = similarity_scores.sort_values(
        ascending=False
    )[1:top_n + 1]
    # Sorting the similar products in descending order (top-bottom) for top 5 products
    return similar_products.index.tolist()


# This block will be executed if the user clicks on the button "Get Recommendations"
if st.button("Get Recommendations"):
    # Storing the similar products for the product_name in the variable (recommendations)
    recommendations = get_similar_products(product_name)
    # This block will be executed if there are recommendations and returns the top 5 products
    if recommendations:
        st.subheader("Top 5 Similar Products")
        for product in recommendations:
            st.markdown(f"✅ {product}")
    # This block will be executed if there are no recommendations
    else:
        st.warning("Product not found. Please check the name.")


# This module to choose the clusters for customers
# Create a header for the Customer Segmentationst.header("Customer Segmentation")
st.header("🎯 Customer Segmentation")

# Creating three variables (recency, frequency and monetary)
recency = st.number_input("Recency (days)", min_value=0)
frequency = st.number_input("Frequency", min_value=0)
monetary = st.number_input("Monetary Value", min_value=0.0)


# Function to select the cluster for the customers
def get_segment_label(cluster):
    return {
        0: "Regular",
        1: "Occasional",
        2: "High-Value",
        3: "At-Risk"
    }.get(cluster, "Unknown")


# This block will be execited of the user clicks on the button "Predict Cluster"
if st.button("Predict Cluster"):
    # "rfm_input" will store the values of recency, frequency, monetary
    rfm_input = pd.DataFrame(
        [[recency, frequency, monetary]],
        columns=["Recency", "Frequency", "Monetary"]
    )

    # Scaling the rfm_input using the loaded rfm_scaler
    rfm_scaled = rfm_scaler.transform(rfm_input)
    # Predicting the clusters using the kmeans model we loaded
    cluster = kmeans_model.predict(rfm_scaled)[0]
    # Selecting the segment
    segment = get_segment_label(cluster)
    # Return the statement with the customer segment
    st.success(f"Customer Segment: **{segment}**")