# 🛒 Shopper Spectrum

Shopper Spectrum is an end-to-end **E-Commerce Analytics and Recommendation System** built using **Machine Learning and Streamlit**.  
The application analyzes customer transaction data to perform **customer segmentation using RFM analysis and K-Means clustering**, and provides **personalized product recommendations** using **item-based collaborative filtering**.

The project demonstrates how raw e-commerce transaction data can be transformed into **actionable business insights** and deployed as an **interactive web application**.

---


## 🧠 Project Highlights
- Customer segmentation into **High-Value, Regular, Occasional, and At-Risk** customers  
- Personalized **product recommendations** based on purchase behavior  
- Interactive **Streamlit web application** with real-time predictions  
- Industry-aligned **MLOps practices** (large data and models excluded from version control)

---

## 🛠 Tech Stack
- **Python**
- **Pandas, NumPy**
- **Scikit-learn**
- **K-Means Clustering**
- **RFM Analysis**
- **Collaborative Filtering (Cosine Similarity)**
- **Streamlit**
- **Matplotlib & Seaborn**

---

## ⚙️ How It Works

### 🎯 Customer Segmentation
1. Customer transaction data is aggregated using **RFM (Recency, Frequency, Monetary)** analysis.
2. RFM values are scaled using **StandardScaler**.
3. **K-Means clustering** groups customers into meaningful segments.
4. Users can input Recency, Frequency, and Monetary values to **predict customer segment in real time**.

---

### 🛍 Product Recommendation
1. Transaction data is converted into a **Product–Customer Matrix**.
2. **Item-based collaborative filtering** is applied.
3. **Cosine similarity** measures similarity between products.
4. When a user enters a product name, the app recommends **Top 5 similar products**.

---

## 📊 Streamlit App Features

### 1️⃣ Product Recommendation Module
- Input a product name
- Get **Top 5 similar product recommendations**

### 2️⃣ Customer Segmentation Module
- Input:
  - Recency (days)
  - Frequency (number of purchases)
  - Monetary value (total spend)
- Get predicted **customer segment label**

---

## 📁 How to Use This Project (Execution Order)

To fully understand and reproduce the project workflow, follow this order:

1️⃣ **Run `data_evaluation.ipynb`**  
- Data exploration  
- Data cleaning  
- Feature engineering  
- RFM calculation and scaling  

2️⃣ **Run `customer_segmentation.ipynb`**  
- K-Means clustering  
- Elbow Curve & Silhouette Score  
- Cluster interpretation and labeling  

3️⃣ **Run `product_recommendation.ipynb`**  
- Product–customer matrix creation  
- Cosine similarity computation  
- Recommendation logic and visualization  

4️⃣ **Run `app.py`**  
- Launch the Streamlit application  
- Interact with customer segmentation and product recommendation modules  

---

## 📌 Important Note
Large datasets and trained model artifacts are **excluded from version control** to follow best MLOps practices.  
All models and similarity matrices are **generated dynamically at runtime** using Streamlit caching.

---

## 📈 Business Use Cases
- Targeted marketing campaigns  
- Personalized product recommendations  
- Customer retention and churn prevention  
- Inventory planning and demand forecasting  

---

## 👨‍💻 Author
**Akash Jalapati**  
Internship Project – E-Commerce & Retail Analytics
