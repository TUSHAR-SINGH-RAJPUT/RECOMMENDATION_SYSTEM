import streamlit as st
import requests

API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="GenAI Recommender",
    layout="wide"
)

st.title("🛋️ GenAI Hyper Personalized Recommender")

# Sidebar
st.sidebar.header("Recommendation Settings")

model_type = st.sidebar.selectbox(
    "Choose Recommendation Type",
    [
        "Collaborative",
        "Embedding",
        "Hybrid",
        "Conversational"
    ]
)

user_id = st.sidebar.number_input(
    "User ID",
    min_value=1,
    value=1
)

query = st.text_input(
    "Enter Your Query",
    placeholder="Example: modern minimalist sofa"
)

# Button
if st.button("Get Recommendations"):

    # Collaborative
    if model_type == "Collaborative":

        endpoint = "/recommend/collaborative/"

        payload = {
            "user_id": user_id,
            "query": query,
            "top_k": 5
        }

        response = requests.post(
            API_BASE_URL + endpoint,
            json=payload
        )

        data = response.json()

        st.subheader("Recommendations")

        for product in data["recommendations"]:

            with st.container():
                st.markdown(f"### {product['product_name']}")
                st.write(f"Category: {product['category']}")
                st.write(f"Score: {product['score']}")
                st.divider()

    # Embedding
    elif model_type == "Embedding":

        endpoint = "/recommend/embedding/"

        payload = {
            "user_id": user_id,
            "query": query,
            "top_k": 5
        }

        response = requests.post(
            API_BASE_URL + endpoint,
            json=payload
        )

        data = response.json()

        st.subheader("Recommendations")

        for product in data["recommendations"]:

            with st.container():
                st.markdown(f"### {product['product_name']}")
                st.write(f"Category: {product['category']}")
                st.write(f"Score: {product['score']}")
                st.divider()

    # Hybrid
    elif model_type == "Hybrid":

        endpoint = "/recommend/hybrid/"

        payload = {
            "user_id": user_id,
            "query": query,
            "top_k": 5
        }

        response = requests.post(
            API_BASE_URL + endpoint,
            json=payload
        )

        data = response.json()

        st.subheader("Recommendations")

        for product in data["recommendations"]:

            with st.container():
                st.markdown(f"### {product['product_name']}")
                st.write(f"Category: {product['category']}")
                st.write(f"Score: {product['score']}")
                st.divider()

    # Conversational
    elif model_type == "Conversational":

        endpoint = "/recommend/conversational/"

        payload = {
            "user_id": user_id,
            "query": query
        }

        response = requests.post(
            API_BASE_URL + endpoint,
            json=payload
        )

        data = response.json()

        st.subheader("AI Response")

        st.success(data["response"])

        st.subheader("Recommended Products")

        for product in data["products"]:

            with st.container():
                st.markdown(f"### {product['product_name']}")
                st.write(f"Category: {product['category']}")
                st.write(f"Score: {product['score']}")
                st.divider()