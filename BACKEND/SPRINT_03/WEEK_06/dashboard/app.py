import streamlit as st
import requests

API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="GenAI Recommender",
    layout="wide"
)

st.title("🛋️ GenAI Hyper Personalized Recommender")

# =========================
# SIDEBAR
# =========================

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

# =========================
# HELPER FUNCTION
# =========================

def display_products(products):

    for product in products:

        with st.container():

            st.markdown(f"### {product['product_name']}")

            st.write(f"Category: {product['category']}")

            if "score" in product:
                st.write(f"Score: {product['score']}")

            st.divider()


# =========================
# BUTTON
# =========================

if st.button("Get Recommendations"):

    try:

        # -------------------------
        # Collaborative
        # -------------------------

        if model_type == "Collaborative":

            endpoint = "/recommend/collaborative/"

            payload = {
                "user_id": user_id,
                "query": query,
                "top_k": 5
            }

        # -------------------------
        # Embedding
        # -------------------------

        elif model_type == "Embedding":

            endpoint = "/recommend/embedding/"

            payload = {
                "user_id": user_id,
                "query": query,
                "top_k": 5
            }

        # -------------------------
        # Hybrid
        # -------------------------

        elif model_type == "Hybrid":

            endpoint = "/recommend/hybrid/"

            payload = {
                "user_id": user_id,
                "query": query,
                "top_k": 5
            }

        # -------------------------
        # Conversational
        # -------------------------

        else:

            endpoint = "/recommend/conversational/"

            payload = {
                "user_id": user_id,
                "query": query
            }

        # =========================
        # API CALL
        # =========================

        response = requests.post(
            API_BASE_URL + endpoint,
            json=payload
        )

        # Handle bad response
        if response.status_code != 200:

            st.error(f"API Error: {response.status_code}")

            st.text(response.text)

        else:

            data = response.json()

            # Conversational
            if model_type == "Conversational":

                st.subheader("AI Response")

                st.success(data["response"])

                st.subheader("Recommended Products")

                display_products(data["products"])

            # Other models
            else:

                st.subheader("Recommendations")

                display_products(data["recommendations"])

    except requests.exceptions.ConnectionError:

        st.error("FastAPI server is not running.")

        st.info(
            "Start FastAPI using:\n\n"
            "uvicorn SPRINT_03.WEEK_06.api.main:app --reload"
        )

    except Exception as e:

        st.error(f"Unexpected Error: {str(e)}")