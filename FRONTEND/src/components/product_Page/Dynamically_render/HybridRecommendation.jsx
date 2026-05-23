import { useState } from "react";

import { getHybridRecommendations }
from "../../../api/recommenderApi";

import RecommendationCard
from "./Dynamically_render/CollaborativeRecommendation";

import styles
from "./Recommendation.module.css";

export default function HybridRecommendation() {

  // =====================================================
  // RECOMMENDATION RESULTS
  // =====================================================
  const [hybridRecommendations,
    setHybridRecommendations] = useState([]);

  // =====================================================
  // FORM STATES
  // =====================================================
  const [userId, setUserId] = useState("");

  const [query, setQuery] = useState("");

  const [topK, setTopK] = useState(10);

  const [category, setCategory] = useState("");

  const [useXGBoost,
    setUseXGBoost] = useState(false);

  // =====================================================
  // LOADING + ERROR STATES
  // =====================================================
  const [loading, setLoading] = useState(false);

  const [error, setError] = useState("");

  // =====================================================
  // FETCH RECOMMENDATIONS
  // =====================================================
  async function handleHybridRecommendation() {

    try {

      setLoading(true);

      setError("");

      // ===============================================
      // REQUEST BODY
      // Matches backend parameters exactly
      // ===============================================
      const requestBody = {

        user_id: Number(userId),

        query: query,

        top_k: Number(topK),

        category: category,

        use_xgboost: useXGBoost
      };

      console.log(
        "Sending Request:",
        requestBody
      );

      // ===============================================
      // API CALL
      // ===============================================
      const response =
        await getHybridRecommendations(
          requestBody
        );

      console.log(response);

      // ===============================================
      // STORE RESULTS
      // ===============================================
      setHybridRecommendations(
        response.data
      );

    }
    catch (err) {

      console.error(err);

      setError(
        "Failed to fetch recommendations"
      );
    }
    finally {

      setLoading(false);
    }
  }

  return (

    <div className={styles.recommendationContainer}>

      {/* TITLE */}
      <h2>
        Hybrid Recommendation System
      </h2>

      {/* =========================================== */}
      {/* INPUT SECTION */}
      {/* =========================================== */}

      <div className={styles.searchContainer}>

        {/* USER ID */}
        <input
          type="number"
          placeholder="Enter User ID"
          value={userId}
          onChange={(e) =>
            setUserId(e.target.value)
          }
        />

        {/* QUERY */}
        <input
          type="text"
          placeholder="Search Product"
          value={query}
          onChange={(e) =>
            setQuery(e.target.value)
          }
        />

        {/* TOP K */}
        <input
          type="number"
          placeholder="Top K"
          value={topK}
          onChange={(e) =>
            setTopK(e.target.value)
          }
        />

        {/* CATEGORY */}
        <input
          type="text"
          placeholder="Category"
          value={category}
          onChange={(e) =>
            setCategory(e.target.value)
          }
        />

        {/* XGBOOST */}
        <label>

          <input
            type="checkbox"
            checked={useXGBoost}
            onChange={(e) =>
              setUseXGBoost(
                e.target.checked
              )
            }
          />

          Use XGBoost

        </label>

        {/* BUTTON */}
        <button
          onClick={
            handleHybridRecommendation
          }
        >
          Get Recommendations
        </button>

      </div>

      {/* =========================================== */}
      {/* LOADING */}
      {/* =========================================== */}

      {loading && (
        <p>
          Loading...
        </p>
      )}

      {/* =========================================== */}
      {/* ERROR */}
      {/* =========================================== */}

      {error && (
        <p>{error}</p>
      )}

      {/* =========================================== */}
      {/* RECOMMENDATIONS */}
      {/* =========================================== */}

      <div className={styles.recommendationGrid}>

        {hybridRecommendations.map(
          (product) => (

            <RecommendationCard
              key={product.product_id}
              product={product}
            />

          )
        )}

      </div>

    </div>
  );
}