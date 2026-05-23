import { useState } from "react";
import { getEmbeddingRecommendations } from "../../api/recommenderApi";

import styles from "./Recommendation.module.css";

function CollaborativeRecommendation() {

  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);

  // ======================================================
  // FETCH RECOMMENDATIONS
  // ======================================================
  async function fetchRecommendations(searchQuery) {

    if (!searchQuery.trim()) return;

    try {

      setLoading(true);

      const data = {
        query: searchQuery,
        top_k: 10,
      };

      const response = await getEmbeddingRecommendations(data);

      setResults(response.recommendations || []);

    } catch (error) {

      console.error("Recommendation Error:", error);

    } finally {

      setLoading(false);
    }
  }

  // ======================================================
  // HANDLE SEARCH BUTTON
  // ======================================================
  async function handleSearch() {

    fetchRecommendations(query);
  }

  return (

    <div className={styles.container}>

      {/* ================================================== */}
      {/* HEADER */}
      {/* ================================================== */}
      <div className={styles.header}>

        <h1 className={styles.title}>
          AI Furniture Recommender
        </h1>

        <p className={styles.subtitle}>
          Discover furniture using AI-powered recommendations
        </p>

      </div>


      {/* ================================================== */}
      {/* SEARCH BAR */}
      {/* ================================================== */}
      <div className={styles.searchContainer}>

        <input
          type="text"
          placeholder="Search furniture..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className={styles.searchInput}
        />

        <button
          onClick={handleSearch}
          className={styles.searchButton}
        >
          Search
        </button>

      </div>


      {/* ================================================== */}
      {/* HELP TEXT */}
      {/* ================================================== */}
      <div className={styles.helpText}>

        <p>
          Try searching with detailed descriptions for better recommendations
        </p>

        <div className={styles.examples}>

          <span
            onClick={() => {
              const value = "modern wooden sofa";
              setQuery(value);
              fetchRecommendations(value);
            }}
          >
            modern wooden sofa
          </span>

          <span
            onClick={() => {
              const value = "minimalist office chair";
              setQuery(value);
              fetchRecommendations(value);
            }}
          >
            minimalist office chair
          </span>

          <span
            onClick={() => {
              const value = "luxury king size bed";
              setQuery(value);
              fetchRecommendations(value);
            }}
          >
            luxury king size bed
          </span>

          <span
            onClick={() => {
              const value = "gaming desk with storage";
              setQuery(value);
              fetchRecommendations(value);
            }}
          >
            gaming desk with storage
          </span>

        </div>

      </div>


      {/* ================================================== */}
      {/* LOADING */}
      {/* ================================================== */}
      {loading && (

        <div className={styles.loading}>
          Loading recommendations...
        </div>

      )}


      {/* ================================================== */}
      {/* RESULTS */}
      {/* ================================================== */}
      <div className={styles.resultsGrid}>

        {results.map((item, index) => (

          <div
            key={index}
            className={styles.card}
          >

            <h2 className={styles.productName}>
              {item.product_name}
            </h2>

            <p className={styles.category}>
              {item.category}
            </p>

            <p className={styles.description}>
              {item.description || "No description available"}
            </p>

            {item.score && (

              <div className={styles.score}>
                Similarity Score: {item.score}
              </div>

            )}

          </div>

        ))}

      </div>

    </div>
  );
}

export default CollaborativeRecommendation;