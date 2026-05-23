import { useState } from 'react';
import Navbar from '../components/Navbar';
import SearchBar from '../components/SearchBar';
import PerformanceMetrics from '../components/PerformanceMetrics';
import RecommendationCard from '../components/RecommendationCard';
import ProductCard from '../components/ProductCard';
import Loader from '../components/Loader';
import Footer from '../components/Footer';
import useRecommendations from '../hooks/useRecommendations';
import productsData from '../data/embedding_metadata.json';
import styles from './Home.module.css';

const PRODUCTS_PER_PAGE = 20;

export default function Home() {
  const {
    selectedModel,
    setSelectedModel,
    results,
    aiResponse,
    loading,
    error,
    metrics,
    userId,
    fetchRecommendations,
  } = useRecommendations();

  const [currentPage, setCurrentPage] = useState(1);

  // Lazy initialize shuffled products state (runs only once)
  const [shuffledProducts] = useState(() => {
    const arr = [...productsData];
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr;
  });

  const totalPages = Math.ceil(shuffledProducts.length / PRODUCTS_PER_PAGE);
  const start = (currentPage - 1) * PRODUCTS_PER_PAGE;
  const currentProducts = shuffledProducts.slice(start, start + PRODUCTS_PER_PAGE);

  function handlePageChange(page) {
    setCurrentPage(page);
    window.scrollTo({ top: document.getElementById('browse-section')?.offsetTop - 100, behavior: 'smooth' });
  }

  return (
    <div className={styles.page}>
      <Navbar selectedModel={selectedModel} onModelChange={setSelectedModel} />

      {/* Hero Section */}
      <section className={styles.hero}>
        <div className={styles.heroGlow} />
        <h1 className={styles.heroTitle}>
          AI-Powered Furniture
          <span className={styles.heroAccent}> Recommender</span>
        </h1>
        <p className={styles.heroSubtitle}>
          Describe what you're looking for and let our AI models find the perfect match.
          Switch between models to compare their performance.
        </p>
        <div className={styles.userBadge}>
          <span>👤 User ID: {userId}</span>
        </div>
        <SearchBar onSearch={fetchRecommendations} loading={loading} />
      </section>

      <main className={styles.main}>
        {/* Error State */}
        {error && (
          <div className={styles.errorBanner}>
            <span className={styles.errorIcon}>⚠️</span>
            <div>
              <strong>Connection Error</strong>
              <p>{error}</p>
            </div>
          </div>
        )}

        {/* Performance Metrics */}
        {metrics && <PerformanceMetrics metrics={metrics} />}

        {/* Loading State */}
        {loading && <Loader />}

        {/* Conversational AI Response */}
        {aiResponse && !loading && (
          <div className={styles.aiResponse}>
            <div className={styles.aiHeader}>
              <span>💬</span>
              <span>AI Assistant</span>
            </div>
            <p>{aiResponse}</p>
          </div>
        )}

        {/* Recommendation Results */}
        {results.length > 0 && !loading && (
          <section className={styles.resultsSection}>
            <h2 className={styles.sectionTitle}>
              <span className={styles.sectionIcon}>✨</span>
              Recommendations
              <span className={styles.resultCount}>{results.length} results</span>
            </h2>
            <div className={styles.resultsGrid}>
              {results.map((product, index) => (
                <RecommendationCard
                  key={`${product.product_id}-${index}`}
                  product={product}
                  rank={index + 1}
                />
              ))}
            </div>
          </section>
        )}

        {/* Browse All Products */}
        <section className={styles.browseSection} id="browse-section">
          <h2 className={styles.sectionTitle}>
            <span className={styles.sectionIcon}>🛍️</span>
            Browse Products
            <span className={styles.resultCount}>{shuffledProducts.length} items</span>
          </h2>
          <div className={styles.productsGrid}>
            {currentProducts.map((product) => (
              <ProductCard key={product.product_id} product={product} />
            ))}
          </div>

          {/* Pagination */}
          <div className={styles.pagination}>
            <button
              className={styles.pageBtn}
              disabled={currentPage === 1}
              onClick={() => handlePageChange(currentPage - 1)}
              id="prev-page-btn"
            >
              ← Previous
            </button>

            <div className={styles.pageInfo}>
              {/* Show max 5 page buttons around current */}
              {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
                let page;
                if (totalPages <= 5) {
                  page = i + 1;
                } else if (currentPage <= 3) {
                  page = i + 1;
                } else if (currentPage >= totalPages - 2) {
                  page = totalPages - 4 + i;
                } else {
                  page = currentPage - 2 + i;
                }
                return (
                  <button
                    key={page}
                    className={`${styles.pageNum} ${currentPage === page ? styles.pageActive : ''}`}
                    onClick={() => handlePageChange(page)}
                  >
                    {page}
                  </button>
                );
              })}
              <span className={styles.pageTotal}>of {totalPages}</span>
            </div>

            <button
              className={styles.pageBtn}
              disabled={currentPage === totalPages}
              onClick={() => handlePageChange(currentPage + 1)}
              id="next-page-btn"
            >
              Next →
            </button>
          </div>
        </section>
      </main>

      <Footer />
    </div>
  );
}