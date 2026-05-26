import { useEffect, useRef, useState } from 'react';
import { gsap } from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import Navbar from '../components/Navbar';
import SearchBar from '../components/SearchBar';
import PerformanceMetrics from '../components/PerformanceMetrics';
import RecommendationCard from '../components/RecommendationCard';
import ProductCard from '../components/ProductCard';
import Loader from '../components/Loader';
import Footer from '../components/Footer';
import Chatbot from '../components/Chatbot';
import useRecommendations from '../hooks/useRecommendations';
import productsData from '../data/embedding_metadata.json';
import styles from './Home.module.css';

const PRODUCTS_PER_PAGE = 20;
gsap.registerPlugin(ScrollTrigger);

export default function Home() {
  const rootRef = useRef(null);
  const {
    selectedModel,
    setSelectedModel,
    results,
    loading,
    error,
    metrics,
    userId,
    fetchRecommendations,
  } = useRecommendations();

  const [currentPage, setCurrentPage] = useState(1);
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

  useEffect(() => {
    const context = gsap.context(() => {
      gsap.from('[data-hero-reveal]', {
        y: 28,
        opacity: 0,
        duration: 0.9,
        stagger: 0.12,
        ease: 'power3.out',
      });

      gsap.utils.toArray('[data-reveal]').forEach((element) => {
        gsap.from(element, {
          y: 36,
          opacity: 0,
          duration: 0.75,
          ease: 'power3.out',
          scrollTrigger: {
            trigger: element,
            start: 'top 86%',
          },
        });
      });

      gsap.utils.toArray('[data-card]').forEach((element) => {
        gsap.from(element, {
          y: 30,
          opacity: 0,
          duration: 0.65,
          ease: 'power2.out',
          scrollTrigger: {
            trigger: element,
            start: 'top 92%',
          },
        });
      });
    }, rootRef);

    return () => context.revert();
  }, [results, currentPage]);

  function handlePageChange(page) {
    setCurrentPage(page);
    window.scrollTo({
      top: document.getElementById('browse-section')?.offsetTop - 100,
      behavior: 'smooth',
    });
  }

  function handleProductSelect(product) {
    const query = [
      product.product_name,
      product.category,
      product.description,
    ].filter(Boolean).join(' ');

    setSelectedModel('hybrid');
    fetchRecommendations(query, {
      model: 'hybrid',
      top_k: 12,
      excludeProductId: product.product_id,
      sourceProduct: product.product_name,
    });

    window.scrollTo({
      top: document.querySelector('main')?.offsetTop - 80,
      behavior: 'smooth',
    });
  }

  return (
    <div className={styles.page} ref={rootRef}>
      <Navbar selectedModel={selectedModel} onModelChange={setSelectedModel} />

      <section className={styles.hero}>
        <div className={styles.heroGlow} />
        <div className={styles.heroKicker} data-hero-reveal>Curated interiors, guided by intelligence</div>
        <h1 className={styles.heroTitle} data-hero-reveal>
          RoomSense Furniture
          <span className={styles.heroAccent}> Recommender</span>
        </h1>
        <p className={styles.heroSubtitle} data-hero-reveal>
          Describe what you are looking for and compare semantic, hybrid, and
          collaborative recommendation models in one interface.
        </p>
        <div className={styles.userBadge} data-hero-reveal>
          <span>User ID: {userId}</span>
        </div>
        <div data-hero-reveal>
          <SearchBar onSearch={fetchRecommendations} loading={loading} />
        </div>
      </section>

      <main className={styles.main}>
        {error && (
          <div className={styles.errorBanner}>
            <span className={styles.errorIcon}>!</span>
            <div>
              <strong>Connection Error</strong>
              <p>{error}</p>
            </div>
          </div>
        )}

        {metrics && (
          <div data-reveal>
            <PerformanceMetrics metrics={metrics} />
          </div>
        )}
        {loading && <Loader />}

        {results.length > 0 && !loading && (
          <section className={styles.resultsSection} data-reveal>
            <h2 className={styles.sectionTitle}>
              <span className={styles.sectionIcon}>*</span>
              {metrics?.sourceProduct ? `Similar to ${metrics.sourceProduct}` : 'Recommendations'}
              <span className={styles.resultCount}>{results.length} results</span>
            </h2>
            <div className={styles.resultsGrid}>
              {results.map((product, index) => (
                <div key={`${product.product_id}-${index}`} data-card>
                  <RecommendationCard
                    product={product}
                    rank={index + 1}
                  />
                </div>
              ))}
            </div>
          </section>
        )}

        <section className={styles.browseSection} id="browse-section" data-reveal>
          <h2 className={styles.sectionTitle}>
            <span className={styles.sectionIcon}>#</span>
            Browse Products
            <span className={styles.resultCount}>{shuffledProducts.length} items</span>
          </h2>
          <div className={styles.productsGrid}>
            {currentProducts.map((product) => (
              <div key={product.product_id} data-card>
                <ProductCard product={product} onSelect={handleProductSelect} />
              </div>
            ))}
          </div>

          <div className={styles.pagination}>
            <button
              className={styles.pageBtn}
              disabled={currentPage === 1}
              onClick={() => handlePageChange(currentPage - 1)}
              id="prev-page-btn"
              type="button"
            >
              Previous
            </button>

            <div className={styles.pageInfo}>
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
                    type="button"
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
              type="button"
            >
              Next
            </button>
          </div>
        </section>
      </main>

      <Footer />
      <Chatbot userId={userId} />
    </div>
  );
}
