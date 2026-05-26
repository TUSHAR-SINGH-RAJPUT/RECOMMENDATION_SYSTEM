import { useState, useCallback } from 'react';
import {
  getEmbeddingRecommendations,
  getCollaborativeRecommendations,
  getHybridRecommendations,
} from '../api/recommenderApi';

export default function useRecommendations() {
  const [selectedModel, setSelectedModel] = useState('embedding');
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [userId] = useState(() => Math.floor(Math.random() * 500) + 1);

  const fetchRecommendations = useCallback(async (query, options = {}) => {
    if (!query?.trim()) return;

    setLoading(true);
    setError(null);

    try {
      let response;
      const model = options.model || selectedModel;

      if (model === 'embedding') {
        response = await getEmbeddingRecommendations({ query });
      } else if (model === 'collaborative') {
        response = await getCollaborativeRecommendations({ user_id: userId, query });
      } else if (model === 'hybrid') {
        response = await getHybridRecommendations({ user_id: userId, query, top_k: options.top_k || 10 });
      } else {
        throw new Error(`Unknown model: ${model}`);
      }

      let products = response.data.recommendations || [];
      if (options.excludeProductId) {
        products = products.filter((product) => product.product_id !== options.excludeProductId);
      }
      setResults(products);
      setMetrics({
        model,
        responseTime: response.responseTime,
        resultCount: products.length,
        timestamp: new Date().toLocaleTimeString(),
        sourceProduct: options.sourceProduct || null,
      });
    } catch (err) {
      console.error('Recommendation fetch error:', err);
      setError(err.message || 'Failed to fetch recommendations. Is the backend running?');
      setResults([]);
      setMetrics(null);
    } finally {
      setLoading(false);
    }
  }, [selectedModel, userId]);

  return {
    selectedModel,
    setSelectedModel,
    results,
    loading,
    error,
    metrics,
    userId,
    fetchRecommendations,
  };
}
