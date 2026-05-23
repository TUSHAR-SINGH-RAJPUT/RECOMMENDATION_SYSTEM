const BASE_URL = 'http://127.0.0.1:8000';

/**
 * Wraps a fetch call with performance timing.
 * Returns { data, responseTime }
 */
async function timedFetch(url, options) {
  const start = performance.now();

  const response = await fetch(url, options);

  if (!response.ok) {
    throw new Error(`API Error: ${response.status} ${response.statusText}`);
  }

  const data = await response.json();
  const responseTime = Math.round(performance.now() - start);

  return { data, responseTime };
}

function postOptions(body) {
  return {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  };
}

// ============================================================
// EMBEDDING (Content-Based)
// ============================================================
export async function getEmbeddingRecommendations({ query, top_k = 10, category }) {
  const body = { query, top_k };
  if (category) body.category = category;

  return timedFetch(
    `${BASE_URL}/recommend/embedding/`,
    postOptions(body)
  );
}

// ============================================================
// COLLABORATIVE FILTERING
// ============================================================
export async function getCollaborativeRecommendations({ user_id, query, top_k = 10 }) {
  return timedFetch(
    `${BASE_URL}/recommend/collaborative/`,
    postOptions({ user_id, query, top_k })
  );
}

// ============================================================
// HYBRID (CF + Embedding)
// ============================================================
export async function getHybridRecommendations({ user_id, query, top_k = 10, category, use_xgboost = true }) {
  const body = { user_id, query, top_k, use_xgboost };
  if (category) body.category = category;

  return timedFetch(
    `${BASE_URL}/recommend/hybrid/`,
    postOptions(body)
  );
}

// ============================================================
// CONVERSATIONAL AI
// ============================================================
export async function getConversationalRecommendations({ user_id, query }) {
  return timedFetch(
    `${BASE_URL}/recommend/conversational/`,
    postOptions({ user_id, query })
  );
}