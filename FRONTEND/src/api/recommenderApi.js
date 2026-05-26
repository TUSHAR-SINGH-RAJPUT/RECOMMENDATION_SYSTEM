const BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

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

export async function getEmbeddingRecommendations({ query, top_k = 10, category }) {
  const body = { query, top_k };
  if (category) body.category = category;
  return timedFetch(`${BASE_URL}/recommend/embedding/`, postOptions(body));
}

export async function getCollaborativeRecommendations({ user_id, query, top_k = 10 }) {
  return timedFetch(
    `${BASE_URL}/recommend/collaborative/`,
    postOptions({ user_id, query, top_k })
  );
}

export async function getHybridRecommendations({
  user_id,
  query,
  top_k = 10,
  category,
  use_xgboost = true,
}) {
  const body = { user_id, query, top_k, use_xgboost };
  if (category) body.category = category;
  return timedFetch(`${BASE_URL}/recommend/hybrid/`, postOptions(body));
}

function parseSseEvents(buffer) {
  const events = [];
  const blocks = buffer.split('\n\n');
  const remainder = blocks.pop() || '';

  for (const block of blocks) {
    let event = 'message';
    let data = '';

    for (const line of block.split('\n')) {
      if (line.startsWith('event:')) event = line.slice(6).trim();
      if (line.startsWith('data:')) data += line.slice(5).trim();
    }

    if (data) events.push({ event, data: JSON.parse(data) });
  }

  return { events, remainder };
}

export async function streamConversationalRecommendations({
  user_id,
  query,
  session_id,
  top_k = 5,
  onProducts,
  onToken,
  onDone,
  onError,
}) {
  const response = await fetch(
    `${BASE_URL}/recommend/conversational/`,
    postOptions({ user_id, query, session_id, top_k })
  );

  if (!response.ok || !response.body) {
    throw new Error(`Chat API Error: ${response.status} ${response.statusText}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const parsed = parseSseEvents(buffer);
    buffer = parsed.remainder;

    for (const item of parsed.events) {
      if (item.event === 'products') onProducts?.(item.data);
      if (item.event === 'token') onToken?.(item.data.token || '');
      if (item.event === 'done') onDone?.(item.data);
      if (item.event === 'error') onError?.(item.data);
    }
  }
}
