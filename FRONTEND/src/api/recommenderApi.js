const BASE_URL = "http://127.0.0.1:8000";


// ======================================================
// EMBEDDING API
// ======================================================
export async function getEmbeddingRecommendations(data) {

    const response = await fetch(
        `${BASE_URL}/recommend/embedding/`,
        {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(data),
        }
    );

    return response.json();
}


// ======================================================
// COLLABORATIVE API
// ======================================================
export async function getCollaborativeRecommendations(data) {

    const response = await fetch(
        `${BASE_URL}/recommend/collaborative/`,
        {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(data),
        }
    );

    return response.json();
}


// ======================================================
// HYBRID API
// ======================================================
export async function getHybridRecommendations(data) {

    const response = await fetch(
        `${BASE_URL}/recommend/hybrid/`,
        {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(data),
        }
    );

    return response.json();
}