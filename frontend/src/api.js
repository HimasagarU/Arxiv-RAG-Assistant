// Remove trailing slash if present to avoid double slashes when combined with paths starting with '/'
const API_BASE = (import.meta.env.VITE_API_URL || '').replace(/\/$/, '');

/**
 * Get the stored access token.
 */
function getToken() {
  return localStorage.getItem('access_token');
}

/**
 * Core fetch wrapper with auth header injection.
 */
async function apiFetch(path, options = {}) {
  const token = getToken();
  const headers = {
    'Content-Type': 'application/json',
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
    ...(options.headers || {}),
  };

  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers,
  });

  if (res.status === 401) {
    // Try refresh
    const refreshed = await tryRefreshToken();
    if (refreshed) {
      // Retry with new token
      headers.Authorization = `Bearer ${getToken()}`;
      const retry = await fetch(`${API_BASE}${path}`, { ...options, headers });
      if (!retry.ok) {
        const err = await retry.json().catch(() => ({}));
        throw new Error(err.detail || `API error ${retry.status}`);
      }
      return retry;
    }
    // Refresh failed — force logout
    localStorage.removeItem('access_token');
    localStorage.removeItem('refresh_token');
    localStorage.removeItem('user');
    window.location.href = '/login';
    throw new Error('Session expired');
  }

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `API error ${res.status}`);
  }

  return res;
}

async function tryRefreshToken() {
  const refreshToken = localStorage.getItem('refresh_token');
  if (!refreshToken) return false;

  try {
    const res = await fetch(`${API_BASE}/auth/refresh`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ refresh_token: refreshToken }),
    });
    if (!res.ok) return false;
    const data = await res.json();
    localStorage.setItem('access_token', data.access_token);
    localStorage.setItem('refresh_token', data.refresh_token);
    return true;
  } catch {
    return false;
  }
}

// ── Auth ─────────────────────────────────────────────────────

export async function register(email, password, displayName) {
  const res = await apiFetch('/auth/register', {
    method: 'POST',
    body: JSON.stringify({ email, password, display_name: displayName }),
  });
  return res.json();
}

export async function login(email, password) {
  const res = await apiFetch('/auth/login', {
    method: 'POST',
    body: JSON.stringify({ email, password }),
  });
  return res.json();
}

export async function getMe() {
  const res = await apiFetch('/auth/me');
  return res.json();
}

// ── Conversations ────────────────────────────────────────────

export async function createConversation(title = 'New Conversation', paperId = null) {
  const res = await apiFetch('/conversations', {
    method: 'POST',
    body: JSON.stringify({ title, paper_id: paperId }),
  });
  return res.json();
}

export async function listConversations(limit = 50) {
  const res = await apiFetch(`/conversations?limit=${limit}`);
  return res.json();
}

export async function getMessages(conversationId) {
  const res = await apiFetch(`/conversations/${conversationId}/messages`);
  return res.json();
}

export async function sendQuery(conversationId, query, topK = 5) {
  const res = await apiFetch(`/conversations/${conversationId}/query`, {
    method: 'POST',
    body: JSON.stringify({ query, top_k: topK }),
  });
  return res.json();
}

export async function deleteConversation(conversationId) {
  await apiFetch(`/conversations/${conversationId}`, { method: 'DELETE' });
}

// ── Documents ────────────────────────────────────────────────

export async function addDocument(arxivId, pdfUrl = null) {
  const res = await apiFetch('/documents/add', {
    method: 'POST',
    body: JSON.stringify({ arxiv_id: arxivId, pdf_url: pdfUrl }),
  });
  return res.json();
}

export async function getDocumentStatus(jobId) {
  const res = await apiFetch(`/documents/status/${jobId}`);
  return res.json();
}

export async function listDocuments() {
  const res = await apiFetch('/documents');
  return res.json();
}

// ── Health ───────────────────────────────────────────────────

export async function healthCheck() {
  const res = await fetch(`${API_BASE}/health`);
  return res.json();
}

// ── Papers ───────────────────────────────────────────────────

export async function getSimilarPapers(paperId, topN = 5) {
  const res = await apiFetch(`/paper/${paperId}/similar?top_n=${topN}`);
  return res.json();
}

