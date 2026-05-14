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

/**
 * Synchronous query (kept as fallback path for streamConversationQuery).
 */
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

export async function cancelDocument(jobId) {
  const res = await apiFetch(`/documents/cancel/${jobId}`, {
    method: 'POST',
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

// ── SSE Streaming Helpers ────────────────────────────────────

/**
 * Parse a raw SSE stream from a ReadableStream reader.
 * Calls the appropriate callback for each parsed SSE data line.
 *
 * @param {ReadableStreamDefaultReader} reader
 * @param {(type: string, payload: object) => void} onEvent
 */
async function _consumeSSEStream(reader, onEvent) {
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      if (buffer.trim()) {
        const raw = buffer.replace(/^data:\s*/, '').trim();
        try {
          const payload = JSON.parse(raw);
          onEvent(payload.type || 'done', payload);
        } catch {}
      }
      break;
    }

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop(); // keep incomplete last line

    let currentEvent = null;
    for (const line of lines) {
      if (line.startsWith('event: ')) {
        currentEvent = line.slice(7).trim();
      } else if (line.startsWith('data: ')) {
        const raw = line.slice(6).trim();
        try {
          const payload = JSON.parse(raw);
          // Named event (chat stream) or typed payload (public stream)
          const eventType = currentEvent || payload.type;
          onEvent(eventType, payload);
        } catch {
          // ignore malformed lines
        }
        currentEvent = null;
      }
    }
  }
}

/**
 * Stream a public (unauthenticated) query via POST /query/stream.
 *
 * SSE event types from server: metadata | token | error | done
 *
 * Falls back to synchronous sendPublicQuery() if the stream fails before
 * the first token arrives, calling onToken once with the full answer.
 *
 * @param {string} query
 * @param {object} opts - { topK, startYear, author, onToken, onMetadata, onError, onDone }
 */
export async function streamPublicQuery(query, {
  topK = 5,
  startYear = null,
  author = null,
  onToken = () => {},
  onMetadata = () => {},
  onStatus = () => {},
  onError = () => {},
  onDone = () => {},
} = {}) {
  let receivedFirstToken = false;

  try {
    const res = await fetch(`${API_BASE}/query/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query,
        top_k: topK,
        start_year: startYear || undefined,
        author: author || undefined,
      }),
    });

    if (!res.ok || !res.body) {
      throw new Error(`Stream request failed: ${res.status}`);
    }

    const reader = res.body.getReader();
    let sources = [];
    let errorOccurred = false;

    await _consumeSSEStream(reader, (type, payload) => {
      if (type === 'metadata') {
        sources = payload.sources || [];
        onMetadata(payload);
      } else if (type === 'token') {
        receivedFirstToken = true;
        onToken(payload.content || '');
      } else if (type === 'status') {
        onStatus(payload.stage || '');
      } else if (type === 'error') {
        errorOccurred = true;
        onError(payload.message || 'Stream error');
      } else if (type === 'done') {
        onDone(sources);
      }
    });

    if (errorOccurred && !receivedFirstToken) {
      throw new Error('Stream error before first token');
    }
  } catch (err) {
    if (receivedFirstToken) {
      // Already streaming — surface the error rather than fallback
      onError(err.message);
      return;
    }

    // Fallback: synchronous query
    console.warn('streamPublicQuery: falling back to synchronous /query', err.message);
    try {
      const data = await sendPublicQuery(query, { topK, startYear, author });
      onMetadata({ sources: data.sources || [], intent: data.retrieval_trace?.intent });
      onToken(data.answer || '');
      onDone(data.sources || []);
    } catch (fallbackErr) {
      onError(fallbackErr.message);
    }
  }
}

/**
 * Synchronous public query (used as fallback by streamPublicQuery).
 */
export async function sendPublicQuery(query, { topK = 5, startYear = null, author = null } = {}) {
  const res = await fetch(`${API_BASE}/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query,
      top_k: topK,
      start_year: startYear || undefined,
      author: author || undefined,
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `API error ${res.status}`);
  }
  return res.json();
}

/**
 * Stream a conversation query via POST /conversations/{id}/query/stream.
 *
 * Named SSE events from server: retrieval_start | retrieval_done | token | error | done
 *
 * Falls back to synchronous sendQuery() if the stream fails before the first
 * token arrives.
 *
 * @param {string} conversationId
 * @param {string} query
 * @param {object} opts - { topK, onToken, onRetrievalStart, onRetrievalDone, onError, onDone }
 */
export async function streamConversationQuery(conversationId, query, {
  topK = 5,
  onToken = () => {},
  onRetrievalStart = () => {},
  onRetrievalDone = () => {},
  onStatus = () => {},
  onError = () => {},
  onDone = () => {},
} = {}) {
  let receivedFirstToken = false;

  const doFetch = async (token) => {
    return fetch(`${API_BASE}/conversations/${conversationId}/query/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({ query, top_k: topK }),
    });
  };

  try {
    let token = getToken();
    let res = await doFetch(token);

    // 401 → try token refresh once
    if (res.status === 401) {
      const refreshed = await tryRefreshToken();
      if (!refreshed) {
        localStorage.removeItem('access_token');
        localStorage.removeItem('refresh_token');
        localStorage.removeItem('user');
        window.location.href = '/login';
        throw new Error('Session expired');
      }
      token = getToken();
      res = await doFetch(token);
    }

    if (!res.ok || !res.body) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Stream request failed: ${res.status}`);
    }

    const reader = res.body.getReader();
    let sources = [];
    let errorOccurred = false;

    await _consumeSSEStream(reader, (type, payload) => {
      if (type === 'retrieval_start') {
        onRetrievalStart(payload);
      } else if (type === 'retrieval_done') {
        onRetrievalDone(payload);
      } else if (type === 'token') {
        receivedFirstToken = true;
        onToken(payload.content || '');
      } else if (type === 'status') {
        onStatus(payload.stage || '');
      } else if (type === 'error') {
        errorOccurred = true;
        onError(payload.message || 'Stream error');
      } else if (type === 'done') {
        sources = payload.sources || [];
        onDone(payload);
      }
    });

    if (errorOccurred && !receivedFirstToken) {
      throw new Error('Stream error before first token');
    }
  } catch (err) {
    if (receivedFirstToken) {
      onError(err.message);
      return;
    }

    // Fallback: synchronous query
    console.warn('streamConversationQuery: falling back to synchronous /query', err.message);
    try {
      const data = await sendQuery(conversationId, query, topK);
      onRetrievalStart({ query, cached: data.cached });
      onRetrievalDone({ num_chunks: data.sources?.length || 0, trace: data.retrieval_trace || {} });
      onToken(data.answer || '');
      onDone({ sources: data.sources || [], message_count: data.message_count, cached: data.cached });
    } catch (fallbackErr) {
      onError(fallbackErr.message);
    }
  }
}
