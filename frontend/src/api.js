const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

async function handleResponse(response) {
  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    const message = data?.detail || "Request failed";
    throw new Error(message);
  }

  return data;
}

export async function getIngestStatus() {
  const response = await fetch(`${API_BASE_URL}/ingest/status`);
  return handleResponse(response);
}

export async function runIngest(options = {}) {
  const params = new URLSearchParams();
  if (options.rebuild) {
    params.set("rebuild", "true");
  }
  if (options.refreshExisting) {
    params.set("refresh_existing", "true");
  }

  const query = params.toString();
  const url = query
    ? `${API_BASE_URL}/ingest?${query}`
    : `${API_BASE_URL}/ingest`;

  const response = await fetch(url, {
    method: "POST",
  });
  return handleResponse(response);
}

export async function getIngestProgress() {
  const response = await fetch(`${API_BASE_URL}/ingest/progress`);
  return handleResponse(response);
}

export async function queryDocs(payload) {
  const response = await fetch(`${API_BASE_URL}/query`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
  return handleResponse(response);
}

export async function runEvaluation() {
  const response = await fetch(`${API_BASE_URL}/eval/run`, {
    method: "POST",
  });
  return handleResponse(response);
}

export async function getEvalQuestions() {
  const response = await fetch(`${API_BASE_URL}/eval/questions`);
  return handleResponse(response);
}
