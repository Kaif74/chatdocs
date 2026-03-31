import { useEffect, useState } from "react";

import { getIngestStatus, runIngest } from "../api";

function IngestPanel() {
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function refreshStatus() {
    try {
      const data = await getIngestStatus();
      setStatus(data);
    } catch (requestError) {
      setError(requestError.message);
    }
  }

  useEffect(() => {
    refreshStatus();
  }, []);

  async function handleIngest() {
    setLoading(true);
    setError("");

    try {
      const data = await runIngest();
      setStatus({
        ready: data.ready,
        total_chunks: data.total_chunks,
        counts_by_source: data.counts_by_source,
      });
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="panel">
      <div className="panel__header">
        <h2>Ingest framework docs</h2>
        <p>Build or rebuild the local Chroma collection from the four LLM-friendly docs sources.</p>
      </div>

      <button type="button" className="button" onClick={handleIngest} disabled={loading}>
        {loading ? "Re-ingesting all sources..." : "Re-ingest All"}
      </button>

      {error ? <div className="notice notice--error">{error}</div> : null}

      {status ? (
        <div className="status-card">
          <div className="status-card__row">
            <span>Vector store ready</span>
            <strong>{status.ready ? "Yes" : "No"}</strong>
          </div>
          <div className="status-card__row">
            <span>Total chunks</span>
            <strong>{status.total_chunks}</strong>
          </div>

          <div className="counts-grid">
            {Object.entries(status.counts_by_source || {}).map(([source, count]) => (
              <div key={source} className="summary-card">
                <span className="summary-card__label">{source}</span>
                <strong>{count}</strong>
              </div>
            ))}
          </div>
        </div>
      ) : null}
    </section>
  );
}

export default IngestPanel;
