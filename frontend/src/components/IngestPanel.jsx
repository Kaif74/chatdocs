import { useEffect, useState } from "react";

import { getIngestProgress, getIngestStatus, runIngest } from "../api";

const ACTIVE_SOURCES = ["langchain", "crewai", "expo"];

function IngestPanel() {
  const [status, setStatus] = useState(null);
  const [progress, setProgress] = useState(null);
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

  useEffect(() => {
    let intervalId;

    async function refreshProgress() {
      try {
        const data = await getIngestProgress();
        setProgress(data);
      } catch (requestError) {
        setError(requestError.message);
      }
    }

    refreshProgress();

    if (loading) {
      intervalId = setInterval(refreshProgress, 2000);
    }

    return () => {
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
  }, [loading]);

  async function handleIngest(options = {}) {
    setLoading(true);
    setError("");

    try {
      const data = await runIngest(options);
      setStatus({
        ready: data.ready,
        total_chunks: data.total_chunks,
        counts_by_source: data.counts_by_source,
      });
      setProgress((previous) => ({
        ...(previous || {}),
        running: false,
        phase: "completed",
        mode: data.mode || previous?.mode || "safe",
        total_chunks: data.total_chunks,
        counts_by_source: data.counts_by_source,
      }));
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  }

  async function handleFullRebuild() {
    const confirmed = window.confirm(
      "Full Rebuild will delete existing vectors and re-ingest all sources. Only continue if required.",
    );
    if (!confirmed) {
      return;
    }

    await handleIngest({ rebuild: true });
  }

  return (
    <section className="panel">
      <div className="panel__header">
        <h2>Ingest framework docs</h2>
        <p>
          Build or rebuild the local Chroma collection from the three
          LLM-friendly docs sources.
        </p>
      </div>

      <div className="row" style={{ gap: "0.75rem", flexWrap: "wrap" }}>
        <button
          type="button"
          className="button"
          onClick={() =>
            handleIngest({ rebuild: false, refreshExisting: false })
          }
          disabled={loading}
        >
          {loading ? "Running ingest..." : "Safe Sync (Missing Only)"}
        </button>
        <button
          type="button"
          className="button"
          onClick={handleFullRebuild}
          disabled={loading}
          title="Deletes and recreates the vector collection before ingest"
        >
          Full Rebuild
        </button>
      </div>

      <div className="notice notice--warning">
        Avoid Full Rebuild unless required. It deletes existing vectors and can
        take significantly longer.
      </div>

      {error ? <div className="notice notice--error">{error}</div> : null}

      {progress ? (
        <div className="status-card">
          <div className="status-card__row">
            <span>Ingestion state</span>
            <strong>{progress.phase || "idle"}</strong>
          </div>
          <div className="status-card__row">
            <span>Mode</span>
            <strong>{progress.mode || "safe"}</strong>
          </div>
          <div className="status-card__row">
            <span>Progress</span>
            <strong>
              {progress.completed_sources || 0}/{progress.total_sources || 0}{" "}
              sources
            </strong>
          </div>
          <div className="status-card__row">
            <span>Current source</span>
            <strong>{progress.current_source || "-"}</strong>
          </div>
          <div className="status-card__row">
            <span>Chunks ingested (session)</span>
            <strong>{progress.total_chunks || 0}</strong>
          </div>
        </div>
      ) : null}

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
            {ACTIVE_SOURCES.map((source) => (
              <div key={source} className="summary-card">
                <span className="summary-card__label">{source}</span>
                <strong>{status.counts_by_source?.[source] || 0}</strong>
              </div>
            ))}
          </div>
        </div>
      ) : null}
    </section>
  );
}

export default IngestPanel;
