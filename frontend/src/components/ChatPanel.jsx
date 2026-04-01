import { useState } from "react";

import { queryDocs } from "../api";
import MarkdownLite from "./MarkdownLite";
import SourceCard from "./SourceCard";

const SOURCE_OPTIONS = [
  { value: "", label: "All sources" },
  { value: "langchain", label: "LangChain" },
  { value: "crewai", label: "CrewAI" },
  { value: "expo", label: "Expo" },
];

function ChatPanel() {
  const [question, setQuestion] = useState("");
  const [sourceFilter, setSourceFilter] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [showSources, setShowSources] = useState(true);

  async function handleSubmit(event) {
    event.preventDefault();
    if (!question.trim()) {
      return;
    }

    setLoading(true);
    setError("");

    try {
      const data = await queryDocs({
        question: question.trim(),
        source_filter: sourceFilter || undefined,
      });
      setResult(data);
    } catch (requestError) {
      setResult(null);
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="panel">
      <div className="panel__header">
        <h2>Ask the docs</h2>
        <p>
          Query the ingested framework documentation and inspect the retrieved
          evidence.
        </p>
      </div>

      <form className="stack" onSubmit={handleSubmit}>
        <label className="field">
          <span>Question</span>
          <textarea
            rows="4"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="How does LCEL support streaming and async execution?"
          />
        </label>

        <label className="field field--inline">
          <span>Source filter</span>
          <select
            value={sourceFilter}
            onChange={(event) => setSourceFilter(event.target.value)}
          >
            {SOURCE_OPTIONS.map((option) => (
              <option key={option.value || "all"} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </label>

        <button type="submit" className="button" disabled={loading}>
          {loading ? "Generating answer..." : "Ask question"}
        </button>
      </form>

      {error ? <div className="notice notice--error">{error}</div> : null}

      {result ? (
        <div className="result-card">
          <div className="result-card__header">
            <h3>Answer</h3>
            <span className="pill">Provider: {result.provider_used}</span>
          </div>
          <MarkdownLite text={result.answer} className="answer-text" />

          <button
            type="button"
            className="link-button"
            onClick={() => setShowSources((current) => !current)}
          >
            {showSources ? "Hide" : "Show"} retrieved chunks (
            {result.sources.length})
          </button>

          {showSources ? (
            <div className="stack">
              {result.sources.map((source) => (
                <SourceCard
                  key={`${source.source}-${source.chunk_id}-${source.score}`}
                  source={source}
                />
              ))}
            </div>
          ) : null}
        </div>
      ) : null}
    </section>
  );
}

export default ChatPanel;
