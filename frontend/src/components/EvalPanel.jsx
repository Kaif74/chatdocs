import { useEffect, useMemo, useState } from "react";

import { getEvalQuestions, runEvaluation } from "../api";

const RUBRIC_STORAGE_KEY = "chatdocs-human-rubrics";
const RUBRIC_FIELDS = [
  { key: "coherence", label: "Coherence" },
  { key: "completeness", label: "Completeness" },
  { key: "factual_correctness", label: "Factual" },
];

function loadStoredRubrics() {
  try {
    const rawValue = localStorage.getItem(RUBRIC_STORAGE_KEY);
    return rawValue ? JSON.parse(rawValue) : {};
  } catch {
    return {};
  }
}

function persistRubrics(rubrics) {
  localStorage.setItem(RUBRIC_STORAGE_KEY, JSON.stringify(rubrics));
}

function mergeRubrics(results, storedRubrics) {
  return results.map((result) => ({
    ...result,
    rubric: {
      ...result.rubric,
      ...(storedRubrics[result.id] || {}),
    },
  }));
}

function EvalPanel() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [summary, setSummary] = useState(null);
  const [results, setResults] = useState([]);
  const [questionCount, setQuestionCount] = useState(0);
  const [storedRubrics, setStoredRubrics] = useState(() => loadStoredRubrics());

  useEffect(() => {
    async function fetchQuestions() {
      try {
        const data = await getEvalQuestions();
        setQuestionCount(data.questions.length);
      } catch (requestError) {
        setError(requestError.message);
      }
    }

    fetchQuestions();
  }, []);

  const hydratedResults = useMemo(
    () => mergeRubrics(results, storedRubrics),
    [results, storedRubrics],
  );

  async function handleRunEvaluation() {
    setLoading(true);
    setError("");

    try {
      const data = await runEvaluation();
      setSummary(data.summary);
      setResults(data.results);
    } catch (requestError) {
      setSummary(null);
      setResults([]);
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  }

  function updateRubric(questionId, field, value) {
    const nextRubrics = {
      ...storedRubrics,
      [questionId]: {
        ...(storedRubrics[questionId] || {}),
        [field]: value ? Number(value) : null,
      },
    };

    setStoredRubrics(nextRubrics);
    persistRubrics(nextRubrics);
  }

  return (
    <section className="panel">
      <div className="panel__header">
        <h2>Evaluate the RAG pipeline</h2>
        <p>
          Run all {questionCount || 15} benchmark questions and score the
          results.
        </p>
      </div>

      <button
        type="button"
        className="button"
        onClick={handleRunEvaluation}
        disabled={loading}
      >
        {loading ? "Running full evaluation..." : "Run Full Evaluation"}
      </button>

      {error ? <div className="notice notice--error">{error}</div> : null}

      {summary ? (
        <div className="summary-grid">
          <div className="summary-card">
            <span className="summary-card__label">
              Average Cosine Similarity
            </span>
            <strong>{summary.average_cosine_similarity}</strong>
          </div>
          <div className="summary-card">
            <span className="summary-card__label">Average Keyword F1</span>
            <strong>{summary.average_keyword_f1}</strong>
          </div>
          <div className="summary-card">
            <span className="summary-card__label">Retrieval Hit Rate</span>
            <strong>{summary.retrieval_hit_rate}</strong>
          </div>
        </div>
      ) : null}

      {hydratedResults.length ? (
        <div className="table-wrapper">
          <table className="results-table">
            <thead>
              <tr>
                <th>Question</th>
                <th>Expected</th>
                <th>Generated</th>
                <th>Provider</th>
                <th>Cos Sim</th>
                <th>KW F1</th>
                <th>Retrieved?</th>
                <th>Human Rubric</th>
              </tr>
            </thead>
            <tbody>
              {hydratedResults.map((result) => (
                <tr key={result.id}>
                  <td>
                    <strong>{result.question}</strong>
                    <div className="cell-meta">{result.source}</div>
                  </td>
                  <td>{result.expected_answer}</td>
                  <td>{result.generated_answer}</td>
                  <td>{result.provider_used}</td>
                  <td>{result.cosine_similarity}</td>
                  <td>{result.keyword_f1}</td>
                  <td>{result.retrieval_hit ? "Yes" : "No"}</td>
                  <td>
                    <div className="rubric-grid">
                      {RUBRIC_FIELDS.map((field) => (
                        <label key={field.key} className="field">
                          <span>{field.label}</span>
                          <select
                            value={result.rubric[field.key] ?? ""}
                            onChange={(event) =>
                              updateRubric(
                                result.id,
                                field.key,
                                event.target.value,
                              )
                            }
                          >
                            <option value="">-</option>
                            {[1, 2, 3, 4, 5].map((score) => (
                              <option key={score} value={score}>
                                {score}
                              </option>
                            ))}
                          </select>
                        </label>
                      ))}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
    </section>
  );
}

export default EvalPanel;
