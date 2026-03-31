import MarkdownLite from "./MarkdownLite";

function SourceCard({ source }) {
  return (
    <article className="source-card">
      <div className="source-card__meta">
        <span className="pill">{source.source}</span>
        <span>Chunk #{source.chunk_id}</span>
        <span>Score: {source.score}</span>
      </div>
      <MarkdownLite text={source.text} className="source-card__text" />
      <a href={source.url} target="_blank" rel="noreferrer" className="source-card__link">
        Open source doc
      </a>
    </article>
  );
}

export default SourceCard;
