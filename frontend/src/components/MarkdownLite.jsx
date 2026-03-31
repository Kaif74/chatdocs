function renderInline(text) {
  const normalizedText = text
    .replace(/<\/?code>/gi, "`")
    .replace(/<\/?strong>/gi, "**")
    .replace(/<\/?em>/gi, "*")
    .replace(/<br\s*\/?>/gi, " ")
    .replace(/<\/?[^>]+>/g, "")
    .replace(/\s+/g, " ")
    .trim();

  const pattern = /(`([^`]+)`|\*\*([^*]+)\*\*|\[([^\]]+)\]\((https?:\/\/[^\s)]+)\))/g;
  const parts = [];
  let lastIndex = 0;
  let match;

  while ((match = pattern.exec(normalizedText)) !== null) {
    if (match.index > lastIndex) {
      parts.push(normalizedText.slice(lastIndex, match.index));
    }

    if (match[2]) {
      parts.push(
        <code key={`inline-code-${match.index}`} className="rich-text__inline-code">
          {match[2]}
        </code>,
      );
    } else if (match[3]) {
      parts.push(<strong key={`bold-${match.index}`}>{match[3]}</strong>);
    } else if (match[4] && match[5]) {
      parts.push(
        <a
          key={`link-${match.index}`}
          href={match[5]}
          target="_blank"
          rel="noreferrer"
          className="rich-text__link"
        >
          {match[4]}
        </a>,
      );
    }

    lastIndex = pattern.lastIndex;
  }

  if (lastIndex < normalizedText.length) {
    parts.push(normalizedText.slice(lastIndex));
  }

  return parts;
}

function isCodeLikeLine(line) {
  if (!line) {
    return false;
  }

  const trimmed = line.trim();
  if (/^(const|let|var|import|export|from|function|class|return)\b/.test(trimmed)) {
    return true;
  }

  if (trimmed.includes("=>") || trimmed.includes("::") || trimmed.includes("</")) {
    return true;
  }

  const codeSignals = [
    /[{}[\]]/,
    /\w+\(/,
    /\)\s*;/,
    /\bnew\s+[A-Z]/,
    /\s=\s/,
    /;\s*\w+/,
  ];

  const hits = codeSignals.reduce((count, pattern) => count + Number(pattern.test(trimmed)), 0);
  return hits >= 3;
}

function parseBlocks(text) {
  const lines = text.split(/\r?\n/);
  const blocks = [];
  let currentBlock = null;
  let inFence = false;
  let fencedLines = [];

  function flushBlock() {
    if (currentBlock && currentBlock.items.length) {
      blocks.push(currentBlock);
    }
    currentBlock = null;
  }

  for (const rawLine of lines) {
    const line = rawLine.trim();

    if (line.startsWith("```")) {
      flushBlock();
      if (inFence) {
        blocks.push({ type: "pre", items: [fencedLines.join("\n")] });
        fencedLines = [];
        inFence = false;
      } else {
        inFence = true;
      }
      continue;
    }

    if (inFence) {
      fencedLines.push(rawLine);
      continue;
    }

    if (!line) {
      flushBlock();
      continue;
    }

    if (/^(\*{3,}|-{3,}|_{3,})$/.test(line)) {
      flushBlock();
      continue;
    }

    if (/^<\/?[A-Za-z][^>]*>$/.test(line)) {
      flushBlock();
      continue;
    }

    if (isCodeLikeLine(rawLine)) {
      flushBlock();
      blocks.push({ type: "pre", items: [rawLine.trim()] });
      continue;
    }

    const headingMatch = line.match(/^(#{1,6})\s+(.*)$/);
    if (headingMatch) {
      flushBlock();
      blocks.push({
        type: "heading",
        level: headingMatch[1].length,
        items: [headingMatch[2]],
      });
      continue;
    }

    const orderedMatch = line.match(/^(\d+)\.\s+(.*)$/);
    if (orderedMatch) {
      if (!currentBlock || currentBlock.type !== "ol") {
        flushBlock();
        currentBlock = { type: "ol", items: [], start: Number(orderedMatch[1]) };
      }
      currentBlock.items.push(orderedMatch[2]);
      continue;
    }

    const unorderedMatch = line.match(/^[-*]\s+(.*)$/);
    if (unorderedMatch) {
      if (!currentBlock || currentBlock.type !== "ul") {
        flushBlock();
        currentBlock = { type: "ul", items: [] };
      }
      currentBlock.items.push(unorderedMatch[1]);
      continue;
    }

    if (!currentBlock || currentBlock.type !== "p") {
      flushBlock();
      currentBlock = { type: "p", items: [] };
    }
    currentBlock.items.push(line);
  }

  flushBlock();

  if (inFence && fencedLines.length) {
    blocks.push({ type: "pre", items: [fencedLines.join("\n")] });
  }

  return blocks;
}

function MarkdownLite({ text, className = "" }) {
  const blocks = parseBlocks(text);

  return (
    <div className={`rich-text ${className}`.trim()}>
      {blocks.map((block, index) => {
        if (block.type === "heading") {
          const HeadingTag = `h${Math.min(block.level || 3, 4)}`;
          return (
            <HeadingTag key={`heading-${index}`} className="rich-text__heading">
              {renderInline(block.items[0])}
            </HeadingTag>
          );
        }

        if (block.type === "code") {
          return (
            <code key={`code-${index}`} className="rich-text__code">
              {block.items[0]}
            </code>
          );
        }

        if (block.type === "pre") {
          return (
            <pre key={`pre-${index}`} className="rich-text__pre">
              <code>{block.items[0]}</code>
            </pre>
          );
        }

        if (block.type === "ol") {
          return (
            <ol
              key={`ol-${index}`}
              start={block.start || 1}
              className="rich-text__list rich-text__list--ordered"
            >
              {block.items.map((item, itemIndex) => (
                <li key={`ol-item-${index}-${itemIndex}`}>{renderInline(item)}</li>
              ))}
            </ol>
          );
        }

        if (block.type === "ul") {
          return (
            <ul key={`ul-${index}`} className="rich-text__list">
              {block.items.map((item, itemIndex) => (
                <li key={`ul-item-${index}-${itemIndex}`}>{renderInline(item)}</li>
              ))}
            </ul>
          );
        }

        return (
          <p key={`p-${index}`} className="rich-text__paragraph">
            {renderInline(block.items.join(" "))}
          </p>
        );
      })}
    </div>
  );
}

export default MarkdownLite;
