// src/components/PathField.tsx
import React, { useState } from "react";

interface PathFieldProps {
  path?: string | null;
  className?: string;
  ariaLabel?: string;
}

/**
 * insertBreakHints: вставляет zero-width space (\u200B) после символов-разделителей,
 * чтобы браузер мог переносить длинные пути по логичным границам.
 */
const insertBreakHints = (s: string) =>
  s.replace(/([\/\\\.\_\-\:])/g, (m) => `${m}\u200B`);

const PathField: React.FC<PathFieldProps> = ({ path, className = "", ariaLabel }) => {
  const [copied, setCopied] = useState(false);
  const safePath = path ?? "—";
  const display = safePath === "—" ? safePath : insertBreakHints(safePath);

  const handleCopy = async (e: React.MouseEvent) => {
    e.stopPropagation();
    if (!path) return;
    try {
      await navigator.clipboard.writeText(path);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      try {
        const ta = document.createElement("textarea");
        ta.value = path;
        document.body.appendChild(ta);
        ta.select();
        document.execCommand("copy");
        ta.remove();
        setCopied(true);
        setTimeout(() => setCopied(false), 1500);
      } catch {
        // ignore
      }
    }
  };

  return (
    <div className={`path-field ${className}`} title={safePath} aria-label={ariaLabel ?? "Path"}>
      <span className="path-text" aria-hidden>
        {display}
      </span>

      {safePath !== "—" && (
        <button
          className="path-copy-btn"
          onClick={handleCopy}
          aria-label={copied ? "Скопировано" : "Скопировать путь"}
          type="button"
        >
          {copied ? "✓" : "⧉"}
        </button>
      )}
    </div>
  );
};

export default PathField;
