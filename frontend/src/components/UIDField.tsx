// src/components/UIDField.tsx
import React, { useState } from "react";

interface UIDFieldProps {
  uid?: string | null;
  className?: string;
  ariaLabel?: string;
}

/**
 * UIDField:
 * - вставляет \u200B (zero-width space) после каждой точки, чтобы дать браузеру точки переноса
 * - отображает uid в monospace
 * - показывает полный uid в title
 * - кнопка копирования копирует оригинальный uid (без ZWSP)
 */
const insertZWS = (s: string) => s.replace(/\./g, ".\u200B");

const UIDField: React.FC<UIDFieldProps> = ({ uid, className = "", ariaLabel }) => {
  const [copied, setCopied] = useState(false);

  const safeUid = uid ?? "—";
  const display = safeUid === "—" ? safeUid : insertZWS(safeUid);

  const handleCopy = async (e: React.MouseEvent) => {
    e.stopPropagation();
    if (!uid) return;
    try {
      await navigator.clipboard.writeText(uid);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // fallback: create temporary textarea
      try {
        const ta = document.createElement("textarea");
        ta.value = uid;
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
    <div className={`uid-field ${className}`} title={safeUid} aria-label={ariaLabel ?? "UID"}>
      <code className="uid-text" aria-hidden>
        {display}
      </code>

      {safeUid !== "—" && (
        <button
          className="uid-copy-btn"
          onClick={handleCopy}
          aria-label={copied ? "Скопировано" : "Скопировать UID"}
          type="button"
        >
          {copied ? "✓" : "⧉"}
        </button>
      )}
    </div>
  );
};

export default UIDField;
