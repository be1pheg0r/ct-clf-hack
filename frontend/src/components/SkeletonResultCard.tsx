// src/components/SkeletonResultCard.tsx
import React from "react";

/**
 * SkeletonResultCard — улучшенный skeleton для ResultCard.
 * Повторяет структуру реальной карточки: хедер, поля, прогресс и кнопка (full-width).
 */
const SkeletonResultCard: React.FC = () => {
  return (
    <div className="skeleton-result-card" role="status" aria-label="Загрузка результатов">
      {/* header */}
      <div className="skeleton-top">
        <div className="skeleton-avatar" aria-hidden />
        <div style={{ flex: 1, marginLeft: 12 }}>
          <div className="skeleton-line short" />
          <div className="skeleton-line tiny" style={{ marginTop: 8 }} />
        </div>
        <div className="skeleton-badge" aria-hidden />
      </div>

      {/* body — имитируем реальные поля карточки */}
      <div className="skeleton-body">
        <div className="skeleton-row">
          <div className="skeleton-label" />
          <div className="skeleton-value" />
        </div>

        <div className="skeleton-row">
          <div className="skeleton-label" />
          <div className="skeleton-value long" />
        </div>

        <div className="skeleton-row">
          <div className="skeleton-label" />
          <div className="skeleton-value long" />
        </div>

        <div className="skeleton-row small">
          <div className="skeleton-label small" />
          <div className="skeleton-value small" />
        </div>

        <div className="skeleton-row small">
          <div className="skeleton-label small" />
          <div className="skeleton-value small" />
        </div>

        {/* probability / progress */}
        <div style={{ marginTop: 6 }} />
        <div className="skeleton-row">
          <div className="skeleton-label" />
          <div style={{ flex: 1 }}>
            <div className="skeleton-progress">
              <div className="skeleton-progress-bar" style={{ marginRight: 12 }}>
                <div className="skeleton-fill" />
              </div>
            </div>
          </div>
        </div>
        {/* spacer */}
        <div style={{ height: 8 }} />

        {/* full-width button skeleton */}
        <div className="skeleton-download-wrapper">
          <div className="skeleton-download" aria-hidden>
            <div className="skeleton-download-icon" />
            <div className="skeleton-download-text" />
            <div className="skeleton-download-badge" />
          </div>
        </div>
      </div>
    </div>
  );
};

export default SkeletonResultCard;
