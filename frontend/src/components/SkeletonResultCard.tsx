// src/components/SkeletonResultCard.tsx
import React from "react";

const SkeletonResultCard: React.FC = () => {
  return (
    <div className="skeleton-result-card" role="status" aria-label="Загрузка результатов">
      <div className="skeleton-top">
        <div className="skeleton-avatar" />
        <div style={{ flex: 1, marginLeft: 12 }}>
          <div className="skeleton-line short" />
          <div className="skeleton-line tiny" style={{ marginTop: 8 }} />
        </div>
        <div className="skeleton-badge" />
      </div>

      <div className="skeleton-body">
        <div className="skeleton-row">
          <div className="skeleton-label" />
          <div className="skeleton-value" />
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
        <div className="skeleton-row small">
          <div className="skeleton-label small" />
          <div className="skeleton-value small" />
        </div>

        <div className="skeleton-progress">
          <div className="skeleton-progress-bar">
            <div className="skeleton-fill" />
          </div>
          <div className="skeleton-percent" />
        </div>
      </div>
    </div>
  );
};

export default SkeletonResultCard;
