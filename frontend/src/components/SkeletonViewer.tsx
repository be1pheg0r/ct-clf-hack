// src/components/SkeletonViewer.tsx
import React from "react";

/**
 * Skeleton для DicomViewer — теперь с двумя "лёгочными" полукруками,
 * синхронизированной линией сканера и пульсирующей точкой.
 */
const SkeletonViewer: React.FC = () => {
  return (
    <div className="skeleton-viewer" style={{ width: 520 }}>
      <div
        className="skeleton-image"
        aria-hidden
        style={{
          height: 360,
          borderRadius: 12,
          overflow: "hidden",
          position: "relative",
        }}
      >
        {/* фон и декоративные элементы управляются через CSS-классов */}
        <div className="scan-bg" />
        <div className="lung-shape" aria-hidden>
          <div className="lung left" />
          <div className="lung right" />
        </div>

        <div className="scan-line" />
        <div className="scan-dot" />
      </div>

      <div style={{ display: "flex", gap: 10, marginTop: 12, alignItems: "center" }}>
        <div className="thumb-glow" style={{backgroundColor : "green", opacity : "0.2"}} />
        <div className="thumb-glow" style={{backgroundColor : "red", opacity : "0.1"}} />
        {/* <div className="small-glow" /> */}
        <div className="thumb-glow" />
        <div className="thumb-glow" />
        <div className="thumb-glow" />
        <div className="thumb-glow" />
        <div className="thumb-glow" />
        <div style={{ flex: 1 }} />
        
      </div>

      <div style={{ display: "flex", gap: 10, marginTop: 10 }}>
        <div className="bar-glow" style={{ width: "60%" }} />
        <div className="bar-glow short" style={{ width: "40%" }} />
      </div>
    </div>
  );
};

export default SkeletonViewer;
