import React, { useEffect, useState, useRef } from "react";
import { ViewerData } from "../types";

interface Props {
  data: ViewerData;
}

const DicomViewer: React.FC<Props> = ({ data }) => {
  const frames = data.frames || [];
  const [index, setIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [intervalMs, setIntervalMs] = useState(120);
  const timerRef = useRef<number | null>(null);

  useEffect(() => {
    if (playing && frames.length > 0) {
      timerRef.current = window.setInterval(() => {
        setIndex((i) => (i + 1) % frames.length);
      }, intervalMs);
      return () => {
        if (timerRef.current) window.clearInterval(timerRef.current);
      };
    } else {
      if (timerRef.current) {
        window.clearInterval(timerRef.current);
        timerRef.current = null;
      }
    }
  }, [playing, intervalMs, frames.length]);

  useEffect(() => {
    setIndex(0);
    setPlaying(false);
  }, [data.frames]);

  if (!frames.length)
    return <div style={{ width: 520 }}>Нет доступных срезов для просмотра.</div>;

  return (
    <div className="dicom-viewer" style={{ width: 520 }}>
      <div
        className="viewer-image"
        style={{
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
          height: 360,
          // background: "#0f172a0f",
          background: "black",
          borderRadius: 8,
        }}
      >
        <img
          src={frames[index]}
          alt={`slice-${index}`}
          style={{ maxWidth: "100%", maxHeight: "100%", borderRadius: 6 }}
        />
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 8 }}>
        <button onClick={() => setIndex((i) => Math.max(0, i - 1))} title="Prev">
          ◀
        </button>
        <button onClick={() => setPlaying((p) => !p)} title="Play/Pause">
          {playing ? "⏸" : "▶"}
        </button>
        <button onClick={() => setIndex((i) => Math.min(frames.length - 1, i + 1))} title="Next">
          ▶
        </button>

        <div style={{ flex: 1 }}>
          <input
            type="range"
            min={0}
            max={frames.length - 1}
            value={index}
            onChange={(e) => setIndex(parseInt(e.target.value))}
            style={{ width: "100%" }}
          />
        </div>

        <div style={{ minWidth: 56, textAlign: "right", fontWeight: 700 }}>
          {index + 1}/{frames.length}
        </div>
      </div>

      <div style={{ display: "flex", gap: 8, alignItems: "center", marginTop: 8 }}>
        <label style={{ fontSize: 12, color: "#6b7280" }}>Speed</label>
        <input
          type="range"
          min={40}
          max={1000}
          value={intervalMs}
          onChange={(e) => setIntervalMs(parseInt(e.target.value))}
        />
        <div style={{ minWidth: 44, textAlign: "right" }}>{intervalMs} ms</div>
      </div>
    </div>
  );
};

export default DicomViewer;
