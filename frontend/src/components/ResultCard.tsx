// src/components/ResultCard.tsx
import React, { useState } from "react";
import { ResultData } from "../types";

interface Props {
  data: ResultData;
}

const percent = (val?: number) =>
  typeof val === "number" && !Number.isNaN(val) ? Math.round(val * 100) : 0;

const ResultCard: React.FC<Props> = ({ data }) => {
  const [downloading, setDownloading] = useState(false);
  const prob = percent(data.probability_of_pathology as unknown as number);
  const pathologyFlag = Number(data.pathology); // 0 or 1

  const downloadReportFromDataUri = async (dataUri: string, filename: string) => {
    try {
      setDownloading(true);
      const parts = dataUri.split(",");
      if (parts.length !== 2) {
        throw new Error("Invalid data URI");
      }
      const meta = parts[0]; // data:...;base64
      const b64 = parts[1];
      const byteString = atob(b64);
      const len = byteString.length;
      const u8 = new Uint8Array(len);
      for (let i = 0; i < len; i++) u8[i] = byteString.charCodeAt(i);
      const mimeMatch = meta.match(/data:(.*?);/);
      const mime = mimeMatch ? mimeMatch[1] : "application/octet-stream";
      const blob = new Blob([u8], { type: mime });

      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      console.error("Download failed", err);
      alert(err instanceof Error ? err.message : "Не удалось скачать отчёт");
    } finally {
      setDownloading(false);
    }
  };

  const handleDownloadClick = () => {
    if (!data.report_xlsx) return;
    const studySafe = (data.study_uid || "report").toString().replace(/[:.]/g, "-");
    const filename = `report_${studySafe}.xlsx`;
    downloadReportFromDataUri(data.report_xlsx, filename);
  };

  return (
    <div className="result-card" role="group" aria-label="Результаты анализа">
      <div className="result-header">
        <div className="result-title">
          <div className="result-logo" aria-hidden>
            🫁
          </div>
          <div>
            <div className="result-main">Результат AI-анализа КТ грудной клетки</div>
            <div className="result-sub">Автоматический скрининг: норма / патология</div>
          </div>
        </div>

        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <div className={`result-badge ${pathologyFlag === 0 ? "normal" : "abnormal"}`}>
            {pathologyFlag === 0 ? "Норма" : "Патология"}
          </div>
          {/* Кнопка убрана из хедера — теперь внизу карточки */}
        </div>
      </div>

      <div className="result-body">
        <div className="row">
          <div className="row-label">Путь</div>
          <div className="row-value">{data.path_to_study || "—"}</div>
        </div>

        <div className="row">
          <div className="row-label">Study UID</div>
          <div className="row-value">{data.study_uid || "—"}</div>
        </div>

        <div className="row">
          <div className="row-label">Series UID</div>
          <div className="row-value">{data.series_uid || "—"}</div>
        </div>

        <div className="row">
          <div className="row-label">Вероятность патологии</div>
          <div className="row-value progress-col">
            <div className="progress-bar" aria-hidden>
              <div
                className={prob < 30 ? "fill low" : prob < 70 ? "fill mid" : "fill high"}
                style={{ width: `${prob}%` }}
              />
            </div>
            <div className="percent-text">{prob}%</div>
          </div>
        </div>

        <div className="row">
          <div className="row-label">Статус обработки</div>
          <div className="row-value">{data.processing_status || "—"}</div>
        </div>

        <div className="row">
          <div className="row-label">Время обработки</div>
          <div className="row-value">{data.time_of_processing || "—"} сек</div>
        </div>

        {/* footer with download button */}
        <div className="result-card-footer" style={{ marginTop: 12 }}>
          <button
            className="download-btn xlsx"
            onClick={handleDownloadClick}
            disabled={downloading || !data.report_xlsx}
            aria-disabled={downloading || !data.report_xlsx}
            title={data.report_xlsx ? "Скачать отчёт (XLSX)" : "Отчёт недоступен"}
          >
            <span className="download-icon" aria-hidden>
              <svg
                width="18"
                height="18"
                viewBox="0 0 24 24"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
                aria-hidden
              >
                <path d="M6 2h7l5 5v13a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2z" fill="#0A6F4C" />
                <path d="M13 2v6h6" fill="#12B886" />
                <path d="M8.5 15.5l1.5-2 1.5 2 1.5-2" stroke="#fff" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" />
                <path d="M8.5 13h6" stroke="#fff" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </span>

            <span className="download-label">{downloading ? "Скачивание..." : "Скачать отчёт"}</span>

            <span className="file-ext-badge" aria-hidden>
              XLSX
            </span>
          </button>
        </div>
      </div>
    </div>
  );
};

export default ResultCard;
