import React from "react";
import { ResultData } from "../types";

interface Props {
  data: ResultData;
}

const percent = (val?: number) =>
  typeof val === "number" && !Number.isNaN(val) ? Math.round(val * 100) : 0;

const ResultCard: React.FC<Props> = ({ data }) => {
  const prob = percent(data.probability_of_pathology as unknown as number);
  const pathologyFlag = Number(data.pathology); // 0 or 1

  return (
    <div className="result-card" role="group" aria-label="Результаты анализа">
      <div className="result-header">
        <div className="result-title">
          <div className="result-logo" aria-hidden>🫁</div>
          <div>
            <div className="result-main">Результат AI-анализа КТ грудной клетки</div>
            <div className="result-sub">Автоматический скрининг: норма / патология</div>
          </div>
        </div>
        <div className={`result-badge ${pathologyFlag === 0 ? "normal" : "abnormal"}`}>
          {pathologyFlag === 0 ? "Норма" : "Патология"}
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
          <div className="row-label">Статус обработки</div>
          <div className="row-value">{data.processing_status || "—"}</div>
        </div>

        <div className="row">
          <div className="row-label">Время обработки</div>
          <div className="row-value">{data.time_of_processing || "—"} сек</div>
        </div>



        <div className="row">
          <div className="row-label">Вероятность патологии</div>
          <div className="row-value progress-col">
            <div className="progress-bar" aria-hidden>
              <div
                className={
                  prob < 30 ? "fill low" : prob < 70 ? "fill mid" : "fill high"
                }
                style={{ width: `${prob}%` }}
              />
            </div>
            <div className="percent-text">{prob}%</div>
          </div>
        </div>

      </div>
    </div>
  );
};

export default ResultCard;
