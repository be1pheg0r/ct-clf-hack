// src/App.tsx
import React, { useState } from "react";
import Chat from "./components/chat";
import FileUpload from "./components/FileUpload";
import "./styles.css";
import { ResultData, ViewerData } from "./types";

export interface Message {
  id: number;
  type: "user" | "system" | "result" | "viewer" | "viewer-loading" | "result-loading";
  content: string | ResultData | ViewerData;
}

const App: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [processing, setProcessing] = useState(false);

  const genId = () => Date.now() + Math.floor(Math.random() * 1000);

  const handleFileUpload = async (file: File) => {
    if (processing) return;
    const isZip =
      file.name.toLowerCase().endsWith(".zip") ||
      file.type === "application/zip" ||
      file.type === "application/x-zip-compressed";

    if (!isZip) {
      setMessages((prev) => [
        ...prev,
        { id: genId(), type: "system", content: "Можно загружать только ZIP-файлы!" },
      ]);
      return;
    }

    setProcessing(true);

    const userMsgId = genId();
    const resultLoadingMsgId = genId();
    const viewerLoadingMsgId = genId();

    setMessages((prev) => [
      ...prev,
      { id: userMsgId, type: "user", content: `📦 ${file.name}` },
      { id: resultLoadingMsgId, type: "result-loading", content: "loading" },
      { id: viewerLoadingMsgId, type: "viewer-loading", content: "loading" },
    ]);

    const formData = new FormData();
    formData.append("file", file);

    try {
      // 1) process-image (metadata + report_xlsx)
      const metaRes = await fetch("http://localhost:8000/process-image", {
        method: "POST",
        body: formData,
      });
      if (!metaRes.ok) {
        const text = await metaRes.text();
        throw new Error(`Meta request failed: ${metaRes.status} ${text}`);
      }
      const meta = await metaRes.json();

      const normalized: ResultData = {
        path_to_study: meta.path_to_study,
        study_uid: meta.study_uid,
        series_uid: meta.series_uid,
        probability_of_pathology:
          typeof meta.probability_of_pathology === "string"
            ? parseFloat(meta.probability_of_pathology)
            : meta.probability_of_pathology,
        pathology:
          typeof meta.pathology === "string" ? parseInt(meta.pathology) : meta.pathology,
        processing_status: meta.processing_status,
        time_of_processing: meta.time_of_processing,
        report_xlsx: typeof meta.report_xlsx === "string" ? meta.report_xlsx : undefined,
      };

      // replace result-loading with actual result (preserve viewer-loading position)
      setMessages((prev) =>
        prev.map((m) => (m.id === resultLoadingMsgId ? { id: genId(), type: "result", content: normalized } : m))
      );

      // 2) viewer — request frames and replace viewer-loading with viewer message
      const viewRes = await fetch("http://localhost:8000/viewer", {
        method: "POST",
        body: formData,
      });

      if (viewRes.ok) {
        const viewData = await viewRes.json();
        const frames = Array.isArray(viewData.frames) ? viewData.frames : [];
        if (frames.length > 0) {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === viewerLoadingMsgId ? ({ id: genId(), type: "viewer", content: { frames } as ViewerData } as Message) : m
            )
          );
        } else {
          // no frames -> remove skeleton
          setMessages((prev) => prev.filter((m) => m.id !== viewerLoadingMsgId));
        }
      } else {
        const text = await viewRes.text();
        setMessages((prev) =>
          prev.map((m) =>
            m.id === viewerLoadingMsgId ? ({ id: genId(), type: "system", content: `❌ Viewer error: ${text}` } as Message) : m
          )
        );
      }
    } catch (error: any) {
      setMessages((prev) => [
        ...prev.filter((m) => m.id !== resultLoadingMsgId && m.id !== viewerLoadingMsgId),
        {
          id: genId(),
          type: "system",
          content:
            error && error.message ? `❌ Ошибка при обработке файла: ${error.message}` : "❌ Ошибка при обработке файла",
        },
      ]);
      console.error("Upload error:", error);
    } finally {
      setProcessing(false);
    }
  };

  const handleClear = () => setMessages([]);

  return (
    <div className="app">
      <header className="app-header">
        <div className="brand">
          <div className="brand-logo">🩺</div>
          <div className="brand-text">
            <h1>AI DICOM Analyzer</h1>
            <span className="brand-sub">КТ грудной клетки — автоматический скрининг</span>
          </div>
        </div>

        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <button className="ghost-btn" onClick={handleClear} title="Очистить чат">
            Очистить
          </button>
        </div>
      </header>

      <main className="main">
        <Chat messages={messages} />
      </main>

      <footer className="app-footer">
        <FileUpload onFileUpload={handleFileUpload} disabled={processing} />
        <div className="small-note">
          {processing ? "Обработка... загрузка временно заблокирована" : "Рекомендуется загружать ZIP с DICOM / подготовленной структурой исследования."}
        </div>
      </footer>
    </div>
  );
};

export default App;
