// src/App.tsx
import React, { useState } from "react";
import Chat from "./components/chat";
import FileUpload from "./components/FileUpload";
import "./styles.css";
import { ResultData, ViewerData } from "./types";

export interface Message {
  id: number;
  type: "user" | "system" | "result" | "viewer";
  content: string | ResultData | ViewerData;
}

const App: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);

  const genId = () => Date.now() + Math.floor(Math.random() * 1000);

  const handleFileUpload = async (file: File) => {
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

    const uploadingMsgId = genId();
    setMessages((prev) => [
      ...prev,
      { id: genId(), type: "user", content: `📦 ${file.name}` },
      { id: uploadingMsgId, type: "system", content: "⏳ Обработка файла..." },
    ]);

    const formData = new FormData();
    formData.append("file", file);

    try {
      // 1. Получаем метаданные
      const metaRes = await fetch("http://localhost:8000/process-image", {
        method: "POST",
        body: formData,
      });
      if (!metaRes.ok) throw new Error("Meta request failed");
      const meta = await metaRes.json();

      const normalized: ResultData = {
        path_to_study: meta.path_to_study,
        study_uid: meta.study_uid,
        series_uid: meta.series_uid,
        probability_of_pathology: meta.probability_of_pathology,
        pathology: meta.pathology,
        processing_status: meta.processing_status,
        time_of_processing: meta.time_of_processing,
      };

      setMessages((prev) => [
        ...prev.filter((m) => m.id !== uploadingMsgId),
        { id: genId(), type: "result", content: normalized },
      ]);

      // 2. Получаем viewer (кадры)
      const viewRes = await fetch("http://localhost:8000/viewer", {
        method: "POST",
        body: formData,
      });
      if (viewRes.ok) {
        const data = await viewRes.json();
        if (data.frames && data.frames.length > 0) {
          setMessages((prev) => [
            ...prev,
            { id: genId(), type: "viewer", content: { frames: data.frames } },
          ]);
        }
      }
    } catch (error: any) {
      setMessages((prev) => [
        ...prev.filter((m) => m.id !== uploadingMsgId),
        {
          id: genId(),
          type: "system",
          content: `❌ Ошибка при обработке файла: ${error.message}`,
        },
      ]);
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
        <FileUpload onFileUpload={handleFileUpload} />
        <div className="small-note">
          Рекомендуется загружать ZIP с DICOM / подготовленной структурой исследования.
        </div>
      </footer>
    </div>
  );
};

export default App;
