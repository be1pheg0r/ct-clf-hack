import React, { useState } from "react";
import Chat from "./components/chat";
import FileUpload from "./components/FileUpload";
import "./styles.css";
import { ResultData } from "./types";

export interface Message {
  id: number;
  type: "user" | "system" | "result";
  content: string | ResultData;
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

    // добавляем сообщение-пользователя и skeleton-статус
    const uploadingMsgId = genId();
    setMessages((prev) => [
      ...prev,
      { id: genId(), type: "user", content: `📦 ${file.name}` },
      { id: uploadingMsgId, type: "system", content: "⏳ Обработка файла..." },
    ]);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("http://localhost:8000/process-image", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();

      // пытаемся привести probability/pathology к числам
      const normalized: ResultData = {
        path_to_study: data.path_to_study,
        study_uid: data.study_uid,
        series_uid: data.series_uid,
        probability_of_pathology:
          typeof data.probability_of_pathology === "string"
            ? parseFloat(data.probability_of_pathology)
            : data.probability_of_pathology,
        pathology:
          typeof data.pathology === "string" ? parseInt(data.pathology) : data.pathology,
        processing_status: data.processing_status,
        time_of_processing: data.time_of_processing,
      };

      // удаляем skeleton-статус и добавляем result-карту
      setMessages((prev) => [
        ...prev.filter((m) => m.id !== uploadingMsgId),
        { id: genId(), type: "result", content: normalized },
      ]);
    } catch (error) {
      setMessages((prev) => [
        ...prev.filter((m) => m.id !== uploadingMsgId),
        { id: genId(), type: "system", content: "❌ Ошибка при обработке файла" },
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
        <div className="small-note">Рекомендуется загружать ZIP с DICOM / подготовленной структурой исследования.</div>
      </footer>
    </div>
  );
};

export default App;
