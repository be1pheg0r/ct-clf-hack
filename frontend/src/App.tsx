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

    // Добавляем сообщение-пользователя и skeleton-статус
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

      if (!res.ok) {
        const text = await res.text();
        throw new Error(`Server returned ${res.status}: ${text}`);
      }

      const data = await res.json();

      // Нормализуем поля результата
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

      // Проверяем frames (массив data:image/png;base64,...)
      const viewer =
        data.frames && Array.isArray(data.frames) && data.frames.length > 0
          ? ({ frames: data.frames as string[] } as ViewerData)
          : null;

      // Собираем массив сообщений, который будем добавить — явно аннотируем как Message[]
      const toAdd: Message[] = [
        { id: genId(), type: "result", content: normalized },
      ];
      if (viewer) {
        toAdd.push({ id: genId(), type: "viewer", content: viewer });
      }

      // Удаляем skeleton-статус и добавляем result + viewer (если есть)
      setMessages((prev) => [
        ...prev.filter((m) => m.id !== uploadingMsgId),
        ...toAdd,
      ]);
    } catch (error: any) {
      // удаляем skeleton и показываем ошибку
      setMessages((prev) => [
        ...prev.filter((m) => m.id !== uploadingMsgId),
        {
          id: genId(),
          type: "system",
          content:
            error && error.message
              ? `❌ Ошибка при обработке файла: ${error.message}`
              : "❌ Ошибка при обработке файла",
        },
      ]);
      console.error("Upload error:", error);
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
