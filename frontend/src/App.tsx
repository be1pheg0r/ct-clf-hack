import React, { useState } from "react";
import Chat from "./components/chat";
import FileUpload from "./components/FileUpload";
import "./styles.css";

export interface Message {
    id: number;
    type: "user" | "system";
    content: string;
}

const App: React.FC = () => {
    const [messages, setMessages] = useState<Message[]>([]);


    const handleFileUpload = async (file: File) => {
        const isZip =
            file.name.toLowerCase().endsWith(".zip") ||
            file.type === "application/zip" ||
            file.type === "application/x-zip-compressed";

        if (!isZip) {
            setMessages((prev) => [
                ...prev,
                { id: Date.now(), type: "system", content: "Можно загружать только ZIP-файлы!" },
            ]);
            return;
        }

        setMessages((prev) => [
            ...prev,
            { id: Date.now(), type: "user", content: `📦 ${file.name}` },
            { id: Date.now() + 1, type: "system", content: "⏳ Обработка файла..." },
        ]);

        const formData = new FormData();
        formData.append("file", file);

        try {
            const res = await fetch("http://localhost:8000/process-image", {
            method: "POST",
            body: formData,
            });

            const data = await res.json();

            const fullText = 
                `${data.path_to_study}\n` +
                `${data.study_uid}\n` +
                `${data.series_uid}\n` +
                `${data.probability_of_pathology}\n` +
                `${data.pathology}\n` +
                `${data.processing_status}\n` +
                `${data.time_of_processing}`;
        

            // Заменяем "⏳" на пустое сообщение-результат
            const resultId = Date.now();
            setMessages((prev) => [
                ...prev.filter((m) => m.content !== "⏳ Обработка файла..."),
                { id: resultId, type: "system", content: "" },
            ]);


            let index = 0;
            const interval = setInterval(() => {
                if (index < fullText.length) {
                    const char = fullText.charAt(index);
                    setMessages((prev) =>
                        prev.map((m) =>
                            m.id === resultId ? { ...m, content: m.content + char } : m
                        )
                    );
                    index++;
                } else {
                    clearInterval(interval);
                }
            }, 40);
        } catch (error) {
            setMessages((prev) => [
                ...prev.filter((m) => m.content !== "⏳ Обработка файла..."),
                { id: Date.now(), type: "system", content: "❌ Ошибка при обработке файла" },
            ]);
        }
    };


    return (
        <div className="app">
            <h1 className="title">DICOM ANALYZER</h1>
            <Chat messages={messages} />
            <FileUpload onFileUpload={handleFileUpload} />
        </div>
    );
};

export default App;
