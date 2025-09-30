// src/components/FileUpload.tsx
import React, { useRef, useState } from "react";

interface FileUploadProps {
  onFileUpload: (file: File) => void;
  disabled?: boolean;
}

const FileUpload: React.FC<FileUploadProps> = ({ onFileUpload, disabled = false }) => {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [dragOver, setDragOver] = useState(false);

  const handleFile = (file: File | null) => {
    if (!file) return;
    onFileUpload(file);
  };

  const onChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) handleFile(e.target.files[0]);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragOver(false);
    if (disabled) return;
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  return (
    <div
      className={`upload-area ${dragOver ? "upload-area--active" : ""} ${disabled ? "upload-area--disabled" : ""}`}
      onDragOver={(e) => {
        e.preventDefault();
        if (!disabled) setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={handleDrop}
    >
      <input
        ref={fileInputRef}
        type="file"
        accept=".zip,application/zip,application/x-zip-compressed"
        style={{ display: "none" }}
        onChange={onChange}
        disabled={disabled}
      />
      <button
        className="upload-btn"
        onClick={() => !disabled && fileInputRef.current?.click()}
        title={disabled ? "Загрузка заблокирована: обработка в процессе" : "Загрузить ZIP"}
        disabled={disabled}
      >

          <svg width="16" height="16" viewBox="0 0 24 24" aria-hidden>
            <path stroke="currentColor" d="M12 2L12 14M12 2L7 7M12 2L17 7M5 20h14" />
          </svg>
        
        <span>{disabled ? "Загрузка заблокирована" : "Загрузить ZIP"}</span>
      </button>
      <div className="upload-hint">{disabled ? "Подождите, идёт обработка..." : "или перетащите ZIP сюда"}</div>
    </div>
  );
};

export default FileUpload;
