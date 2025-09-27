import React, { useRef, useState } from "react";

interface FileUploadProps {
  onFileUpload: (file: File) => void;
}

const FileUpload: React.FC<FileUploadProps> = ({ onFileUpload }) => {
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
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  return (
    <div
      className={`upload-area ${dragOver ? "upload-area--active" : ""}`}
      onDragOver={(e) => {
        e.preventDefault();
        setDragOver(true);
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
      />
      <button
        className="upload-btn"
        onClick={() => fileInputRef.current?.click()}
        title="Загрузить ZIP"
      >
        <svg width="16" height="16" viewBox="0 0 24 24" aria-hidden>
          <path fill="currentColor" d="M12 2L12 14M12 2L7 7M12 2L17 7M5 20h14" />
        </svg>
        <span>Загрузить ZIP</span>
      </button>
      <div className="upload-hint">или перетащите ZIP сюда</div>
    </div>
  );
};

export default FileUpload;
