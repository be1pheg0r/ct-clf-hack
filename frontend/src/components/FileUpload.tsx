import React, { useRef } from "react";

interface FileUploadProps {
  onFileUpload: (file: File) => void;
}

const FileUpload: React.FC<FileUploadProps> = ({ onFileUpload }) => {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      onFileUpload(e.target.files[0]);
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      onFileUpload(e.dataTransfer.files[0]);
    }
  };

  return (
    <div
      className="upload-area"
      onDragOver={(e) => e.preventDefault()}
      onDrop={handleDrop}
    >
      <input
        type="file"
        accept=".zip"
        ref={fileInputRef}
        style={{ display: "none" }}
        onChange={handleFileChange}
      />
      <button onClick={() => fileInputRef.current?.click()}>
        Загрузить ZIP
      </button>
      <p>или перетащите файл сюда (плохо работает)</p>
    </div>
  );
};

export default FileUpload;
