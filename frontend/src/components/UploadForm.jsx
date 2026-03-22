import { useState, useRef } from "react";
import { Upload, FileText, Image } from "lucide-react";
import { uploadPrescription, analyzeText } from "../api";

export default function UploadForm({ onResult, onLoading, loading }) {
  const [tab, setTab] = useState("image");
  const [text, setText] = useState("");
  const [dragOver, setDragOver] = useState(false);
  const fileRef = useRef();

  const handleFile = (file) => {
    if (!file) return;
    onLoading(true);
    onResult(null);
    uploadPrescription(file)
      .then((res) => onResult(res.data))
      .catch((err) =>
        onResult({ errors: [err.response?.data?.detail || "Upload failed"] })
      )
      .finally(() => onLoading(false));
  };

  const handleText = () => {
    if (!text.trim()) return;
    onLoading(true);
    onResult(null);
    analyzeText(text)
      .then((res) => onResult(res.data))
      .catch((err) =>
        onResult({ errors: [err.response?.data?.detail || "Analysis failed"] })
      )
      .finally(() => onLoading(false));
  };

  return (
    <div className="card">
      {/* Tabs */}
      <div className="tab-row">
        <button
          className={`tab ${tab === "image" ? "tab-active" : ""}`}
          onClick={() => setTab("image")}
        >
          <Image size={15} /> Image Upload
        </button>
        <button
          className={`tab ${tab === "text" ? "tab-active" : ""}`}
          onClick={() => setTab("text")}
        >
          <FileText size={15} /> Plain Text
        </button>
      </div>

      {tab === "image" ? (
        <div
          className={`dropzone ${dragOver ? "dropzone-active" : ""}`}
          onClick={() => fileRef.current.click()}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragOver(false);
            handleFile(e.dataTransfer.files[0]);
          }}
        >
          <Upload size={32} className="upload-icon" />
          <p className="dropzone-text">
            Drag & drop a prescription image, or <span className="link">browse</span>
          </p>
          <p className="dropzone-hint">JPEG · PNG · WEBP · HEIC</p>
          <input
            ref={fileRef}
            type="file"
            accept="image/jpeg,image/png,image/webp,image/heic,.jfif"
            style={{ display: "none" }}
            onChange={(e) => handleFile(e.target.files[0])}
          />
        </div>
      ) : (
        <div className="text-input-area">
          <textarea
            className="textarea"
            rows={6}
            placeholder="Paste or type prescription text here..."
            value={text}
            onChange={(e) => setText(e.target.value)}
          />
          <button
            className="btn-primary"
            onClick={handleText}
            disabled={loading || !text.trim()}
          >
            Analyze Text
          </button>
        </div>
      )}

      {loading && (
        <div className="loading-bar">
          <div className="loading-pulse" />
          <span>Processing prescription through AI pipeline...</span>
        </div>
      )}
    </div>
  );
}
