import { useState } from "react";
import { Stethoscope } from "lucide-react";
import UploadForm from "./components/UploadForm";
import ResultDisplay from "./components/ResultDisplay";
import ChatInterface from "./components/ChatInterface";
import "./App.css";

export default function App() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const hasResult = result && (result.structured_data || result.errors?.length);

  return (
    <div className="app">
      <header className="header">
        <Stethoscope size={28} className="header-icon" />
        <div>
          <h1 className="header-title">PrescriptoAI</h1>
          <p className="header-sub">
            AI-powered prescription analysis — OCR · Structure · Interpret · Risk
          </p>
        </div>
      </header>

      <main className="main">
        <div className="left-col">
          <UploadForm onResult={setResult} onLoading={setLoading} loading={loading} />
          {hasResult && <ResultDisplay result={result} />}
        </div>

        {hasResult && (
          <div className="right-col">
            <ChatInterface prescriptionContext={result?.structured_data} />
          </div>
        )}
      </main>
    </div>
  );
}
