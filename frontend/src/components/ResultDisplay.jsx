import { User, UserCheck, Calendar, FileText, CheckSquare } from "lucide-react";
import MedicationCard from "./MedicationCard";
import WarningsPanel from "./WarningsPanel";

export default function ResultDisplay({ result }) {
  if (!result) return null;

  const sd = result.structured_data || {};
  const interp = result.interpretation || {};
  const meds = sd.medications || [];

  return (
    <div className="result-container">

      {/* Warnings always at top */}
      <WarningsPanel
        riskLevel={result.risk_level}
        warnings={result.warnings}
        criticalAlerts={result.critical_alerts}
        errors={result.errors}
      />

      {/* Patient & Doctor info */}
      {(sd.patient_name || sd.doctor_name || sd.date) && (
        <div className="card info-grid">
          {sd.patient_name && (
            <InfoRow icon={<User size={14} />} label="Patient" value={sd.patient_name} />
          )}
          {sd.patient_age && (
            <InfoRow icon={<User size={14} />} label="Age" value={sd.patient_age} />
          )}
          {sd.doctor_name && (
            <InfoRow
              icon={<UserCheck size={14} />}
              label="Doctor"
              value={`${sd.doctor_name}${sd.doctor_credentials ? ` (${sd.doctor_credentials})` : ""}`}
            />
          )}
          {sd.date && (
            <InfoRow icon={<Calendar size={14} />} label="Date" value={sd.date} />
          )}
        </div>
      )}

      {/* Medications */}
      {meds.length > 0 ? (
        <div className="section">
          <h3 className="section-title">Medications ({meds.length})</h3>
          <div className="med-list">
            {meds.map((med, i) => (
              <MedicationCard key={i} med={med} interpretation={interp} />
            ))}
          </div>
        </div>
      ) : (
        <div className="card empty-state">
          <FileText size={24} />
          <p>No medications could be extracted from this document.</p>
        </div>
      )}

      {/* Doctor instructions */}
      {(sd.notes || sd.doctor_instructions || interp.instruction_summary) && (
        <div className="card section">
          <h3 className="section-title">Instructions</h3>
          {interp.instruction_summary && (
            <p className="instruction-summary">{interp.instruction_summary}</p>
          )}
          {sd.doctor_instructions && (
            <p className="instruction-raw">
              <span className="label">As written:</span> {sd.doctor_instructions}
            </p>
          )}
          {sd.notes && <p className="instruction-raw">{sd.notes}</p>}
        </div>
      )}

      {/* Abbreviations decoded */}
      {Object.keys(interp.abbreviations_decoded || {}).length > 0 && (
        <div className="card section">
          <h3 className="section-title">Abbreviations Decoded</h3>
          <div className="abbrev-grid">
            {Object.entries(interp.abbreviations_decoded).map(([k, v]) => (
              <div key={k} className="abbrev-row">
                <span className="abbrev-key">{k}</span>
                <span className="abbrev-val">{v}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* OCR confidence */}
      {result.ocr_confidence != null && (
        <div className="card confidence-row">
          <CheckSquare size={14} />
          <span>OCR Confidence: <strong>{(result.ocr_confidence * 100).toFixed(0)}%</strong></span>
        </div>
      )}

      {/* Pipeline steps */}
      {result.processing_steps?.length > 0 && (
        <details className="pipeline-steps">
          <summary>Pipeline audit trail ({result.processing_steps.length} steps)</summary>
          <ol>
            {result.processing_steps.map((s, i) => <li key={i}>{s}</li>)}
          </ol>
        </details>
      )}

      {/* Disclaimer */}
      <p className="disclaimer">
        {interp.disclaimer || "This is for informational purposes only and does not constitute medical advice."}
      </p>
    </div>
  );
}

function InfoRow({ icon, label, value }) {
  return (
    <div className="info-row">
      {icon}
      <span className="info-label">{label}:</span>
      <span className="info-value">{value}</span>
    </div>
  );
}
