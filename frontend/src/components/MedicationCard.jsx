import { Pill } from "lucide-react";

export default function MedicationCard({ med, interpretation }) {
  const interp = interpretation?.medication_interpretations?.find(
    (m) => m.medication_name?.toLowerCase() === med.name?.toLowerCase()
  );

  return (
    <div className="med-card">
      <div className="med-header">
        <Pill size={16} className="med-icon" />
        <span className="med-name">{med.name || "Unknown Medication"}</span>
      </div>

      <div className="med-grid">
        <Detail label="Dosage" value={med.dosage} />
        <Detail label="Frequency" value={med.frequency} />
        <Detail label="Duration" value={med.duration} />
        <Detail label="Route" value={med.route} />
      </div>

      {med.instructions && (
        <p className="med-instructions">
          <span className="label">Instructions:</span> {med.instructions}
        </p>
      )}

      {interp && (
        <div className="med-interpretation">
          <p className="interp-use">
            <span className="label">Common use:</span> {interp.common_use}
          </p>
          <p className="interp-dosage">
            <span className="label">In plain terms:</span> {interp.dosage_explanation}
          </p>
          {interp.important_notes && (
            <p className="interp-notes">{interp.important_notes}</p>
          )}
        </div>
      )}
    </div>
  );
}

function Detail({ label, value }) {
  if (!value) return null;
  return (
    <div className="detail">
      <span className="detail-label">{label}</span>
      <span className="detail-value">{value}</span>
    </div>
  );
}
