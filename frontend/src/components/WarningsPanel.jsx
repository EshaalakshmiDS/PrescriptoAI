import { AlertTriangle, AlertCircle, Info, CheckCircle } from "lucide-react";

const LEVEL_CONFIG = {
  high:   { icon: AlertCircle,   cls: "risk-high",   label: "High Risk" },
  medium: { icon: AlertTriangle, cls: "risk-medium",  label: "Medium Risk" },
  low:    { icon: CheckCircle,   cls: "risk-low",     label: "Low Risk" },
};

export default function WarningsPanel({ riskLevel, warnings, criticalAlerts, errors }) {
  const cfg = LEVEL_CONFIG[riskLevel?.toLowerCase()] || LEVEL_CONFIG.low;
  const Icon = cfg.icon;

  const hasIssues =
    criticalAlerts?.length || warnings?.length || errors?.length;

  if (!riskLevel && !hasIssues) return null;

  return (
    <div className="warnings-panel">
      {riskLevel && (
        <div className={`risk-badge ${cfg.cls}`}>
          <Icon size={16} />
          <span>{cfg.label}</span>
        </div>
      )}

      {criticalAlerts?.length > 0 && (
        <AlertList
          icon={<AlertCircle size={14} />}
          title="Critical Alerts"
          items={criticalAlerts}
          cls="alert-critical"
        />
      )}

      {warnings?.length > 0 && (
        <AlertList
          icon={<AlertTriangle size={14} />}
          title="Warnings"
          items={warnings}
          cls="alert-warning"
        />
      )}

      {errors?.length > 0 && (
        <AlertList
          icon={<Info size={14} />}
          title="Processing Notes"
          items={errors}
          cls="alert-info"
        />
      )}
    </div>
  );
}

function AlertList({ icon, title, items, cls }) {
  return (
    <div className={`alert-list ${cls}`}>
      <p className="alert-title">
        {icon} {title}
      </p>
      <ul>
        {items.map((item, i) => (
          <li key={i}>{item}</li>
        ))}
      </ul>
    </div>
  );
}
