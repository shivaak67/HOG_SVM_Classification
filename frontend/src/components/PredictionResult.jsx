import { CheckCircle2 } from "lucide-react";

export default function PredictionResult({ result }) {
  if (!result) {
    return (
      <div className="rounded-lg border border-slate-200 bg-white p-5">
        <p className="text-sm font-medium text-slate-500">Prediction</p>
        <p className="mt-3 text-slate-700">Upload an image to see the classifier result.</p>
      </div>
    );
  }

  const label = result.prediction.charAt(0).toUpperCase() + result.prediction.slice(1);
  const confidence =
    typeof result.confidence === "number"
      ? `${Math.round(result.confidence * 100)}%`
      : "Not available";

  return (
    <div className="rounded-lg border border-teal-200 bg-white p-5 shadow-sm">
      <div className="flex items-center gap-3 text-teal-700">
        <CheckCircle2 size={22} />
        <p className="text-sm font-semibold uppercase tracking-wide">Prediction ready</p>
      </div>
      <p className="mt-4 text-4xl font-bold text-slate-950">{label}</p>
      <p className="mt-2 text-slate-600">Confidence: {confidence}</p>
    </div>
  );
}
