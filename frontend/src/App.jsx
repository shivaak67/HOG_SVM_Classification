import { useRef, useState } from "react";
import { Loader2, Wand2 } from "lucide-react";

import ImageUploader from "./components/ImageUploader.jsx";
import PredictionResult from "./components/PredictionResult.jsx";
import { predictImage } from "./services/api.js";

const VALID_IMAGE_TYPES = ["image/jpeg", "image/png", "image/bmp", "image/webp"];

export default function App() {
  const fileInputRef = useRef(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  function setImageFile(file) {
    setError("");
    setResult(null);

    if (!file) {
      return;
    }

    if (!VALID_IMAGE_TYPES.includes(file.type)) {
      setSelectedFile(null);
      setPreviewUrl("");
      setError("Please choose a JPG, PNG, BMP, or WebP image.");
      return;
    }

    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
  }

  function clearImage() {
    setSelectedFile(null);
    setPreviewUrl("");
    setResult(null);
    setError("");
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  }

  async function handlePredict() {
    if (!selectedFile) {
      setError("Upload an image before running prediction.");
      return;
    }

    setIsLoading(true);
    setError("");

    try {
      const prediction = await predictImage(selectedFile);
      setResult(prediction);
    } catch (requestError) {
      setResult(null);
      setError(requestError.message);
    } finally {
      setIsLoading(false);
    }
  }

  function handleDrop(event) {
    event.preventDefault();
    setIsDragging(false);
    setImageFile(event.dataTransfer.files?.[0]);
  }

  return (
    <main className="min-h-screen px-4 py-8 sm:px-6 lg:px-8">
      <div className="mx-auto max-w-5xl">
        <header className="mb-8">
          <p className="text-sm font-semibold uppercase tracking-wide text-teal-700">
            HOG + SVM image classifier
          </p>
          <h1 className="mt-2 text-3xl font-bold text-slate-950 sm:text-4xl">
            Cat vs Dog Prediction
          </h1>
          <p className="mt-3 max-w-2xl text-slate-600">
            Upload a pet image and run it through your trained HOG feature extractor and SVM model.
          </p>
        </header>

        <div className="grid gap-6 lg:grid-cols-[1.5fr_1fr]">
          <section>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/jpeg,image/png,image/bmp,image/webp"
              className="hidden"
              onChange={(event) => setImageFile(event.target.files?.[0])}
            />

            <ImageUploader
              previewUrl={previewUrl}
              isDragging={isDragging}
              onBrowse={() => fileInputRef.current?.click()}
              onClear={clearImage}
              onDrop={handleDrop}
              onDragOver={(event) => {
                event.preventDefault();
                setIsDragging(true);
              }}
              onDragLeave={() => setIsDragging(false)}
            />
          </section>

          <aside className="space-y-4">
            <button
              type="button"
              onClick={handlePredict}
              disabled={!selectedFile || isLoading}
              className="flex w-full items-center justify-center gap-2 rounded-md bg-teal-700 px-5 py-3 font-semibold text-white hover:bg-teal-800 disabled:cursor-not-allowed disabled:bg-slate-300"
            >
              {isLoading ? <Loader2 className="animate-spin" size={20} /> : <Wand2 size={20} />}
              {isLoading ? "Predicting..." : "Predict"}
            </button>

            {error && (
              <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-700">
                {error}
              </div>
            )}

            <PredictionResult result={result} />
          </aside>
        </div>
      </div>
    </main>
  );
}
