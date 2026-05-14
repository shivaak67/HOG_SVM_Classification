import { ImagePlus, X } from "lucide-react";

export default function ImageUploader({
  previewUrl,
  isDragging,
  onBrowse,
  onClear,
  onDrop,
  onDragOver,
  onDragLeave,
}) {
  return (
    <div
      onDrop={onDrop}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      className={[
        "relative flex min-h-[360px] items-center justify-center rounded-lg border-2 border-dashed bg-white p-6 transition",
        isDragging ? "border-teal-500 bg-teal-50" : "border-slate-300",
      ].join(" ")}
    >
      {previewUrl ? (
        <>
          <button
            type="button"
            onClick={onClear}
            className="absolute right-4 top-4 rounded-full bg-white p-2 text-slate-600 shadow-sm ring-1 ring-slate-200 hover:text-slate-900"
            aria-label="Remove selected image"
            title="Remove selected image"
          >
            <X size={18} />
          </button>
          <img
            src={previewUrl}
            alt="Selected upload preview"
            className="max-h-[320px] max-w-full rounded-md object-contain"
          />
        </>
      ) : (
        <div className="text-center">
          <div className="mx-auto mb-5 flex h-16 w-16 items-center justify-center rounded-full bg-teal-100 text-teal-700">
            <ImagePlus size={30} />
          </div>
          <p className="text-lg font-semibold text-slate-900">Drop an image here</p>
          <p className="mt-2 text-sm text-slate-500">JPG, PNG, BMP, or WebP</p>
          <button
            type="button"
            onClick={onBrowse}
            className="mt-6 rounded-md bg-slate-900 px-4 py-2 text-sm font-medium text-white hover:bg-slate-800"
          >
            Choose image
          </button>
        </div>
      )}
    </div>
  );
}
