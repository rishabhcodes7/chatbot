// components/UploadPDF.tsx
"use client";

import { useState } from "react";

export default function UploadPDF() {
  const [uploading, setUploading] = useState(false);
  const [message, setMessage] = useState("");

  async function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    setUploading(true);
    setMessage("");

    try {
      const res = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      if (res.ok) {
        setMessage(`✅ ${data.message}`);
      } else {
        setMessage(`❌ ${data.error || "Upload failed"}`);
      }
    } catch (err) {
      setMessage("❌ Upload error");
    } finally {
      setUploading(false);
    }
  }

  return (
    <div className="flex flex-col items-center gap-2 p-4 border rounded-md">
      <label className="cursor-pointer px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700">
        {uploading ? "Uploading..." : "Upload PDF"}
        <input
          type="file"
          accept="application/pdf"
          onChange={handleFileChange}
          hidden
        />
      </label>
      {message && <p className="text-sm">{message}</p>}
    </div>
  );
}
