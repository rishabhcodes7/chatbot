// pages/api/upload.ts
import { NextApiRequest, NextApiResponse } from "next";
import formidable, { File } from "formidable";
import fs from "fs";
import path from "path";

export const config = {
  api: {
    bodyParser: false,
  },
};

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse
) {
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  const uploadDir = path.join(process.cwd(), "docs");
  if (!fs.existsSync(uploadDir)) {
    fs.mkdirSync(uploadDir, { recursive: true });
  }

  const form = formidable({
    multiples: false,
    uploadDir,
    keepExtensions: true,
  });

  form.parse(req, (err, fields, files) => {
    if (err) return res.status(500).json({ error: "File upload failed" });

    console.log("Uploaded files:", files);

    const uploadedFile = Array.isArray(files.file) ? files.file[0] : files.file;
    if (!uploadedFile)
      return res.status(400).json({ error: "No file uploaded" });

    const newPath = path.join(
      uploadDir,
      uploadedFile.originalFilename || uploadedFile.newFilename
    );
    fs.renameSync(uploadedFile.filepath, newPath);

    return res.status(200).json({
      message: "File uploaded successfully",
      filename: uploadedFile.originalFilename,
    });
  });
}
