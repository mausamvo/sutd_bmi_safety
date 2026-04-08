import { mkdir, writeFile } from "fs/promises";
import path from "path";

const SAVE_ROOT = path.resolve(
  process.cwd(),
  "..",
  "sutd_bmi_safety_data"
);

const sanitizeSegment = (value) =>
  String(value || "capture")
    .replace(/[^a-zA-Z0-9._-]/g, "_")
    .replace(/_+/g, "_")
    .replace(/^_+|_+$/g, "")
    .slice(0, 120) || "capture";

const sanitizeSubfolderPath = (value) => {
  const raw = String(value || "adc_live_auto").replace(/\\\\/g, "/");
  const parts = raw
    .split("/")
    .map((part) => sanitizeSegment(part))
    .filter(Boolean);
  return parts.length ? parts.join("/") : "adc_live_auto";
};

const parseDataUrl = (dataUrl) => {
  const match = /^data:(.+);base64,(.+)$/.exec(dataUrl || "");
  if (!match) {
    throw new Error("Invalid image data URL.");
  }
  return Buffer.from(match[2], "base64");
};

export async function POST(request) {
  try {
    const {
      baseFilename,
      csv,
      chartImages = [],
      saveSubfolder = "adc_live_auto",
    } = await request.json();

    if (!csv || typeof csv !== "string") {
      return Response.json({ error: "CSV content is required." }, { status: 400 });
    }

    const safeBase = sanitizeSegment(baseFilename);
    const safeSubfolder = sanitizeSubfolderPath(saveSubfolder);
    const saveDir = path.join(SAVE_ROOT, safeSubfolder);
    await mkdir(saveDir, { recursive: true });

    const files = [];
    const csvName = `${safeBase}_adc.csv`;
    const csvPath = path.join(saveDir, csvName);
    await writeFile(csvPath, csv, "utf8");
    files.push(csvName);

    for (const image of chartImages) {
      if (!image?.dataUrl || !image?.name) continue;
      const pngBuffer = parseDataUrl(image.dataUrl);
      const imageName = `${safeBase}_${sanitizeSegment(image.name)}.png`;
      const imagePath = path.join(saveDir, imageName);
      await writeFile(imagePath, pngBuffer);
      files.push(imageName);
    }

    return Response.json({
      ok: true,
      saveDirectory: `src/testbed/sutd_bmi_safety_data/${safeSubfolder}`,
      files,
    });
  } catch (error) {
    return Response.json(
      { error: error?.message || "Failed to save capture." },
      { status: 500 }
    );
  }
}
