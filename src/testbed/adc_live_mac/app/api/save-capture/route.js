// Saves one capture (CSV + optional chart PNGs) under SAVE_ROOT/<subfolder>/.
// From the macOS-optimised adc-live: validates everything before writing,
// never overwrites a recording, refuses path tricks and symlinked folders,
// and only accepts requests from this page.
import fs from "node:fs/promises";
import path from "node:path";

export const runtime = "nodejs";

// server.js sets SAVE_ROOT (default new/data/recordings) and its display label.
const SAVE_ROOT = path.resolve(process.env.SAVE_ROOT || path.join(process.cwd(), "..", "..", "data", "recordings"));
const SAVE_ROOT_LABEL = process.env.SAVE_ROOT_LABEL || SAVE_ROOT;
const MAX_BODY_BYTES = 32 * 1024 * 1024;
const PNG_SIGNATURE = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);

function invalid(message, status = 400) {
  return Object.assign(new Error(message), { status });
}

function sanitizeSegment(value) {
  const clean = String(value || "").replace(/[^a-zA-Z0-9._-]/g, "_")
    .replace(/_+/g, "_").replace(/^_+|_+$/g, "").slice(0, 120);
  if (!clean || clean === "." || clean === "..") throw invalid("Invalid filename or folder segment.");
  return clean;
}

function sanitizeSubfolder(value = "adc_live_auto") {
  if (typeof value !== "string" || value.length > 512) throw invalid("Invalid capture subfolder.");
  const parts = value.replace(/\\/g, "/").split("/");
  if (parts.length > 8 || parts.some((part) => !part)) throw invalid("Use a relative capture subfolder.");
  return parts.map(sanitizeSegment);
}

function parsePng(dataUrl) {
  if (typeof dataUrl !== "string" || dataUrl.length > 8 * 1024 * 1024) throw invalid("Chart image is too large.", 413);
  const match = /^data:image\/png;base64,([A-Za-z0-9+/]+={0,2})$/.exec(dataUrl);
  if (!match || match[1].length % 4 !== 0) throw invalid("Chart image must be a PNG data URL.");
  const bytes = Buffer.from(match[1], "base64");
  if (!bytes.subarray(0, 8).equals(PNG_SIGNATURE)) throw invalid("Chart image has no PNG signature.");
  return bytes;
}

async function readJson(request) {
  const declaredLength = Number(request.headers.get("content-length"));
  if (declaredLength > MAX_BODY_BYTES) throw invalid("Capture request is too large.", 413);
  if (!request.body) throw invalid("Capture request body is required.");
  const reader = request.body.getReader();
  const chunks = [];
  let length = 0;
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      length += value.byteLength;
      if (length > MAX_BODY_BYTES) {
        await reader.cancel();
        throw invalid("Capture request is too large.", 413);
      }
      chunks.push(Buffer.from(value));
    }
  } finally {
    reader.releaseLock();
  }
  try { return JSON.parse(Buffer.concat(chunks).toString("utf8")); }
  catch { throw invalid("Capture request is not valid JSON."); }
}

async function saveCapture(payload, saveRoot = SAVE_ROOT, rootLabel = SAVE_ROOT_LABEL) {
  if (!payload || typeof payload !== "object") throw invalid("Invalid capture request.");
  const { baseFilename, csv, chartImages = [], saveSubfolder } = payload;
  if (typeof csv !== "string" || !csv.trim()) throw invalid("CSV content is required.");
  if (Buffer.byteLength(csv) > 16 * 1024 * 1024) throw invalid("CSV is too large.", 413);
  if (!Array.isArray(chartImages) || chartImages.length > 4) throw invalid("At most four chart images are allowed.");
  const base = sanitizeSegment(baseFilename || "capture");
  const parts = sanitizeSubfolder(saveSubfolder);
  // Validate every output before creating any file.
  const outputs = [{ name: `${base}_adc.csv`, bytes: Buffer.from(csv) }];
  for (const image of chartImages) {
    if (!image || typeof image.name !== "string") throw invalid("Invalid chart image.");
    outputs.push({ name: `${base}_${sanitizeSegment(image.name)}.png`, bytes: parsePng(image.dataUrl) });
  }
  if (new Set(outputs.map((item) => item.name)).size !== outputs.length) throw invalid("Duplicate output filenames.");

  await fs.mkdir(saveRoot, { recursive: true });
  let saveDir = await fs.realpath(saveRoot);
  // Refuse symlinked subfolders rather than following them out of the data root.
  for (const part of parts) {
    saveDir = path.join(saveDir, part);
    await fs.mkdir(saveDir).catch((error) => { if (error.code !== "EEXIST") throw error; });
    if (!(await fs.lstat(saveDir)).isDirectory()) throw invalid("Capture subfolder is not a regular directory.");
  }

  const created = [];
  try {
    for (const output of outputs) {
      const filename = path.join(saveDir, output.name);
      const handle = await fs.open(filename, "wx"); // never overwrite a recording
      created.push(filename);
      try {
        await handle.writeFile(output.bytes);
        await handle.sync();
      } finally { await handle.close(); }
    }
  } catch (error) {
    await Promise.all(created.map((filename) => fs.unlink(filename).catch(() => {})));
    if (error.code === "EEXIST") throw invalid("A capture with this filename already exists. Choose a new filename.", 409);
    throw error;
  }
  return { ok: true, saveDirectory: `${rootLabel}/${parts.join("/")}`, files: outputs.map((item) => item.name) };
}

async function handleSaveRequest(request, saveRoot = SAVE_ROOT, rootLabel = SAVE_ROOT_LABEL) {
  try {
    const origin = request.headers.get("origin");
    if (origin) {
      let sameOrigin = false;
      try {
        const source = new URL(origin);
        const target = new URL(request.url);
        // The custom server can normalise request.url; the browser-facing
        // host is the incoming Host header.
        sameOrigin = source.host === (request.headers.get("host") || target.host) &&
          source.protocol === target.protocol;
      } catch { sameOrigin = false; }
      if (!sameOrigin) throw invalid("Cross-origin save requests are not allowed.", 403);
    }
    if (!(request.headers.get("content-type") || "").toLowerCase().startsWith("application/json")) {
      throw invalid("Use application/json for capture requests.", 415);
    }
    return Response.json(await saveCapture(await readJson(request), saveRoot, rootLabel));
  } catch (error) {
    return Response.json({ error: error.status ? error.message : "Failed to save capture. Check disk space and folder permissions." },
      { status: error.status || 500 });
  }
}

export async function POST(request) {
  return handleSaveRequest(request);
}
