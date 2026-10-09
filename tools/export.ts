// Dump frames of an ascii.rest piece as JSON lines: {text, color?}.
// usage: node tools/export.ts <piece> <fps> <seconds> [t0]
import { writeSync } from "node:fs";

const [piece, fps, seconds, t0 = "0"] = process.argv.slice(2);
const mod = await import(`./pieces/${piece}.ts`);
const frame = mod.default();
const { cols, rows, palette } = mod.meta;
const n = Math.round(Number(fps) * Number(seconds));
for (let i = 0; i < n; i++) {
  const t = Number(t0) + i / Number(fps);
  const color = palette ? new Uint8Array(cols * rows) : undefined;
  const text = frame(t, { color });
  const rec = { text, color: color ? Buffer.from(color).toString("base64") : undefined };
  writeSync(1, JSON.stringify(rec) + "\n");
}
