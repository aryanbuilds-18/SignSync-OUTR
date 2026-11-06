const express = require("express");
const fs = require("fs");
const path = require("path");

const app = express();
const PORT = process.env.PORT || 3000;
const DATA = path.join(__dirname, "data");
const MODEL_FILE = path.join(DATA, "model.json");
const REPORT_FILE = path.join(DATA, "reports.json");
const CLASSES = ["A", "B", "C", "N"];
fs.mkdirSync(DATA, { recursive: true });

const read = (f, fallback) => { try { return JSON.parse(fs.readFileSync(f, "utf8")); } catch { return fallback; } };
const write = (f, v) => { fs.writeFileSync(f + ".tmp", JSON.stringify(v)); fs.renameSync(f + ".tmp", f); };

app.use(express.json({ limit: "50mb" }));

app.get("/api/health", (_q, r) => {
  const d = read(MODEL_FILE, { samples: [] });
  r.json({
    ok: true,
    samples: d.samples.length,
    handModelPresent: fs.existsSync(path.join(__dirname, "models", "hand_landmarker.task")),
  });
});

app.get("/api/model", (_q, r) => r.json(read(MODEL_FILE, { samples: [], refs: {} })));

app.put("/api/model", (q, r) => {
  const b = q.body;
  const valid = b && Array.isArray(b.samples) && b.samples.every(
    (s) => s && Array.isArray(s.f) && s.f.length === 129 && CLASSES.includes(s.l)
  );
  if (!valid) return r.status(400).json({ ok: false, error: "Invalid model format" });
  write(MODEL_FILE, { samples: b.samples, refs: b.refs || {} });
  r.json({ ok: true, samples: b.samples.length });
});

app.post("/api/report", (q, r) => {
  const all = read(REPORT_FILE, []);
  all.push(q.body);
  write(REPORT_FILE, all.slice(-100));
  r.json({ ok: true });
});
app.get("/api/reports", (_q, r) => r.json(read(REPORT_FILE, [])));

// Hand-tracking runtime and model are served locally, so no CDN is needed.
app.use("/vendor/mediapipe", express.static(path.join(__dirname, "node_modules/@mediapipe/tasks-vision")));
app.use("/models", express.static(path.join(__dirname, "models")));
app.use(express.static(path.join(__dirname, "public")));

app.listen(PORT, () => {
  console.log(`SignSync running at http://localhost:${PORT}`);
  if (!fs.existsSync(path.join(__dirname, "models", "hand_landmarker.task")))
    console.warn("WARNING: models/hand_landmarker.task is missing. Run: npm run setup (see README).");
});
