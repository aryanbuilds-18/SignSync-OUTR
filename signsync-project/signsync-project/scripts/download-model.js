const fs = require("fs");
const path = require("path");
const URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const dir = path.join(__dirname, "..", "models");
const out = path.join(dir, "hand_landmarker.task");

(async () => {
  fs.mkdirSync(dir, { recursive: true });
  if (fs.existsSync(out) && fs.statSync(out).size > 1e6) return console.log("Hand model already present.");
  console.log("Downloading hand model (~7 MB)...");
  const r = await fetch(URL);
  if (!r.ok) throw new Error("HTTP " + r.status);
  fs.writeFileSync(out, Buffer.from(await r.arrayBuffer()));
  console.log("Saved", out);
})().catch((e) => {
  console.error("\nCould not download the hand model: " + e.message);
  console.error("Download it manually (mobile data is fine):\n  " + URL);
  console.error("and save it as models/hand_landmarker.task, then run: npm start\n");
});
