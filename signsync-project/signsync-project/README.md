# SignSync - ISL (A, B, C) communication bridge
Team The Innovatrix · PS-05 SignBridge

## Run in VS Code
1. Install Node.js 18 or newer.
2. Open this folder in VS Code, then open a terminal (Ctrl+`).
3. `npm install` (also downloads models/hand_landmarker.task)
4. `npm start`, then open http://localhost:3000 in Chrome or Edge.

If the download fails (college wifi), fetch the file from the URL printed in the terminal using mobile data
and save it as `models/hand_landmarker.task`.

## Use
1. Start camera. Train tab -> **Guided auto-train**. It records training and held-out test samples for A, B, C and Other.
2. Evaluate tab shows held-out accuracy and a confusion matrix.
3. Sign to text: spells letters and speaks them. Text to sign: shows the A/B/C reference signs.

## Backend
- GET/PUT `/api/model`: samples saved to `data/model.json`
- POST `/api/report`, GET `/api/reports`: evaluation history in `data/reports.json`
- GET `/api/health`

Scope: ISL letters A, B and C only. No claim of general ISL translation.
