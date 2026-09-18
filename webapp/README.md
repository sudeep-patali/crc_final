# CRC-ID — CRC Whole Slide Analyzer (Web Frontend)

A local web app: upload a `.svs` whole slide image in your browser, watch it
get tiled and classified live, and see the dominant invasion stage +
recommendation as a result.

## 1. Where this folder goes

This `webapp/` folder must sit **inside your existing `crc_project` folder**,
next to `config.py`, `models/`, and `utils/` — it imports those directly.

Your final local folder structure should look like:

```
crc_project/
├── config.py
├── models/
├── utils/
├── scripts/
├── checkpoints/              <-- put binary_best.pt + stage_best.pt HERE
│   ├── binary_best.pt
│   └── stage_best.pt
└── webapp/                   <-- this folder
    ├── app.py
    ├── inference_runner.py
    ├── requirements.txt
    ├── templates/index.html
    └── static/style.css, script.js
```

**Based on your screenshots**, you downloaded a `crc-project-models` folder
from Kaggle containing `checkpoints/binary_best.pt` and
`checkpoints/stage_best.pt`. Copy that entire `checkpoints` folder into your
`crc_project` folder (so it sits at `crc_project/checkpoints/`, matching the
diagram above) — that's the only manual step needed for the backend to find
your trained models, since `config.py`'s `CHECKPOINT_DIR` points there by
default.

## 2. Install dependencies

From inside `crc_project/webapp/`:

```bash
pip install -r requirements.txt --break-system-packages
```

You also need everything from the main project's `requirements.txt` (torch,
albumentations, opencv, openslide) already installed, since this reuses
`config.py`, `models/`, and `utils/` directly:

```bash
pip install -r ../requirements.txt --break-system-packages
```

Plus the OpenSlide system library (needed to read `.svs` files):
```bash
# Ubuntu/Debian:
sudo apt-get install openslide-tools
# macOS:
brew install openslide
```

## 3. Run it

```bash
cd crc_project/webapp
python app.py
```

Then open **http://localhost:5000** in your browser.

## 4. Use it

1. Drag & drop a `.svs` file onto the upload area (or click to browse)
2. Watch the live tile grid fill in as your binary + stage models classify
   each tissue patch in real time
3. See the result: dominant invasion stage, tumor fraction, per-stage
   distribution, and a recommendation (flagging "consult a pathologist" if
   any regions couldn't be confidently staged)

## Notes

- This runs on CPU if no GPU is available locally — for a large WSI
  (thousands of patches) that will be noticeably slower than on Kaggle's T4.
  It still works, just budget more time for the classification step.
- This is a local single-user development server (Flask's built-in server)
  — fine for your project demo, not intended for public deployment.
- Uploaded slides are saved to `webapp/uploads/` and per-job tiles are
  written to `webapp/jobs/<job_id>/` then cleaned up after each run.
