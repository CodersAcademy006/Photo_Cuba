# Photo Cuba v2 Runbook (Windows)

This guide is the single source to run the full pipeline end-to-end.

It covers:
- What the notebook does
- What happens if you run all notebook cells
- How to run with local photos
- How to run with Google Drive (cloud) photos
- Exact commands and expected outputs

---

## 1) First, understand your Cell 2 error

Error seen:

```text
ModuleNotFoundError: No module named 'google'
```

Meaning:
- You ran the notebook in a local Python kernel (VS Code/Jupyter), not in Google Colab runtime.
- `google.colab` exists only inside Colab.

Current notebook behavior (already adjusted):
- In Colab: Cell 2 mounts Drive and changes directory.
- Local kernel: Cell 2 skips Drive mount and continues.

So this error is expected in old Cell 2 code, and fixed in the updated notebook logic.

---

## 2) Put the right values in `.env`

Use these keys in `.env`:

```env
API_URL=http://localhost:8000
API_PORT=8000
INFERENCE_API_URL=http://localhost:8001

QDRANT_URL=https://5cdbcbdd-35a8-4f2f-a3cc-ec9fd251a77b.us-west-1-0.aws.cloud.qdrant.io
QDRANT_API_KEY=<YOUR_FULL_QDRANT_API_KEY>
SIM_THRESHOLD=0.45

GOOGLE_CREDENTIALS_PATH=credentials.json
DRIVE_ROOT_FOLDER_ID=1VwGSXAL6A6QZhyfg-3h3BfK_YI6loZkx

TELEGRAM_BOT_TOKEN=
```

Notes:
- `DRIVE_ROOT_FOLDER_ID` must NOT contain `?usp=drive_link`.
- Paste the full API key from Qdrant (copy button), not a truncated value.
- Keep `TELEGRAM_BOT_TOKEN` empty for now if you are not testing bot yet.

---

## 3) What each notebook cell does

Notebook: `colab/worker.ipynb`

Cell 1:
- Installs inference dependencies.
- Chooses CPU/GPU runtime packages based on CUDA availability.

Cell 2:
- If Colab: mounts Google Drive and `cd` into `/content/drive/MyDrive/TANTRA 2026`.
- If local: skips Drive mount and prints current working directory.

Cell 3:
- Loads InsightFace detector.
- Loads AdaFace IR-50 model (enabled).
- Downloads AdaFace checkpoint once, then uses cache.

Cell 4:
- Defines `get_embeddings()`:
  - detect faces
  - align crop
  - quality gate (blur + det score)
  - produce normalized 512-d embeddings

Cell 5:
- Starts inference FastAPI server on `0.0.0.0:8001`.
- Endpoints:
  - `GET /` health
  - `POST /embed` embedding extraction

Cell 6:
- Starts ngrok tunnel to expose port 8001 publicly.
- Prints `INFERENCE WORKER URL` for `.env`.

### If you run the complete notebook

Local kernel (VS Code):
- Worker runs on `http://localhost:8001`.
- You can skip Cell 6 unless you need a public URL.

Colab runtime:
- Worker runs in Colab VM on port 8001.
- Cell 6 gives public ngrok URL.
- You must paste that URL into local `.env` as `INFERENCE_API_URL`.

---

## 4) Run Mode A: Local inference worker + local photo folder

Use this first for fastest debugging.

## Step A1: Install dependencies

```powershell
cd <your-path>\photo_cuba_v2
pip install -r requirements.txt
```

Expected:
- Packages install without errors.

## Step A2: Start notebook worker

Open `colab/worker.ipynb` in VS Code and run cells in order:
1. Cell 1
2. Cell 2
3. Cell 3
4. Cell 4
5. Cell 5

Keep notebook running.

Quick health check (new terminal):

```powershell
python -c "import httpx; print(httpx.get('http://localhost:8001/').json())"
```

Expected:

```text
{'status': 'ok', 'device': 'cuda' or 'cpu', 'adaface': True}
```

## Step A3: Start laptop API (port 8000)

```powershell
cd <your-path>\photo_cuba_v2
uvicorn api.main:app --reload --port 8000
```

Expected:
- Uvicorn startup logs.
- API root at `http://127.0.0.1:8000`.

## Step A4: Smoke test worker reachability

```powershell
cd <your-path>\photo_cuba_v2
python tests/test_search.py --smoke
```

Expected key line:

```text
[PASS] Worker reachable  -> device=...  adaface=True
```

## Step A5: Create event

```powershell
python scripts/setup_event.py ^
  --event-id test_event_001 ^
  --event-name "Test Event" ^
  --date 2026-03-13 ^
  --photographer "Srijan"
```

Expected block includes:

```text
Event ID      : test_event_001
Qdrant        : ✅ Created   (or Already existed)
Drive Folder  : ✅ <folder_id>  (or warning if drive auth not done)
```

## Step A6: Batch index local photos

Set once in `.env` (already configured in your workspace):

```env
EVENT_PHOTOS_DIR=E:\Tantra 2k26\TANTRA 2K26 Memories\Images
```

Dry run first:

```powershell
python scripts/batch_index.py --event test_event_001 --dry-run
```

Expected:

```text
[DRY RUN] Would index N photos. Exiting.
```

Actual indexing:

```powershell
python scripts/batch_index.py --event test_event_001 --workers 3
```

Expected summary:

```text
Batch Index Complete — test_event_001
Total photos  : ...
✅ Indexed     : ...
⏭️  Skipped     : ...
❌ Failed      : ...
👤 Faces added : ...
```

## Step A7: Real search test with selfie

```powershell
python tests/test_search.py --event test_event_001 --selfie "E:\Tantra 2k26\TANTRA 2K26 Memories\Images\your_face.jpg"
```

Expected:
- Multiple PASS lines.
- Important section:

```text
[7] Real Search — test_event_001
[PASS] Search completed
[PASS] Results returned
Top match: ...
```

If Step A7 passes, core pipeline is working.

---

## 5) Run Mode B: Colab worker + Drive/Cloud photos

Use this when you want GPU inference in Colab and photos from Google Drive.

## Step B1: In Colab runtime, run notebook Cells 1 to 6

- Cell 2 must show:
  - Drive mounted
  - working dir `/content/drive/MyDrive/TANTRA 2026`
- Cell 6 prints ngrok URL:

```text
INFERENCE WORKER URL:
https://xxxx.ngrok-free.app
```

## Step B2: Update local `.env`

Set:

```env
INFERENCE_API_URL=https://xxxx.ngrok-free.app
```

## Step B3: Start local API

```powershell
cd <your-path>\photo_cuba_v2
uvicorn api.main:app --reload --port 8000
```

## Step B4: Index photos from Drive data

### Option B4.1 (recommended): Drive for Desktop path on laptop

If Drive for Desktop is installed and your folder is visible locally:

```powershell
python scripts/batch_index.py --event tantra_2026
```

Uses `EVENT_PHOTOS_DIR` from `.env` (e.g. `E:\Tantra 2k26\TANTRA 2K26 Memories\Images`).

### Option B4.2: Direct indexing inside Colab

Use a dedicated Colab indexing cell to read `/content/drive/MyDrive/TANTRA 2026/Assets_extracted` and upsert into Qdrant.

When to choose:
- Choose B4.2 if Drive for Desktop is not available on laptop.
- Choose B4.1 if you want to use existing project scripts directly.

## Step B5: Validate search

```powershell
python tests/test_search.py --event tantra_2026 --selfie "E:\Tantra 2k26\TANTRA 2K26 Memories\Images\your_face.jpg"
```

Expected:
- PASS for worker health
- PASS for collection exists + faces
- PASS in Real Search with returned matches

---

## 6) Should you reuse v1 embedding zip or re-infer?

Check zip contents first in Colab:

```python
import zipfile
z = zipfile.ZipFile('/content/drive/MyDrive/TANTRA 2026/Assets_extracted_emb....zip')
print(z.namelist()[:50])
```

Reuse v1 embeddings only if ALL are true:
- Embedding dimension is 512
- Embeddings are L2-normalized
- You have reliable mapping to original image path/URL
- Distance metric is cosine-compatible

If any of the above is unclear, re-run inference from scratch (safer).

---

## 7) Recommended terminal layout

Terminal 1:
- `uvicorn api.main:app --reload --port 8000`

Terminal 2:
- setup/index/test commands

Notebook kernel:
- worker notebook cells running (local or Colab)

Optional Terminal 3 (later):
- watcher

```powershell
python watcher/photo_watcher.py --event test_event_001 --folder D:\TestPhotos\Live
```

Expected when new photo appears:

```text
Indexed: photo.jpg (job_id=xxxx)
```

---

## 8) Common failures and quick fixes

Worker unreachable:
- Wrong `INFERENCE_API_URL` or notebook server not running.
- Fix: rerun notebook Cell 5 (and Cell 6 if using Colab ngrok).

No faces detected:
- Low quality / profile angle / dark image.
- Fix: use clear front-facing query image; keep AdaFace enabled.

Qdrant auth/connection error:
- Wrong URL/key or key truncated.
- Fix: copy full endpoint + full API key again from Qdrant dashboard.

Drive auth popup:
- First-time OAuth required.
- Fix: complete browser auth once; `token.json` will be reused.

---

## 9) Minimum success criteria

You are done when these 3 pass:
1. `python tests/test_search.py --smoke`
2. `python scripts/batch_index.py ...` adds faces
3. `python tests/test_search.py --event ... --selfie ...` returns real matches

After that, Telegram and watcher are just integration layers.
