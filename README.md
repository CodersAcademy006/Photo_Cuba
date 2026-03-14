# FotoOwl v2

AI-powered event photo search. The laptop-side API stores face embeddings in Qdrant, uploads photos to Google Drive, and asks a Colab worker to do the heavy face detection and embedding extraction.

## Main pieces

- `api/`: FastAPI endpoints for event creation, indexing, search, gallery, and live feed
- `storage/`: Google Drive upload helper
- `watcher/`: real-time folder watcher for incoming event photos
- `bot/`: Telegram bot for guest selfie search
- `scripts/`: local helpers for creating an event and batch-indexing a folder
- `colab/worker.ipynb`: GPU-side inference worker notebook

## Local setup

1. Create `.env` from `.env.example`.
2. Install dependencies with `pip install -r requirements.txt`.
3. Start the Colab worker and copy its public URL into `INFERENCE_API_URL`.
4. Run the API with `uvicorn api.main:app --reload --port 8000` from inside `fotoowl_v2`.

## Useful commands

- `python scripts/setup_event.py --event demo_event --name "Demo Event"`
- `python scripts/batch_index.py --event demo_event --folder ../test`
- `python scripts/batch_index.py --event demo_event` (uses `EVENT_PHOTOS_DIR` from `.env`)
- `python watcher/photo_watcher.py --event demo_event --folder ../test`
- `python watcher/photo_watcher.py --event demo_event --recursive` (uses `EVENT_PHOTOS_DIR`)
- `python bot/telegram_bot.py`
