"""
telegram_bot.py — Guest-facing bot. Deploy to Railway (free).

No FastAPI router. No local laptop. Calls Colab ngrok directly.

Environment variables needed on Railway:
  TELEGRAM_BOT_TOKEN  = from @BotFather
  INFERENCE_API_URL   = ngrok URL from Colab Cell 6
  QDRANT_URL          = Qdrant Cloud URL
  QDRANT_API_KEY      = Qdrant Cloud API key
  DEFAULT_EVENT_ID    = tantra_2026  (or set per-user flow)

Deploy to Railway:
  1. Push this file to a GitHub repo (just bot/ folder is enough)
  2. railway.app → New Project → Deploy from GitHub
  3. Add env vars in Railway dashboard
  4. Done — bot runs 24/7 for free
"""

import os
import asyncio
import httpx
import numpy as np
from loguru import logger
from dotenv import load_dotenv
from telegram import Update, InputMediaPhoto
from telegram.ext import (
    Application, CommandHandler,
    MessageHandler, filters, ContextTypes,
)
from qdrant_client import QdrantClient

load_dotenv()

BOT_TOKEN        = os.getenv("TELEGRAM_BOT_TOKEN", "")
INFERENCE_URL    = os.getenv("INFERENCE_API_URL", "")
QDRANT_URL       = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY   = os.getenv("QDRANT_API_KEY", "")
DEFAULT_EVENT_ID = os.getenv("DEFAULT_EVENT_ID", "tantra_2026")
SIM_THRESHOLD    = float(os.getenv("SIM_THRESHOLD", "0.45"))
TOP_K            = 2000

# Qdrant client (CPU-only operations — just vector search)
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# In-memory state: chat_id → event_id
# For production → replace with Redis or Supabase
_guest_events: dict[str, str] = {}


# ── Helpers ───────────────────────────────────────────────────

async def get_embedding(image_bytes: bytes, filename: str) -> list | None:
    """Call Colab ngrok /embed endpoint."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(
                f"{INFERENCE_URL}/embed",
                files={"image": (filename, image_bytes, "image/jpeg")},
                data={"augment": "true"},
            )
            r.raise_for_status()
            data = r.json()
            return data.get("embeddings", [])
    except httpx.ConnectError:
        logger.error(f"Colab worker unreachable: {INFERENCE_URL}")
        return None
    except Exception as e:
        logger.error(f"Embed error: {e}")
        return None


def search_qdrant(event_id: str, embeddings: list) -> list[dict]:
    """
    Search Qdrant directly from bot — no FastAPI middleman.
    Mean-of-top-3 aggregation per image for stable scoring.
    """
    if not embeddings:
        return []

    image_scores: dict[str, list[float]] = {}
    image_meta:   dict[str, dict]        = {}

    for emb in embeddings:
        hits = qdrant.search(
            collection_name=event_id,
            query_vector=emb,
            limit=TOP_K,
            score_threshold=SIM_THRESHOLD * 0.8,
        )
        for hit in hits:
            path  = hit.payload.get("image_path", "")
            score = float(hit.score)
            if path not in image_scores:
                image_scores[path] = []
                image_meta[path]   = hit.payload
            image_scores[path].append(score)

    results = []
    for path, scores in image_scores.items():
        scores.sort(reverse=True)
        agg = float(np.mean(scores[:3]))
        if agg < SIM_THRESHOLD:
            continue
        meta = image_meta[path]
        results.append({
            "image_path": path,
            "image_url":  meta.get("image_url", ""),
            "filename":   meta.get("filename", ""),
            "score":      round(agg, 4),
        })

    results.sort(key=lambda x: -x["score"])
    return results


# ── Command handlers ──────────────────────────────────────────

async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.message.chat_id)
    # Auto-join default event — no manual event ID step for guests
    _guest_events[chat_id] = DEFAULT_EVENT_ID

    await update.message.reply_text(
        "👋 Welcome to FotoOwl!\n\n"
        "Send me a clear selfie and I'll find all your\n"
        "photos from the event 📸\n\n"
        "Make sure your face is well-lit and front-facing."
    )


async def handle_photo(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    chat_id  = str(update.message.chat_id)
    event_id = _guest_events.get(chat_id, DEFAULT_EVENT_ID)

    # Step 1: acknowledge immediately
    msg = await update.message.reply_text("🔍 Searching for you...")

    # Step 2: download photo (highest resolution)
    photo      = update.message.photo[-1]
    tg_file    = await ctx.bot.get_file(photo.file_id)
    img_bytes  = bytes(await tg_file.download_as_bytearray())

    # Step 3: get embedding from Colab
    embeddings = await get_embedding(img_bytes, "selfie.jpg")

    if embeddings is None:
        await msg.edit_text(
            "⚠️ Our AI server is warming up. Try again in 30 seconds."
        )
        return

    if not embeddings:
        await msg.edit_text(
            "😕 No face detected in your photo.\n\n"
            "Tips:\n"
            "• Use a clear front-facing selfie\n"
            "• Good lighting (no backlighting)\n"
            "• Remove sunglasses / masks"
        )
        return

    # Step 4: search Qdrant
    results = search_qdrant(event_id, embeddings)

    if not results:
        await msg.edit_text(
            "📭 No matching photos found yet.\n\n"
            "The photographer may still be uploading.\n"
            "Try again in a few minutes!"
        )
        return

    await msg.edit_text(f"🎉 Found {len(results)} photo(s) of you!")

    # Step 5: send photos
    # Strategy: try media group first (10 at a time), fall back to individual
    urls   = [r["image_url"] for r in results if r["image_url"]]
    paths  = [r["filename"]  for r in results]

    if not urls:
        # No Drive URLs yet (photos indexed from local path only)
        # Send filenames so guest knows which photos to request
        photo_list = "\n".join(f"📷 {p}" for p in paths[:20])
        await update.message.reply_text(
            f"Found {len(results)} photos.\n\nPhoto names:\n{photo_list}"
            + (f"\n... and {len(results) - 20} more" if len(results) > 20 else "")
        )
        return

    sent = 0
    for i in range(0, min(len(urls), 30), 10):
        batch = urls[i:i + 10]
        try:
            media = [InputMediaPhoto(url) for url in batch]
            await ctx.bot.send_media_group(chat_id=chat_id, media=media)
            sent += len(batch)
        except Exception:
            # Fall back to sending one by one
            for url in batch:
                try:
                    await ctx.bot.send_photo(chat_id=chat_id, photo=url)
                    sent += 1
                except Exception as e:
                    logger.warning(f"Photo send failed: {e}")

    if len(results) > 30:
        await update.message.reply_text(
            f"Showing {sent} of {len(results)} matches.\n"
            "Send your selfie again to refresh — "
            "more photos may have been added!"
        )


async def handle_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Allow manually switching event by typing an event ID."""
    chat_id  = str(update.message.chat_id)
    text     = update.message.text.strip().lower().replace(" ", "_")

    try:
        existing = [c.name for c in qdrant.get_collections().collections]
        if text in existing:
            _guest_events[chat_id] = text
            await update.message.reply_text(
                f"✅ Switched to event: *{text}*\n"
                "Now send your selfie!",
                parse_mode="Markdown"
            )
        else:
            await update.message.reply_text(
                "Just send me a selfie and I'll find your photos! 📸"
            )
    except Exception:
        await update.message.reply_text(
            "Just send me a selfie and I'll find your photos! 📸"
        )


# ── Main ─────────────────────────────────────────────────────

def main():
    if not BOT_TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN not set")
    if not INFERENCE_URL:
        raise ValueError("INFERENCE_API_URL not set")
    if not QDRANT_URL:
        raise ValueError("QDRANT_URL not set")

    logger.info(f"Starting FotoOwl bot — event: {DEFAULT_EVENT_ID}")
    logger.info(f"Inference: {INFERENCE_URL}")

    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    logger.info("Bot running. Ctrl+C to stop.")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()