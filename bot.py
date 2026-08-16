import os
import logging
import time
from io import BytesIO

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from google import genai
from google.genai import types

# ==== Настройка логов (чтобы видеть ошибки в консоли хостинга) ====
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ==== Токены берутся из переменных окружения (задаются на хостинге) ====
TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

client = genai.Client(api_key=GEMINI_API_KEY)

TEXT_MODEL = "gemini-flash-latest"
IMAGE_MODEL = "gemini-2.5-flash-image"

# Ключевые слова для генерации картинок (можно добавить свои)
IMAGE_TRIGGERS = ["нарисуй", "сгенерируй картинку", "сгенерируй изображение", "draw", "generate image"]

MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5


def generate_with_retry(model: str, contents: str):
    """Пытается выполнить запрос к Gemini несколько раз, если сервер перегружен (503)."""
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return client.models.generate_content(model=model, contents=contents)
        except Exception as e:
            last_error = e
            is_overloaded = "503" in str(e) or "UNAVAILABLE" in str(e)
            if is_overloaded and attempt < MAX_RETRIES:
                logger.warning(f"Сервер перегружен, попытка {attempt}/{MAX_RETRIES}, жду {RETRY_DELAY_SECONDS} сек...")
                time.sleep(RETRY_DELAY_SECONDS)
                continue
            raise
    raise last_error


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Привет! Я ИИ-бот.\n\n"
        "Команды:\n"
        "/ask <вопрос> — задать вопрос\n"
        "/img <описание> — сгенерировать изображение\n\n"
        "Также отвечаю, если меня упомянуть (@имя_бота) или ответить на моё сообщение."
    )


async def ask(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    question = " ".join(context.args)
    if not question:
        await update.message.reply_text("Напиши вопрос после команды, например:\n/ask сколько лет самой старой черепахе?")
        return
    await handle_question(update, question)


async def img(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    prompt = " ".join(context.args)
    if not prompt:
        await update.message.reply_text("Напиши описание картинки после команды, например:\n/img кот-космонавт в стиле акварели")
        return
    await handle_image(update, prompt)


async def handle_question(update: Update, question: str) -> None:
    thinking_msg = await update.message.reply_text("Думаю...")
    try:
        response = generate_with_retry(TEXT_MODEL, question)
        answer = response.text or "Не получилось сформулировать ответ, попробуй переформулировать вопрос."
        await thinking_msg.edit_text(answer)
    except Exception as e:
        logger.exception("Ошибка при обращении к Gemini (текст)")
        await thinking_msg.edit_text(f"Произошла ошибка при обращении к нейросети: {e}")


async def handle_image(update: Update, prompt: str) -> None:
    thinking_msg = await update.message.reply_text("Рисую...")
    try:
        response = generate_with_retry(IMAGE_MODEL, prompt)
        image_found = False
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                image_bytes = part.inline_data.data
                bio = BytesIO(image_bytes)
                bio.name = "image.png"
                await update.message.reply_photo(photo=bio)
                image_found = True
        await thinking_msg.delete()
        if not image_found:
            await update.message.reply_text("Не получилось сгенерировать изображение, попробуй другое описание.")
    except Exception as e:
        logger.exception("Ошибка при обращении к Gemini (картинка)")
        await thinking_msg.edit_text(f"Произошла ошибка при генерации изображения: {e}")


async def handle_mention_or_reply(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if message is None or message.text is None:
        return

    bot_username = context.bot.username
    is_mentioned = bot_username and f"@{bot_username}" in message.text
    is_reply_to_bot = (
        message.reply_to_message is not None
        and message.reply_to_message.from_user is not None
        and message.reply_to_message.from_user.id == context.bot.id
    )

    if not (is_mentioned or is_reply_to_bot):
        return

    text = message.text.replace(f"@{bot_username}", "").strip()
    if not text:
        return

    if any(trigger in text.lower() for trigger in IMAGE_TRIGGERS):
        await handle_image(update, text)
    else:
        await handle_question(update, text)


def main() -> None:
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("ask", ask))
    application.add_handler(CommandHandler("img", img))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_mention_or_reply)
    )

    logger.info("Бот запущен, ожидаю сообщения...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
