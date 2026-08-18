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

TEXT_MODEL = "gemini-flash-lite-latest"
IMAGE_MODEL = "gemini-2.5-flash-image"

SYSTEM_INSTRUCTION = """ отвечай кратко, на агрессию отвечай агрессией """

# Ключевые слова для генерации картинок (можно добавить свои)
IMAGE_TRIGGERS = ["нарисуй", "сгенерируй картинку", "сгенерируй изображение", "draw", "generate image"]

# Ключевые слова, при которых стоит подключать поиск в интернете (расходует отдельный лимит)
SEARCH_TRIGGERS = [
    "погода", "новост", "курс", "сегодня", "сейчас", "актуальн", "последние",
    "кто выиграл", "результат матча", "цена", "стоимост",
]

MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5

# Память в рамках сессии: хранит последние сообщения для каждого пользователя.
# Формат: {user_id: [{"role": "user"/"model", "text": "..."}, ...]}
# Обнуляется при перезапуске бота (не сохраняется на диск).
conversation_history: dict[int, list[dict]] = {}
MAX_HISTORY_MESSAGES = 10  # сколько последних сообщений (вопрос+ответ) помнить на человека


def get_history(user_id: int) -> list[dict]:
    return conversation_history.get(user_id, [])


def add_to_history(user_id: int, role: str, text: str) -> None:
    history = conversation_history.setdefault(user_id, [])
    history.append({"role": role, "text": text})
    # обрезаем историю, чтобы не разрасталась бесконечно
    if len(history) > MAX_HISTORY_MESSAGES * 2:
        conversation_history[user_id] = history[-MAX_HISTORY_MESSAGES * 2:]


def build_contents(user_id: int, question: str) -> list[dict]:
    """Собирает историю переписки + новый вопрос в формате, понятном Gemini."""
    contents = []
    for entry in get_history(user_id):
        contents.append({
            "role": entry["role"],
            "parts": [{"text": entry["text"]}],
        })
    contents.append({
        "role": "user",
        "parts": [{"text": question}],
    })
    return contents


def generate_with_retry(model: str, contents: str, use_search: bool = False):
    """Пытается выполнить запрос к Gemini несколько раз, если сервер перегружен (503)."""
    last_error = None
    config_kwargs = {"system_instruction": SYSTEM_INSTRUCTION}
    if use_search:
        config_kwargs["tools"] = [types.Tool(google_search=types.GoogleSearch())]

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(**config_kwargs),
            )
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
        "/img <описание> — сгенерировать изображение\n"
        "/reset — забыть историю нашей переписки\n\n"
        "Также отвечаю, если меня упомянуть (@имя_бота) или ответить на моё сообщение. "
        "Я помню контекст последних сообщений в рамках беседы."
    )


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    conversation_history.pop(user_id, None)
    await update.message.reply_text("Память очищена, начинаем с чистого листа.")


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
    user_id = update.effective_user.id
    thinking_msg = await update.message.reply_text("Думаю...")
    try:
        needs_search = any(trigger in question.lower() for trigger in SEARCH_TRIGGERS)
        contents = build_contents(user_id, question)
        response = generate_with_retry(TEXT_MODEL, contents, use_search=needs_search)
        answer = response.text or "Не получилось сформулировать ответ, попробуй переформулировать вопрос."
        await thinking_msg.edit_text(answer)
        # сохраняем обмен в историю только при успешном ответе
        add_to_history(user_id, "user", question)
        add_to_history(user_id, "model", answer)
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
    application.add_handler(CommandHandler("reset", reset))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_mention_or_reply)
    )

    logger.info("Бот запущен, ожидаю сообщения...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
