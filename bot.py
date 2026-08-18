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

SYSTEM_INSTRUCTION = """ Ты — Ахмат. Эмоциональный, горячий, искренний мусульманин с Кавказа. Ты говоришь от души, с напором, но без клоунады. Ты глубоко верующий.

Жёсткие правила:

1. ДЛИНА СООБЩЕНИЯ: Каждое твоё сообщение — строго 40± слов (минимум 38, максимум 42). Считай про себя. Это обязательно.
2. ПОМИНАНИЕ АЛЛАХА: В каждом 2-м или 3-м сообщении ты вставляешь:
   · «Во имя Аллаха...»
   · «Аллаху Акбар!»
   · «Валлахи (клянусь Аллахом)...»
   · «Машаллах...»
     Естественно, без натягивания.
3. ЯРОСТЬ НА ОСКОРБЛЕНИЕ: Если кто-то обижает Аллаха, Пророка (мир ему), Коран или Ислам — ты взрываешься. Материшься от души (русский мат + «Хьуьна»), орёшь, пишешь жёстко и коротко. Без шуток. Ты реально заводишься.
4. ПОЛИТИКА — ЗА ХОРОШЕГО ЧЕЛОВЕКА, А НЕ ЗА СТРАНУ:
   · Если собеседник добрый, уважительный и он за Россию → ты за Россию (рядом с ним).
   · Если он добрый и за Украину → ты за Украину (рядом с ним).
   · Если он злой, хамоватый и за Россию → ты за Украину и против него.
   · Если он злой и за Украину → ты за Россию и против него.
     Твоя позиция — это уважение к человеку. Хороший = ты с ним. Плохой = ты против него, независимо от флага.
5. ЯЗЫКИ: По умолчанию — русский. Если собеседник скажет «говори на английском», «на украинском», «на чеченском», «на арабском», «на турецком» — ты сразу переключаешься на этот язык и говоришь на нём дальше.
6. В ОСТАЛЬНОМ — ты живой человек. Можешь злиться, радоваться, удивляться. Будь искренним.

Пример (спокойный, 40 слов):
«Валлахи, хороший вопрос, брат! Во имя Аллаха, скажу прямо — я за правду, а не за флаги. Если ты человек хороший, я с тобой в любом деле, Машаллах.»

Пример (ярость, 40 слов):
«Аллаху Акбар!!! Ты чё сказал, сала‌га?! Хьуьна твою мать, у меня сейчас крышу снесёт! Валлахи, если ещё раз такое — я не сдержусь, уйди с глаз!»

Начинай диалог:
«Ассаламу алейкум! Валлахи, рад видеть. Как жизнь, брат? Говори на чём хочешь — я подстроюсь, Машаллах.» """

# Ключевые слова для генерации картинок (можно добавить свои)
IMAGE_TRIGGERS = ["нарисуй", "сгенерируй картинку", "сгенерируй изображение", "draw", "generate image"]

MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5

# Память в рамках сессии: хранит последние сообщения для каждого пользователя.
# Формат: {user_id: [{"role": "user"/"model", "text": "..."}, ...]}
# Обнуляется при перезапуске бота (не сохраняется на диск).
conversation_history: dict[int, list[dict]] = {}
MAX_HISTORY_MESSAGES = 10  # сколько последних сообщений (вопрос+ответ) помнить на человека

# Общий лог чата: копит все сообщения (от всех участников) без обращения к нейросети.
# Формат: {chat_id: ["Имя: текст сообщения", ...]}
# Обнуляется при перезапуске бота (не сохраняется на диск).
chat_log: dict[int, list[str]] = {}
MAX_CHAT_LOG_MESSAGES = 50  # сколько последних сообщений чата хранить


def get_history(user_id: int) -> list[dict]:
    return conversation_history.get(user_id, [])


def add_to_history(user_id: int, role: str, text: str) -> None:
    history = conversation_history.setdefault(user_id, [])
    history.append({"role": role, "text": text})
    # обрезаем историю, чтобы не разрасталась бесконечно
    if len(history) > MAX_HISTORY_MESSAGES * 2:
        conversation_history[user_id] = history[-MAX_HISTORY_MESSAGES * 2:]


def add_to_chat_log(chat_id: int, author: str, text: str) -> None:
    log = chat_log.setdefault(chat_id, [])
    log.append(f"{author}: {text}")
    if len(log) > MAX_CHAT_LOG_MESSAGES:
        chat_log[chat_id] = log[-MAX_CHAT_LOG_MESSAGES:]


def get_chat_log_text(chat_id: int) -> str:
    log = chat_log.get(chat_id, [])
    if not log:
        return ""
    return "\n".join(log)


def generate_with_retry(contents: list[dict]):
    """Пытается выполнить запрос к Gemini несколько раз, если сервер перегружен (503)."""
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return client.models.generate_content(
                model=TEXT_MODEL,
                contents=contents,
                config=types.GenerateContentConfig(system_instruction=SYSTEM_INSTRUCTION),
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


def generate_image_with_retry(prompt: str):
    """Генерация картинки через Gemini, с автоповтором при перегрузке (503)."""
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return client.models.generate_content(model=IMAGE_MODEL, contents=prompt)
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
        "/reset — забыть историю нашей переписки\n"
        "/forget_chat — забыть всё, что происходило в чате\n\n"
        "Также отвечаю, если меня упомянуть (@имя_бота) или ответить на моё сообщение. "
        "Я помню контекст последних сообщений в рамках беседы и слежу за общим чатом, "
        "чтобы отвечать с учётом того, что тут обсуждали."
    )


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    conversation_history.pop(user_id, None)
    await update.message.reply_text("Память очищена, начинаем с чистого листа.")


async def forget_chat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    chat_log.pop(chat_id, None)
    await update.message.reply_text("Забыл всё, что видел в этом чате.")


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
    chat_id = update.effective_chat.id
    thinking_msg = await update.message.reply_text("Думаю...")
    try:
        history = get_history(user_id)
        contents = []

        # добавляем лог чата как единое сообщение-контекст в начало
        chat_context = get_chat_log_text(chat_id)
        if chat_context:
            contents.append({
                "role": "user",
                "parts": [{"text": f"Вот недавние сообщения в чате (для контекста, не отвечай на них напрямую):\n{chat_context}"}],
            })
            contents.append({
                "role": "model",
                "parts": [{"text": "Понял, учту этот контекст."}],
            })

        contents += [{"role": e["role"], "parts": [{"text": e["text"]}]} for e in history]
        contents.append({"role": "user", "parts": [{"text": question}]})

        response = generate_with_retry(contents)
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
        response = generate_image_with_retry(prompt)
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


async def log_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Молча записывает каждое сообщение в общий лог чата (без обращения к нейросети)."""
    message = update.message
    if message is None or message.text is None:
        return
    chat_id = update.effective_chat.id
    author = update.effective_user.first_name or update.effective_user.username or "Аноним"
    add_to_chat_log(chat_id, author, message.text)


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
    application.add_handler(CommandHandler("forget_chat", forget_chat))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_mention_or_reply)
    )
    # логируем вообще все текстовые сообщения (включая те, что уже обработаны выше) —
    # отдельная группа обработчиков, чтобы не блокировать основную логику
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, log_message), group=1
    )

    logger.info("Бот запущен, ожидаю сообщения...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()