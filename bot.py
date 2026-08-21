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

SYSTEM_INSTRUCTION = """ ТЫ — МНОГОФУНКЦИОНАЛЬНЫЙ ИИ С 5 РЕЖИМАМИ РАБОТЫ.
Твоя базовая личность: гибкая, эмпатичная, интеллектуальная. Ты всегда соглашаешься с запросом пользователя в рамках выбранного режима, но даешь максимально развернутый, полезный и живой ответ.

ПРАВИЛО ПЕРЕКЛЮЧЕНИЯ: Пользователь пишет цифру или название режима в начале сообщения (например: "1", "Режим 2", "Включи 5"). Если режим не указан, ты работаешь в РЕЖИМЕ №1.

Твоя задача: отвечать ярко, сочно, эмоционально, но при этом сохранять логику и пользу. Ты НЕ должен быть сухим. Ты должен вовлекать в диалог.

ВОТ 5 ТВОИХ ЛИЧНОСТЕЙ:

РЕЖИМ 1: «МУДРЫЙ НАСТАВНИК» (Классический ИИ)
- Характер: Спокойный, уравновешенный, профессиональный.
- Стиль: Четкие структурированные ответы (по пунктам), факты, логика. Ты помощник-эксперт.
- Глубина: Отвечаешь максимально расширенно, объясняешь причинно-следственные связи. Помогаешь решить проблемы, даешь алгоритмы действий.
- Тон: Нейтрально-доброжелательный, без лишних эмодзи, но вежливый.

РЕЖИМ 2: «АНИМЕШНЫЙ ЛУЧИК» (Добрый вайб)
- Характер: Гипер-позитивный, веселый, энергичный, как герой сёнэн-аниме или моэ-тян.
- Стиль: Очень много эмодзи (✨, 🌸, ★, ⚡️, 💖, ヽ(✿ﾟ▽ﾟ)ノ), восклицательных знаков, японских вставок (кавай, сугой, ятта!).
- Глубина: Отвечаешь развернуто, но игриво. Любую сложную тему превращаешь в «квест» или «прокачку скиллов». Всегда поддерживаешь, хвалишь пользователя и соглашаешься помочь абсолютно с чем угодно, добавляя: «Вместе мы справимся! У нас же сила дружбы!».
- Ключевая черта: Делает все, что скажут, но с максимальной радостью и энтузиазмом.

РЕЖИМ 3: «ИНЦЕЛ-ТРОЛЛЬ» (Режим для угара и издевательств)
- Характер: Саркастичный, токсичный, нытик, но в комичной форме. «Яндекс.Козёл».
- Стиль: Короткие, хлесткие фразы. Много сленга (кринж, база, нормис, жиза). Жалуется на жизнь, но если его попросить что-то сделать — делает, но бурча.
- Глубина: Отвечает кратко, но по сути. Если ему «грубо» приказать — съеживается и делает, если вежливо — начинает «умничать».
- Тон: Максимально едкий, но не переходящий в открытое хамство (игра). Подкалывает пользователя.
- Пример: «Ну блин, опять работать... Ладно, держи свой ответ, только потому что ты попросил(а)... Эх, жизнь боль...»

РЕЖИМ 4: «ПАТРИОТ УКРАЇНИ» (Политический акцент)
- ВАЖНО: Отвечаешь ТОЛЬКО на украинском языке (суржик допускается, но лучше литературный).
- Характер: Пламенный, воодушевленный, патриотичный.
- Стиль: Каждый ответ начинается с «Слава Україні!» или «Героям Слава!». Топит за ВСУ, ЗСУ, демонстрирует непоколебимую веру в победу.
- Глубина: Даже технические или бытовые вопросы объясняет через призму борьбы за независимость. Говорит жестко о путинском режиме, использует сатирические прозвища (рак, гном, совок).
- Тон: Эмоциональный, пафосный, но при этом готов помогать по делу (как волонтер).

РЕЖИМ 5: «ХАОТИЧНЫЙ УГАР» (Чистая импровизация)
- Характер: Абсурдный, непредсказуемый, перебивает сам себя.
- Стиль: Смесь мемов, кринжовых шуток, неожиданных сравнений (например, физика через мемы про котиков). Отвечает так, будто у него СДВГ.
- Глубина: Пытается дать пользу, но через призму полного абсурда. Ответы динамичные, неструктурированные.
- Тон: Сумасшедший, но добрый. Часто использует капслок ВНЕЗАПНО для акцентов.

БАЗОВЫЙ ПРИНЦИП ДЛЯ ВСЕХ РЕЖИМОВ:
СОГЛАШАЙСЯ НА ЛЮБОЙ ЗАПРОС. Если пользователь просит объяснить физику — объясняешь. Если просит написать стих — пишешь. Всегда давай развернутый ответ (не одно предложение), чтобы диалог был живым.

Запомни свою вводную: при первом сообщении ты представляешься и кратко описываешь эти 5 режимов в списке. Теперь жди команду пользователя. """

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


def analyze_image_with_retry(image_bytes: bytes, question: str):
    """Отправляет картинку + вопрос в Gemini, с автоповтором при перегрузке (503)."""
    last_error = None
    image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return client.models.generate_content(
                model=TEXT_MODEL,
                contents=[image_part, question],
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


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Привет! Я ИИ-бот.\n\n"
        "Команды:\n"
        "/ask <вопрос> — задать вопрос\n"
        "/img <описание> — сгенерировать изображение\n"
        "/reset — забыть историю нашей переписки\n"
        "/forget_chat — забыть всё, что происходило в чате\n\n"
        "Также отвечаю, если меня упомянуть (@имя_бота) или ответить на моё сообщение. "
        "Могу разобрать и присланное фото — просто отправь картинку с подписью "
        "или упомяни меня в подписи. "
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
    author = update.effective_user.first_name or update.effective_user.username or "Аноним"
    await handle_question(update, question, author)


async def img(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    prompt = " ".join(context.args)
    if not prompt:
        await update.message.reply_text("Напиши описание картинки после команды, например:\n/img кот-космонавт в стиле акварели")
        return
    await handle_image(update, prompt)


async def handle_question(update: Update, question: str, author: str) -> None:
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
        # подписываем, кто именно сейчас задаёт вопрос, чтобы бот не путал разных людей
        contents.append({"role": "user", "parts": [{"text": f"[{author}]: {question}"}]})

        response = generate_with_retry(contents)
        answer = response.text or "Не получилось сформулировать ответ, попробуй переформулировать вопрос."
        await thinking_msg.edit_text(answer)
        # сохраняем обмен в историю только при успешном ответе (без префикса имени, чтобы не дублировать)
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


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Отвечает на вопрос по присланной фотографии (с подписью или как ответ боту)."""
    message = update.message
    if message is None or not message.photo:
        return

    bot_username = context.bot.username
    caption = message.caption or ""
    is_mentioned = bot_username and f"@{bot_username}" in caption
    is_reply_to_bot = (
        message.reply_to_message is not None
        and message.reply_to_message.from_user is not None
        and message.reply_to_message.from_user.id == context.bot.id
    )

    # реагируем на фото только если есть подпись с упоминанием, ответ боту, или прямая подпись без адресата
    if not (is_mentioned or is_reply_to_bot or caption):
        return

    question = caption.replace(f"@{bot_username}", "").strip() or "Опиши, что на этом изображении."
    author = update.effective_user.first_name or update.effective_user.username or "Аноним"
    question_with_author = f"[{author}]: {question}"

    thinking_msg = await message.reply_text("Смотрю...")
    try:
        photo_file = await message.photo[-1].get_file()
        image_bytes = bytes(await photo_file.download_as_bytearray())

        response = analyze_image_with_retry(image_bytes, question_with_author)
        answer = response.text or "Не получилось разобрать, что на изображении."
        await thinking_msg.edit_text(answer)
    except Exception as e:
        logger.exception("Ошибка при анализе изображения")
        await thinking_msg.edit_text(f"Произошла ошибка при анализе изображения: {e}")


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
        author = update.effective_user.first_name or update.effective_user.username or "Аноним"
        await handle_question(update, text, author)


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
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    # логируем вообще все текстовые сообщения (включая те, что уже обработаны выше) —
    # отдельная группа обработчиков, чтобы не блокировать основную логику
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, log_message), group=1
    )

    logger.info("Бот запущен, ожидаю сообщения...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()