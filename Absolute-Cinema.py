import os
import cv2
import numpy as np
import tempfile
import logging
import asyncio
import random
import gc
from uuid import uuid4
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
from moviepy.editor import VideoFileClip, AudioFileClip

# Настройка логгирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Конфигурация
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB
TEMP_DIR = tempfile.gettempdir()
user_sessions = {}
REQUEST_TIMEOUT = 30  # Увеличиваем таймаут до 30 секунд

def validate_shapes(img1: np.ndarray, img2: np.ndarray):
    """Проверка совпадения размеров и количества каналов"""
    if img1.shape != img2.shape:
        raise ValueError(f"Shape mismatch: {img1.shape} vs {img2.shape}")

def create_scratch_texture(height: int, width: int) -> np.ndarray:
    """Генерация текстуры царапин"""
    texture = np.zeros((height, width, 3), dtype=np.uint8)
    for _ in range(random.randint(2, 5)):
        y = random.randint(0, height)
        thickness = random.randint(1, 3)
        cv2.line(texture, (0, y), (width, y), (200, 200, 200), thickness)
    return texture

def apply_old_film(frame: np.ndarray) -> np.ndarray:
    """Эффект старой кинопленки"""
    h, w, c = frame.shape
    
    # Сепия
    sepia_kernel = np.array([
        [0.272, 0.534, 0.131],
        [0.349, 0.686, 0.168],
        [0.393, 0.769, 0.189]
    ], dtype=np.float32)
    
    processed = cv2.transform(frame.astype(np.float32), sepia_kernel)
    processed = np.clip(processed, 0, 255).astype(np.uint8)

    # Добавление шума
    noise = np.random.normal(0, 15, (h, w, c)).astype(np.uint8)
    validate_shapes(processed, noise)
    processed = cv2.add(processed, noise)

    # Случайные царапины
    if random.random() < 0.3:
        texture = create_scratch_texture(h, w)
        if texture.shape != processed.shape:
            texture = cv2.resize(texture, (w, h))
        validate_shapes(processed, texture)
        processed = cv2.addWeighted(processed, 0.9, texture, 0.1, 0)

    # Виньетирование
    kernel_x = cv2.getGaussianKernel(w, w//3)
    kernel_y = cv2.getGaussianKernel(h, h//3)
    kernel = np.outer(kernel_y, kernel_x)
    mask = (kernel * 0.7 + 0.3)[:, :, np.newaxis]
    mask = np.repeat(mask, c, axis=2).astype(np.float32)
    
    validate_shapes(processed, mask)
    return cv2.multiply(processed.astype(np.float32), mask, dtype=cv2.CV_8UC3)

def apply_brightness(frame: np.ndarray) -> np.ndarray:
    """Повышение яркости с увеличенным эффектом"""
    return cv2.convertScaleAbs(frame, alpha=1.2, beta=50)  # Увеличены оба параметра

def apply_negative(frame: np.ndarray) -> np.ndarray:
    """Негатив"""
    return cv2.bitwise_not(frame)

def apply_pixelation(frame: np.ndarray) -> np.ndarray:
    """Пикселизация"""
    h, w = frame.shape[:2]
    new_w = max(1, w // 10)
    new_h = max(1, h // 10)
    small = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

def apply_edge_detection(frame: np.ndarray) -> np.ndarray:
    """Выделение границ"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

def apply_vignette(frame: np.ndarray) -> np.ndarray:
    """Исправленное виньетирование с правильными размерами маски"""
    h, w = frame.shape[:2]
    
    # Создаём маску с тремя каналами
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    
    # Создаём 3-канальную маску
    mask = 1 - np.clip(R / 0.8, 0, 1)
    mask = (mask * 0.7 + 0.3)
    mask = np.repeat(mask[..., np.newaxis], 3, axis=2)  # Добавляем 3 канала

    # Убедимся в совпадении размеров
    if mask.shape != frame.shape:
        mask = cv2.resize(mask, (w, h))

    # Нормализуем и применяем
    return cv2.multiply(frame, mask.astype(np.float32), dtype=cv2.CV_8UC3)

def process_video_with_audio(input_path: str, output_path: str, effect_func):
    """Оптимизированная обработка видео"""
    with VideoFileClip(input_path) as video_clip:
        # Уменьшаем разрешение для ускорения обработки
        video_clip = video_clip.resize(height=480)
        
        audio = video_clip.audio
        processed_frames = []
        
        for frame in video_clip.iter_frames():
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            processed = effect_func(frame_bgr)
            processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            processed_frames.append(processed_rgb)
        
        processed_clip = video_clip.set_make_frame(
            lambda t: processed_frames[min(int(t * video_clip.fps), len(processed_frames)-1)]
        )
        
        if audio:
            processed_clip = processed_clip.set_audio(audio)
        
        # Оптимизированные параметры экспорта
        processed_clip.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            preset='fast',  # Ускорение кодирования
            bitrate='500k', # Контроль битрейта
            logger=None,
            threads=4
        )
        processed_clip.close()

async def apply_effect(update: Update, context: ContextTypes.DEFAULT_TYPE, effect_func, effect_name: str):
    user_id = update.message.from_user.id
    output_path = None
    attempts = 3  # Количество попыток отправки
    
    try:
        if user_id not in user_sessions or not os.path.exists(user_sessions[user_id]['input_path']):
            await update.message.reply_text("⚠️ Сначала отправьте видео!")
            return

        session = user_sessions[user_id]
        output_path = os.path.join(TEMP_DIR, f"processed_{session['file_id']}.mp4")
        
        # Оптимизированная обработка видео
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: process_video_with_audio(session['input_path'], output_path, effect_func)
        )

        # Проверка размера файла
        if os.path.getsize(output_path) > MAX_FILE_SIZE:
            await update.message.reply_text("⚠️ Результат слишком большой для отправки!")
            return

        # Отправка с повторными попытками
        for attempt in range(attempts):
            try:
                await context.bot.send_video(
                    chat_id=update.message.chat_id,
                    video=output_path,
                    caption=f"✅ {effect_name}",
                    write_timeout=REQUEST_TIMEOUT,
                    connect_timeout=REQUEST_TIMEOUT
                )
                break
            except Exception as send_error:
                if attempt == attempts - 1:
                    raise send_error
                await asyncio.sleep(2)  # Пауза перед повторной попыткой
                
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        await update.message.reply_text("⚠️ Ошибка обработки или отправки видео")
        
    finally:
        # Очистка с повторными попытками
        for path in [output_path, user_sessions.get(user_id, {}).get('input_path')]:
            if path and os.path.exists(path):
                for _ in range(3):
                    try:
                        os.remove(path)
                        break
                    except:
                        await asyncio.sleep(0.5)
        
        if user_id in user_sessions:
            del user_sessions[user_id]
        
        gc.collect()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start"""
    await update.message.reply_text(
        "📹 Отправьте видео для обработки (максимум 20MB)\n"
        "Доступные команды после загрузки видео:\n"
        "/old_film - Эффект старой пленки\n"
        "/brightness - Повышение яркости\n"
        "/negative - Негатив\n"
        "/pixelate - Пикселизация\n"
        "/edges - Определение границ\n"
        "/vignette - Виньетирование"
    )

async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка полученного видео"""
    try:
        user_id = update.message.from_user.id
        
        # Проверка размера файла
        if update.message.video.file_size > MAX_FILE_SIZE:
            await update.message.reply_text("⚠️ Файл слишком большой! Максимальный размер: 20MB")
            return

        # Скачивание видео
        video = update.message.video
        file = await video.get_file()
        file_id = str(uuid4())
        input_path = os.path.join(TEMP_DIR, f"{user_id}_{file_id}.mp4")
        
        await file.download_to_drive(input_path)
        user_sessions[user_id] = {
            'input_path': input_path,
            'file_id': file_id
        }

        await update.message.reply_text(
            "✅ Видео получено! Выберите эффект:\n"
            "/old_film /brightness /negative /pixelate /edges /vignette"
        )

    except Exception as e:
        logger.error(f"Ошибка загрузки видео: {e}", exc_info=True)
        await update.message.reply_text("❌ Ошибка при загрузке видео")

# Регистрация обработчиков команд
async def old_film(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await apply_effect(update, context, apply_old_film, "Старая пленка")

async def brightness(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await apply_effect(update, context, apply_brightness, "Повышение яркости")

async def negative(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await apply_effect(update, context, apply_negative, "Негатив")

async def pixelate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await apply_effect(update, context, apply_pixelation, "Пикселизация")

async def edges(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await apply_effect(update, context, apply_edge_detection, "Определение границ")

async def vignette(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await apply_effect(update, context, apply_vignette, "Виньетирование")

def main():
    """Запуск бота"""
    application = Application.builder().token("7674173232:AAFBkten7YJZOh2sHwhvyKqdzuTAnrGfwJs").build()
    
    # Регистрация обработчиков
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.VIDEO, handle_video))
    application.add_handler(CommandHandler("old_film", old_film))
    application.add_handler(CommandHandler("brightness", brightness))
    application.add_handler(CommandHandler("negative", negative))
    application.add_handler(CommandHandler("pixelate", pixelate))
    application.add_handler(CommandHandler("edges", edges))
    application.add_handler(CommandHandler("vignette", vignette))
    
    application.run_polling()

if __name__ == "__main__":
    main()
