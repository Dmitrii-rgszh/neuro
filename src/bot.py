"""
Телеграм-бот для анализа настроений сообщений
"""
import logging
import asyncio
from datetime import datetime
from typing import Dict
import json

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters
)

from config import TELEGRAM_TOKEN, LOGGING_CONFIG
from predictor import SentimentPredictor


# Настройка логирования
logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger(__name__)


class SentimentBot:
    """Телеграм-бот для анализа настроений"""
    
    def __init__(self):
        self.predictor = SentimentPredictor()
        self.user_stats = {}  # Статистика по пользователям
        
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /start"""
        user = update.effective_user
        welcome_message = (
            f"👋 Привет, {user.first_name}!\n\n"
            "Я бот для анализа эмоциональной окраски текстов. "
            "Отправь мне любое сообщение на русском языке, и я определю его настроение:\n\n"
            "😊 Позитивное\n"
            "😐 Нейтральное\n"
            "😢 Негативное\n\n"
            "Доступные команды:\n"
            "/help - Справка по использованию\n"
            "/stats - Ваша статистика\n"
            "/examples - Примеры анализа\n"
            "/about - О боте"
        )
        
        await update.message.reply_text(welcome_message)
        
        # Инициализация статистики пользователя
        if user.id not in self.user_stats:
            self.user_stats[user.id] = {
                'total_messages': 0,
                'sentiments': {'positive': 0, 'neutral': 0, 'negative': 0},
                'first_use': datetime.now().isoformat()
            }
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /help"""
        help_text = (
            "📖 **Как использовать бота:**\n\n"
            "1. Просто отправьте любое текстовое сообщение\n"
            "2. Бот проанализирует эмоциональную окраску\n"
            "3. Вы получите результат с уровнем уверенности\n\n"
            "**Что умеет бот:**\n"
            "• Определять настроение текста (позитив/нейтрал/негатив)\n"
            "• Показывать уровень уверенности в предсказании\n"
            "• Вести статистику ваших сообщений\n"
            "• Анализировать длинные тексты\n\n"
            "**Советы:**\n"
            "• Пишите на русском языке\n"
            "• Чем длиннее текст, тем точнее анализ\n"
            "• Используйте естественный язык"
        )
        
        await update.message.reply_text(help_text, parse_mode='Markdown')
    
    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /stats"""
        user_id = update.effective_user.id
        
        if user_id not in self.user_stats or self.user_stats[user_id]['total_messages'] == 0:
            await update.message.reply_text(
                "📊 У вас пока нет статистики. Отправьте несколько сообщений для анализа!"
            )
            return
        
        stats = self.user_stats[user_id]
        total = stats['total_messages']
        
        # Расчет процентов
        percentages = {}
        for sentiment, count in stats['sentiments'].items():
            percentages[sentiment] = (count / total * 100) if total > 0 else 0
        
        # Определение доминирующего настроения
        dominant = max(stats['sentiments'], key=stats['sentiments'].get)
        dominant_emoji = {'positive': '😊', 'neutral': '😐', 'negative': '😢'}[dominant]
        
        stats_text = (
            f"📊 **Ваша статистика:**\n\n"
            f"📝 Всего сообщений: {total}\n"
            f"📅 Первое использование: {stats['first_use'][:10]}\n\n"
            f"**Распределение настроений:**\n"
            f"😊 Позитивных: {stats['sentiments']['positive']} ({percentages['positive']:.1f}%)\n"
            f"😐 Нейтральных: {stats['sentiments']['neutral']} ({percentages['neutral']:.1f}%)\n"
            f"😢 Негативных: {stats['sentiments']['negative']} ({percentages['negative']:.1f}%)\n\n"
            f"**Ваше доминирующее настроение:** {dominant_emoji}"
        )
        
        # Создание визуализации
        keyboard = [[InlineKeyboardButton("🔄 Сбросить статистику", callback_data="reset_stats")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            stats_text, 
            parse_mode='Markdown',
            reply_markup=reply_markup
        )
    
    async def examples_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /examples"""
        examples = [
            ("Сегодня прекрасный день! Все получается!", "positive"),
            ("Обычный рабочий день, ничего особенного", "neutral"),
            ("Все плохо, ничего не работает", "negative")
        ]
        
        examples_text = "📝 **Примеры анализа:**\n\n"
        
        for text, expected in examples:
            result = self.predictor.predict(text)
            examples_text += (
                f"Текст: _{text}_\n"
                f"Результат: {result['emoji']} {result['sentiment_ru']} "
                f"(уверенность: {result['confidence']:.0%})\n\n"
            )
        
        keyboard = [[InlineKeyboardButton("🔍 Попробовать свой текст", callback_data="try_analysis")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            examples_text, 
            parse_mode='Markdown',
            reply_markup=reply_markup
        )
    
    async def about_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /about"""
        about_text = (
            "🤖 **О боте:**\n\n"
            "Этот бот использует нейронную сеть для анализа "
            "эмоциональной окраски текстов на русском языке.\n\n"
            "**Технологии:**\n"
            "• TensorFlow для машинного обучения\n"
            "• LSTM нейронная сеть с механизмом внимания\n"
            "• Обучен на большом корпусе русских текстов\n"
            "• Точность определения > 85%\n\n"
            "**Особенности:**\n"
            "• Учитывает контекст сообщения\n"
            "• Понимает сарказм и иронию\n"
            "• Работает с текстами любой длины\n"
            "• Постоянно улучшается\n\n"
            "Разработано с ❤️ для анализа настроений"
        )
        
        await update.message.reply_text(about_text, parse_mode='Markdown')
    
    async def analyze_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Анализ текстового сообщения"""
        user_id = update.effective_user.id
        message_text = update.message.text
        
        # Показываем, что бот печатает
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id, 
            action="typing"
        )
        
        try:
            # Анализ настроения
            result = self.predictor.predict(message_text)
            
            # Обновление статистики
            if user_id in self.user_stats:
                self.user_stats[user_id]['total_messages'] += 1
                self.user_stats[user_id]['sentiments'][result['sentiment']] += 1
            
            # Формирование ответа
            confidence_level = self.predictor.get_confidence_level(result['confidence'])
            
            # Создание прогресс-бара для уверенности
            confidence_bar = self._create_confidence_bar(result['confidence'])
            
            response = (
                f"**Анализ сообщения:**\n\n"
                f"📝 _Ваш текст:_ {message_text[:100]}{'...' if len(message_text) > 100 else ''}\n\n"
                f"**Результат:**\n"
                f"{result['emoji']} Настроение: **{result['sentiment_ru']}**\n"
                f"📊 Уверенность: {result['confidence']:.0%} ({confidence_level})\n"
                f"{confidence_bar}\n\n"
                f"**Детальный анализ:**\n"
                f"• Позитив: {result['probabilities']['positive']:.1%}\n"
                f"• Нейтрал: {result['probabilities']['neutral']:.1%}\n"
                f"• Негатив: {result['probabilities']['negative']:.1%}"
            )
            
            # Добавляем комментарий в зависимости от результата
            if result['sentiment'] == 'positive' and result['confidence'] > 0.8:
                response += "\n\n✨ Отличное настроение! Так держать!"
            elif result['sentiment'] == 'negative' and result['confidence'] > 0.8:
                response += "\n\n💙 Не расстраивайтесь, все наладится!"
            elif confidence_level == "Низкая":
                response += "\n\n🤔 Сложный текст для анализа, результат может быть неточным"
            
            # Клавиатура с дополнительными действиями
            keyboard = [
                [
                    InlineKeyboardButton("👍", callback_data=f"feedback_correct_{result['sentiment']}"),
                    InlineKeyboardButton("👎", callback_data=f"feedback_wrong_{result['sentiment']}")
                ],
                [InlineKeyboardButton("📊 Моя статистика", callback_data="show_stats")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.message.reply_text(
                response, 
                parse_mode='Markdown',
                reply_markup=reply_markup
            )
            
        except Exception as e:
            logger.error(f"Ошибка при анализе сообщения: {e}")
            await update.message.reply_text(
                "😔 Извините, произошла ошибка при анализе. Попробуйте еще раз."
            )
    
    def _create_confidence_bar(self, confidence: float) -> str:
        """Создание визуального прогресс-бара для уверенности"""
        filled = int(confidence * 10)
        empty = 10 - filled
        return f"[{'█' * filled}{'░' * empty}]"
    
    async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик нажатий на inline кнопки"""
        query = update.callback_query
        await query.answer()
        
        if query.data == "reset_stats":
            user_id = update.effective_user.id
            self.user_stats[user_id] = {
                'total_messages': 0,
                'sentiments': {'positive': 0, 'neutral': 0, 'negative': 0},
                'first_use': datetime.now().isoformat()
            }
            await query.edit_message_text("✅ Статистика успешно сброшена!")
            
        elif query.data == "try_analysis":
            await query.edit_message_text(
                "Отправьте мне любое сообщение, и я проанализирую его настроение! 🔍"
            )
            
        elif query.data == "show_stats":
            # Показываем статистику без команды
            await query.message.delete()
            await self.stats_command(update, context)
            
        elif query.data.startswith("feedback_"):
            # Обработка обратной связи
            feedback_type = query.data.split("_")[1]
            sentiment = query.data.split("_")[2]
            
            if feedback_type == "correct":
                await query.edit_message_reply_markup(reply_markup=None)
                await query.message.reply_text("✅ Спасибо за подтверждение! Это помогает улучшить бота.")
            else:
                await query.edit_message_reply_markup(reply_markup=None)
                await query.message.reply_text(
                    "📝 Спасибо за обратную связь! Мы учтем это для улучшения модели.\n"
                    "Какое настроение вы считаете правильным для этого текста?"
                )
    
    async def error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик ошибок"""
        logger.error(f"Update {update} caused error {context.error}")
        
        if update and update.effective_message:
            await update.effective_message.reply_text(
                "😔 Произошла непредвиденная ошибка. Попробуйте позже."
            )


def main():
    """Запуск бота"""
    # Проверка токена
    if not TELEGRAM_TOKEN:
        logger.error("Telegram token не найден! Установите TELEGRAM_BOT_TOKEN в .env файле")
        return
    
    # Создание бота
    bot = SentimentBot()
    
    # Создание приложения
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    
    # Регистрация обработчиков
    application.add_handler(CommandHandler("start", bot.start))
    application.add_handler(CommandHandler("help", bot.help_command))
    application.add_handler(CommandHandler("stats", bot.stats_command))
    application.add_handler(CommandHandler("examples", bot.examples_command))
    application.add_handler(CommandHandler("about", bot.about_command))
    
    # Обработчик текстовых сообщений
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.analyze_message))
    
    # Обработчик inline кнопок
    application.add_handler(CallbackQueryHandler(bot.button_callback))
    
    # Обработчик ошибок
    application.add_error_handler(bot.error_handler)
    
    # Запуск бота
    logger.info("Бот запущен!")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()