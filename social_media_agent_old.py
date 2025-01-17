import os
import logging
import asyncio
import json
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
import openai
from telegram import (
    Update, 
    InlineKeyboardButton, 
    InlineKeyboardMarkup,
    File
)
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# Load environment variables
load_dotenv()

# Configure logging with detailed format and both file and console output
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)
BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')

# Debugging: Print token and its type
print(f"Retrieved BOT_TOKEN from environment: {BOT_TOKEN} (Type: {type(BOT_TOKEN)})")

@dataclass
class Config:
    """Configuration class for the bot with validation."""
    
    def __init__(self):
        # Retrieving environment variables with debugging
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
        self.TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.MEDIA_DIR = "media"
        self.MAX_TOKENS = 300
        self.MODEL_NAME = "gpt-3.5-turbo"
        self.ALLOWED_FILE_TYPES = None
        self.MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB
        self.RATE_LIMIT_DELAY = 1  # Seconds between API calls

        # Debugging: Print retrieved values and types
        print(f"Retrieved OPENAI_API_KEY: {self.OPENAI_API_KEY} (Type: {type(self.OPENAI_API_KEY)})")
        print(f"Retrieved TELEGRAM_BOT_TOKEN: {self.TELEGRAM_BOT_TOKEN} (Type: {type(self.TELEGRAM_BOT_TOKEN)})")
        
        if not self.OPENAI_API_KEY:
            raise ValueError("OpenAI API key not found in environment variables")
        if not self.TELEGRAM_BOT_TOKEN:
            raise ValueError("Telegram Bot token not found in environment variables")
        
        # Initialize allowed file types
        self.ALLOWED_FILE_TYPES = ['.jpg', '.jpeg', '.png', '.gif', '.mp4', '.pdf', '.doc', '.docx']

        # Debugging: Confirm the values are properly initialized
        print(f"Config initialized with TELEGRAM_BOT_TOKEN: {self.TELEGRAM_BOT_TOKEN} (Type: {type(self.TELEGRAM_BOT_TOKEN)})")
        print(f"Config initialized with OPENAI_API_KEY: {self.OPENAI_API_KEY} (Type: {type(self.OPENAI_API_KEY)})")


class Analytics:
    """Analytics tracking for bot usage."""
    def __init__(self):
        self.user_sessions: Dict[int, Dict[str, Any]] = {}
        self.start_time = datetime.now()

    def track_activity(self, user_id: int, activity_type: str) -> None:
        """Track user activity."""
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = {
                'first_seen': datetime.now(),
                'activities': []
            }
        
        self.user_sessions[user_id]['activities'].append({
            'type': activity_type,
            'timestamp': datetime.now().isoformat()
        })
        
        logger.info(f"User {user_id} performed {activity_type}")

    def get_user_stats(self, user_id: int) -> Dict[str, Any]:
        """Get statistics for a specific user."""
        if user_id not in self.user_sessions:
            return None
        
        user_data = self.user_sessions[user_id]
        return {
            'first_seen': user_data['first_seen'],
            'total_activities': len(user_data['activities']),
            'last_activity': user_data['activities'][-1] if user_data['activities'] else None
        }
    

class ContentGenerator:
    """Enhanced content generation using OpenAI API."""
    def __init__(self, api_key: str):
        self.client = openai  # Corrected client initialization
        self.api_key = api_key
        self.last_request_time = 0

    async def generate_content(self, prompt: str, max_tokens: int = 300) -> str:
        """Generate content with rate limiting and error handling."""
        try:
            # Rate limiting
            current_time = datetime.now().timestamp()
            time_since_last_request = current_time - self.last_request_time
            if time_since_last_request < Config.RATE_LIMIT_DELAY:
                await asyncio.sleep(Config.RATE_LIMIT_DELAY - time_since_last_request)

            response = await asyncio.to_thread(
                self.client.ChatCompletion.create,
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a professional social media content creator."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )
            
            self.last_request_time = datetime.now().timestamp()
            return response['choices'][0]['message']['content'].strip()
        except Exception as e:
            logger.error(f"Content generation error: {e}")
            return "Sorry, I couldn't generate content at this time."

class SocialMediaBot:
    def __init__(self, token: str):
        """Initialize the bot with the provided token."""
        print(f"Initializing SocialMediaBot with token: {token} (Type: {type(token)})")  # Debugging token
        
        # Check if token is a string
        if isinstance(token, str):
            print("Token is confirmed to be a string.")  # Confirming token type
        else:
            print(f"Error: Token is not a string. It is of type: {type(token)}")  # Type checking
            # Forcefully convert to string if not a string
            print("Converting token to string.")
            token = str(token)
        
        # Initialize the bot with the token
        self.application = Application.builder().token(token).build()
        self.analytics = Analytics()  # Initialize analytics
        self._running = False  # Track the bot's running state
        print("Bot initialized successfully.")  # Confirmation message

    async def setup(self):
        """Set up the bot application and handlers."""
        if not self.application:
            try:
                # Initialize bot application if not already initialized
                print("Initializing bot application...")  # Debugging initialization
                self.application = Application.builder().token(self.config.TELEGRAM_BOT_TOKEN).build()
                print("Bot application initialized.")  # Confirmation
            except Exception as e:
                print(f"Error initializing bot application: {e}")  # Debugging initialization error
                logger.error(f"Error initializing bot application: {e}", exc_info=True)
                raise

        try:
            # Add handlers
            print("Setting up the bot handlers...")  # Debugging handler setup
            self.application.add_handler(CommandHandler("start", self.start_command))
            print("Start command handler added.")  # Debugging handler addition

            self.application.add_handler(CommandHandler("help", self.help_command))
            print("Help command handler added.")  # Debugging handler addition

            self.application.add_handler(CommandHandler("stats", self.stats_command))
            print("Stats command handler added.")  # Debugging handler addition

            self.application.add_handler(MessageHandler(filters.PHOTO | filters.Document.ALL, self.handle_media))
            print("Media handler added.")  # Debugging media handler addition

            self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
            print("Message handler added.")  # Debugging message handler addition

            self.application.add_handler(CallbackQueryHandler(self.handle_button))
            print("Button handler added.")  # Debugging button handler addition

            self.application.add_error_handler(self.error_handler)
            print("Error handler added.")  # Debugging error handler addition

            # Log and print that the handlers are set up
            logger.info("Bot setup completed successfully.")
            print("Handlers added successfully.")  # Confirm handlers added

        except Exception as e:
            print(f"Error setting up handlers: {e}")  # Debugging error during handler setup
            logger.error(f"Error setting up handlers: {e}", exc_info=True)
            raise

    async def start(self):
        """Start the bot."""
        try:
            print("Starting the bot...")  # Debugging start process
            await self.setup()  # Set up handlers
            print("Handlers set up.")  # Confirm if handlers are set up
            await self.application.initialize()
            print("Bot initialized.")  # Confirm if bot is initialized
            await self.application.start()
            print("Bot started.")  # Confirm if bot started
            self._running = True
            print("Bot is now ready to receive messages.")  # Ready to receive
            await self.application.run_polling()  # Start polling for updates
            print("Polling is running.")  # Confirm polling is started
        except Exception as e:
            print(f"Error starting bot: {e}")  # Debugging start error
            logger.error(f"Error starting bot: {e}", exc_info=True)
            raise

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle incoming text messages with detailed debugging."""
        # Debugging: Log the received message text
        print(f"Received text message: {update.message.text}")  # Debugging message received
        logger.info(f"Received text message: {update.message.text}")  # Logging the received message

        user_id = update.effective_user.id
        self.analytics.track_activity(user_id, 'message_received')  # Track the activity in analytics
        print(f"User ID {user_id} message received and activity tracked.")  # Debugging activity tracking

        try:
            # Prepare a reply text
            reply_text = f"Received your message: {update.message.text}"
            print(f"Reply text prepared: {reply_text}")  # Debugging reply preparation

            # Send the reply back to the user
            await update.message.reply_text(reply_text, parse_mode=ParseMode.MARKDOWN)
            print(f"Replied to user with: {reply_text}")  # Debugging reply sent

        except Exception as e:
            # Error handling: Log and reply with an error message
            print(f"Error handling message: {e}")  # Debugging error message
            logger.error(f"Error handling message: {e}", exc_info=True)  # Logging the error with traceback
            await update.message.reply_text("❌ Error processing your message. Please try again.")
            print("Error response sent.")  # Debugging error response sent

        # Debugging: Print confirmation that the method has finished processing the message
        print(f"Finished processing message from user ID {user_id}")


    async def stop(self):
        """Stop the bot gracefully."""
        try:
            print("Stopping the bot...")  # Debugging stop process
            if self.application and self.application.running:
                await self.application.stop()
                await self.application.shutdown()
                self._running = False
            logger.info("Bot stopped successfully.")
            print("Bot stopped successfully.")  # Confirm bot stopped
        except Exception as e:
            print(f"Error stopping bot: {e}")  # Debugging stop error
            logger.error(f"Error stopping bot: {e}", exc_info=True)
            raise
    async def handle_media(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Process media uploads with validation and error handling."""
        print("Received media...")  # Debugging message when media is received
        user_id = update.effective_user.id
        self.analytics.track_activity(user_id, 'media_upload')
    
        try:
            print("Preparing to process media...")  # Debugging

            processing_msg = await update.message.reply_text("🔄 Processing your media file...")
        
            # Handle photo
            if update.message.photo:
                print("Photo received.")  # Debugging photo received
                file = update.message.photo[-1]  # Get the highest resolution photo
                file_extension = ".jpg"
            # Handle document
            elif update.message.document:
                print(f"Document received: {update.message.document.file_name}")  # Debugging document received
                file = update.message.document
                file_extension = os.path.splitext(file.file_name)[1] if hasattr(file, "file_name") else ""
            else:
                await processing_msg.edit_text("❌ Unsupported media type.")
                print("Unsupported media type received.")  # Debugging unsupported type
                return

            print(f"Received file: {file.file_id}, Extension: {file_extension}, Size: {file.file_size} bytes")  # Debugging file details

            # Validate file size
            if file.file_size > self.config.MAX_FILE_SIZE:
                print(f"File size too large: {file.file_size} bytes")  # Debugging file size issue
                await processing_msg.edit_text("❌ File too large (max 20MB)")
                return

            # Validate file type
            if file_extension.lower() not in self.config.ALLOWED_FILE_TYPES:
                print(f"Unsupported file type: {file_extension}")  # Debugging unsupported file type
                await processing_msg.edit_text(f"❌ Unsupported file type. Allowed types: {', '.join(self.config.ALLOWED_FILE_TYPES)}")
                return

            # Process file
            file_id = file.file_id
            file_name = f"{file_id}{file_extension}"
            file_path = os.path.join(self.config.MEDIA_DIR, file_name)
            print(f"Processing file... Saving to: {file_path}")  # Debugging file save path

            try:
                # Download the file from Telegram
                await context.bot.get_file(file_id).download_to_drive(file_path)
                print(f"File downloaded to: {file_path}")  # Debugging file download success
            except Exception as e:
                print(f"File download error: {e}")  # Debugging download error
                logger.error(f"File download error: {e}", exc_info=True)
                await processing_msg.edit_text("❌ Error downloading the file. Please try again.")
                return
        
            # Save media information in user_data (optional)
            context.user_data['media_file'] = {
                'path': file_path,
                'type': file_extension,
                'timestamp': datetime.now().isoformat()
            }
            print(f"Media file saved: {file_path}")  # Debugging media file save

            await processing_msg.edit_text(
                "✅ Media uploaded successfully!\n"
                "Send me a topic to generate content.",
                parse_mode=ParseMode.MARKDOWN
            )
            print("Successfully processed media and sent confirmation.")  # Debugging successful completion

        except Exception as e:
            print(f"Error processing media: {e}")  # Debugging media handling error
            logger.error(f"Media handling error: {e}", exc_info=True)
            await update.message.reply_text("❌ Error processing media. Please try again.")
            print("Error processing media, sending failure message.")  # Debugging failure message sent

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle the /start command with welcome message."""
        print("Received /start command")  # Debugging command reception
        user_id = update.effective_user.id
        self.analytics.track_activity(user_id, 'start_command')

        welcome_text = (
            "👋 Hi I'm Jimmy your Social Media Assistant!\n\n"
            "🎯 Here's what I can do:\n"
            "• Generate engaging social media posts\n"
            "• Create video content ideas\n"
            "• Help with hashtag research\n"
            "• Process media uploads\n\n"
            "📎 Send me a description or upload media to get started!\n\n"
            "Commands:\n"
            "/start - Show this message\n"
            "/help - Show detailed help\n"
            "/stats - Show your usage statistics"
        )

        await update.message.reply_text(welcome_text, parse_mode=ParseMode.MARKDOWN)

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Provide detailed help information."""
        print("Handling /help command.")  # Debugging
        help_text = (
            "📚 *Detailed Help Guide*\n\n"
            "*Content Generation:*\n"
            "• Send a topic or description\n"
            "• Choose your target platform\n"
            "• Get optimized content\n\n"
            "*Media Handling:*\n"
            "• Upload photos or documents\n"
            "• Supported formats: JPG, PNG, GIF, MP4, PDF\n"
            "• Max size: 20MB\n\n"
            "*Available Platforms:*\n"
            "🐦 *Twitter*\n"
            "• Character limit: 280\n"
            "• Hashtag optimization\n"
            "• Engagement focus\n\n"
            "🎥 *YouTube*\n"
            "• Script outlines\n"
            "• SEO optimization\n"
            "• Call-to-action\n\n"
            "📱 *TikTok*\n"
            "• Trend awareness\n"
            "• Hook creation\n"
            "• Sound suggestions"
        )
        await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)

    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Show user statistics."""
        print("Handling /stats command.")  # Debugging
        user_id = update.effective_user.id
        stats = self.analytics.get_user_stats(user_id)

        if not stats:
            await update.message.reply_text("No statistics available yet.")
            return

        stats_text = (
            "📊 *Your Usage Statistics*\n\n"
            f"First seen: {stats['first_seen'].strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Total activities: {stats['total_activities']}\n"
            "Keep creating awesome content! 🚀"
        )

        await update.message.reply_text(stats_text, parse_mode=ParseMode.MARKDOWN)

async def main():
    """Main function to run the bot."""
    try:
        print("Initializing the bot configuration...")  # Debugging config initialization
        config = Config()  # Initialize the Config object

        # Debugging: Print token and its type
        print(f"Config initialized with TELEGRAM_BOT_TOKEN: {config.TELEGRAM_BOT_TOKEN} (Type: {type(config.TELEGRAM_BOT_TOKEN)})")

        # Pass only the token string to SocialMediaBot
        bot = SocialMediaBot(config.TELEGRAM_BOT_TOKEN)  # Pass only the token string
        print("SocialMediaBot object created.")  # Confirmation after bot object is created

        print("Starting bot...")  # Debugging start
        await bot.start()  # Start the bot
        print("Bot started successfully.")  # Debugging successful start

    except Exception as e:
        print(f"Fatal error in main: {e}")  # Additional debug info
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        if bot._running:  # Check if bot is running before stopping
            await bot.stop()  # Ensure proper cleanup
            print("Bot stopped.")  # Debugging stop
        else:
            print("Bot was not running, skipping stop.")


async def handle_media(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Process media uploads with validation and error handling."""
    if not self._running:
        return

    user_id = update.effective_user.id
    self.analytics.track_activity(user_id, 'media_upload')

    try:
        processing_msg = await update.message.reply_text("🔄 Processing your media file...")
        
        if update.message.photo:
            file = update.message.photo[-1]
            file_extension = ".jpg"
        else:
            file = update.message.document
            file_extension = os.path.splitext(file.file_name)[1] if hasattr(file, "file_name") else ""
        
        # Validate file
        if file.file_size > self.config.MAX_FILE_SIZE:
            await processing_msg.edit_text("❌ File too large (max 20MB)")
            return
            
        if file_extension.lower() not in self.config.ALLOWED_FILE_TYPES:
            await processing_msg.edit_text("❌ Unsupported file type")
            return

        # Process file
        file_id = file.file_id
        file_name = f"{file_id}{file_extension}"
        file_path = os.path.join(self.config.MEDIA_DIR, file_name)
        
        try:
            await context.bot.get_file(file_id).download_to_drive(file_path)
        except Exception as e:
            logger.error(f"File download error: {e}", exc_info=True)
            await processing_msg.edit_text("❌ Error downloading the file. Please try again.")
            return
        
        context.user_data['media_file'] = {
            'path': file_path,
            'type': file_extension,
            'timestamp': datetime.now().isoformat()
        }
        
        await processing_msg.edit_text(
            "✅ Media uploaded successfully!\n"
            "Send me a topic to generate content.",
            parse_mode=ParseMode.MARKDOWN
        )

    except Exception as e:
        logger.error(f"Media handling error: {e}", exc_info=True)
        await update.message.reply_text("❌ Error processing media. Please try again.")

    def _get_platform_prompt(self, platform: str, topic: str, has_media: bool = False) -> str:
        """Generate platform-specific prompts."""
        media_context = "considering the attached media" if has_media else ""
        
        prompts = {
            'twitter': (
                f"Create an engaging Twitter post about: {topic} {media_context}. "
                "Include trending hashtags. Keep under 280 characters. "
                "Focus on maximizing engagement."
            ),
            'youtube': (
                f"Create a YouTube video script outline about: {topic} {media_context}. "
                "Include:\n1) Attention-grabbing hook\n2) Key points\n"
                "3) Call to action\n4) SEO optimization\n5) Thumbnail ideas"
            ),
            'tiktok': (
                f"Create a TikTok/Reels content idea about: {topic} {media_context}. "
                "Include:\n1) Viral hook\n2) Trending sound suggestions\n"
                "3) Popular hashtags\n4) Engagement triggers"
            )
        }
        return prompts.get(platform, "")

    async def generate_platform_content(self, topic: str, message) -> dict:
        """Generate content for multiple platforms."""
        platforms = {
            'twitter': "Twitter post",
            'youtube': "YouTube script",
            'tiktok': "TikTok idea"
        }
        
        content = {}
        progress_msg = await message.reply_text("🔄 Starting content generation...")
        
        try:
            for i, (platform, desc) in enumerate(platforms.items(), 1):
                await progress_msg.edit_text(
                    f"🔄 Generating {desc}... ({i}/{len(platforms)})"
                )
                
                prompt = self._get_platform_prompt(platform, topic)
                content[platform] = await self.content_generator.generate_content(prompt)
            
            await progress_msg.edit_text("✅ Content generation completed!")
            return content
            
        except Exception as e:
            logger.error(f"Content generation error: {e}")
            await progress_msg.edit_text("❌ Error generating content. Please try again.")
            return {}

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Process user messages and generate content."""
        if not self._running:
            return

        user_id = update.effective_user.id
        self.analytics.track_activity(user_id, 'message')
        
        user_input = update.message.text
        
        try:
            processing_msg = await update.message.reply_text(
                "🎯 Got your topic! Starting content generation..."
            )
            
            content = await self.generate_platform_content(user_input, update.message)
            if content:
                context.user_data.update(content)
                
                keyboard = [
                    [InlineKeyboardButton("Twitter Post 🐦", callback_data="twitter")],
                    [InlineKeyboardButton("YouTube Script 🎥", callback_data="youtube")],
                    [InlineKeyboardButton("TikTok Idea 📱", callback_data="tiktok")]
                ]
                
                await processing_msg.edit_text(
                    "🎨 Content ready! Choose a platform:",
                    reply_markup=InlineKeyboardMarkup(keyboard)
                )
        except Exception as e:
            logger.error(f"Message handling error: {e}")
            await update.message.reply_text("❌ Error processing request. Please try again.")

    async def handle_button(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle platform selection buttons."""
        if not self._running:
            return

        query = update.callback_query
        await query.answer()
        
        user_id = query.from_user.id
        self.analytics.track_activity(user_id, f'view_{query.data}')
        
        platform_content = context.user_data.get(query.data, "No content available.")
        platform_emojis = {
            'twitter': '🐦',
            'youtube': '🎥',
            'tiktok': '📱'
        }
        
        message_text = (
            f"{platform_emojis.get(query.data, '📝')} *{query.data.title()} Content:*\n\n"
            f"{platform_content}\n\n"
            "✨ Ready to use! Copy and share!"
        )
        
        await query.edit_message_text(text=message_text, parse_mode=ParseMode.MARKDOWN)

    async def error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle errors in the bot."""
        logger.error(f"Update {update} caused error {context.error}")
        if update and update.effective_message:
            await update.effective_message.reply_text(
                "❌ Sorry, something went wrong.\n"
                "The error has been logged and will be addressed.\n"
                "Please try again later or contact support if the issue persists."
            )



if __name__ == "__main__":
    asyncio.run(main())  # This runs the main function