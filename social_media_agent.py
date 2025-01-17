import os
import logging
import asyncio
from dataclasses import dataclass
from typing import List, Optional
from dotenv import load_dotenv
import openai
from telegram import (
    Update, 
    InlineKeyboardButton, 
    InlineKeyboardMarkup
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

# Configure logging (at the top of the file)
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    level=logging.DEBUG  # Change this from INFO to DEBUG
)
logger = logging.getLogger(__name__)

def debug_loop_state():
    try:
        loop = asyncio.get_event_loop()
        print(f"DEBUG: Current loop ID: {id(loop)}, Running: {loop.is_running()}")
    except Exception as e:
        print(f"DEBUG: Error getting loop info: {e}")


# 1. Simplified Config Class
@dataclass
class Config:
    """Simplified configuration class for the bot."""
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN")
    MAX_TOKENS: int = 300
    MODEL_NAME: str = "gpt-3.5-turbo"

# 2. Simplified Content Generator
class ContentGenerator:
    """Handles content generation using OpenAI API."""
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
    
    async def generate_content(self, prompt: str, max_tokens: int = 300) -> str:
        """Generate content using OpenAI API."""
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating content: {e}")
            return "Sorry, I couldn't generate content at this time."
        
class LoopDebugger:
    @staticmethod
    def debug_loop_info(location: str):
        try:
            current_loop = asyncio.get_running_loop()
            logger.debug(f"[{location}] Current loop id: {id(current_loop)}, Running: {current_loop.is_running()}")
        except RuntimeError:
            logger.debug(f"[{location}] No running event loop")


# 3. Simplified SocialMediaBot Class
class SocialMediaBot:
    """Simplified bot class focusing on content generation."""
    def __init__(self, config: Config):
        self.config = config
        self.content_generator = ContentGenerator(config.OPENAI_API_KEY)
        self.application = None

    def _get_platform_prompt(self, platform: str, topic: str) -> str:
        """Get platform-specific prompt."""
        prompts = {
            'twitter': (
                f"Create an engaging Twitter post about: {topic}. "
                "Include relevant hashtags. Keep it under 280 characters."
            )   ,
            'facebook': (
                f"Create an engaging Facebook post about: {topic}. "
                "Include emojis and hashtags. Aim for 1-2 paragraphs with "
                "a strong call to action."
            ),
            'linkedin': (
                f"Create a professional LinkedIn post about: {topic}. "
                "Include industry-relevant hashtags, use professional tone, "
                "and format with line breaks for readability. Focus on business value."
            ),
            'instagram': (
                f"Create an Instagram caption about: {topic}. "
                "Include relevant hashtags (max 30), emojis, and engaging questions. "
                "Make it visually appealing with line breaks."
            ),
            'youtube': (
                f"Create a YouTube video script outline about: {topic}. "
                "Include: 1) Attention-grabbing intro 2) Main points "
                "3) Call to action. Keep it concise but comprehensive."
            ),
            'youtube_shorts': (
                f"Create a YouTube Shorts content idea about: {topic}. "
                "Include: 1) Hook (first 3 seconds) 2) Quick-paced content "
                "3) Call to action. Keep it under 60 seconds."
            ),
            'tiktok': (
                f"Create a TikTok content idea about: {topic}. "
                "Include: 1) Hook 2) Main content structure "
                "3) Trending elements or sounds that could work well."
            ),
            'facebook_reels': (
                f"Create a Facebook Reels content idea about: {topic}. "
                "Include: 1) Attention-grabbing opener 2) Quick content delivery "
                "3) Trending music suggestion 4) Call to action. Keep it engaging and fast-paced."
            )
        }
        return prompts.get(platform, "")

    async def generate_platform_content(self, topic: str, message) -> dict:
        """Generate content for different platforms."""
        platforms = {
            'twitter': "Twitter post",
            'facebook': "Facebook post",
            'linkedin': "LinkedIn post",
            'instagram': "Instagram caption",
            'youtube': "YouTube script",
            'youtube_shorts': "YouTube Shorts idea",
            'tiktok': "TikTok idea",
            'facebook_reels': "Facebook Reels idea"
        }

        content = {}
        progress_msg = await message.reply_text("🔄 Starting content generation...")

        try:
            for i, (platform, desc) in enumerate(platforms.items(), 1):
                await progress_msg.edit_text(
                    f"🔄 Generating content... ({i}/{len(platforms)})\n"
                    f"Currently working on: {desc}"
                )

                prompt = self._get_platform_prompt(platform, topic)
                generated_content = await self.content_generator.generate_content(prompt)
                content[platform] = generated_content

                # Debugging: Print the generated content for each platform
                logger.debug(f"Generated content for {platform}: {generated_content}")

            await progress_msg.edit_text("✅ Content generation completed!")
            return content

        except Exception as e:
            logger.error(f"Error in content generation: {e}")
            await progress_msg.edit_text(
                "❌ Error occurred during content generation.\n"
                "Please try again later."
            )
            return {}
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Simplified start command."""
        try:
            await update.message.reply_text(
                "👋 Hi! I'm Jimmy, your Social Media Content Generator!\n\n"
                "Simply send me a topic or description, and I'll generate content for:\n"
                "• Twitter\n"
                "• Facebook\n"
                "• LinkedIn\n"
                "• Instagram\n"
                "• YouTube\n"
                "• YouTube Shorts\n"
                "• TikTok\n"
                "• Facebook Reels\n\n"
                "Just type your topic to get started!"
            )
        except Exception as e:
            logger.error(f"Error in start command: {e}")
            await update.message.reply_text("❌ Sorry, something went wrong. Please try again later.")

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Simplified message handler focusing on content generation."""
        user_input = update.message.text
    
        try:
            processing_msg = await update.message.reply_text(
                "🎯 Got your topic! Starting content generation...\n"
                "This might take a few moments."
            )
    
            content = await self.generate_platform_content(user_input, update.message)
            if content:
                # Debug log to verify content
                logger.debug(f"Generated content keys: {content.keys()}")
            
                # Store content in user_data
                context.user_data.update(content)
                logger.debug(f"Stored content keys in user_data: {context.user_data.keys()}")



                # Verify storage
                logger.debug(f"Stored content keys in user_data: {context.user_data.keys()}")  # Make sure content is stored in context

                keyboard = [
                    [InlineKeyboardButton("Twitter Post 🐦", callback_data="twitter")],
                    [InlineKeyboardButton("Facebook Post 👥", callback_data="facebook")],
                    [InlineKeyboardButton("LinkedIn Post 💼", callback_data="linkedin")],
                    [InlineKeyboardButton("Instagram Caption 📸", callback_data="instagram")],
                    [InlineKeyboardButton("YouTube Script 🎥", callback_data="youtube")],
                    [InlineKeyboardButton("YouTube Shorts 📱", callback_data="youtube_shorts")],
                    [InlineKeyboardButton("TikTok Idea 🎵", callback_data="tiktok")],
                    [InlineKeyboardButton("Facebook Reels 🎬", callback_data="facebook_reels")]
            ]
                
                await processing_msg.edit_text(
                    "🎨 Content generation successful!\n\n"
                    "*Available Content:*\n"
                    "• Twitter Post\n"
                    "• Facebook Post\n"
                    "• LinkedIn Post\n"
                    "• Instagram Caption\n"
                    "• YouTube Script\n"
                    "• YouTube Shorts\n"
                    "• TikTok Content\n"
                    "• Facebook Reels\n\n"
                    "Select a platform to view the content:",
                    reply_markup=InlineKeyboardMarkup(keyboard),
                    parse_mode=ParseMode.MARKDOWN
                )
        except Exception as e:
            logger.error(f"Error generating content: {e}")
            await update.message.reply_text(
                "❌ Sorry, I encountered an error while generating content.\n"
                "Please try again later."
            )


    async def handle_button(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle platform selection buttons and display content."""
        try:
            query = update.callback_query
            await query.answer()

            # Debugging: Log available content keys to ensure the correct content is accessible
            print("\n=== Available Content Keys ===")
            print(context.user_data.keys())
    
            # Fetch the content for the selected platform
            platform_content = context.user_data.get(query.data, "No content available.")

            # Debugging: Log the content being accessed
            print(f"\n=== Content for {query.data} ===")
            print(platform_content)
            print("============================\n")

            # If the content is found, send it to the user
            if platform_content != "No content available.":
                platform_emojis = {
                    'twitter': '🐦',
                    'facebook': '👥',
                    'linkedin': '💼',
                    'instagram': '📸',
                    'youtube': '🎥',
                    'youtube_shorts': '📱',
                    'tiktok': '🎵',
                    'facebook_reels': '🎬'
                }

                platform_name = query.data.replace('_', ' ').title()

                # Format the message with content
                message_text = (
                    f"{platform_emojis.get(query.data, '📝')} *{platform_name} Content:*\n\n"
                    f"{platform_content}\n\n"
                    "✨ Content has been generated and is ready to use!\n"
                    "💡 You can now copy and paste this content to your preferred platform."
                )

                # Debugging: Print the message being sent
                print("\n=== Attempting to send message ===")
                print(message_text)
                print("================================\n")
        
                # Send the formatted message to the user
                await query.edit_message_text(
                    text=message_text,
                    parse_mode=ParseMode.MARKDOWN
                )
            else:
                # If no content is found, send a message indicating no content available
                await query.edit_message_text(
                    "❌ No content available for this platform. Please try generating content again."
                )

        except Exception as e:
            print(f"\n=== Error in button handler ===")
            print(f"Error: {str(e)}")
            print("=============================\n")
            logger.error(f"Error in button handler: {e}")
            try:
                await query.edit_message_text(
                    "❌ Error displaying content. Please try generating content again."
                )
            except Exception as edit_error:
                logger.error(f"Error sending error message: {edit_error}")

    async def setup(self) -> None:
        """Simplified setup with only necessary handlers."""
        try:
            self.application = Application.builder().token(self.config.TELEGRAM_BOT_TOKEN).build()
            
            # Add only necessary handlers
            self.application.add_handler(CommandHandler("start", self.start_command))
            self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
            self.application.add_handler(CallbackQueryHandler(self.handle_button))
            
            logger.info("Bot setup completed successfully")
        except Exception as e:
            logger.error(f"Error during bot setup: {e}")
            raise

    async def start(self) -> None:
        """Start the bot."""
        try:
            print("DEBUG: Starting bot setup...")
            current_loop = asyncio.get_event_loop()
            print(f"DEBUG: Current loop ID in start(): {id(current_loop)}")
        
            await self.setup()
            logger.debug(f"Setup complete. Loop ID: {id(current_loop)}")
            print(f"DEBUG: Setup complete. Loop ID: {id(current_loop)}")
        
            logger.info("Bot started successfully")
            print("DEBUG: About to start polling...")
        
            # Modified polling approach
            await self.application.initialize()
            await self.application.start()
            await self.application.updater.start_polling()
        
            # Keep the bot running
            while True:
                await asyncio.sleep(1)
            
        except Exception as e:
            print(f"DEBUG: Error in start(): {e}")
            logger.error(f"Error starting bot: {e}")
            await self.stop()

    async def stop(self) -> None:
        """Stop the bot with detailed loop debugging."""
        if self.application:
            logger.info("Stopping bot...")
            LoopDebugger.debug_loop_info("Bot Stop - Before Shutdown")
            try:
                if self.application.running:
                    await self.application.stop()
                await self.application.shutdown()
                logger.info("Bot stopped successfully")
            except Exception as e:
                logger.error(f"Error stopping bot: {e}")

    async def error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle errors occurring in the bot."""
        logger.error(f"Update {update} caused error {context.error}")
        try:
            if update and update.effective_message:
                await update.effective_message.reply_text(
                    "❌ Sorry, something went wrong. Please try again later."
                )
        except Exception as e:
            logger.error(f"Error sending error message: {e}")

    def cleanup(self) -> None:
        """Perform any necessary cleanup."""
        logger.info("Performing cleanup...")
        try:
            # Add any cleanup operations here if needed
            pass
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


async def main() -> None:
    """Main function to run the bot."""
    bot = None
    try:
        # Initialize configuration
        config = Config()

        # Validate configuration
        if not config.TELEGRAM_BOT_TOKEN:
            logger.error("No telegram bot token provided. Please set TELEGRAM_BOT_TOKEN environment variable.")
            return
        if not config.OPENAI_API_KEY:
            logger.error("No OpenAI API key provided. Please set OPENAI_API_KEY environment variable.")
            return

        # Create bot instance
        bot = SocialMediaBot(config)

        # Start the bot
        await bot.start()

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        if bot:
            await bot.stop()


if __name__ == "__main__":
    print("DEBUG: Starting main process...")
    
    try:
        print("DEBUG: Creating new event loop...")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        print("DEBUG: Set as default event loop")
        
        # Run the main function in the loop
        try:
            loop.run_until_complete(main())
        except KeyboardInterrupt:
            print("DEBUG: Keyboard interrupt received")
            # Gracefully stop the bot
            if 'bot' in locals():
                loop.run_until_complete(bot.stop())
        finally:
            loop.close()
            
    except Exception as e:
        print(f"DEBUG: Fatal error: {e}")
    finally:
        print("DEBUG: Shutdown complete")
