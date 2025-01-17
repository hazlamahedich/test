import os
import asyncio
import logging
from datetime import datetime
from dotenv import load_dotenv

# Pydantic Models
from models import (
    ConfigModel, 
    PlatformContent, 
    ContentGenerationRequest, 
    validate_config
)

# Research and Content Generation Modules
from web_research import WebResearchAgent
from video_cross_poster import VideoCrossPoster

# Telegram Bot Imports
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# OpenAI for Content Generation
import openai

# Load environment variables
load_dotenv()

# Configure comprehensive logging
def setup_logging():
    """
    Set up detailed logging configuration
    """
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s',
        handlers=[
            # Console handler
            logging.StreamHandler(),
            # File handler with detailed logs
            logging.FileHandler('logs/social_media_bot.log', encoding='utf-8', mode='a')
        ]
    )

    # Set up loggers for specific modules
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('telegram').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)

    return logging.getLogger(__name__)

# Setup logger
logger = setup_logging()

class LoopDebugger:
    @staticmethod
    def debug_loop_info(location: str):
        try:
            current_loop = asyncio.get_running_loop()
            logger.debug(f"[{location}] Current loop id: {id(current_loop)}, Running: {current_loop.is_running()}")
        except RuntimeError:
            logger.debug(f"[{location}] No running event loop")

class ContentGenerator:
    """Handles content generation using OpenAI API"""
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
    
    async def generate_content(self, prompt: str, max_tokens: int = 300, platform: str = 'generic') -> str:
        """
        Generate content using OpenAI API with detailed logging
        
        :param prompt: Input prompt for content generation
        :param max_tokens: Maximum tokens for the generated content
        :param platform: Platform for which content is being generated
        :return: Generated content
        """
        logger.info(f"Starting content generation for {platform} platform")
        logger.debug(f"Prompt length: {len(prompt)} characters")
        
        try:
            # Log the start of OpenAI API call
            start_time = datetime.now()
            logger.info(f"Calling OpenAI API for {platform} content generation")
            
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens
            )
            
            # Calculate API response time
            response_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"OpenAI API response received for {platform}. Response time: {response_time:.2f} seconds")
            
            # Extract and log generated content
            generated_content = response.choices[0].message.content.strip()
            logger.info(f"Content generated for {platform}. Length: {len(generated_content)} characters")
            logger.debug(f"Generated content preview: {generated_content[:100]}...")
            
            return generated_content
        
        except Exception as e:
            logger.error(f"Error generating content for {platform}: {e}", exc_info=True)
            return f"Sorry, I couldn't generate content for {platform} at this time."

class SocialMediaBot:
    def __init__(self, config: ConfigModel):
        """
        Initialize the Social Media Bot with comprehensive configuration and components

        :param config: Configuration model containing API keys and settings
        """
        try:
            # Log initialization start
            logger.info("Initializing Social Media Bot")

            # Store configuration
            self.config = config
            logger.debug(f"Loaded configuration: Model {config.MODEL_NAME}, Max Tokens {config.MAX_TOKENS}")

            # Initialize OpenAI Client
            self.openai_client = openai.OpenAI(api_key=config.OPENAI_API_KEY)

            # Initialize content generation components
            self.content_generator = ContentGenerator(
                api_key=config.OPENAI_API_KEY,
                max_tokens=config.MAX_TOKENS,
                model_name=config.MODEL_NAME
            )
            logger.info("Content Generator initialized")

            # Initialize research agent
            self.research_agent = WebResearchAgent(
                openai_api_key=config.OPENAI_API_KEY
            )
            logger.info("Web Research Agent initialized")

            # Initialize media handling
            self.media_handler = MediaHandler(
                openai_api_key=config.OPENAI_API_KEY
            )
            logger.info("Media Handler initialized")

            # Initialize video cross-posting
            self.video_cross_poster = VideoCrossPoster()
            logger.info("Video Cross-Poster initialized")

            # Telegram application placeholder
            self.application = None

            # User session management
            self.user_sessions = {}

            # Platforms configuration
            self.platforms = {
                'twitter': {
                    'name': 'Twitter 🐦',
                    'max_images': 4,
                    'max_text_length': 280
                },
                'facebook': {
                    'name': 'Facebook 👥',
                    'max_images': 10,
                    'max_text_length': 5000
                },
                'linkedin': {
                    'name': 'LinkedIn 💼',
                    'max_images': 9,
                    'max_text_length': 3000
                },
                'instagram': {
                    'name': 'Instagram 📸',
                    'max_images': 10,
                    'max_text_length': 2200
                }   ,
                'youtube': {
                    'name': 'YouTube 🎥',
                    'max_video_length': 15 * 60  # 15 minutes
                },
                'youtube_shorts': {
                    'name': 'YouTube Shorts 📱',
                    'max_video_length': 60  # 60 seconds
                },
                'tiktok': {
                    'name': 'TikTok 🎵',
                    'max_video_length': 180  # 3 minutes
             },
                'facebook_reels': {
                    'name': 'Facebook Reels 🎬',
                    'max_video_length': 60  # 60 seconds
                }
            }

            # Logging additional configuration details
            logger.debug(f"Bot initialized with OpenAI model: {config.MODEL_NAME}")
            logger.debug(f"Maximum token limit: {config.MAX_TOKENS}")

            # Optional: Health check for API connections
            self._perform_initial_health_checks()

        except Exception as e:
            # Comprehensive error logging for initialization failures
            logger.critical(f"Fatal error during bot initialization: {e}", exc_info=True)
            raise

    def _perform_initial_health_checks(self):
        """
        Perform initial health checks for API connections
        """
        try:
            # OpenAI API health check
            test_response = self.content_generator.generate_content(
                "System health check", 
                max_tokens=10
            )
            logger.info("OpenAI API connection verified")

            # Optional: Add more health checks
            # Research Agent health check
            research_test = self.research_agent.web_search("test", num_results=1)
            logger.info("Web Research component initialized")

        except Exception as e:
            logger.warning(f"Initial health check failed: {e}")

    def _get_platform_prompt(self, platform: str, topic: str) -> str:
        """
        Generate a comprehensive platform-specific content generation prompt
    
        :param platform: Target social media platform
        :param topic: Content topic
        :return: Detailed, context-rich prompt for content generation
        """
        try:
            # Retrieve platform-specific configuration
            platform_config = config_loader.get_platform_instructions(platform)
        
            # Log the platform configuration for debugging
            logger.debug(f"Platform Configuration for {platform}: {platform_config}")

            # Determine max length (use platform-specific or default)
            max_length = platform_config.get('max_length', {
                'twitter': 280,
                'facebook': 5000,
                'linkedin': 3000,
                'instagram': 2200,
                'youtube': 5000,
                'youtube_shorts': 280,
                'tiktok': 280,
                'facebook_reels': 280
            }.get(platform, 500))

            # Comprehensive prompt template
            prompt_template = (
                "Content Generation Task:\n"
                f"Platform: {platform.replace('_', ' ').title()}\n"
                f"Topic: {topic}\n"
                "Objectives:\n"
                "1. Create engaging, platform-specific content\n"
                "2. Maintain the specified tone and style\n"
                "3. Adhere to platform-specific content guidelines\n\n"
            )

            # Platform-specific content generation guidelines
            platform_guidelines = {
                'twitter': (
                    f"{prompt_template}"
                    "Twitter Guidelines:\n"
                    "- Keep content concise and punchy\n"
                    "- Use relevant, trending hashtags\n"
                    "- Encourage immediate engagement\n"
                    f"- Maximum length: {max_length} characters\n"
                ),
                'facebook': (
                    f"{prompt_template}"
                    "Facebook Content Guidelines:\n"
                    "- Create a storytelling approach\n"
                    "- Use conversational tone\n"
                    "- Encourage comments and shares\n"
                    "- Include relevant emojis\n"
                    f"- Maximum length: {max_length} characters\n"
                ),
                'linkedin': (
                    f"{prompt_template}"
                    "LinkedIn Professional Guidelines:\n"
                    "- Maintain a professional, authoritative tone\n"
                    "- Focus on industry insights\n"
                    "- Provide professional value\n"
                    "- Use business-appropriate language\n"
                    f"- Maximum length: {max_length} characters\n"
                ),
                'instagram': (
                    f"{prompt_template}"
                    "Instagram Content Guidelines:\n"
                    "- Create visually inspiring content\n"
                    "- Use trendy, energetic language\n"
                    "- Include relevant, popular hashtags\n"
                    "- Encourage visual storytelling\n"
                    f"- Maximum length: {max_length} characters\n"
                ),
                'youtube': (
                    f"{prompt_template}"
                    "YouTube Content Guidelines:\n"
                    "- Develop a clear narrative structure\n"
                    "- Include educational or entertaining elements\n"
                    "- Create a compelling intro\n"
                    "- End with a strong call-to-action\n"
                    "- Consider viewer engagement\n"
                ),
                'youtube_shorts': (
                    f"{prompt_template}"
                    "YouTube Shorts Guidelines:\n"
                    "- Grab attention in the first 3 seconds\n"
                    "- Keep content fast-paced and dynamic\n"
                    "- Use trending sounds or visual effects\n"
                    "- Ensure quick, clear message delivery\n"
                    f"- Maximum length: {max_length} characters\n"
                ),
                'tiktok': (
                    f"{prompt_template}"
                    "TikTok Content Guidelines:\n"
                    "- Follow current trends\n"
                    "- Use popular sound bites\n"
                    "- Create visually engaging content\n"
                    "- Encourage viral potential\n"
                    f"- Maximum length: {max_length} characters\n"
                ),
                'facebook_reels': (
                    f"{prompt_template}"
                    "Facebook Reels Guidelines:\n"
                    "- Create entertaining, shareable content\n"
                    "- Use quick, dynamic storytelling\n"
                    "- Incorporate trending music or effects\n"
                    "- Aim for viral potential\n"
                    f"- Maximum length: {max_length} characters\n"
                )
            }

            # Start with platform-specific guidelines
            base_prompt = platform_guidelines.get(platform, prompt_template)

            # Incorporate custom instructions from config
            custom_instructions = platform_config.get('instructions', '')
            if custom_instructions:
                base_prompt += f"\nCustom Instructions:\n{custom_instructions}\n"

            # Final prompt with additional context
            final_prompt = (
                f"{base_prompt}\n"
                "Additional Considerations:\n"
                "- Ensure content is original and engaging\n"
                "- Align with the latest trends and audience expectations\n"
                "- Proofread for clarity and impact\n\n"
                "Generated Content:"
            )

            return final_prompt

        except Exception as e:
            # Fallback to default prompt if any error occurs
            logger.error(f"Error generating platform prompt for {platform}: {e}")
            return (
                f"Generate engaging content about {topic} for {platform}. "
                "Ensure the content is relevant, concise, and platform-appropriate."
            )

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Handle the /start command in the Telegram bot
        
        :param update: Incoming update from Telegram
        :param context: Context for the current conversation
        """
        try:
            # Log the start command interaction
            logger.info(f"Start command received from user {update.effective_user.id}")
            
            await update.message.reply_text(
                "👋 Hi! I'm Jimmy, your AI Social Media Content Generator!\n\n"
                "🚀 What I can do for you:\n"
                "• Generate content for multiple platforms\n"
                "• Provide AI-powered research insights\n"
                "• Create engaging posts across:\n"
                "  - Twitter 🐦\n"
                "  - Facebook 👥\n"
                "  - LinkedIn 💼\n"
                "  - Instagram 📸\n"
                "  - YouTube 🎥\n"
                "  - YouTube Shorts 📱\n"
                "  - TikTok 🎵\n"
                "  - Facebook Reels 🎬\n\n"
                "💡 Simply send me a topic, and I'll create awesome content!"
            )
            
            # Log successful start command response
            logger.info("Start command response sent successfully")
        
        except Exception as e:
            # Log any errors during start command
            logger.error(f"Error in start command: {e}", exc_info=True)
            
            try:
                await update.message.reply_text(
                    "❌ Sorry, something went wrong. Please try again later."
                )
            except Exception as reply_error:
                logger.error(f"Error sending error message: {reply_error}")

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Enhanced message handler with comprehensive content generation and media integration
    
        Workflow:
        1. Generate content for multiple platforms
        2. Check for existing media files
        3. Offer AI image generation if no media
        4. Prepare platform selection keyboard
        """
        user_input = update.message.text
        user_id = update.effective_user.id
    
        try:
            # Log the incoming message
            logger.info(f"Received message from user {user_id}: {user_input}")
        
            # Start processing message
            processing_msg = await update.message.reply_text(
                "🎯 Got your topic! Starting content generation...\n"
                "This might take a few moments."
            )
    
            # Generate content for multiple platforms
            content = await self.generate_platform_content(user_input, update.message)
        
            if content:
                # Log successful content generation
                logger.info(f"Content generated successfully for user {user_id}")
            
                # Store full content in user_data
                context.user_data['generated_content'] = content.model_dump()
                context.user_data['last_topic'] = user_input
            
                # Check for attached media files from previous interaction
                media_files = context.user_data.get('attached_media', [])
            
                # Determine media handling flow
                if not media_files:
                    # No media files attached, offer AI image generation
                    keyboard = [
                        [
                            InlineKeyboardButton("Generate AI Image 🖼️", callback_data="generate_ai_image"),
                            InlineKeyboardButton("No Image 🚫", callback_data="no_image")
                        ]
                    ]
                
                    await processing_msg.edit_text(
                        "🖼️ Would you like to add an image to your posts?\n\n"
                        "• Generate an AI image based on your content\n"
                        "• Skip image generation\n"
                        "• Or upload your own image in the next message",
                        reply_markup=InlineKeyboardMarkup(keyboard)
                    )
                else:
                    # Media files already attached, validate and proceed
                    try:
                        # Validate media files for each platform
                        validated_media = {}
                        for platform in ['twitter', 'facebook', 'linkedin', 'instagram']:
                            try:
                                validated_media[platform] = await self.media_handler.validate_media_files(
                                    media_files, 
                                    platform
                                )
                            except ValueError as ve:
                                logger.warning(f"Media validation error for {platform}: {ve}")
                    
                        # Store validated media
                        context.user_data['validated_media'] = validated_media
                    
                        # Proceed to platform selection
                        await self._show_platform_selection(processing_msg)
                
                    except Exception as media_error:
                        logger.error(f"Media validation error: {media_error}")
                        await processing_msg.edit_text(
                            "❌ Error processing media files. Please try again or upload new files."
                        )
            else:
                # Content generation failed
                await processing_msg.edit_text(
                    "❌ Failed to generate content. Please try a different topic."
                )
    
        except Exception as e:
            # Comprehensive error handling
            logger.error(f"Error in message handling for user {user_id}: {e}", exc_info=True)
        
            await update.message.reply_text(
                "❌ Sorry, I encountered an unexpected error.\n"
                "Please try generating content again."
            )

    async def handle_media_upload(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Handle media file uploads
        """
        try:
            # Get uploaded files
            files = update.message.photo or update.message.document
        
            if files:
                # Download and save files
                media_files = []
                for file in files:
                    file_path = await file.get_file()
                    downloaded_file = await file_path.download()
                    media_files.append(downloaded_file)
            
                # Store media files in user context
                context.user_data['attached_media'] = media_files
            
                await update.message.reply_text(
                    "Media files received! They will be used with your posts."
                )
        
        except Exception as e:
            logger.error(f"Error handling media upload: {e}")

async def _show_platform_selection(self, processing_msg) -> None:
    """
    Show platform selection keyboard
    
    :param processing_msg: Message to edit with platform selection
    """
    try:
        # Prepare platform selection keyboard
        keyboard = []
        platforms = [
            ('twitter', "Twitter 🐦"),
            ('facebook', "Facebook 👥"),
            ('linkedin', "LinkedIn 💼"),
            ('instagram', "Instagram 📸"),
            ('youtube', "YouTube 🎥"),
            ('youtube_shorts', "YouTube Shorts 📱"),
            ('tiktok', "TikTok 🎵"),
            ('facebook_reels', "Facebook Reels 🎬")
        ]
        
        # Create keyboard rows
        for platform, platform_name in platforms:
            keyboard.append([
                InlineKeyboardButton(f"View {platform_name}", callback_data=f"view_{platform}"),
                InlineKeyboardButton("Regenerate 🔄", callback_data=f"regen_{platform}"),
                InlineKeyboardButton("Edit ✏️", callback_data=f"edit_{platform}")
            ])
        
        await processing_msg.edit_text(
            "🎨 Content generation successful!\n\n"
            "*Available Platforms:*\n"
            "Select a platform to view or modify content:",
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode=ParseMode.MARKDOWN
        )
    
    except Exception as e:
        logger.error(f"Error showing platform selection: {e}")
        await processing_msg.edit_text(
            "❌ Error preparing platform selection. Please try again."
        )

    async def handle_button(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Enhanced button handler with multiple functionalities:
        - AI Image Generation
        - Platform Content View
        - Content Regeneration
        - Content Editing
        - Navigation
        """
        try:
            query = update.callback_query
            await query.answer()

            # Extract action and platform (if applicable)
            callback_data = query.data
            parts = callback_data.split('_', 1)
            action = parts[0]
            platform = parts[1] if len(parts) > 1 else None

            # Retrieve stored content
            generated_content = context.user_data.get('generated_content', {})
            last_topic = context.user_data.get('last_topic', '')

            # AI Image Generation Handling
            if action == 'generate':
                # Use research context or generated content for image prompt
                research_context = generated_content.get('research_context', {})
                prompt = research_context.get('research_summary', last_topic)
            
                # Generate AI image
                image_path = await self.media_handler.generate_dalle_image(prompt)
            
                if image_path:
                    # Store generated image
                    context.user_data['ai_generated_image'] = image_path
                
                    # Prepare platform selection keyboard
                    keyboard = self._create_platform_keyboard()
                
                    await query.edit_message_text(
                        "🖼️ AI Image Generated Successfully!\n"
                        "Select a platform to view or modify content:",
                        reply_markup=InlineKeyboardMarkup(keyboard),
                        parse_mode=ParseMode.MARKDOWN
                    )
                else:
                    await query.edit_message_text(
                        "❌ Failed to generate AI image. Would you like to try again?",
                        reply_markup=InlineKeyboardMarkup([
                            [
                                InlineKeyboardButton("Try Again 🔄", callback_data="generate_ai_image"),
                                InlineKeyboardButton("Skip Image 🚫", callback_data="no_image")
                            ]
                        ])
                    )

            # No Image Option
            elif action == 'no':
                # Prepare platform selection keyboard
                keyboard = self._create_platform_keyboard()
            
                await query.edit_message_text(
                    "🎨 Proceeding without an image.\n"
                    "Select a platform to view or modify content:",
                    reply_markup=InlineKeyboardMarkup(keyboard),
                    parse_mode=ParseMode.MARKDOWN
            )

            # View Platform Content
            elif action == 'view' and platform:
                # Retrieve specific platform content
                content = generated_content.get(platform, "No content available.")
            
                # Prepare view keyboard with actions
                view_keyboard = [
                    [
                        InlineKeyboardButton("Regenerate 🔄", callback_data=f"regen_{platform}"),
                        InlineKeyboardButton("Edit ✏️", callback_data=f"edit_{platform}"),
                        InlineKeyboardButton("Post 📤", callback_data=f"post_{platform}")
                    ],
                    [
                        InlineKeyboardButton("Back to Platforms 🔙", callback_data="back_to_platforms"),
                        InlineKeyboardButton("Back to Start 🏠", callback_data="back_to_start")
                    ]
                ]

                await query.edit_message_text(
                    f"*{platform.replace('_', ' ').title()} Content:*\n\n"
                    f"{content}\n\n"
                    "Choose an action:",
                    reply_markup=InlineKeyboardMarkup(view_keyboard),
                    parse_mode=ParseMode.MARKDOWN
                )

            # Regenerate Platform Content
            elif action == 'regen' and platform:
                # Generate new content for the specific platform
                prompt = self._get_platform_prompt(platform, last_topic)
                new_content = await self.content_generator.generate_content(prompt)
            
                # Update content in user data
                generated_content[platform] = new_content
                context.user_data['generated_content'] = generated_content
            
                # Prepare regeneration keyboard
                regen_keyboard = [
                    [
                        InlineKeyboardButton("View 👀", callback_data=f"view_{platform}"),
                        InlineKeyboardButton("Edit ✏️", callback_data=f"edit_{platform}"),
                        InlineKeyboardButton("Post 📤", callback_data=f"post_{platform}")
                    ],
                    [
                        InlineKeyboardButton("Back to Platforms 🔙", callback_data="back_to_platforms"),
                        InlineKeyboardButton("Back to Start 🏠", callback_data="back_to_start")
                    ]
                ]

                await query.edit_message_text(
                    f"*Regenerated {platform.replace('_', ' ').title()} Content:*\n\n"
                    f"{new_content}\n\n"
                    "Choose an action:",
                    reply_markup=InlineKeyboardMarkup(regen_keyboard),
                    parse_mode=ParseMode.MARKDOWN
                )

            # Edit Platform Content
            elif action == 'edit' and platform:
                await query.edit_message_text(
                    f"*Editing {platform.replace('_', ' ').title()} Content:*\n\n"
                    "Send me the updated content for this platform.",
                    reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("Cancel ❌", callback_data=f"view_{platform}")]
                    ]),
                    parse_mode=ParseMode.MARKDOWN
                )
                # Store editing context
                context.user_data['editing_platform'] = platform

            # Post to Platform (Placeholder)
            elif action == 'post' and platform:
                await query.edit_message_text(
                    f"🚧 Posting to {platform.replace('_', ' ').title()} is not implemented yet.\n"
                    "Future versions will support direct social media posting.",
                    reply_markup=InlineKeyboardMarkup([
                        [
                            InlineKeyboardButton("Back to Content 🔙", callback_data=f"view_{platform}"),
                            InlineKeyboardButton("Back to Start 🏠", callback_data="back_to_start")
                        ]
                    ]),
                    parse_mode=ParseMode.MARKDOWN
                )

            # Navigation Actions
            elif callback_data == 'back_to_platforms':
                # Return to platform selection
                keyboard = self._create_platform_keyboard()
                await query.edit_message_text(
                    "🎨 Select a platform to view or modify content:",
                    reply_markup=InlineKeyboardMarkup(keyboard),
                    parse_mode=ParseMode.MARKDOWN
                )

            elif callback_data == 'back_to_start':
                # Clear user data and return to start
                context.user_data.clear()
                await query.edit_message_text(
                    "🏠 Returned to start. Send a new topic to begin.",
                    reply_markup=ReplyKeyboardRemove()
                )

            else:
                # Unexpected callback
                await query.edit_message_text(
                    "❓ Unexpected action. Please try again.",
                    parse_mode=ParseMode.MARKDOWN
                )

        except Exception as e:
            logger.error(f"Error in button handler: {e}", exc_info=True)
            await query.edit_message_text(
                "❌ An unexpected error occurred. Please try again.",
                parse_mode=ParseMode.MARKDOWN
            )

def _create_platform_keyboard(self):
    """
    Create platform selection keyboard
    
    :return: List of keyboard buttons
    """
    platforms = [
        ('twitter', "Twitter 🐦"),
        ('facebook', "Facebook 👥"),
        ('linkedin', "LinkedIn 💼"),
        ('instagram', "Instagram 📸"),
        ('youtube', "YouTube 🎥"),
        ('youtube_shorts', "YouTube Shorts 📱"),
        ('tiktok', "TikTok 🎵"),
        ('facebook_reels', "Facebook Reels 🎬")
    ]
    
    keyboard = []
    for platform, platform_name in platforms:
        keyboard.append([
            InlineKeyboardButton(f"View {platform_name}", callback_data=f"view_{platform}"),
            InlineKeyboardButton("Regenerate 🔄", callback_data=f"regen_{platform}"),
            InlineKeyboardButton("Edit ✏️", callback_data=f"edit_{platform}")
        ])
    
    return keyboard

    async def handle_edit_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Handle message sent during editing mode
        """
        try:
            # Check if user is in editing mode
            editing_platform = context.user_data.get('editing_platform')
            if not editing_platform:
                return

            # Get the new content
            new_content = update.message.text

            # Update the content in user_data
            generated_content = context.user_data.get('generated_content', {})
            generated_content[editing_platform] = new_content
            context.user_data['generated_content'] = generated_content

            # Clear editing mode
            del context.user_data['editing_platform']

            # Prepare edit confirmation keyboard
            edit_keyboard = [
                [
                    InlineKeyboardButton("View 👀", callback_data=f"view_{editing_platform}"),
                    InlineKeyboardButton("Regenerate 🔄", callback_data=f"regen_{editing_platform}"),
                    InlineKeyboardButton("Post 📤", callback_data=f"post_{editing_platform}")
                ],
                [
                    InlineKeyboardButton("Back to Start 🏠", callback_data="back_to_start")
                ]
            ]

            await update.message.reply_text(
                f"*{editing_platform.replace('_', ' ').title()} Content Updated:*\n\n"
                f"{new_content}\n\n"
                "Choose an action:",
                reply_markup=InlineKeyboardMarkup(edit_keyboard),
                parse_mode=ParseMode.MARKDOWN
            )

        except Exception as e:
            logger.error(f"Error handling edit: {e}")
            await update.message.reply_text(
                "❌ An error occurred while editing. Please try again.",
                parse_mode=ParseMode.MARKDOWN
            )

    async def setup(self) -> None:
        """Set up the Telegram bot with media handling"""
        try:
            self.application = Application.builder().token(self.config.TELEGRAM_BOT_TOKEN).build()
        
            # Add handlers
            self.application.add_handler(CommandHandler("start", self.start_command))
            self.application.add_handler(MessageHandler(
                filters.TEXT & ~filters.COMMAND & ~filters.UpdateType.EDITED_MESSAGE, 
                self.handle_message
            ))
            # Media upload handler
            self.application.add_handler(MessageHandler(
                filters.PHOTO | filters.Document.IMAGE, 
                self.handle_media_upload
            ))
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
        config = validate_config()

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