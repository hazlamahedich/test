# Social Media Content Generation Bot 🤖📱

## Overview
An advanced AI-powered Telegram bot that generates tailored content for multiple social media platforms using OpenAI and cutting-edge research techniques.

## 🌟 Features
- Multi-platform content generation
- AI-powered research
- Customizable content instructions
- Media file support
- Image generation with DALL-E
- Cross-platform content creation

## 🛠 Prerequisites
- Python 3.8+
- Telegram Bot Token
- OpenAI API Key

## 📦 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/social-media-content-bot.git
cd social-media-content-bot

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

##3. Install Dependencies
```bash
pip install -r requirements.txt

### 4. Configuration
Environment Variables
Create a .env file with:
CopyTELEGRAM_BOT_TOKEN=your_telegram_bot_token
OPENAI_API_KEY=your_openai_api_key
Custom Content Instructions
Edit config.yaml to customize content generation for different platforms.

#### 🚀 Running the Bot
``bash
python main.py

#### 📝 Usage

Start the bot on Telegram
Send a topic
Select platforms
Generate and customize content

#### 🔧 Customization

Modify config.yaml for platform-specific instructions
Adjust prompts in _get_platform_prompt
Configure media handling in media_handler.py

####🤝 Contributing

Fork the repository
Create your feature branch
Commit changes
Push to the branch
Create a Pull Request

📄 License
[Your License Here - e.g., MIT]
🙏 Acknowledgements

OpenAI
Telegram
Python Community
