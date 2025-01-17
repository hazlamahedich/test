**DOCUMENTATION.md**:
```markdown
# Social Media Content Generation Bot - Detailed Documentation

## 1. System Architecture

### 1.1 Core Components
- **ContentGenerator**: Handles AI-powered content creation
- **WebResearchAgent**: Performs web research and information gathering
- **MediaHandler**: Manages media file processing and AI image generation
- **ConfigurationLoader**: Manages custom configuration and platform instructions

## 2. Configuration Management

### 2.1 Environment Variables
Key environment variables:
- `TELEGRAM_BOT_TOKEN`: Telegram bot authentication token
- `OPENAI_API_KEY`: OpenAI API access key

### 2.2 Configuration File (`config.yaml`)
Allows customization of:
- Global content instructions
- Platform-specific guidelines
- Tone and style preferences

#### Example Configuration
```yaml
global_instructions:
  tone: "Professional"
  style: "Engaging"

platform_instructions:
  twitter:
    specific_instructions: "Use trending hashtags"
    max_length: 280
3. Content Generation Workflow
3.1 Steps

User sends topic
Web research performed
Content generated for multiple platforms
Optional AI image generation
User can review and edit content

3.2 Platform Support

Twitter
Facebook
LinkedIn
Instagram
YouTube
YouTube Shorts
TikTok
Facebook Reels

4. Media Handling
4.1 Supported Media Types

Images: JPG, PNG, GIF
Video files

4.2 Platform Limitations

Twitter: 4 images max
Facebook: 10 images max
LinkedIn: 9 images max
Instagram: 10 images max

5. Advanced Features
5.1 AI Image Generation

Uses DALL-E for generating platform-specific images
Prompt derived from content research

5.2 Web Research

Multi-method web searching
Concurrent web scraping
Intelligent content summarization

6. Error Handling and Logging
6.1 Logging Levels

INFO: Standard operations
DEBUG: Detailed diagnostics
ERROR: Critical issues
WARNING: Potential problems

6.2 Error Recovery

Graceful error handling
Fallback to default configurations
Comprehensive error logging

7. Security Considerations

API key protection
Input sanitization
Rate limiting
Secure file handling

8. Performance Optimization

Asynchronous processing
Caching mechanisms
Efficient API calls

9. Extensibility

Modular architecture
Easy to add new platforms
Configurable content generation

10. Troubleshooting
Common Issues

API Key Problems
Network Connectivity
Rate Limit Exceeded

Recommended Diagnostics

Check .env file
Verify internet connection
Review logs in logs/ directory

11. Future Roadmap

Direct social media posting
Enhanced multi-language support
Advanced analytics