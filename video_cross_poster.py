# video_cross_poster.py
import os
import logging
from typing import Optional, Dict

import google_auth_oauthlib.flow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import facebook

class VideoCrossPoster:
    def __init__(self):
        """
        Initialize cross-posting capabilities
        """
        # YouTube API Setup
        self.youtube_client = self._setup_youtube_client()
        
        # Facebook API Setup
        self.facebook_client = self._setup_facebook_client()
        
        # Logging
        self.logger = logging.getLogger(__name__)

    def _setup_youtube_client(self) -> Optional[object]:
        """
        Set up YouTube API client
        """
        try:
            # YouTube authentication flow
            client_secrets_file = os.getenv('YOUTUBE_CLIENT_SECRETS')
            
            if not client_secrets_file:
                self.logger.warning("YouTube credentials not found")
                return None
            
            # Create flow instance to manage the OAuth 2.0 Authorization Grant
            flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(
                client_secrets_file, 
                ['https://www.googleapis.com/auth/youtube.upload']
            )
            
            # Run the flow to get credentials
            credentials = flow.run_local_server()
            
            # Build YouTube service
            youtube = build('youtube', 'v3', credentials=credentials)
            return youtube
        
        except Exception as e:
            self.logger.error(f"YouTube client setup error: {e}")
            return None

    def _setup_facebook_client(self) -> Optional[object]:
        """
        Set up Facebook Graph API client
        """
        try:
            FB_ACCESS_TOKEN = os.getenv('FACEBOOK_ACCESS_TOKEN')
            FB_PAGE_ID = os.getenv('FACEBOOK_PAGE_ID')
            
            if not FB_ACCESS_TOKEN or not FB_PAGE_ID:
                self.logger.warning("Facebook credentials not found")
                return None
            
            # Initialize Facebook Graph API
            graph = facebook.GraphAPI(FB_ACCESS_TOKEN)
            return graph
        except Exception as e:
            self.logger.error(f"Facebook client setup error: {e}")
            return None

    def cross_post_video(self, video_path: str, title: str, description: str) -> Dict[str, Optional[str]]:
        """
        Cross-post video to YouTube Shorts and Facebook Reels
        
        :param video_path: Path to the video file
        :param title: Title of the video
        :param description: Description for the video
        :return: Dictionary of upload results
        """
        results = {
            'youtube_shorts': None,
            'facebook_reels': None
        }

        try:
            # Upload to YouTube Shorts
            if self.youtube_client:
                results['youtube_shorts'] = self._upload_to_youtube_shorts(
                    video_path, title, description
                )

            # Upload to Facebook Reels
            if self.facebook_client:
                results['facebook_reels'] = self._upload_to_facebook_reels(
                    video_path, title
                )

        except Exception as e:
            self.logger.error(f"Cross-posting error: {e}")

        return results

    def _upload_to_youtube_shorts(self, video_path: str, title: str, description: str) -> Optional[str]:
        """
        Upload video to YouTube Shorts
        """
        try:
            body = {
                'snippet': {
                    'title': title,
                    'description': description,
                    'categoryId': '22'  # Entertainment category
                },
                'status': {
                    'privacyStatus': 'public'
                }
            }

            # Media upload
            media = MediaFileUpload(
                video_path, 
                mimetype='video/mp4', 
                resumable=True
            )

            # Perform the upload
            request = self.youtube_client.videos().insert(
                part='snippet,status',
                body=body,
                media_body=media
            )
            response = request.execute()

            # Construct video URL
            video_id = response['id']
            return f"https://youtube.com/shorts/{video_id}"

        except Exception as e:
            self.logger.error(f"YouTube Shorts upload error: {e}")
            return None

    def _upload_to_facebook_reels(self, video_path: str, caption: str) -> Optional[str]:
        """
        Upload video to Facebook Reels
        """
        try:
            FB_PAGE_ID = os.getenv('FACEBOOK_PAGE_ID')
            
            # Open the video file
            with open(video_path, 'rb') as video_file:
                # Upload video to Facebook
                post = self.facebook_client.put_video(
                    video_file, 
                    page_id=FB_PAGE_ID,
                    description=caption,
                    published=True
                )
            
            # Construct post URL
            post_id = post.get('id')
            return f"https://facebook.com/{FB_PAGE_ID}/posts/{post_id}" if post_id else None

        except Exception as e:
            self.logger.error(f"Facebook Reels upload error: {e}")
            return None