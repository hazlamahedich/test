# web_research.py
import os
import asyncio
import logging
import json
from typing import List, Dict, Any, Optional

import aiohttp
from bs4 import BeautifulSoup
from googlesearch import search
import requests
import openai

class WebResearchAgent:
    """Advanced web research agent for content generation"""
    def __init__(self, openai_api_key: str):
        """
        Initialize research capabilities
        
        :param openai_api_key: OpenAI API key for processing
        """
        # OpenAI Client
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
    def get_trending_topics(self) -> List[str]:
        """
        Retrieve trending topics from various sources
        
        :return: List of trending topics
        """
        trending_topics = []
        
        try:
            # Use Google News for trending topics
            google_news_url = "https://news.google.com/topstories"
            
            response = requests.get(google_news_url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            })
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract headlines as potential trending topics
                headlines = soup.find_all('h3')
                trending_topics = [
                    headline.get_text().strip() 
                    for headline in headlines 
                    if headline.get_text().strip()
                ][:10]  # Limit to top 10
        
        except Exception as e:
            self.logger.error(f"Error fetching trending topics: {e}")
        
        # Fallback to AI-generated trending topics if web scraping fails
        if not trending_topics:
            try:
                trending_response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a trend analyzer."},
                        {"role": "user", "content": "List 10 current trending topics across various domains."}
                    ],
                    max_tokens=200
                )
                
                # Parse the AI-generated trends
                trending_topics = [
                    topic.strip() 
                    for topic in trending_response.choices[0].message.content.split('\n') 
                    if topic.strip()
                ]
            except Exception as e:
                self.logger.error(f"Error generating AI trends: {e}")
        
        return trending_topics

    async def web_search(self, query: str, num_results: int = 5) -> List[Dict[str, str]]:
        """
        Perform web search and extract key information
        
        :param query: Search query
        :param num_results: Number of search results to retrieve
        :return: List of search result dictionaries
        """
        results = []
        try:
            # Use Google search
            search_results = list(search(query, num_results=num_results))
            
            # Async web scraping
            async with aiohttp.ClientSession() as session:
                async def fetch_page_content(url):
                    try:
                        async with session.get(url, timeout=10) as response:
                            if response.status == 200:
                                html = await response.text()
                                soup = BeautifulSoup(html, 'html.parser')
                                
                                # Extract key information
                                title = soup.title.string if soup.title else url
                                # Try to get meta description
                                meta_desc = soup.find('meta', attrs={'name': 'description'})
                                description = meta_desc['content'] if meta_desc else ''
                                
                                return {
                                    'url': url,
                                    'title': title,
                                    'description': description
                                }
                    except Exception as e:
                        self.logger.error(f"Error scraping {url}: {e}")
                        return None
                
                # Gather results concurrently
                tasks = [fetch_page_content(url) for url in search_results]
                results = await asyncio.gather(*tasks)
                
            # Filter out None results
            return [r for r in results if r]
        
        except Exception as e:
            self.logger.error(f"Web search error: {e}")
            return []

    async def research_topic(self, topic: str) -> Dict[str, Any]:
        """
        Comprehensive topic research
        
        :param topic: Topic to research
        :return: Detailed research summary
        """
        try:
            # Web search
            web_results = await self.web_search(topic)
            
            # Get trending topics for context
            trending_topics = self.get_trending_topics()
            
            # Use OpenAI to summarize and contextualize research
            research_summary = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a comprehensive research assistant. Analyze the following web search results and trending topics to provide a detailed, insightful summary."},
                    {"role": "user", "content": f"Topic: {topic}\n\n"
                     f"Web Search Results:\n{json.dumps(web_results, indent=2)}\n\n"
                     f"Current Trending Topics:\n{', '.join(trending_topics)}\n\n"
                     "Please provide a comprehensive research summary including:"
                     "1. Key insights\n"
                     "2. Current context\n"
                     "3. Potential angles for content creation\n"
                     "4. How this relates to current trends"}
                ],
                max_tokens=500
            )
            
            # Extract summary
            summary = research_summary.choices[0].message.content
            
            return {
                'topic': topic,
                'web_results': web_results,
                'trending_topics': trending_topics,
                'research_summary': summary
            }
        
        except Exception as e:
            self.logger.error(f"Comprehensive research error: {e}")
            return {
                'topic': topic,
                'error': str(e)
            }