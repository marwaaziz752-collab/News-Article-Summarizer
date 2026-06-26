"""
Article Extraction Module
Handles URL-based and text-based article extraction
"""

from newspaper import Article
from urllib.parse import urlparse
import re
from datetime import datetime


class ArticleExtractor:
    """
    Extracts article content from URLs and processes text input.
    """

    def __init__(self):
        """Initialize the ArticleExtractor."""
        self.article = None
        self.extracted_data = {
            'title': '',
            'author': '',
            'publish_date': '',
            'article_text': '',
            'source_url': '',
            'top_image': '',
            'word_count': 0,
            'sentence_count': 0
        }

    def is_valid_url(self, url):
        """
        Validate if the input string is a valid URL.
        
        Args:
            url (str): URL to validate
            
        Returns:
            bool: True if valid URL, False otherwise
        """
        try:
            result = urlparse(url)
            return all([result.scheme in ['http', 'https'], result.netloc])
        except Exception:
            return False

    def extract_from_url(self, url):
        """
        Extract article content from a given URL.
        
        Args:
            url (str): URL of the news article
            
        Returns:
            dict: Dictionary containing extracted article data or error info
        """
        try:
            if not self.is_valid_url(url):
                return {
                    'success': False,
                    'error': 'Invalid URL format. Please provide a valid HTTP/HTTPS URL.'
                }

            # Download and parse the article
            article = Article(url)
            article.download()
            article.parse()

            # Check if article content was successfully extracted
            if not article.text or len(article.text.strip()) < 50:
                return {
                    'success': False,
                    'error': 'Failed to extract article content. Please check the URL and try again.'
                }

            # Populate extracted data
            self.extracted_data['title'] = article.title or 'No Title Found'
            self.extracted_data['author'] = article.authors[0] if article.authors else 'Unknown Author'
            self.extracted_data['publish_date'] = self._format_date(article.publish_date)
            self.extracted_data['article_text'] = article.text
            self.extracted_data['source_url'] = url
            self.extracted_data['top_image'] = article.top_image or ''
            self.extracted_data['word_count'] = len(article.text.split())
            self.extracted_data['sentence_count'] = self._count_sentences(article.text)

            return {
                'success': True,
                'data': self.extracted_data
            }

        except Exception as e:
            return {
                'success': False,
                'error': f'Error extracting article: {str(e)}'
            }

    def extract_from_text(self, text, title='Manual Input'):
        """
        Extract and process manually pasted article text.
        
        Args:
            text (str): Article text provided by user
            title (str): Optional title for the article
            
        Returns:
            dict: Dictionary containing processed article data or error info
        """
        try:
            # Clean and validate text
            cleaned_text = self._clean_text(text)

            if len(cleaned_text.split()) < 20:
                return {
                    'success': False,
                    'error': 'Article text too short. Please provide at least 20 words.'
                }

            # Populate extracted data
            self.extracted_data['title'] = title
            self.extracted_data['author'] = 'Manual Input'
            self.extracted_data['publish_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self.extracted_data['article_text'] = cleaned_text
            self.extracted_data['source_url'] = 'Manual Text Input'
            self.extracted_data['word_count'] = len(cleaned_text.split())
            self.extracted_data['sentence_count'] = self._count_sentences(cleaned_text)

            return {
                'success': True,
                'data': self.extracted_data
            }

        except Exception as e:
            return {
                'success': False,
                'error': f'Error processing text: {str(e)}'
            }

    @staticmethod
    def _clean_text(text):
        """
        Clean and normalize text input.
        
        Args:
            text (str): Raw text to clean
            
        Returns:
            str: Cleaned text
        """
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove common artifacts
        text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
        text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
        text = re.sub(r'[^\w\s.!?,-]', '', text)  # Remove special characters except punctuation
        
        return text.strip()

    @staticmethod
    def _count_sentences(text):
        """
        Count the number of sentences in text.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            int: Number of sentences
        """
        # Simple sentence splitting on periods, exclamation marks, and question marks
        sentences = re.split(r'[.!?]+', text)
        return len([s for s in sentences if s.strip()])

    @staticmethod
    def _format_date(date_obj):
        """
        Format date object to readable string.
        
        Args:
            date_obj: Date object to format
            
        Returns:
            str: Formatted date string
        """
        if date_obj:
            try:
                return date_obj.strftime('%Y-%m-%d %H:%M:%S')
            except Exception:
                return 'Date Not Available'
        return 'Date Not Available'

    def get_extracted_data(self):
        """
        Get the currently extracted article data.
        
        Returns:
            dict: Extracted article data
        """
        return self.extracted_data

    def reset(self):
        """Reset the extracted data."""
        self.extracted_data = {
            'title': '',
            'author': '',
            'publish_date': '',
            'article_text': '',
            'source_url': '',
            'word_count': 0,
            'sentence_count': 0
        }
