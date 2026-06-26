"""
Summarization Module
Implements extractive and abstractive summarization techniques with ROUGE scoring
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
import warnings

warnings.filterwarnings('ignore')

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class TextSummarizer:
    """
    Implements multiple summarization techniques:
    - Extractive summarization using NLTK
    - Abstractive summarization using Hugging Face Transformers
    - ROUGE score calculation for evaluation
    """

    def __init__(self):
        """Initialize the TextSummarizer with models and configurations."""
        self.abstractive_model = None
        self.stop_words = set(stopwords.words('english'))
        self.max_summary_length = 150  # Maximum length for abstractive summary
        self.model_loaded = False
        self._load_abstractive_model()

    def _load_abstractive_model(self):
        """Load the Hugging Face abstractive summarization model."""
        try:
            self.abstractive_model = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=-1  # Use CPU (-1) or GPU (0) if available
            )
            self.model_loaded = True
        except Exception as e:
            print(f"Warning: Could not load abstractive model: {str(e)}")
            print("Abstractive summarization will not be available.")
            self.model_loaded = False

    def extract_summary(self, text, num_sentences=5):
        """
        Extractive summarization using TF-IDF scoring.
        
        Args:
            text (str): Input text to summarize
            num_sentences (int): Number of sentences to include in summary
            
        Returns:
            dict: Summary data including text, metrics, and analysis
        """
        try:
            # Tokenize sentences
            sentences = sent_tokenize(text)

            # Ensure we don't request more sentences than available
            num_sentences = min(num_sentences, len(sentences))

            if num_sentences == 0:
                return {
                    'success': False,
                    'error': 'Text too short for summarization'
                }

            # Preprocess and score sentences
            scored_sentences = self._score_sentences(sentences)

            # Select top sentences while maintaining order
            top_indices = sorted(
                sorted(range(len(scored_sentences)), 
                       key=lambda i: scored_sentences[i], 
                       reverse=True)[:num_sentences]
            )

            summary_sentences = [sentences[i] for i in top_indices]
            summary_text = ' '.join(summary_sentences)

            # Calculate metrics
            metrics = self._calculate_metrics(text, summary_text)

            return {
                'success': True,
                'summary': summary_text,
                'metrics': metrics
            }

        except Exception as e:
            return {
                'success': False,
                'error': f'Extractive summarization error: {str(e)}'
            }

    def abstract_summary(self, text, max_length=150, min_length=50):
        """
        Abstractive summarization using Hugging Face Transformers.
        
        Args:
            text (str): Input text to summarize
            max_length (int): Maximum length of summary
            min_length (int): Minimum length of summary
            
        Returns:
            dict: Summary data including text, metrics, and analysis
        """
        try:
            if not self.model_loaded:
                return {
                    'success': False,
                    'error': 'Abstractive model not loaded. Check internet connection and try again.'
                }

            # Truncate text if too long (BART has token limits)
            words = text.split()
            if len(words) > 1024:
                text = ' '.join(words[:1024])

            # Generate summary
            summary_result = self.abstractive_model(
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )

            summary_text = summary_result[0]['summary_text']

            # Calculate metrics
            metrics = self._calculate_metrics(text, summary_text)

            return {
                'success': True,
                'summary': summary_text,
                'metrics': metrics
            }

        except Exception as e:
            return {
                'success': False,
                'error': f'Abstractive summarization error: {str(e)}'
            }

    def _score_sentences(self, sentences):
        """
        Score sentences using TF-IDF to identify the most important sentences.
        
        Args:
            sentences (list): List of sentences to score
            
        Returns:
            list: Scores for each sentence
        """
        cleaned_sentences = []
        for sentence in sentences:
            cleaned_tokens = self._clean_and_tokenize(sentence)
            cleaned_sentences.append(' '.join(cleaned_tokens))

        try:
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(cleaned_sentences)
            scores = tfidf_matrix.sum(axis=1)
            return [float(score) for score in scores.A1]
        except Exception:
            # Fallback to frequency scoring if TF-IDF fails
            words = []
            for sentence in sentences:
                cleaned_words = self._clean_and_tokenize(sentence)
                words.extend(cleaned_words)
            word_freq = FreqDist(words)
            sentence_scores = []
            for sentence in sentences:
                cleaned_words = self._clean_and_tokenize(sentence)
                score = sum(word_freq[word] for word in cleaned_words)
                sentence_scores.append(score)
            return sentence_scores

    def _clean_and_tokenize(self, text):
        """
        Clean and tokenize text, removing stopwords.
        
        Args:
            text (str): Text to process
            
        Returns:
            list: List of cleaned tokens
        """
        # Lowercase and remove punctuation
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)

        # Tokenize and remove stopwords
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in self.stop_words and len(word) > 2]

        return tokens

    @staticmethod
    def _calculate_metrics(original, summary):
        """
        Calculate summarization metrics.
        
        Args:
            original (str): Original text
            summary (str): Summarized text
            
        Returns:
            dict: Dictionary containing various metrics
        """
        original_words = original.split()
        summary_words = summary.split()
        original_sentences = sent_tokenize(original)
        summary_sentences = sent_tokenize(summary)

        compression_ratio = len(summary_words) / len(original_words) if original_words else 0

        return {
            'original_word_count': len(original_words),
            'original_sentence_count': len(original_sentences),
            'summary_word_count': len(summary_words),
            'summary_sentence_count': len(summary_sentences),
            'compression_ratio': round(compression_ratio * 100, 2),
            'reduction_percentage': round((1 - compression_ratio) * 100, 2)
        }

    @staticmethod
    def calculate_rouge_score(original, summary):
        """
        Calculate ROUGE scores between original and summary.
        
        ROUGE-1: Unigram overlap
        ROUGE-2: Bigram overlap
        ROUGE-L: Longest common subsequence
        
        Args:
            original (str): Original text
            summary (str): Summarized text
            
        Returns:
            dict: ROUGE scores (precision, recall, f-score)
        """
        try:
            # Tokenize
            original_tokens = word_tokenize(original.lower())
            summary_tokens = word_tokenize(summary.lower())

            # ROUGE-1 (Unigram)
            rouge1 = TextSummarizer._calculate_rouge_n(original_tokens, summary_tokens, 1)

            # ROUGE-2 (Bigram)
            rouge2 = TextSummarizer._calculate_rouge_n(original_tokens, summary_tokens, 2)

            # ROUGE-L (Longest Common Subsequence)
            rougeL = TextSummarizer._calculate_rouge_l(original_tokens, summary_tokens)

            return {
                'ROUGE-1': rouge1,
                'ROUGE-2': rouge2,
                'ROUGE-L': rougeL
            }

        except Exception as e:
            return {
                'error': f'ROUGE calculation error: {str(e)}'
            }

    @staticmethod
    def _calculate_rouge_n(reference, candidate, n):
        """
        Calculate ROUGE-N score.
        
        Args:
            reference (list): Reference tokens
            candidate (list): Candidate tokens
            n (int): N-gram size
            
        Returns:
            dict: Precision, recall, and F-score
        """
        # Generate n-grams
        ref_ngrams = set(zip(*[reference[i:] for i in range(n)]))
        cand_ngrams = set(zip(*[candidate[i:] for i in range(n)]))

        # Calculate overlap
        overlap = len(ref_ngrams & cand_ngrams)

        # Precision and Recall
        precision = overlap / len(cand_ngrams) if cand_ngrams else 0
        recall = overlap / len(ref_ngrams) if ref_ngrams else 0

        # F-score
        f_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f_score': round(f_score, 4)
        }

    @staticmethod
    def _calculate_rouge_l(reference, candidate):
        """
        Calculate ROUGE-L (Longest Common Subsequence) score.
        
        Args:
            reference (list): Reference tokens
            candidate (list): Candidate tokens
            
        Returns:
            dict: Precision, recall, and F-score
        """
        # Calculate LCS
        lcs_length = TextSummarizer._lcs_length(reference, candidate)

        # Precision and Recall
        precision = lcs_length / len(candidate) if candidate else 0
        recall = lcs_length / len(reference) if reference else 0

        # F-score
        f_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f_score': round(f_score, 4)
        }

    @staticmethod
    def _lcs_length(ref, cand):
        """
        Calculate the length of the longest common subsequence.
        
        Args:
            ref (list): Reference tokens
            cand (list): Candidate tokens
            
        Returns:
            int: Length of LCS
        """
        m, n = len(ref), len(cand)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref[i - 1] == cand[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[m][n]

    @staticmethod
    def generate_comparison_stats(original_text, extractive_summary, abstractive_summary):
        """
        Generate detailed comparison statistics.
        
        Args:
            original_text (str): Original article
            extractive_summary (str): Extractive summary
            abstractive_summary (str): Abstractive summary
            
        Returns:
            dict: Comprehensive comparison statistics
        """
        try:
            orig_words = original_text.split()
            extract_words = extractive_summary.split()
            abstract_words = abstractive_summary.split()

            orig_sents = len(sent_tokenize(original_text))
            extract_sents = len(sent_tokenize(extractive_summary))
            abstract_sents = len(sent_tokenize(abstractive_summary))

            return {
                'original': {
                    'word_count': len(orig_words),
                    'sentence_count': orig_sents,
                    'avg_word_length': round(sum(len(w) for w in orig_words) / len(orig_words), 2) if orig_words else 0
                },
                'extractive': {
                    'word_count': len(extract_words),
                    'sentence_count': extract_sents,
                    'compression': round((1 - len(extract_words) / len(orig_words)) * 100, 2) if orig_words else 0
                },
                'abstractive': {
                    'word_count': len(abstract_words),
                    'sentence_count': abstract_sents,
                    'compression': round((1 - len(abstract_words) / len(orig_words)) * 100, 2) if orig_words else 0
                },
                'comparison': {
                    'extractive_vs_abstractive_ratio': round(len(extract_words) / len(abstract_words), 2) if abstract_words else 0,
                    'extractive_shorter': len(extract_words) < len(abstract_words)
                }
            }

        except Exception as e:
            return {'error': f'Comparison stats error: {str(e)}'}
