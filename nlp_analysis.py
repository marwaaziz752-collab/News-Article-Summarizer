"""
NLP Analysis Module
Performs named entity recognition, keyword extraction,
category classification, and summary statistics.
"""

import re
import math
from markupsafe import escape

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from transformers import pipeline

try:
    import spacy
except ImportError:
    spacy = None


class NLPAnalyzer:
    """
    Encapsulates NLP analysis tasks for article intelligence.
    """

    def __init__(self):
        self.ner_model = None
        self.category_pipeline = None
        self.category_labels = [
            'Politics', 'Technology', 'Sports', 'Business', 'Health',
            'Entertainment', 'Science', 'World News'
        ]

        self._load_spacy_model()
        self._load_category_pipeline()

    def _load_spacy_model(self):
        """Load spaCy model with graceful fallback."""
        try:
            if spacy is None:
                raise ImportError('spaCy is not installed')
            self.ner_model = spacy.load('en_core_web_sm')
        except Exception as exc:
            print(f'Warning: spaCy model could not be loaded: {exc}')
            self.ner_model = None

    def _load_category_pipeline(self):
        """Load zero-shot classification pipeline for news categories."""
        try:
            self.category_pipeline = pipeline(
                'zero-shot-classification',
                model='facebook/bart-large-mnli',
                device=-1
            )
        except Exception as exc:
            print(f'Warning: Category pipeline could not be loaded: {exc}')
            self.category_pipeline = None

    def perform_ner(self, text):
        """
        Perform Named Entity Recognition and return entity groups.
        """
        if not self.ner_model:
            return {
                'entities': {},
                'highlighted_text': escape(text),
                'error': 'NER model unavailable'
            }

        try:
            doc = self.ner_model(text)
            entities = {
                'People': [],
                'Organizations': [],
                'Locations': [],
                'Dates': [],
                'Events': []
            }

            entity_map = {
                'PERSON': 'People',
                'ORG': 'Organizations',
                'GPE': 'Locations',
                'LOC': 'Locations',
                'DATE': 'Dates',
                'EVENT': 'Events'
            }

            for ent in doc.ents:
                category = entity_map.get(ent.label_)
                if category:
                    entities[category].append(ent.text)

            # Deduplicate while preserving order
            for category, values in entities.items():
                unique = []
                seen = set()
                for value in values:
                    if value not in seen:
                        unique.append(value)
                        seen.add(value)
                entities[category] = unique

            highlighted_text = self._highlight_entities(text, doc.ents)

            return {
                'entities': entities,
                'highlighted_text': highlighted_text
            }

        except Exception as exc:
            return {
                'entities': {},
                'highlighted_text': escape(text),
                'error': f'NER processing error: {exc}'
            }

    def _highlight_entities(self, text, entities):
        """
        Inject HTML spans around entity text for highlighting.
        """
        if not entities:
            return escape(text)

        sorted_entities = sorted(entities, key=lambda item: item.start_char)
        highlighted_parts = []
        last_index = 0

        for ent in sorted_entities:
            start = ent.start_char
            end = ent.end_char
            if start < last_index:
                continue
            highlighted_parts.append(escape(text[last_index:start]))
            label = ent.label_.lower()
            css_class = 'entity-' + label
            entity_text = escape(text[start:end])
            highlighted_parts.append(
                f'<span class="entity-tag {css_class}" title="{ent.label_}">{entity_text}</span>'
            )
            last_index = end

        highlighted_parts.append(escape(text[last_index:]))
        return ''.join(highlighted_parts)

    def extract_keywords(self, text, top_n=12):
        """
        Extract top keywords from text using TF-IDF and frequency analysis.
        """
        try:
            vectorizer = TfidfVectorizer(
                stop_words='english',
                ngram_range=(1, 2),
                max_features=100
            )
            tfidf_matrix = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            keyword_scores = sorted(
                zip(feature_names, scores),
                key=lambda item: item[1],
                reverse=True
            )[:top_n]

            frequency_vectorizer = CountVectorizer(
                stop_words='english',
                ngram_range=(1, 2),
                max_features=200
            )
            token_counts = frequency_vectorizer.fit_transform([text]).toarray()[0]
            freq_names = frequency_vectorizer.get_feature_names_out()
            counts = dict(zip(freq_names, token_counts))

            keywords = []
            for keyword, score in keyword_scores:
                keywords.append({
                    'keyword': keyword,
                    'score': round(float(score), 4),
                    'frequency': int(counts.get(keyword, 0))
                })

            return keywords

        except Exception as exc:
            return {
                'error': f'Keyword extraction error: {exc}'
            }

    def classify_category(self, text):
        """
        Classify article category using zero-shot classification.
        """
        if not self.category_pipeline:
            return {
                'category': 'Unknown',
                'score': 0.0,
                'details': [],
                'error': 'Category classification unavailable'
            }

        try:
            output = self.category_pipeline(
                text[:512],
                candidate_labels=self.category_labels,
                multi_label=False
            )
            label = output.get('labels', ['Unknown'])[0]
            scores = output.get('scores', [0.0])
            score = float(scores[0]) if scores else 0.0

            details = []
            for lbl, score_value in zip(output.get('labels', []), output.get('scores', [])):
                details.append({
                    'label': lbl,
                    'confidence': round(float(score_value) * 100, 2)
                })

            return {
                'category': label,
                'score': round(score * 100, 2),
                'details': details
            }

        except Exception as exc:
            return {
                'category': 'Unknown',
                'score': 0.0,
                'details': [],
                'error': f'Category classification error: {exc}'
            }

    def generate_statistics(self, original_text, extractive_summary, abstractive_summary):
        """
        Generate summary statistics and dashboard values.
        """
        try:
            original_words = original_text.split()
            extractive_words = extractive_summary.split()
            abstractive_words = abstractive_summary.split()

            original_sentences = len(re.split(r'[.!?]+', original_text))
            extractive_sentences = len(re.split(r'[.!?]+', extractive_summary))
            abstractive_sentences = len(re.split(r'[.!?]+', abstractive_summary))

            original_word_count = len(original_words)
            extractive_word_count = len(extractive_words)
            abstractive_word_count = len(abstractive_words)

            reading_time = math.ceil(original_word_count / 200) if original_word_count else 0
            compression_ratio = round((abstractive_word_count / original_word_count) * 100, 2) if original_word_count else 0
            reduction_percentage = round((1 - (abstractive_word_count / original_word_count)) * 100, 2) if original_word_count else 0

            return {
                'original_word_count': original_word_count,
                'original_sentence_count': original_sentences,
                'extractive_word_count': extractive_word_count,
                'extractive_sentence_count': extractive_sentences,
                'abstractive_word_count': abstractive_word_count,
                'abstractive_sentence_count': abstractive_sentences,
                'reading_time_minutes': reading_time,
                'compression_ratio': compression_ratio,
                'reduction_percentage': reduction_percentage
            }

        except Exception as exc:
            return {
                'error': f'Statistics generation error: {exc}'
            }
