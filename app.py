"""
NLP News Article Summarizer - Flask Application
Main application file with routes and request handling
"""

from flask import Flask, render_template, request, jsonify, send_file
from article_extractor import ArticleExtractor
from summarizer import TextSummarizer
from nlp_analysis import NLPAnalyzer
from pdf_export import build_pdf_report
import os
from datetime import datetime
import secrets
from io import BytesIO


def _to_json_serializable(obj):
    """Convert NumPy/Pandas scalars and arrays to native Python types."""
    # Primitive safe types
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    # Dicts preserve keys and recursively convert values
    if isinstance(obj, dict):
        return {str(key): _to_json_serializable(value) for key, value in obj.items()}

    # Lists, tuples, sets become lists
    if isinstance(obj, (list, tuple, set)):
        return [_to_json_serializable(value) for value in obj]

    # NumPy scalars and arrays
    try:
        import numpy as np
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass

    # Pandas objects if pandas is installed
    try:
        import pandas as pd
        if isinstance(obj, (pd.Series, pd.Index)):
            return obj.tolist()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        if isinstance(obj, (pd.Timestamp, pd.Timedelta, pd.Period)):
            return str(obj)
    except Exception:
        pass

    # Fallback for any iterable types except strings
    if hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        return [_to_json_serializable(value) for value in obj]

    return str(obj)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = secrets.token_hex(16)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Initialize components
extractor = ArticleExtractor()
summarizer = TextSummarizer()

# NLP analysis utilities
nlp_analyzer = NLPAnalyzer()


@app.route('/')
def index():
    """Render the home page."""
    return render_template('index.html')


@app.route('/api/summarize', methods=['POST'])
def summarize():
    """
    API endpoint for article summarization.
    
    Accepts:
    - URL-based article summarization
    - Text-based article summarization
    
    Returns:
    - JSON response with summaries, metrics, and statistics
    """
    try:
        data = request.json
        input_type = data.get('type', '').lower()
        
        # Validate input type
        if input_type not in ['url', 'text']:
            return jsonify({
                'success': False,
                'error': 'Invalid input type. Must be "url" or "text".'
            }), 400

        # Extract article
        if input_type == 'url':
            url = data.get('url', '').strip()
            if not url:
                return jsonify({
                    'success': False,
                    'error': 'Please provide a URL.'
                }), 400

            extraction_result = extractor.extract_from_url(url)

        else:  # text input
            text = data.get('text', '').strip()
            title = data.get('title', 'Manual Input').strip() or 'Manual Input'
            
            if not text:
                return jsonify({
                    'success': False,
                    'error': 'Please provide article text.'
                }), 400

            extraction_result = extractor.extract_from_text(text, title)

        # Check extraction success
        if not extraction_result.get('success'):
            return jsonify(extraction_result), 400

        extracted_data = extraction_result['data']

        # Generate summaries
        length_choice = data.get('length', 'medium').lower()
        sentence_map = {'short': 3, 'medium': 5, 'detailed': 8}
        num_sentences = sentence_map.get(length_choice, 5)

        extractive_result = summarizer.extract_summary(
            extracted_data['article_text'],
            num_sentences=num_sentences
        )

        abstractive_result = summarizer.abstract_summary(
            extracted_data['article_text'],
            max_length=200 if length_choice == 'detailed' else 120,
            min_length=60 if length_choice == 'detailed' else 35
        )

        # Check if summaries were generated successfully
        extractive_summary = extractive_result.get('summary', '') if extractive_result.get('success') else ''
        extractive_metrics = extractive_result.get('metrics', {}) if extractive_result.get('success') else {}
        extractive_error = extractive_result.get('error') if not extractive_result.get('success') else None

        abstractive_summary = abstractive_result.get('summary', '') if abstractive_result.get('success') else ''
        abstractive_metrics = abstractive_result.get('metrics', {}) if abstractive_result.get('success') else {}
        abstractive_error = abstractive_result.get('error') if not abstractive_result.get('success') else None

        # NLP analysis features
        ner_result = nlp_analyzer.perform_ner(extracted_data['article_text'])
        keywords_result = nlp_analyzer.extract_keywords(extracted_data['article_text'])
        category_result = nlp_analyzer.classify_category(extracted_data['article_text'])
        statistics_result = nlp_analyzer.generate_statistics(
            extracted_data['article_text'],
            extractive_summary,
            abstractive_summary
        )

        # Calculate ROUGE between extractive and abstractive summaries
        rouge_scores = summarizer.calculate_rouge_score(
            extractive_summary if extractive_summary else extracted_data['article_text'],
            abstractive_summary if abstractive_summary else extractive_summary
        )

        # Generate comparison statistics
        comparison_stats = summarizer.generate_comparison_stats(
            extracted_data['article_text'],
            extractive_summary if extractive_summary else extracted_data['article_text'],
            abstractive_summary if abstractive_summary else extracted_data['article_text']
        )

        # Prepare response
        response = {
            'success': True,
            'article': {
                'title': extracted_data['title'],
                'author': extracted_data['author'],
                'publish_date': extracted_data['publish_date'],
                'source_url': extracted_data['source_url'],
                'top_image': extracted_data.get('top_image', ''),
                'content': extracted_data['article_text'],
                'word_count': extracted_data['word_count'],
                'sentence_count': extracted_data['sentence_count']
            },
            'extractive': {
                'summary': extractive_summary,
                'metrics': extractive_metrics,
                'error': extractive_error,
                'length_label': length_choice.title()
            },
            'abstractive': {
                'summary': abstractive_summary,
                'metrics': abstractive_metrics,
                'error': abstractive_error,
                'length_label': length_choice.title()
            },
            'ner': ner_result,
            'keywords': keywords_result,
            'category': category_result,
            'statistics': statistics_result,
            'rouge_scores': rouge_scores,
            'comparison_stats': comparison_stats,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        return jsonify(_to_json_serializable(response)), 200

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500


@app.route('/result')
def result():
    """Render the result page."""
    return render_template('result.html')


@app.route('/download-report', methods=['POST'])
def download_report():
    """Generate and return the PDF summary report."""
    try:
        data = request.json
        if not data:
            return jsonify({'success': False, 'error': 'No data provided for PDF generation.'}), 400

        pdf_bytes = build_pdf_report(data)
        buffer = BytesIO(pdf_bytes)
        buffer.seek(0)

        return send_file(
            buffer,
            as_attachment=True,
            download_name='news_summary_report.pdf',
            mimetype='application/pdf'
        )
    except Exception as exc:
        return jsonify({'success': False, 'error': f'PDF export failed: {exc}'}), 500


@app.route('/error')
def error():
    """Render the error page."""
    error_message = request.args.get('message', 'An unknown error occurred.')
    return render_template('error.html', error=error_message)


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'abstractive_model_ready': summarizer.model_loaded
    }), 200


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return render_template('error.html', error='Page not found'), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return render_template('error.html', error='Internal server error'), 500


if __name__ == '__main__':
    # Create static and templates directories if they don't exist
    os.makedirs('static', exist_ok=True)
    os.makedirs('templates', exist_ok=True)

    # Run the Flask app
    # For production, use a WSGI server like Gunicorn
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )
