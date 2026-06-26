# NLP News Article Summarizer

A comprehensive Python-based web application for automatic news article summarization using Natural Language Processing (NLP) and Deep Learning techniques.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technologies](#technologies)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [How It Works](#how-it-works)
- [Screenshots](#screenshots)
- [Statistics & Metrics](#statistics--metrics)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

The NLP News Article Summarizer is a cutting-edge application designed to automatically generate concise summaries of news articles using two complementary approaches:

1. **Extractive Summarization** - Identifies and extracts key sentences from the original text
2. **Abstractive Summarization** - Generates new sentences that capture the essence of the article

The application provides a modern web interface for users to input articles via URL or direct text input, and presents detailed comparison metrics including ROUGE scores for summary evaluation.

## ✨ Features

### A. URL-Based Article Extraction

- **Automatic Content Extraction**: Uses `newspaper3k` to extract article metadata and content
- **Metadata Retrieval**: Captures title, author, publication date automatically
- **Error Handling**: Gracefully handles invalid URLs and extraction failures
- **Support for Multiple News Sources**: Works with most major news websites

### B. Text-Based Input

- **Direct Paste Support**: Users can paste article text directly
- **Custom Titles**: Option to add custom titles to manually pasted articles
- **Validation**: Ensures minimum text length (20 words) for quality summarization
- **Format Flexibility**: Handles various text formats

### C. Extractive Summarization (NLTK-based)

- **TF-IDF Scoring**: Uses term frequency-inverse document frequency for sentence ranking
- **Configurable Output**: Adjustable number of sentences in summary
- **Stopword Removal**: Filters common English words
- **Sentence Order Preservation**: Maintains original sentence sequence

**Algorithm Steps:**
1. Tokenize text into sentences
2. Tokenize sentences into words
3. Remove stopwords and normalize
4. Calculate word frequencies
5. Score each sentence based on word frequency
6. Select top-scoring sentences
7. Maintain original order

### D. Abstractive Summarization (Hugging Face BART)

- **State-of-the-Art Model**: Uses `facebook/bart-large-cnn` from Hugging Face
- **Human-Like Summaries**: Generates novel sentences not directly from source
- **Configurable Length**: Adjustable summary length (50-150 tokens)
- **Production Ready**: Uses latest transformer technology

**Model Capabilities:**
- Understands context and semantic meaning
- Paraphrases and reformulates content
- Generates grammatically correct summaries
- Requires GPU for optimal performance (CPU supported)

### E. ROUGE Score Analysis

Implements ROUGE metrics for comprehensive summary evaluation:

- **ROUGE-1**: Unigram (single word) overlap
  - Measures word-level recall and precision
  
- **ROUGE-2**: Bigram (two-word phrase) overlap
  - Captures phrase-level similarities
  
- **ROUGE-L**: Longest Common Subsequence
  - Evaluates longest matching sequences

**Metrics Provided:**
- Precision: How many summary words match the original
- Recall: How many original words are captured in summary
- F-Score: Harmonic mean of precision and recall

### F. Comprehensive Statistics

- **Word Count Analysis**:
  - Original article words
  - Summary words
  - Compression ratio
  - Reduction percentage

- **Sentence Count Analysis**:
  - Original sentences
  - Summary sentences
  - Average sentence length

- **Comparison Metrics**:
  - Side-by-side comparison table
  - Compression ratios for each method
  - Method comparison (extractive vs abstractive)

### G. Responsive Web Interface

- **Modern Bootstrap UI**: Beautiful, responsive design
- **Tab-Based Navigation**: Organized content presentation
- **Multiple Views**: Original, extractive, abstractive, comparison
- **Mobile Optimized**: Fully responsive on all devices
- **Copy to Clipboard**: Quick copy functionality for summaries
- **Real-time Processing**: Instant visual feedback

## 🛠️ Technologies

### Backend
- **Python 3.8+**: Core programming language
- **Flask 2.3.3**: Web framework for routing and API
- **NLTK 3.8.1**: Natural Language Toolkit for extractive summarization
- **Transformers 4.32.0**: Hugging Face library for abstractive summarization
- **Newspaper3k 0.2.8**: Article extraction from URLs
- **PyTorch 2.0.1**: Deep learning framework (required by Transformers)

### Frontend
- **HTML5**: Semantic markup
- **Bootstrap 5.3.0**: Responsive UI framework
- **CSS3**: Modern styling with gradients and animations
- **JavaScript (Vanilla)**: Dynamic interactions and API calls
- **Bootstrap Icons**: Icon library

### Additional Libraries
- **BeautifulSoup4 4.12.2**: HTML parsing
- **Requests 2.31.0**: HTTP library
- **lxml 4.9.3**: XML/HTML processing

## 📁 Project Structure

```
news_summarizer/
├── app.py                      # Main Flask application
├── article_extractor.py        # URL/text extraction logic
├── summarizer.py              # Summarization algorithms
├── requirements.txt           # Python dependencies
├── README.md                  # This file
│
├── templates/                 # HTML templates
│   ├── index.html            # Home page
│   ├── result.html           # Results display page
│   └── error.html            # Error page
│
└── static/                    # Static assets
    └── style.css             # Stylesheet

```

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git
- 2GB free disk space (for models)

### Step-by-Step Installation

1. **Clone or Download the Project**
   ```bash
   cd path/to/news_summarizer
   ```

2. **Create a Virtual Environment** (Recommended)
   ```bash
   # On Windows
   python -m venv venv
   venv\Scripts\activate
   
   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK Data** (Automatic on first run, or manual)
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
   ```

5. **First-Time Model Download**
   - The BART model (~1.6GB) will download automatically on first use
   - Ensure stable internet connection
   - This happens only once

6. **Run the Application**
   ```bash
   python app.py
   ```

7. **Access the Web Interface**
   - Open your browser
   - Navigate to: `http://localhost:5000`

## 🚀 Usage

### Using URL-Based Summarization

1. Go to the home page
2. Select the "Article URL" tab (default)
3. Enter a news article URL (e.g., from BBC, CNN, Reuters)
4. Click "Summarize Article"
5. View results in tabs

**Supported Sites:**
- BBC News
- CNN
- Reuters
- The Guardian
- The New York Times
- Most major news websites

### Using Text-Based Summarization

1. Go to the home page
2. Select the "Paste Text" tab
3. (Optional) Enter a custom title
4. Paste your article text
5. Click "Summarize Article"
6. View results in tabs

**Text Requirements:**
- Minimum: 20 words
- Format: Plain text
- Length: No maximum limit (recommended: under 10,000 words)

### Interpreting Results

The results page includes:

1. **Article Information**
   - Title, author, publication date
   - Source URL
   - Original word/sentence counts

2. **Original Article Tab**
   - Full original text
   - Statistics

3. **Extractive Summary Tab**
   - Key sentences extracted
   - Compression percentage
   - NLTK-based extraction

4. **Abstractive Summary Tab**
   - AI-generated summary
   - Compression percentage
   - BART model output

5. **Comparison & ROUGE Tab**
   - Side-by-side metrics table
   - ROUGE-1, ROUGE-2, ROUGE-L scores
   - Detailed precision/recall/F-score

## 📡 API Documentation

### Endpoint: `/api/summarize`

**Method:** POST

**Content-Type:** application/json

#### Request Body (URL-based)
```json
{
  "type": "url",
  "url": "https://example.com/article"
}
```

#### Request Body (Text-based)
```json
{
  "type": "text",
  "text": "Article text content here...",
  "title": "Optional article title"
}
```

#### Response Structure
```json
{
  "success": true,
  "article": {
    "title": "Article Title",
    "author": "Author Name",
    "publish_date": "2024-01-01 12:00:00",
    "source_url": "https://example.com",
    "content": "Full article text...",
    "word_count": 1000,
    "sentence_count": 50
  },
  "extractive": {
    "summary": "Extractive summary text...",
    "metrics": {
      "original_word_count": 1000,
      "original_sentence_count": 50,
      "summary_word_count": 150,
      "summary_sentence_count": 10,
      "compression_ratio": 15.0,
      "reduction_percentage": 85.0
    }
  },
  "abstractive": {
    "summary": "Abstractive summary text...",
    "metrics": {
      "original_word_count": 1000,
      "original_sentence_count": 50,
      "summary_word_count": 140,
      "summary_sentence_count": 8,
      "compression_ratio": 14.0,
      "reduction_percentage": 86.0
    }
  },
  "rouge_scores": {
    "ROUGE-1": {
      "precision": 0.85,
      "recall": 0.78,
      "f_score": 0.81
    },
    "ROUGE-2": {
      "precision": 0.72,
      "recall": 0.65,
      "f_score": 0.68
    },
    "ROUGE-L": {
      "precision": 0.88,
      "recall": 0.80,
      "f_score": 0.84
    }
  },
  "comparison_stats": {
    "original": {
      "word_count": 1000,
      "sentence_count": 50,
      "avg_word_length": 5.2
    },
    "extractive": {
      "word_count": 150,
      "sentence_count": 10,
      "compression": 85.0
    },
    "abstractive": {
      "word_count": 140,
      "sentence_count": 8,
      "compression": 86.0
    }
  },
  "timestamp": "2024-01-01 12:30:00"
}
```

#### Error Response
```json
{
  "success": false,
  "error": "Error description here"
}
```

## 🧠 How It Works

### Extractive Summarization Process

```
Original Text
    ↓
[Sentence Tokenization]
    ↓
[Word Tokenization]
    ↓
[Stopword Removal]
    ↓
[Calculate Word Frequencies]
    ↓
[Score Each Sentence]
    ↓
[Select Top Sentences]
    ↓
[Preserve Original Order]
    ↓
Extractive Summary
```

### Abstractive Summarization Process

```
Original Text
    ↓
[Tokenization & Encoding]
    ↓
[BART Model Encoding]
    ↓
[Attention Mechanisms]
    ↓
[Decoder Generation]
    ↓
[Token Selection]
    ↓
Abstractive Summary
```

### ROUGE Scoring

```
Original Text          Summary
    ↓                    ↓
[N-gram Extraction] [N-gram Extraction]
    ↓                    ↓
    ←─── [Comparison] ───→
    ↓
[Precision & Recall Calculation]
    ↓
[F-Score Computation]
    ↓
ROUGE Metrics
```

## 📸 Screenshots

### Home Page
- Clean, modern interface
- Tab-based input selection
- Feature highlights
- Error handling display

### Results Page
- Article information card
- Statistics overview
- Multi-tab result display
- Comparison metrics
- ROUGE score visualization

### Features Showcase
- Four main feature cards
- Feature descriptions
- Visual icons

## 📊 Statistics & Metrics

### Supported Metrics

1. **Word Count**
   - Original: Total words in article
   - Summary: Total words in summary
   - Reduction: Percentage decrease

2. **Sentence Count**
   - Original: Total sentences
   - Summary: Extracted/Generated sentences
   - Ratio: Compression ratio

3. **ROUGE Scores**
   - ROUGE-1: Unigram overlap
   - ROUGE-2: Bigram overlap
   - ROUGE-L: LCS-based score

4. **Compression Metrics**
   - Compression Ratio: Summary words / Original words
   - Reduction Percentage: 1 - Compression Ratio
   - Quality: Based on ROUGE scores

## 🚀 Future Enhancements

### Planned Features

1. **Multi-Language Support**
   - Support for Spanish, French, German, Chinese
   - Automatic language detection
   - Language-specific models

2. **Advanced Models**
   - T5-based summarization
   - Pegasus model support
   - Custom fine-tuned models
   - Hybrid approaches

3. **User Accounts & History**
   - User registration and login
   - Save favorite articles
   - Summarization history
   - Comparison exports

4. **Database Integration**
   - Store summaries in database
   - User preferences
   - Performance analytics

5. **Advanced Features**
   - Keyword extraction
   - Named Entity Recognition (NER)
   - Topic modeling
   - Multi-document summarization

6. **API Enhancements**
   - Batch processing
   - Webhook support
   - Rate limiting
   - API key authentication

7. **Export Options**
   - PDF export
   - JSON export
   - Markdown export
   - Email delivery

8. **Performance Optimization**
   - GPU acceleration
   - Caching mechanisms
   - Asynchronous processing
   - Model quantization

9. **UI/UX Improvements**
   - Dark mode
   - Advanced search
   - Customizable themes
   - Keyboard shortcuts

10. **Analytics & Monitoring**
    - Performance metrics
    - User analytics
    - Model statistics
    - Error tracking

## 🔧 Troubleshooting

### Common Issues

**Issue:** Model download fails
- **Solution:** Check internet connection, ensure 2GB free space

**Issue:** Summarization too slow
- **Solution:** Use GPU, reduce article length, use CPU-optimized models

**Issue:** Article extraction fails
- **Solution:** Try different URL, check if site blocks scraping

**Issue:** Memory errors
- **Solution:** Run on machine with 8GB+ RAM, reduce model size

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📝 License

This project is open source and available under the MIT License.

## 👨‍💻 Author

Built with ❤️ for academic excellence and NLP research.

## 🙏 Acknowledgments

- Hugging Face for the BART model and Transformers library
- NLTK for NLP toolkit
- Newspaper3k for article extraction
- Bootstrap community for UI framework

## 📞 Support

For issues, questions, or suggestions:
- Create an issue on GitHub
- Check documentation
- Review troubleshooting section

## 🎓 Educational Use

This project is ideal for:
- Learning NLP concepts
- Understanding summarization techniques
- Web application development
- Machine learning model deployment
- Academic research

---

**Last Updated:** January 2024
**Version:** 1.0.0
**Status:** Production Ready

Happy Summarizing! 📰✨
