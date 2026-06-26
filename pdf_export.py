"""
PDF Export Module
Generates a PDF report from summarization results.
"""

from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle


def build_pdf_report(result_data):
    """
    Build a PDF summary report from the result data.

    Args:
        result_data (dict): Summary analysis data

    Returns:
        bytes: Generated PDF as bytes
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            rightMargin=0.7 * inch,
                            leftMargin=0.7 * inch,
                            topMargin=0.8 * inch,
                            bottomMargin=0.8 * inch)

    styles = {
        'title': ParagraphStyle(
            name='Title',
            fontSize=20,
            leading=24,
            spaceAfter=12,
            alignment=1,
            textColor=colors.darkblue),
        'heading': ParagraphStyle(
            name='Heading',
            fontSize=14,
            leading=18,
            spaceBefore=12,
            spaceAfter=8,
            textColor=colors.black),
        'normal': ParagraphStyle(
            name='Normal',
            fontSize=11,
            leading=15,
            spaceAfter=6)
    }

    elements = []
    elements.append(Paragraph('News Summary Report', styles['title']))
    elements.append(Spacer(1, 12))

    article = result_data.get('article', {})
    extractive = result_data.get('extractive', {})
    abstractive = result_data.get('abstractive', {})
    ner = result_data.get('ner', {})
    keywords = result_data.get('keywords', [])
    category = result_data.get('category', {})
    rouge = result_data.get('rouge_scores', {})
    stats = result_data.get('statistics', {})

    # Article metadata
    elements.append(Paragraph('Article Information', styles['heading']))
    metadata_table = [
        ['Title:', article.get('title', 'N/A')],
        ['Author:', article.get('author', 'N/A')],
        ['Published:', article.get('publish_date', 'N/A')],
        ['URL:', article.get('source_url', 'N/A')],
        ['Category:', category.get('category', 'Unknown')],
        ['Category Confidence:', f"{category.get('score', 0.0)}%"],
    ]
    table = Table(metadata_table, colWidths=[2.2*inch, 4.8*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
        ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 12))

    # Keywords
    elements.append(Paragraph('Keywords', styles['heading']))
    keyword_text = ', '.join([f"{item.get('keyword')} ({item.get('frequency', 0)})" for item in keywords])
    elements.append(Paragraph(keyword_text or 'N/A', styles['normal']))
    elements.append(Spacer(1, 12))

    # Entities
    entities = ner.get('entities', {})
    elements.append(Paragraph('Named Entities', styles['heading']))
    for category_name, items in entities.items():
        line = f"<strong>{category_name}:</strong> {', '.join(items) if items else 'N/A'}"
        elements.append(Paragraph(line, styles['normal']))
    elements.append(Spacer(1, 12))

    # Summary statistics
    elements.append(Paragraph('Summary Statistics', styles['heading']))
    stats_table = [
        ['Original Word Count', stats.get('original_word_count', 'N/A')],
        ['Original Sentence Count', stats.get('original_sentence_count', 'N/A')],
        ['Extractive Word Count', stats.get('extractive_word_count', 'N/A')],
        ['Abstractive Word Count', stats.get('abstractive_word_count', 'N/A')],
        ['Reading Time (min)', stats.get('reading_time_minutes', 'N/A')],
        ['Compression Ratio', f"{stats.get('compression_ratio', 'N/A')}%"],
        ['Reduction Percentage', f"{stats.get('reduction_percentage', 'N/A')}%"],
    ]
    table = Table(stats_table, colWidths=[2.4*inch, 4.6*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 12))

    # ROUGE scores
    elements.append(Paragraph('ROUGE Evaluation', styles['heading']))
    rouge_table = [
        ['Metric', 'Precision', 'Recall', 'F-Score'],
        ['ROUGE-1', rouge.get('ROUGE-1', {}).get('precision', 'N/A'), rouge.get('ROUGE-1', {}).get('recall', 'N/A'), rouge.get('ROUGE-1', {}).get('f_score', 'N/A')],
        ['ROUGE-2', rouge.get('ROUGE-2', {}).get('precision', 'N/A'), rouge.get('ROUGE-2', {}).get('recall', 'N/A'), rouge.get('ROUGE-2', {}).get('f_score', 'N/A')],
        ['ROUGE-L', rouge.get('ROUGE-L', {}).get('precision', 'N/A'), rouge.get('ROUGE-L', {}).get('recall', 'N/A'), rouge.get('ROUGE-L', {}).get('f_score', 'N/A')],
    ]
    table = Table(rouge_table, colWidths=[2.4*inch, 1.8*inch, 1.8*inch, 1.8*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 12))

    # Extractive summary
    elements.append(Paragraph('Extractive Summary', styles['heading']))
    elements.append(Paragraph(extractive.get('summary', 'N/A'), styles['normal']))
    elements.append(Spacer(1, 12))

    # Abstractive summary
    elements.append(Paragraph('Abstractive Summary', styles['heading']))
    elements.append(Paragraph(abstractive.get('summary', 'N/A'), styles['normal']))
    elements.append(Spacer(1, 12))

    doc.build(elements)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
