from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import os
import re
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import nltk
import base64
from io import BytesIO
import datetime
import json
import matplotlib.pyplot as plt
import logging
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download required NLTK data safely
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
except Exception as e:
    logger.error(f"Failed to download NLTK data: {e}")

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', "sentimax_secret_key")
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize sentiment analyzer
try:
    sia = SentimentIntensityAnalyzer()
except Exception as e:
    logger.error(f"Failed to initialize SentimentIntensityAnalyzer: {e}")
    sia = None

# Define topics for categorization
topics = {
    'Topic 1': ['user', 'interface', 'ui', 'design', 'layout'],
    'Topic 2': ['feature', 'function', 'capability', 'tool', 'option'],
    'Topic 3': ['chat', 'conversation', 'response', 'message', 'communication'],
    'Topic 4': ['performance', 'speed', 'fast', 'slow', 'efficient'],
    'Topic 5': ['content', 'accuracy', 'correct', 'wrong', 'information']
}

# Define emotions for detection
emotions = ['joy', 'trust', 'anger', 'anticipation', 'disgust', 'fear', 'sadness', 'surprise']

def allowed_file(filename):
    """Check if file has allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'txt'

def preprocess_text(text):
    """Preprocess text for analysis."""
    try:
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and digits
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        # Tokenize the text
        tokens = word_tokenize(text)
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
        return tokens, ' '.join(tokens)
    except Exception as e:
        logger.error(f"Error preprocessing text: {e}")
        return [], ''

def analyze_sentiment(text):
    """Analyze sentiment of text using VADER."""
    try:
        if sia:
            sentiment_score = sia.polarity_scores(text)
            return sentiment_score['compound']
        return 0
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {e}")
        return 0

def categorize_comment(comment, topics):
    """Categorize comment based on keyword matching."""
    try:
        comment_lower = comment.lower()
        scores = {}
        
        for topic, keywords in topics.items():
            score = sum(1 for keyword in keywords if keyword in comment_lower)
            scores[topic] = score
        
        # If no keywords are matched, categorize as the most common topic
        if all(score == 0 for score in scores.values()):
            return max(topics.keys(), key=lambda k: len(topics[k]))
        
        return max(scores, key=scores.get)
    except Exception as e:
        logger.error(f"Error categorizing comment: {e}")
        return list(topics.keys())[0]  # Return first topic as default

def categorize_feedback(sentiment_score, comment_text):
    """Categorize feedback type based on sentiment and keywords."""
    try:
        if sentiment_score > 0.5:
            return "Praise"
        elif sentiment_score < -0.3:
            return "Complaint"
        elif "feature" in comment_text.lower() or "request" in comment_text.lower():
            return "Feature Request"
        else:
            return "Neutral"
    except Exception as e:
        logger.error(f"Error categorizing feedback: {e}")
        return "Neutral"

def detect_emotions(text):
    """Detect emotions in text based on keyword matching."""
    try:
        emotion_counts = {emotion: 0 for emotion in emotions}
        
        # Simple keyword matching for emotions
        emotion_keywords = {
            'joy': ['happy', 'great', 'excellent', 'good', 'joy', 'love', 'enjoy'],
            'trust': ['trust', 'reliable', 'believe', 'confident', 'faith'],
            'anger': ['angry', 'mad', 'furious', 'annoyed', 'frustrated'],
            'anticipation': ['expect', 'anticipate', 'look forward', 'hope'],
            'disgust': ['disgust', 'hate', 'dislike', 'awful', 'terrible'],
            'fear': ['fear', 'afraid', 'scary', 'worried', 'anxious'],
            'sadness': ['sad', 'unhappy', 'disappointed', 'depressed', 'miss'],
            'surprise': ['surprise', 'amazed', 'astonished', 'unexpected', 'wow']
        }
        
        for emotion, keywords in emotion_keywords.items():
            for keyword in keywords:
                if keyword in text.lower():
                    emotion_counts[emotion] += 1
        
        return emotion_counts
    except Exception as e:
        logger.error(f"Error detecting emotions: {e}")
        return {emotion: 0 for emotion in emotions}

def generate_plots(comments_data):
    """Generate all visualization plots from the data."""
    plots = {}
    
    try:
        # Create sentiment distribution histogram with Plotly
        fig_sentiment = px.histogram(
            comments_data, 
            x='sentiment', 
            nbins=15,
            title='Sentiment Distribution'
        )
        fig_sentiment.add_vline(x=0, line_color='red', line_dash='dash')
        fig_sentiment.update_layout(
            xaxis_title='Sentiment Score',
            yaxis_title='Count',
            width=600,
            height=400
        )
        plots['sentiment_dist'] = get_plotly_url(fig_sentiment)
        
        # Create feedback type distribution pie chart with Plotly
        feedback_counts = comments_data['feedback_type'].value_counts()
        fig_feedback = px.pie(
            values=feedback_counts.values,
            names=feedback_counts.index,
            title='Feedback Type Distribution'
        )
        fig_feedback.update_layout(
            width=800,
            height=600
        )
        plots['feedback_dist'] = get_plotly_url(fig_feedback)
        
        # Create top 15 words bar chart with Plotly
        word_counts = Counter()
        for tokens in comments_data['tokens']:
            word_counts.update(tokens)
        
        top_words = dict(word_counts.most_common(15))
        fig_top_words = px.bar(
            x=list(top_words.values()),
            y=list(top_words.keys()),
            orientation='h',
            title='Top 15 Words'
        )
        fig_top_words.update_layout(
            xaxis_title='Frequency',
            yaxis_title='',
            width=600,
            height=400
        )
        plots['top_words'] = get_plotly_url(fig_top_words)
        
        # Create comments by topic horizontal bar chart with Plotly
        topic_counts = comments_data['topic'].value_counts().reset_index()
        topic_counts.columns = ['topic', 'count']
        fig_topics = px.bar(
            topic_counts,
            x='count',
            y='topic',
            orientation='h',
            title='Comments by Topic'
        )
        fig_topics.update_layout(
            xaxis_title='Count',
            yaxis_title='',
            width=600,
            height=400
        )
        plots['comments_by_topic'] = get_plotly_url(fig_topics)
        
        # Create sentiment trend line chart with Plotly
        # Group by week and calculate average sentiment
        comments_data['date'] = pd.to_datetime(comments_data['date'])
        if len(comments_data) > 1:  # Only create trend if we have multiple data points
            weekly_sentiment = comments_data.groupby(pd.Grouper(key='date', freq='W'))['sentiment'].mean().reset_index()
            
            fig_trend = px.line(
                weekly_sentiment,
                x='date',
                y='sentiment',
                markers=True,
                title='Sentiment Trend'
            )
            fig_trend.add_hline(y=0, line_color='gray', line_dash='dash', opacity=0.7)
            fig_trend.update_layout(
                xaxis_title='Week',
                yaxis_title='Avg Sentiment',
                width=600,
                height=400
            )
            plots['sentiment_trend'] = get_plotly_url(fig_trend)
        
        # Create emotional content bar chart with Plotly
        emotion_data = {emotion: sum(comments_data[emotion]) for emotion in emotions}
        emotion_df = pd.DataFrame({'emotion': list(emotion_data.keys()), 'frequency': list(emotion_data.values())})
        fig_emotions = px.bar(
            emotion_df,
            x='frequency',
            y='emotion',
            orientation='h',
            title='Emotional Content'
        )
        fig_emotions.update_layout(
            xaxis_title='Frequency',
            yaxis_title='',
            width=600,
            height=400
        )
        plots['emotional_content'] = get_plotly_url(fig_emotions)
        
        # Create sentiment by topic bar chart with Plotly
        topic_sentiment = comments_data.groupby('topic')['sentiment'].mean().reset_index()
        fig_topic_sentiment = px.bar(
            topic_sentiment,
            x='topic',
            y='sentiment',
            title='Sentiment by Topic'
        )
        fig_topic_sentiment.update_layout(
            xaxis_title='Topic',
            yaxis_title='Avg Sentiment',
            width=600,
            height=400
        )
        plots['sentiment_by_topic'] = get_plotly_url(fig_topic_sentiment)
        
        # Create word cloud (keeping matplotlib for this since Plotly doesn't have a native wordcloud)
        if comments_data['processed_text'].str.cat(sep=' '):  # Check if we have text to create wordcloud
            all_text = ' '.join(comments_data['processed_text'])
            wordcloud = WordCloud(width=600, height=400, background_color='white', colormap='viridis', 
                                max_font_size=100, max_words=100).generate(all_text)
            
            plt.figure(figsize=(6, 4))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.tight_layout()
            plots['wordcloud'] = get_matplotlib_url()
    except Exception as e:
        logger.error(f"Error generating plots: {e}")
    
    return plots

def get_plotly_url(fig):
    """Get a Plotly figure as a base64 encoded URL."""
    try:
        img_bytes = pio.to_image(fig, format='png')
        image_base64 = base64.b64encode(img_bytes).decode('utf-8')
        return f"data:image/png;base64,{image_base64}"
    except Exception as e:
        logger.error(f"Error converting Plotly figure to image: {e}")
        return ""

def get_matplotlib_url():
    """Get the current matplotlib plot as a base64 encoded URL."""
    try:
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        
        image_base64 = base64.b64encode(image_png).decode('utf-8')
        return f"data:image/png;base64,{image_base64}"
    except Exception as e:
        logger.error(f"Error converting Matplotlib figure to image: {e}")
        return ""

@app.route('/')
def index():
    """Render the index page."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Process and analyze user comments."""
    comments = []
    
    try:
        # Check if the post request has the file part and it's not empty
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            if file and allowed_file(file.filename):
                # Save the file temporarily
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(file_path)
                
                # Read file content
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                    
                # Split the content into individual comments (assuming one comment per line)
                comments = [comment.strip() for comment in file_content.split('\n') if comment.strip()]
                
                # Delete the temporary file
                os.remove(file_path)
                logger.info(f"Processed file: {file.filename} with {len(comments)} comments")
        
        # If there's text input, process it as well
        elif request.form.get('text_input', '').strip():
            text_input = request.form['text_input']
            # Split the input into individual comments (assuming one comment per line)
            comments = [comment.strip() for comment in text_input.split('\n') if comment.strip()]
            logger.info(f"Processed text input with {len(comments)} comments")
        
        if not comments:
            flash('No valid input provided. Please upload a text file or enter text.', 'error')
            logger.warning("No valid input provided")
            return redirect(url_for('index'))
        
        # Process each comment
        data = []
        for comment in comments:
            tokens, processed_text = preprocess_text(comment)
            sentiment_score = analyze_sentiment(comment)
            topic = categorize_comment(comment, topics)
            feedback_type = categorize_feedback(sentiment_score, comment)
            emotion_counts = detect_emotions(comment)
            
            # Add current date for time series
            current_date = datetime.datetime.now()
            
            data.append({
                'comment': comment,
                'tokens': tokens,
                'processed_text': processed_text,
                'sentiment': sentiment_score,
                'topic': topic,
                'feedback_type': feedback_type,
                'date': current_date,
                **emotion_counts
            })
        
        # Convert to DataFrame for analytics
        comments_data = pd.DataFrame(data)
        
        # Generate plots
        plots = generate_plots(comments_data)
        
        # Generate summary statistics
        avg_sentiment = comments_data['sentiment'].mean()
        sentiment_counts = {
            'positive': len(comments_data[comments_data['sentiment'] > 0.05]),
            'negative': len(comments_data[comments_data['sentiment'] < -0.05]),
            'neutral': len(comments_data[(comments_data['sentiment'] >= -0.05) & (comments_data['sentiment'] <= 0.05)])
        }
        
        feedback_counts = comments_data['feedback_type'].value_counts().to_dict()
        topic_counts = comments_data['topic'].value_counts().to_dict()
        
        # Calculate most frequent emotions
        emotions_sum = {emotion: comments_data[emotion].sum() for emotion in emotions}
        dominant_emotion = max(emotions_sum, key=emotions_sum.get) if any(emotions_sum.values()) else "None"
        
        summary = {
            'total_comments': len(comments_data),
            'avg_sentiment': avg_sentiment,
            'sentiment_counts': sentiment_counts,
            'feedback_counts': feedback_counts,
            'topic_counts': topic_counts,
            'dominant_emotion': dominant_emotion
        }
        
        logger.info(f"Analysis complete: {len(comments_data)} comments processed")
        return render_template('results.html', 
                            plots=plots, 
                            summary=summary, 
                            comments=comments_data.to_dict('records'))
    
    except Exception as e:
        logger.error(f"Error in analyze route: {e}")
        flash(f'An error occurred during analysis: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error."""
    flash('File too large. Maximum size is 16MB.', 'error')
    return redirect(url_for('index'))

@app.errorhandler(500)
def internal_server_error(error):
    """Handle internal server error."""
    flash('An internal server error occurred. Please try again.', 'error')
    return redirect(url_for('index'))

# Add these route functions to your app.py file, below the existing routes
# but before the if __name__ == '__main__' statement

@app.route('/about')
def about():
    """Render the about page."""
    return render_template('about.html')

@app.route('/contact')
def contact():
    """Render the contact page."""
    return render_template('contact.html')

@app.route('/submit_contact', methods=['POST'])
def submit_contact():
    """Process the contact form submission."""
    try:
        # Get form data
        name = request.form.get('name', '')
        email = request.form.get('email', '')
        subject = request.form.get('subject', '')
        message = request.form.get('message', '')
        
        # Here you would typically save this information to a database
        # or send an email with the contact information
        
        # For now, just log it
        logger.info(f"Contact form submission from {name} ({email}): {subject}")
        
        # Flash a success message
        flash('Your message has been sent successfully! We will get back to you soon.', 'success')
        
        # Redirect back to the contact page
        return redirect(url_for('contact'))
    
    except Exception as e:
        logger.error(f"Error processing contact form: {e}")
        flash('An error occurred while sending your message. Please try again.', 'error')
        return redirect(url_for('contact'))

if __name__ == '__main__':
    app.run(debug=True)
