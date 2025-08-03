print("--- Script is starting ---")

import os
from dotenv import load_dotenv
load_dotenv()
from flask import Flask, request, render_template, jsonify
from chatbot import get_chatbot_response
import re


app = Flask(__name__)

def extract_video_id(url):
    """Extracts the YouTube video ID from various URL formats."""
    regex = r"(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})"
    match = re.search(regex, url)
    return match.group(1) if match else None

@app.route('/', methods=['GET', 'POST'])
def index():
    """Handles the web interface for the chatbot."""
    answer = None
    if request.method == 'POST':
        video_url = request.form.get('video_url')
        question = request.form.get('question')

        if video_url and question:
            video_id = extract_video_id(video_url)
            if video_id:
                answer = get_chatbot_response(video_id, question)
            else:
                answer = "Invalid YouTube URL. Please enter a valid URL."
        else:
            answer = "Please provide both a YouTube URL and a question."

    return render_template('index.html', answer=answer)

@app.route('/chat', methods=['POST'])
def chat_api():
    """Provides an API endpoint for the chatbot."""
    data = request.get_json()
    video_url = data.get('video_url')
    question = data.get('question')

    if not video_url or not question:
        return jsonify({'error': 'Please provide both a video_url and a question.'}), 400

    video_id = extract_video_id(video_url)
    if not video_id:
        return jsonify({'error': 'Invalid YouTube URL provided.'}), 400

    response = get_chatbot_response(video_id, question)
    return jsonify({'answer': response})

if __name__ == '__main__':
    # This block is for local development; Render uses Gunicorn.
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)