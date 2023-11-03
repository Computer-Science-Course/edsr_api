from flask import Flask, request, send_file
from flask_cors import CORS
import os

def create_app():
    app = Flask(__name__)
    CORS(app)

    UPLOAD_FOLDER = './src'
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

    @app.route('/upload', methods=['POST'])
    def upload_file():
        if 'file' not in request.files:
            return 'No file part'

        file = request.files['file']
        if file.filename == '':
            return 'No selected file'

        if file:
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            tail = save_path.split('.')[-1]
            file.save(save_path)
            return send_file(save_path, mimetype=f'image/{tail}')

    return app
