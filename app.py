from flask import Flask, request
import os

def create_app():
    app = Flask(__name__)

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
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
            return 'File uploaded successfully'

    return app