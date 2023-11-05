import io
from flask import Flask, request, send_file
from flask_cors import CORS
from PIL import Image

from utils.strange import predict


def create_app():
    app = Flask(__name__)
    CORS(app)

    UPLOAD_FOLDER = './src'
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

    @app.route('/upload', methods=['POST'])
    def upload_file():
        model = request.form.get('model', '500_200')

        if 'file' not in request.files:
            return 'No file part'

        file = request.files['file']
        if file.filename == '':
            return 'No selected file'

        if file:
            # Pass the uploaded image to the predict function
            new_image_tensor = predict(file, model)

            # Convert the tensor back to a numpy array
            new_image_array = new_image_tensor.numpy()

            # Convert the numpy array to a PIL image
            new_image_pil = Image.fromarray(new_image_array.astype('uint8'))

            # Create a BytesIO object to send the image as a response
            new_image_stream = io.BytesIO()
            new_image_pil.save(new_image_stream, format='JPEG')
            new_image_stream.seek(0)

            return send_file(new_image_stream, mimetype='image/jpeg')

    return app
