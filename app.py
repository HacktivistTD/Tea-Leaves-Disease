from flask import Flask, render_template, request, redirect, url_for
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)
model = load_model('tea_leaves_disease_model.h5')

# Define class labels
class_labels = {0: 'Class1', 1: 'Class2', 2: 'Class3', 3: 'Class4', 4: 'Class5', 5: 'Class6', 6: 'Class7', 7: 'Healthy'}

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]

    return predicted_class

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = file.filename
            filepath = os.path.join('static', filename)
            file.save(filepath)
            prediction = predict_image(filepath)
            return render_template('index.html', prediction=prediction, image_path=filepath)
    return render_template('index.html', prediction=None, image_path=None)

if __name__ == '__main__':
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True)
