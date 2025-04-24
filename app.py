from flask import Flask, render_template, request
from keras.models import load_model
import numpy as np
import cv2
import os

app = Flask(__name__, template_folder='templates')
model = load_model("model/greenclassify_cnn_model.h5")

# Update with your class names from training
class_names = ['Beans', 'Brinjal', 'Broccoli', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Cucumber', 'Paprika', 'Potato', 'Pumpkin', 'Radish', 'Tomato', 'Turnip']

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")

@app.route('/classify', methods=['GET', 'POST'])
def classify():
    prediction = None
    if request.method == 'POST':
        file = request.files['image']
        if file:
            img_path = os.path.join("static", "upload.jpg")
            file.save(img_path)

            img = cv2.imread(img_path)
            img = cv2.resize(img, (128, 128))
            img = img.astype("float32") / 255.0
            img = np.expand_dims(img, axis=0)

            pred = model.predict(img)
            prediction = class_names[np.argmax(pred)]

    return render_template("classify.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
