from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import tensorflow as tf
import cv2
import imghdr

app = Flask(__name__)

# Load the drowsiness detection model
drowsiness_model = tf.keras.models.load_model("open-close-eyes")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detectDrowsiness():
    img = request.files['image']
    link = "temp." + imghdr.what(img)
    img.save(link)
    
    CATEGORIES = ["Driver is Sleeping", "Driver is Awake"]

    def prepare(filepath):
        IMG_SIZE = 50 
        img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    
    prediction = drowsiness_model.predict([prepare(link)])
    prediction = int(prediction)
    print(prediction)

    return render_template('result.html', result=prediction)

if __name__ == '__main__':
    app.run(debug=True)
