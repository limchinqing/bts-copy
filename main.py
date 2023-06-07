import PIL
import cv2
import requests
import numpy as np
from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO
from convert import convert_to_braille_unicode, parse_xywh_and_class, map_characters
from braille_to_bw import preprocess_image

def load_model(model_path):
    """load model from path"""
    model = YOLO(model_path)
    return model

# Define function to load image
def load_image(image_path):
    """load image from path"""
    image = PIL.Image.open(image_path)
    return image

# Define function to process image
def process_image(image_path, model):
    CONF = 0.35
    
    # Fetch the image data from the URL
    response = requests.get(image_path)
    image_data = response.content
    
    # Save the image data to a local file
    image_path = 'input.jpg'
    with open(image_path, 'wb') as file:
        file.write(image_data)
    
    # Convert the image data to a NumPy array
    nparr = np.frombuffer(image_data, np.uint8)
    
    # Decode the NumPy array into an OpenCV image
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    image_path = preprocess_image(image)
    image = load_image(image_path)
    res = model.predict(image, save=True, save_txt=True, exist_ok=True, conf=CONF)
    boxes = res[0].boxes  # first image
    list_boxes = parse_xywh_and_class(boxes)

    result = ""
    for box_line in list_boxes:
        str_left_to_right = ""
        box_classes = box_line[:, -1]
        for each_class in box_classes: 
            str_left_to_right += convert_to_braille_unicode(model.names[int(each_class)])
        result += str_left_to_right + "\n"

    result = map_characters(result, "utils/map_alphabet.json")
    return result


# Load the model
MODEL_PATH = "yolov8_braille.pt"
model = load_model(MODEL_PATH)

# Initialize Flask app
app = Flask("Braille to Speech")

@app.route('/')
def index():
    return "Hi"

# Define route to handle the image path
@app.route('/process_image', methods=['POST'])
def process_image_route():
    if 'image_path' in request.form:
        image_path = request.form['image_path']
        result = process_image(image_path, model)
        return jsonify({'result': result})
    else:
        return jsonify({'error': 'No image path provided'})


# Run the Flask app
if __name__ == '__main__':
    app.run()
