# # !pip install flask_ngrok

from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
#from tensorflow.keras.applications.resnet152V2 import ResNet152V2
#from keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
import threading

app = Flask(__name__)


# Model saved with Keras model.save()
MODEL_PATH ='/home/shagunaggarwal/Desktop/crop_predict/models/model_vgg19.h5'

# Load your trained model
model = load_model(MODEL_PATH)



def model_predict(img_path, model):
    print(img_path)
    
    img = image.load_img(img_path, target_size=(180,180))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x=x/255
    x = np.expand_dims(x, axis=0)
   

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    # x = preprocess_input(x)

    preds = model.predict(x)
    preds=np.argmax(preds, axis=1)
    if preds==0:
        preds="jute"
    elif preds==1:
        preds="maize"
    elif preds==2:
        preds="rice"
    elif preds==3:
        preds="sugarcane"
    else:
        preds="wheat"
        
    
    
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/crop', methods=['GET'])
def crop():
    # Main page
    return render_template('crop.html')

@app.route('/about', methods=['GET'])
def about():
    # Main page
    return render_template('about.html')

@app.route('/wheat', methods=['GET'])
def wheat():
    # Main page
    return render_template('wheat.html')

@app.route('/rice', methods=['GET'])
def rice():
    # Main page
    return render_template('rice.html')

@app.route('/sugarcane', methods=['GET'])
def sugarcane():
    # Main page
    return render_template('sugarcane.html')

@app.route('/jute', methods=['GET'])
def jute():
    # Main page
    return render_template('jute.html')

@app.route('/maize', methods=['GET'])
def maize():
    # Main page
    return render_template('maize.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f=request.files['file']
        file_path=os.path.join("/home/shagunaggarwal/Desktop/crop_predict/uploads" ,secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        
        
        preds = model_predict(file_path, model)
        result=preds
        return result
        
    return None

if __name__ == '__main__':
 
  app.run()
# threading.Thread(target=app.run, kwargs={'host':'127.0.0.1','port':5001}).start()

    