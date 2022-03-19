from flask import Blueprint,render_template
from tensorflow.keras.models import load_model
from keras.preprocessing import image
from flask import request
import numpy as np
import cv2
from numpy import array
from flask_login import login_required,current_user
views = Blueprint('views', __name__)
dic = {


    0:'Covid-19 Infected', 1:'Normal'
}
def Covid_prediction(imagetobetested):
    model = load_model('D:/covid webapp/website/covid_mdel.h5')
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    img1= cv2.imread(imagetobetested)
    img1 = cv2.resize(img1,(224, 224))
    img1 = np.reshape(img1,[1, 224, 224, 3])
    #print(img1.shape)
    p = model.predict_classes(img1)
    return dic[p[0][0]]

@views.route('/')
@login_required
def home():
    return render_template("home.html", user=current_user)
@views.route('/upload-image', methods=['POST','GET'], endpoint='my_upload-image')
def uploadImage():
    if request.method == 'POST':
        file = request.files["image"]
        #file_path = file.filename
        file.save(file.filename)
        p = Covid_prediction(file.filename)
        return render_template('prediction.html', prediction=p)