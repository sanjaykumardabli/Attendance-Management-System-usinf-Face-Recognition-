from flask import Flask,render_template, request, redirect
import pickle
import alignment
from sklearn import svm 
import cv2
import face_recognition 
import training
import os
import numpy as np
from PIL import Image
import pandas as pd
import datetime
import time
import pickle
import csv_creater
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import testing

app = Flask(__name__)





@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        global file_path
        file_path = os.path.join(
            basepath, 'student_images', secure_filename(f.filename))
        f.save(file_path)
        
        dir="training_image"
        img=file_path
        result = testing.face_recognize(dir,img)         
        return result
    return None


@app.route('/')
def dude():
    return render_template('home.html')





@app.route('/login', methods=['GET', 'POST'])
def login_form():
    #capt()
    return render_template('login.html')
@app.route('/faculty_home', methods=['GET', 'POST'])
def faccc():
    if request.method == 'POST':
        
        admin = request.form['user']
        password = int(request.form['password'])
        name=request.form['name']
        
        if admin=='admin' and password==1234:
                return render_template('trying.html',prediction_text=f"Welcome {name}",text=f"Logged in as : {name}")#,)
        
        return render_template("login.html",text="Username and Password doesnot match" )
        


@app.route('/register_student', methods=['GET', 'POST'])
def abfo():
    return render_template('register.html')



@app.route('/done', methods=['GET', 'POST'])
def contacts():
    
    
    if request.method == 'POST':
        global roll_num
        global Name
        
        roll_num=request.form['roll_num']
        Name=request.form['name']
        
        
        dir="training_image/"
        directory=Name+str(roll_num)
        path = os.path.join(dir, directory)
        os.mkdir(path)
        cam = cv2.VideoCapture(0)
        detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        #capture(roll_num,Name,path)
        sampleNum = 0
        while (True):
            
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                            
                names=Name + ""  + str(sampleNum) + ".jpg"
                            
                cv2.imwrite(f"{path}/{names}"  ,gray[y:y + h, x:x + w])
                cv2.imshow('Frame', img)
                sampleNum = sampleNum + 1
                        
            k=cv2.waitKey(1)
            if k%256 == 27 or k%256 == 32: 
                break
            elif sampleNum > 60 :
                break
                    #SPACE pressed
                        
        cam.release()
        cv2.destroyAllWindows()

    
    return render_template('trying.html',prediction_text="Student Details Saved")

@app.route('/face_alignmnent', methods=['GET', 'POST'])
def homses():
    
    
    dir="training_image/"
    loc=Name+str(roll_num)
    alignment.align(dir,loc,Name,roll_num)
    #engines(text="faces are aligned")
    
    return render_template('trying.html',prediction_text="Faces Are Now Aligned")
    
        
    
@app.route('/training', methods=['GET', 'POST'])
def thans():
    dir="training_image"
    training.face_recognize(dir)
    
    return render_template('trying.html',prediction_text="Done Training")

@app.route('/attendance', methods=['GET', 'POST'])
def resume():
    
        return render_template('index.html')
    
if __name__ == "__main__":
    app.run(debug=False)
    #app.run(debug=True)
