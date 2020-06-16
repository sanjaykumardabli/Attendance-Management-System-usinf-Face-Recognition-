
import cv2

import os
import numpy as np
from PIL import Image
import pandas as pd
import datetime
import time
import os



roll_num=7
cam = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
Name='a'
dir="training_image/"
directory=Name
path = os.path.join(dir, directory)
os.mkdir(path)
sampleNum = 0
while (True):
        
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # incrementing sample number
            
            # saving the captured face in the dataset folder
            names=Name + "." + str(roll_num) + '.' + str(sampleNum) + ".jpg"
            
            cv2.imwrite(f"{path}/{names}"  ,gray[y:y + h, x:x + w])
            cv2.imshow('Frame', img)
            sampleNum = sampleNum + 1
        # wait for 100 miliseconds
        #if cv2.waitKey(0) & 0xFF == ord('q'):
         #   break
        # break if the sample number is morethan 100
        #if sampleNum > 30 or cv2.waitKey(0):
         #   break
        k=cv2.waitKey(1)
        if k%256 == 27 or k%256 == 32: 
            break
        elif sampleNum > 30 :
            break
    #SPACE pressed
        
cam.release()
cv2.destroyAllWindows()


