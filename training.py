import face_recognition 
#import docopt 
from sklearn import svm 
import os
import pickle



def face_recognize(dir): 
    
    encodings = [] 
    names = [] 

    # Training directory 
    if dir[-1]!='/': 
        dir += '/'
    train_dir = os.listdir(dir) 

    # Loop through each person in the training directory 
    for person in train_dir: 
        pix = os.listdir(dir + person) 

        # Loop through each training image for the current person 
        for person_img in pix: 
            
            face = face_recognition.load_image_file(person_img) 
            face_bounding_boxes = face_recognition.face_locations(face) 

             
            if len(face_bounding_boxes) == 1: 
                face_enc = face_recognition.face_encodings(face)[0] 
                 
                encodings.append(face_enc) 
                names.append(person) 
            else: 
                print(person + "/" + person_img + " can't be used for training") 

# Create and train the SVC classifier 
    clf = svm.SVC(gamma ='scale') 
    model=clf.fit(encodings, names) 
    filename='model/s.pickle'
    pickle.dump(model,open(filename, 'wb'))

