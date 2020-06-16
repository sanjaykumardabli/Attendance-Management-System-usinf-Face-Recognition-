import face_recognition 
import docopt 
from sklearn import svm 
import os 
import pickle
  
def face_recognize(dir, test): 
    
    nam=pickle.load(open('model/s.pickle','rb'))
    test_image = face_recognition.load_image_file(test) 
  
    # Find all the faces in the test image using the default HOG-based model 
    face_locations = face_recognition.face_locations(test_image) 
    no = len(face_locations) 
    #print("Number of faces detected: ", no) 
    final_names=[]
    # Predict all the faces in the test image using the trained classifier 
    #print("Found:") 
    for i in range(no): 
        test_image_enc = face_recognition.face_encodings(test_image)[i] 
        name = nam.predict([test_image_enc]) 
        #print(*name)
        final_names.append(*name)
    if no ==0:
        result= "Oops...... Try Again With Different Images"
    else:
        result= f"number of faces detected {no} \n and the names are {final_names}"
    return result


                      
    
