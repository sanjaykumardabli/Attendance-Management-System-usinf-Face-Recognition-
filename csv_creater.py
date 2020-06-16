import face_recognition 
import os 
import datetime
import os
import pickle
import re
import csv
def csv_create(nm):
    #drive_letter = r'C:\\' 
    folder_name = 'attendance files\\'
    folder_time = datetime.datetime.now().strftime("%Y-%m-%d_%I-%M-%S_%p")
    folder_to_save_files = folder_name + folder_time+'.csv'

    with open(folder_to_save_files, 'w') as file:
        fieldnames = ['Student_name', 'Roll_number','Present']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        
        j,k=0,0
        for i in  range(len(nm)):
            writer.writerow({'Student_name': nm[j][k], 'Roll_number': nm[j][k+1],'Present':'Yes'})
            
            j+=1
        
        
            
#csv_create(name,roll_num)
def face_recognize(dir, test): 
    
    nam=pickle.load(open('model/s.pickle','rb'))
    test_image = face_recognition.load_image_file(test) 

    face_locations = face_recognition.face_locations(test_image) 
    no = len(face_locations) 
     
    final_names=[]
    
    for i in range(no): 
        test_image_enc = face_recognition.face_encodings(test_image)[i] 
        name = nam.predict([test_image_enc]) 
        
        final_names.append(*name)
    
    nm=[]
    for j in final_names:
        temp = re.compile("([a-zA-Z]+)([0-9]+)") 
        res = temp.match(j).groups() 
        nm.append(res)
    csv_create(nm)
    print(final_names)
    print(nm)
    if no ==0:
        result= "Oops...... Try Again With Different Images"
    else:
        result= f"number of faces detected {no} \n and the names are {final_names}"
    return result
    
dir="training_image"
#img=r"C:\Users\ADMIN\Pictures\Camera Roll\WIN_20191118_18_45_17_Pro.jpg"
#face_recognize(dir,img)                      
    
