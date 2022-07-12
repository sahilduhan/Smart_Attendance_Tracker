import cv2
import numpy as np 
import os
import face_recognition
from datetime import datetime


path = 'images'
images = []
personName = []
myList = os.listdir(path)
# print(myList)

for curImage in myList:
    currentImage = cv2.imread(f'{path}/{curImage}')
    images.append(currentImage)
    personName.append(os.path.splitext(curImage)[0])

# print(personName)

def faceEncoding(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img)[0]
        encodeList.append(encodings)
    return encodeList

#using hog algorithm 
# this face encoding is encoding nearly by 128 points of the image
encodeListKnown = faceEncoding(images)
print("All the encoding is done")


def Attendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        myNameList = []
        for line in myDataList: 
            entry = line.split(',')
            myNameList.append(entry[0])
        if name not in myNameList:
            time_Now = datetime.now()
            timeStr = time_Now.strftime('%H:%M:%S')
            dateStr = time_Now.strftime('%d/%m/%Y')
            f.writelines(f'{name},{timeStr},{dateStr}')

cap = cv2.VideoCapture(0)

while True:
    ret , frame = cap.read()
    faces = cv2.resize(frame,(0,0),None,0.25, 0.25)
    faces = cv2.cvtColor(faces,cv2.COLOR_BGR2RGB)
    facesCurrentFrame = face_recognition.face_locations(faces)
    enocdesCurrFrame = face_recognition.face_encodings(faces,facesCurrentFrame)

    for encodeFace, faceLocation in zip(enocdesCurrFrame,facesCurrentFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)

        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = personName[matchIndex].upper()
            # print(name)
            y1, x2, y2, x1 = faceLocation
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),0)
            cv2.rectangle(frame,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(frame, name, (x1 + 6, y2 - 6) , cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
            Attendance(name)
    cv2.imshow("Camera",frame)
    if cv2.waitKey(2) == 13:
        break
cap.release()
cv2.destroyAllWindows()