#from charset_normalizer import detect
import cv2
import random
import os




def faceBox(faceNet,frame):
    #print(frame)
    frameHeight=frame.shape[0]
    frameWidth=frame.shape[1]
    blob=cv2.dnn.blobFromImage(frame, 1.0, (300,300), [104,117,123], swapRB=False)
    faceNet.setInput(blob)
    detection=faceNet.forward()
    bboxs=[]
    for i in range(detection.shape[2]):
        confidence=detection[0,0,i,2]
        if confidence>0.7:
            x1=int(detection[0,0,i,3]*frameWidth)
            y1=int(detection[0,0,i,4]*frameHeight)
            x2=int(detection[0,0,i,5]*frameWidth)
            y2=int(detection[0,0,i,6]*frameHeight)
            bboxs.append([x1,y1,x2,y2])
            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0), 1)

    #print(detection.shape)
    #return detection
    return frame, bboxs



faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"


ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"


faceNet=cv2.dnn.readNet(faceModel, faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
#ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
ageList = ['0', '1', '10', '20', '30', '43', '60', '70']
genderList = ['Male', 'Female']


video=cv2.VideoCapture(0)

padding =20

while True:
    ret,frame=video.read()
    #detect=faceBox(faceNet,frame)
    frame,bboxs =faceBox(faceNet,frame)
    for bbox in bboxs:
        face=frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]
        blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPred=genderNet.forward()
        gender=genderList[genderPred[0].argmax()]


        ageNet.setInput(blob)
        agePred=ageNet.forward()
        age=ageList[agePred[0].argmax()]


        label="{},{}".format(gender,age)
        cv2.rectangle(frame,(bbox[0], bbox[1]-30), (bbox[2], bbox[1]), (0,0,0),-1) 
        cv2.putText(frame, label, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        print(age)

        if (int(age)<100) :
            print("Hello")
            filename=random.choice(os.listdir("0-100/"))
            cap = cv2.VideoCapture("0-100/"+filename)

            # Read until video is completed
            while(cap.isOpened()):
            # Capture frame-by-frame
                ret, frame1 = cap.read()
                if ret == True:
                # Display the resulting frame
                            cv2.imshow('Frame',frame1)
                            # Press Q on keyboard to  exit
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                             break

        if((int(age)>=0)and(int(age)<=2)):
            print("Hello")
            filename=random.choice(os.listdir("0-2/"))
            cap = cv2.VideoCapture("0-2/"+filename)

        if((int(age)>=4)and(int(age)<=6)):
            print("Hello")
            filename=random.choice(os.listdir("4-6/"))
            cap = cv2.VideoCapture("4-6/"+filename)

        if((int(age)>=8)and(int(age)<=12)):
            print("Hello")
            filename=random.choice(os.listdir("8-12/"))
            cap = cv2.VideoCapture("0-2/"+filename)

        if((int(age)>=8)and(int(age)<=12)):
            print("Hello")
            filename=random.choice(os.listdir("8-12/"))
            cap = cv2.VideoCapture("8-12/"+filename)

        if((int(age)>=15)and(int(age)<=20)):
            print("Hello")
            filename=random.choice(os.listdir("15-20/"))
            cap = cv2.VideoCapture("15-20/"+filename)

        if((int(age)>=25)and(int(age)<=32)):
            print("Hello")
            filename=random.choice(os.listdir("25-32/"))
            cap = cv2.VideoCapture("25-32/"+filename)

        if((int(age)>=38)and(int(age)<=43)):
            print("Hello")
            filename=random.choice(os.listdir("38-43/"))
            cap = cv2.VideoCapture("38-43/"+filename)

        if((int(age)>=48)and(int(age)<=53)):
            print("Hello")
            filename=random.choice(os.listdir("48-53/"))
            cap = cv2.VideoCapture("48-53/"+filename)

        if((int(age)>=60)and(int(age)<=100)):
            print("Hello")
            filename=random.choice(os.listdir("60-100/"))
            cap = cv2.VideoCapture("60-100/"+filename)

    # Read until video is completed
            while(cap.isOpened()):
            # Capture frame-by-frame
                ret, frame1 = cap.read()
                if ret == True:
                # Display the resulting frame
                            cv2.imshow('Frame',frame1)
                            # Press Q on keyboard to  exit
                            if cv2.waitKey(25) & 0xFF == ord('q'):
                             break
                            else:
                                break

            


    

    cv2.imshow("Age-Gender",frame)
   # k=cv2.waitKey(1)
    if cv2.waitKey(1) & 0x30 == ord('0'):
        break
video.release()
cap.release()
cv2.destroyAllWindows()
def main():
    face=faceBox(faceNet,frame)

if __name__=="_main_":
    main()