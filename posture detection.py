from flask import Flask, Response, render_template
import cv2
from tensorflow import keras
import numpy as np

app = Flask(__name__)

model = keras.models.load_model('model.h5')
classes = ['bad', 'good']
face_cascade = cv2.CascadeClassifier('myenv\cv\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')


cam = cv2.VideoCapture(0)
cam.set(3,640) # set Width
cam.set(4,400) # set Height

def generate_frames():
    while True:
        success, frame = cam.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #cascade classifier works on grayscale images
            faces = face_cascade.detectMultiScale(gray,1.3,5) #scaleFactor=1.3, minNeighbors=5 
            for (x,y,w,h) in faces:
                frame1 = cv2.resize(frame, (128,128))
                np_frame = np.array(frame1)
                np_frame1 = np_frame.reshape((1,np_frame.shape[0], np_frame.shape[1],np_frame.shape[2]))
        #         cv2.putText(frame,str(np_frame1.shape),(x+w,y),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
                y_pred = model.predict(np_frame1)
                y_pred_i = y_pred.astype(int)
                y_x = y_pred_i.shape[0]
                y_pred_i_r = y_pred_i.reshape((y_x,))
                output = [classes[y] for y in y_pred_i_r]
        #         cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.putText(frame,output[0],(280,300),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            k = cv2.waitKey(30) & 0xff
            if(k==27):
                break
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype= 'multipart/x-mixed-replace; boundary= frame')

if __name__ == "__main__":
    app.run(debug= True)






