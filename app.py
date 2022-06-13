

from flask import Flask, render_template, Response
import cv2
app=Flask(__name__)



camera = cv2.VideoCapture(0)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


def generate_frames():  
    
    while True:
        read,frame = camera.read()  # read the camera frame
        
        if not read:
            break
        
        else:
            
            detect_face=cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')
            faces_in_frame =detect_face.detectMultiScale(frame,1.1,8)
          
            
             #Draw the rectangle around each face
            for (x, y, w, h) in faces_in_frame:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



if __name__=='__main__':
    app.run()