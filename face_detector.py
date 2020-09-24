import cv2
import torch
import numpy as np
import time
from datetime import datetime
from facenet_pytorch import MTCNN
from imutils.video import WebcamVideoStream
from face_recognition_classifier import frc

class FaceDetector(object):
    def __init__(self):

        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

        self.mtcnn = mtcnn = MTCNN(
        image_size = 160,
        min_face_size = 60,
        post_process = True,
        thresholds = [0.6, 0.7, 0.7],
        margin = 20,
        factor = 0.6,
        keep_all = True,
        device = self.device)

        print(f'Detection module is running on {self.device}')
        #dictionary of faces know to the network.
        
    #drawing boxes, landmarks and text
    def draw(self, frame, boxes, probs, landmarks, faces):
        try:
            for box, probability, landmark, face in zip(boxes, probs, landmarks, faces):
                cv2.rectangle(frame,
                              (box[0], box[1]),
                              (box[2], box[3]),
                              (0, 100, 100),
                              thickness=1)
                #recognizing every face in a frame to output the persons' name
                recognized_person, confidence = frc.recognize(face)
                
                cv2.putText(frame, '{:},{:.4f}'.format(recognized_person,confidence), (box[1], box[3]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 1, cv2.LINE_AA)
                
                cv2.circle(frame, tuple(landmark[0]), 2, (0, 100, 255), -1)
                cv2.circle(frame, tuple(landmark[1]), 2, (0, 100, 255), -1)
                cv2.circle(frame, tuple(landmark[2]), 2, (0, 100, 255), -1)
                cv2.circle(frame, tuple(landmark[3]), 2, (0, 100, 255), -1)
                cv2.circle(frame, tuple(landmark[4]), 2, (0, 100, 255), -1)
        except:
            pass
        return frame

    def run(self):
        #cap = cv2.VideoCapture(0)
        cap = WebcamVideoStream(src=0).start()
        index = 0
        while True:
            #ret, frame = cap.read()
            frame = cap.read()
            #fps.update()
            since = time.time()
            try:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                #saving aligned faces to the directory, uncomment if needed
                #mtcnn(frame, save_path=f'./images/queen/{str(time.time())}.jpg')
                boxes, probs, landmarks = self.mtcnn.detect(frame, landmarks=True)
                faces = self.mtcnn(frame)
                self.draw(frame, boxes, probs, landmarks, faces)
                index =+ 1

            except:
                pass

            print(f'Time for one frame: {time.time() - since}')

            cv2.imshow('Face Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cv2.destroyAllWindows()

detector = FaceDetector()
detector.run()
