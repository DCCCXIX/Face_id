import cv2
import torch
import numpy as np
import time
from facenet_pytorch import MTCNN
from imutils.video import WebcamVideoStream
from data_collector import DataCollector

class FaceDetector(object):
    def __init__(self):

        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

        self.mtcnn = mtcnn = MTCNN(
            image_size=160,
            min_face_size=60,
            post_process=True,
            thresholds=[0.6, 0.7, 0.7],
            margin=20,
            factor=0.7,
            keep_all=False,
            select_largest=True,
            device=self.device)

        self.detects = 0

        print(f'Detection module is running on {self.device}')
        # dictionary of faces know to the network.

    # drawing boxes, landmarks and text
    def draw(self, frame, boxes, probs, landmarks):
        try:
            for box, probability, landmark in zip(boxes, probs, landmarks):

                cv2.rectangle(frame,
                              (box[0], box[1]),
                              (box[2], box[3]),
                              (0, 100, 100),
                              thickness=1)
                cv2.putText(frame, f'Samples collected: {self.detects}', (box[1], box[3]),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 1, cv2.LINE_AA)

        except:
            pass
        return frame

    def run(self):
        cap = WebcamVideoStream(src=0).start()
        while True:
            frame = cap.read()
            try:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.mtcnn(frame, save_path=f'./images/qsss/{str(time.time())}.jpg')
                self.detects =+ 1
                boxes, probs, landmarks = self.mtcnn.detect(frame, landmarks=True)
                self.draw(frame, boxes, probs, landmarks)
            except:
                pass

            cv2.imshow('Face Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

detector = FaceDetector()
detector.run()
        