#not implemented#
------------------------------------
Python is needed to run this script:
https://www.python.org

Lines marked with '>' are to be entered into the command prompt from the scripts' origin path.
For Linux, input 'python3' instead of 'python' and 'pip3' instead of 'pip'

Installing dependancies:
>pip install -r requirements.txt
------------------------------------

Recording face images from a camera:

>python webcam_capture.py

Folder name should be changed at line 55:
55 self.mtcnn(frame, save_path=f'./images/YOUR_FOLDER_NAME/{str(time.time())}.jpg')

This script will keep saving detected faces to the selected folder.
Around 1500 images are needed for acceptable accuracy.
Getting more samples in different lighting conditions is advised.
Press 'q' to close the script prematurely.
------------------------------------

Proccessing data for model input:

>python data_collector.py

This script will automatically create a dictionary if known faces
depending on the contents of ./images and save it as facedict.json.
It will also create a training_data.npy, which is used for model training.
------------------------------------

To lauch face detection and recognition demonstration:

>python face_detector.py

This script will lauch a demo which will show bounding boxes and landmarks for detected faces,
as well as labels and confidence. Press 'q' to close the script prematurely.


