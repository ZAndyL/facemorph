# facemorph
Morph one face to another using facial landmark detection, delaunay triangles and affine transformations. Based on https://www.learnopencv.com/face-morph-using-opencv-cpp-python/

Dependencies:
dlib, opencv

How to use:
python facemorph.py shape_predictor_68_face_landmarks.dat hillary_clinton.jpg ted_cruz.jpg 

Sample output:

![Alt text](sample_output/0.1.jpg?raw=true "0.1")
![Alt text](sample_output/0.3.jpg?raw=true "0.3")
![Alt text](sample_output/0.5.jpg?raw=true "0.5")
![Alt text](sample_output/0.7.jpg?raw=true "0.7")
![Alt text](sample_output/0.9.jpg?raw=true "0.9")

Limitations:
Images have to be the same size, only one face per image.

To Do:
Auto center and crop so images of different sizes are supported
Allow multiple faces per image and choosing which face to morph
Port to mobile app
