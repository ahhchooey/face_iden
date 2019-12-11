#face identification application

##about
this is a simple facial identification tool. it is currently a skeleton where you can put images to train the model.

##how to use

####to install locally

1. clone the directory
2. in directory root, run ```python3 -m venv venv``` in terminal to create a virtual environment called venv
3. run ```source venv/bin/activate``` to activate the virtual environment
4. in directory root, run ```pip install -r requirements.txt```
5. run ```pip freeze``` to make sure that the correct dependencies are installed
6. run ```python test/camera_test.py``` to make sure that the camera and opencv are functional, q to cancel


####to run facial recognition

1. run ```python src/face.py```


####to train model

1. make images directory in src
2. make a directory with a name (ie. Alex) in the images directory
3. fill that directory with images of a person
4. from directory root, run ```python src/learn.py```
5. run ```python src/face.py```, you should see a name above the box if the model learned the face


##dependencies
* python3
* pip
* pillow => python imaging library
* opencv
