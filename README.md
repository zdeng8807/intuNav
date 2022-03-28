# In this repository, a Learning-based navigation generation method is introduced for Nasotracheal intubation. 

## Environment configuration：
* Python 3.6/3.7/3.8
* Pytorch 1.6
* pycocotools(Linux:```pip install pycocotools```; Windows:```pip install pycocotools-windows```)
* Ubuntu(Windows is not recommended)
* Best to use GPU training

## File structure：
```
├── src: Relevant modules that implement the model    
│     ├── resnet.py:Use resnet network as the backbone
│     ├── model.py:network structure file
│     └── utils.py:Implementation of some functions used in the training process
├── train_utils: Training and validation related modules (including cocotools)  
├── intu_dataset.py: Custom dataset is used to read the our dataset   
├── train.py: Using resnet as the backbone of the SSD network for training     
├── predict.py: Prediction script, using trained weights to make predictions    
├── intu_names.json: our dataset label file  
├── transforms.py: Preprocess the image
└── draw_box_utils.py: Used to draw prediction boxes
```

## Training method
* Prepare the dataset in advance
* Use the train.py script for training
