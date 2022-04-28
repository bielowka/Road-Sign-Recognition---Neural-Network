from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os

import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img


import numpy as np

model = load_model("model_saved.h5")

classes = ['maszpierwszenstwo','rondo','stop','ustap','zakazzatrzymywania']


def preprocess(path):
    img_path = os.path.join(path)
    image = load_img(img_path, target_size=(224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)
    data = []
    data.append(image)
    data = np.array(data,dtype="float32")
    return data

def checkSign(path,cat):
    isRight = 0
    global model
    img = preprocess(path)
    prediction = model.predict(img)
    prediction = prediction.tolist()
    print()
    print(classes)
    print(prediction[0])
    print(path)
    res = max(prediction[0])
    for i in range(len(prediction[0])):
        if prediction[0][i] == res:
            print("Classified to: ",classes[i])
            if i == cat:
                isRight = 1
    if isRight:
        print("Correct")
    else:
        print("Not correct")

    print()
    return isRight

full = 17
right = 0

right += checkSign("testy/maszpierszenstwo1.png",0)
right += checkSign("testy/maszpierszenstwo2.jpg",0)
right += checkSign("testy/maszpierszenstwo3.jpg",0)
right += checkSign("testy/maszpierszenstwo4.png",0)
right += checkSign("testy/rondo.jpeg",1)
right += checkSign("testy/rondo1.jpg",1)
right += checkSign("testy/stop1.jpg",2)
right += checkSign("testy/stop2.jpeg",2)
right += checkSign("testy/stop3.jpg",2)
right += checkSign("testy/stop4.jpg",2)
right += checkSign("testy/stop5.jpeg",2)
right += checkSign("testy/stop6.png",2)
right += checkSign("testy/ustap1.jpg",3)
right += checkSign("testy/ustap2.jpg",3)
right += checkSign("testy/ustap2.jpg",3)
right += checkSign("testy/zakaz.jpeg",4)
right += checkSign("testy/zakaz1.jpeg",4)

print("Result: ",right/full*100,"% correct")