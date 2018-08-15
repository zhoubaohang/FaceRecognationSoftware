#### PART OF THIS CODE IS USING CODE FROM VICTOR SY WANG: https://github.com/iwantooxxoox/Keras-OpenFace/blob/master/utils.py ####

import numpy as np
from keras.models import Model

_FLOATX = 'float32'

def img_to_encoding(img1, model):
    img = img1[...,::-1]
    img = np.around(np.transpose(img, (2,0,1))/255.0, decimals=12)
    x_train = np.array([img])
    embedding = model.predict_on_batch(x_train)
    return embedding

def who_is_it(image, database, model):
    encoding = img_to_encoding(image, model)
    min_dist = 100
    for (name, db_enc) in database.items():
        dist = np.linalg.norm(encoding - db_enc)
        if dist < min_dist:
            min_dist = dist
            identity = name
    if min_dist > 0.7:
        identity = "Unknown"
        # print("Not in the database")
    else:
        print("it's {0}, the distance is {1}".format(identity, min_dist))
    return min_dist, identity