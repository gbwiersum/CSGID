#! usr/bin/env python3

def make_prediction(model, image):
    pred = model.predict(image)
    return pred
