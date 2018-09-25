import random
import string
from keras.models import load_model
from PIL import Image, ImageDraw, ImageFont
import numpy as np


characters = string.digits + string.ascii_uppercase
width, height, n_len, n_class = 440, 80, 17, len(characters)

def gen_vin():
    v1 = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(11)).upper()
    v2 = ''.join(random.choice(string.digits) for _ in range(6))
    return v1 + v2
                
def gen_vin_img(vin):
    img = Image.new('RGB', (440, 80), color = (73, 109, 137))
    fnt = ImageFont.truetype('courier_new.ttf', 40)
    d = ImageDraw.Draw(img)
    d.text((15,20), vin, font=fnt, fill=(255, 255, 0))
    return img

def decode(y):
    characters = string.digits + string.ascii_uppercase
    y = np.argmax(np.array(y), axis=2)[:,0]
    return ''.join([characters[x] for x in y])

def gen(batch_size=32):
    X = np.zeros((batch_size, height, width, 3), dtype=np.uint8)
    y = [np.zeros((batch_size, n_class), dtype=np.uint8) for i in range(n_len)]
    #generator = ImageCaptcha(width=width, height=height, fonts=fonts)
    while True:
        for i in range(batch_size):
            #random_str = ''.join([random.choice(characters) for j in range(4)])
            random_str = gen_vin()
            img = gen_vin_img(random_str)
            img.save('test.png')
            X[i] = img
            for j, ch in enumerate(random_str):
                y[j][i, :] = 0
                y[j][i, characters.find(ch)] = 1
        yield X, y

model = load_model('vin.h5')
X, y = next(gen(1))
y_pred = model.predict(X)
print('real: %s\npred:%s'%(decode(y), decode(y_pred)))

