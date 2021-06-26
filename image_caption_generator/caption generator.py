#import keras package for preprocessing the text.
#import pickle package for saving and loading the file.
#cv2 package is imported for image preprocessing.
from keras.preprocessing.text import Tokenizer
from pickle import dump
from pickle import load
import cv2



def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text


def load_set(filename):
    doc = load_doc(filename)
    dataset = list()
    for line in doc.split('\n'):
        if len(line) < 1:
            continue
        identifier = line.split('.')[0]
        dataset.append(identifier)
    return set(dataset)


def to_lines(descriptions):
    all_desc = list()
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc


def create_tokenizer(descriptions):
    lines = to_lines(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


def load_clean_descriptions(filename, dataset):
    doc = load_doc(filename)
    descriptions = dict()
    for line in doc.split('\n'):
        tokens = line.split()
        image_id, image_desc = tokens[0], tokens[1:]
        if image_id in dataset:
            if image_id not in descriptions:
                descriptions[image_id] = list()
            desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
            descriptions[image_id].append(desc)
    return descriptions


filename = r"/home/user/Desktop/bbq/Flickr8k_text/Flickr_8k.trainImages.txt"
train = load_set(filename)
print('Dataset: %d' % len(train))
train_descriptions = load_clean_descriptions(r'/home/user/Desktop/bbq/results/descriptions.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))
tokenizer = create_tokenizer(train_descriptions)
dump(tokenizer, open('tokenizer.pkl', 'wb'))

from numpy import argmax
from keras.preprocessing.sequence import pad_sequences
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.models import load_model


def extract_features(filename):
    model = VGG16()
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)

    image = load_img(filename, target_size=(224,224))


    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], 3))
    image = preprocess_input(image)
    feature = model.predict(image, verbose=0)
    return feature


def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = argmax(yhat)
        word = word_for_id(yhat, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text


# load the tokenizer
tokenizer = load(open('/home/user/Desktop/bbq/tokenizer.pkl', 'rb'))
# pre-define the max sequence length (from training)
max_length = 34
# load the model
model = load_model(r'/home/user/Desktop/bbq/results/model_17.h5')
# load and prepare the photograph
photo = extract_features('/home/user/Desktop/bbq/pictures/second.jpg')
# generate description
description = generate_desc(model, tokenizer, photo, max_length)
print(description)

description = description.replace('startseq', '').replace('endseq', '')
text = description

import gtts
from playsound import playsound
from gtts import gTTS
import os

language = 'en'

tts = gtts.gTTS(text=text, lang="en")
tts.save("text.mp3")
playsound("text.mp3")

# speech = gTTS(text=text, lang=language, slow=False)
# speech.save('text.mp3')
# os.system("start text.mp3")