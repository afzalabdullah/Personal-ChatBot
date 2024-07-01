import random
import numpy as np
import pickle
import json
import nltk
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import Flask-CORS

# Ensure NLTK data is available
nltk.download('punkt')

lemmatizer = WordNetLemmatizer()

# chat initialization
model = load_model("chatbot_model.h5")
intents = json.loads(open("intents.json").read())
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))

app = Flask(__name__)
CORS(app)  # Enable CORS for your Flask app

@app.route("/get", methods=["POST"])
def chatbot_response():
    data = request.get_json()
    msg = data['msg']
    
    if msg.startswith('my name is'):
        name = msg[11:]
        ints = predict_class(msg, model)
        res1 = getResponse(ints, intents)
        res = res1.replace("{n}", name)
    elif msg.startswith('hi my name is'):
        name = msg[14:]
        ints = predict_class(msg, model)
        res1 = getResponse(ints, intents)
        res = res1.replace("{n}", name)
    else:
        ints = predict_class(msg, model)
        if not ints:  # Check if ints is empty
            res = "I'm sorry, I didn't understand that."
        else:
            res = getResponse(ints, intents)
    
    return jsonify({"response": res})

# chat functionalities
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]["intent"]
    list_of_intents = intents_json["intents"]
    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i["responses"])
            break
    return result

if __name__ == "__main__":
    app.run(debug=True)
