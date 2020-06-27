import numpy as np
import csv
import string
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    print(request.form.get("message"))

    message = request.form.get("message")

    #Read in features.csv to match what we have to it
    with open('features.csv', 'r') as read_obj:
        features = list(csv.reader(read_obj, delimiter=','))
        feat_count = sum(1 for row in features)
        print(feat_count)

        #Define output with zeros to start
        input_features = np.zeros(feat_count)

        #Need to send it same features used in the ML model
        
        #Create the two calculated fields
        def count_punct(text):
            count = sum([1 for char in text if char in string.punctuation])
            return round(count/(len(text) - text.count(" ")), 3)*100

        body_len = len(message) - message.count(" ")
        punct = count_punct(message)

        input_features[0] = body_len
        input_features[1] = punct

        # Skip body_len, punct, and empty string ('')
        cnt = 0
        for row in features:
            if cnt >= 3:
                if message.count(str(row[0])) > 0:
                    input_features[cnt] = message.count(str(row[0]))
            cnt = cnt + 1
        print(cnt)
            
    output = model.predict([input_features])

    print(output)

    if str(output).count('ham') > 0:
        output = 'LEGIT'
    else:
        output = 'SPAM'


    return render_template('index.html', prediction_text='Text is considered {}'.format(str(output)))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)