import pickle
import numpy as np
from flask import Flask, request, render_template
import sklearn
from sklearn.metrics.pairwise import euclidean_distances

appli = Flask(__name__, static_url_path='/static')
application = appli

#Load the model from the pickle file
model = pickle.load(open('ourModel.pickle','rb'))

@appli.route('/')
def hello():
    return render_template('test.html')

@appli.route('/predication', methods=['POST'])
def predication():
    #Retrieve the values from the form and convert them to integers
    getValueInt = [int(x) for x in request.form.values()]

    # Validation of the values
    if any(x < 0 or x > 10 for x in getValueInt):
        return render_template('test.html', prediction_text='Please enter values between 0 and 10.')

    final_features = [np.array(getValueInt)]

    #Make a prediction with the loaded model
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)

    # Return the response as text
    return render_template('test.html', prediction_text='Your response {}'.format(output))

if __name__ == "__main__":
    appli.run(debug=True)
