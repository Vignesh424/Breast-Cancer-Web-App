from flask import Flask, render_template, request
import pickle
import pandas as pd
import sklearn
import numpy as np

app = Flask(__name__)
model = pickle.load(open('breast_cancer.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('main.html')


@app.route('/predict', methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    features_name = ['clump_thickness', 'uniform_cell_size', 'uniform_cell_shape',
                     'marginal_adhesion', 'single_epithelial_size', 'bare_nuclei',
                     'bland_chromatin', 'normal_nucleoli', 'mitoses']
    data = pd.DataFrame(final_features, columns=features_name)
    output = model.predict(data)
    if output == 4:
        value = "Breast cancer"
    else:
        value = "No Breast cancer"

    return render_template("main.html", prediction_text='You have {}'.format(value))


if __name__ == "__main__":
    app.run(debug=True)
