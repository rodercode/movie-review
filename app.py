from flask import Flask, request as req, jsonify
import pandas as pd
import numpy as np
import joblib
import model_test

# Create flask app
app = Flask(__name__)

# Load model_joblib
model = joblib.load('model_joblib')


@app.route("/predict/reviews", methods=["POST"])
def pred():
    df = pd.DataFrame(req.json)
    reviews = np.array(df['review'])
    reviews_vect = model_test.cv.transform(reviews)

    pred_list = list(model.predict(reviews_vect))

    for i in range(len(pred_list)):
        if pred_list[i] == 1:
            pred_list[i] = "Pos"
        else:
            pred_list[i] = "Neg"

    return jsonify({"Prediction": pred_list})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
