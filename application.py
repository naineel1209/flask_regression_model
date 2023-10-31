from flask import Flask, render_template, request, redirect, url_for, session, flash
import pandas as pd
import pickle

model = pickle.load(open("LinearRegressionModel.pkl", "rb"))

app = Flask(__name__)
car = pd.read_csv("cleaned data.csv")


@app.route("/")
def index():
    companies = sorted(car["company"].unique())
    car_models = sorted(car["name"].unique())
    year = sorted(car["year"].unique(), reverse=True)
    fuel_type = car["fuel_type"].unique()
    return render_template(
        "index.html",
        companies=companies,
        car_models=car_models,
        years=year,
        fuel_types=fuel_type,
    )


@app.route("/predict", methods=["POST"])
def predict():
    print("inside predict function")
    data = request.form
    print(data)
    # data = request.get_json()  # This method parses JSON data from the request
    # # Now 'data' will contain the JSON data sent in the request
    company = data.get("company")
    car_model = data.get("model")
    year = int(data.get("year"))
    fuel_type = data.get("fuel_type")
    kms_driven = int(data.get("kms_driven"))
    # print(company, car_model, year, fuel_type, kms_driven)

    prediction = model.predict(
        pd.DataFrame(
            [[car_model, company, year, kms_driven, fuel_type]],
            columns=[
                "name",
                "company",
                "year",
                "kms_driven",
                "fuel_type",
            ],
        )
    )

    return str(prediction[0])


if __name__ == "__main__":
    app.run(debug=True)
