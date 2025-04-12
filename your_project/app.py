from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression
import warnings

# Suppressing warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Load dataset
df = pd.read_csv("jaipur.csv")
df['datetime'] = pd.to_datetime(df['datetime'], format='%d-%m-%Y')
df['year'] = df['datetime'].dt.year
df['month'] = df['datetime'].dt.month

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('data.html')

@app.route('/recommendation', methods=['POST'])
def recommendation():
    travel_date = request.form['travel_date']
    date_components = travel_date.split('-')
    year = int(date_components[0])
    month = int(date_components[1])
    day = int(date_components[2])

    df_year = df[df['year'] == year]
    predictions_dict = {}

    if len(df_year) == 0:
        X_train = df[df['year'] != year][['month', 'humidity']]
        y_train = df[df['year'] != year]['temp']

        model = LinearRegression()
        model.fit(X_train, y_train)

        avg_humidity_by_month = df.groupby('month')['humidity'].mean().to_dict()

        for m in range(1, 13):
            prediction = model.predict([[m, avg_humidity_by_month[m]]])[0]
            predictions_dict[m] = prediction
    else:
        X = df_year[['month', 'humidity']]
        y = df_year['temp']

        model = LinearRegression()
        model.fit(X, y)

        avg_humidity_by_month = df_year.groupby('month')['humidity'].mean().to_dict()

        for m in range(1, 13):
            prediction = model.predict([[m, avg_humidity_by_month[m]]])[0]
            predictions_dict[m] = prediction

    prediction = predictions_dict.get(month, "Invalid month")

    if prediction != "Invalid month":
        if prediction > 80:
            suggestion = "Ooty, Tamil Nadu"
        elif 78 <= prediction <= 79:
            suggestion = "Munnar, Kerala"
        elif 77 <= prediction < 80:
            suggestion = "Rishikesh, Uttarakhand"
        elif 75 <= prediction <= 77:
            suggestion = "Jaipur, Rajasthan"
        else:
            suggestion = "No suitable place found based on the predicted temperature."
    else:
        suggestion = "Invalid month."

    return render_template('data.html', prediction=prediction, suggestion=suggestion)

if __name__ == '__main__':
    app.run(debug=True)
