import requests
from flask import Flask, render_template, request, redirect, url_for, flash, session,jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from newsapi import NewsApiClient
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os
import joblib
import numpy as np

# Load dataset for crop recommendation
data = pd.read_csv('Crop_recommendation.csv')
X = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = data['label']

# Train crop recommendation model
crop_model = RandomForestClassifier()
crop_model.fit(X, y)


app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for flash messages

# Dummy users data (mobile numbers and passwords)
users = {}


# Route for the root URL ("/")
@app.route('/')
def index():
    return redirect(url_for('login'))  # Redirecting to login page by default

# Route for the signup page
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        mobile_number = request.form['mobile_number']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        # Backend check if passwords match
        if password != confirm_password:
            flash('Passwords do not match. Please try again.')
            return redirect(url_for('signup'))

        # Backend check if mobile number is exactly 10 digits
        if len(mobile_number) != 10 or not mobile_number.isdigit():
            flash('Please enter a valid 10-digit mobile number.')
            return redirect(url_for('signup'))

        # Check if the mobile number is already registered
        if mobile_number in users:
            flash('Mobile number already registered. Please login.')
            return redirect(url_for('login'))
        
        # Add the mobile number and password to the users dictionary
        users[mobile_number] = password
        flash('Signup successful! Please login.')
        return redirect(url_for('login'))
    
    return render_template('signup.html')

# Route for the login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        mobile_number = request.form['mobile_number']
        password = request.form['password']
        
        # Validate login using mobile number
        if mobile_number in users and users[mobile_number] == password:
            flash('Login successful!')
            return redirect(url_for('home'))
        else:
            flash('Invalid mobile number or password. Please try again.')
            return redirect(url_for('login'))
    
    return render_template('login.html')

# Route for the home page (after login)
@app.route('/home')
def home():
    return render_template('home.html')


# Route for Crop Recommendation page
@app.route('/crop-recommendation', methods=['GET', 'POST'])
def crop_recommendation():
    prediction = None  # Initialize prediction variable
    if request.method == 'POST':
        # Get data from form
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # Prepare input data for the crop recommendation model
        input_data = [[N, P, K, temperature, humidity, ph, rainfall]]
        prediction = crop_model.predict(input_data)[0]  # Make a prediction

    return render_template('crop_recommendation.html', prediction=prediction)



# Path to the local CSV file
CSV_FILE_PATH = 'weather_data.csv'  # Ensure this file is in your project directory

def fetch_weather_data():
    try:
        return pd.read_csv(CSV_FILE_PATH)
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        return None

def irrigation_recommendation(precipitation):
    if precipitation > 80 :
        return "High"
    elif precipitation < 80 and precipitation >= 30:
        return "Medium"
    else:
        return "Low"

def plot_weather_data(past_days):
    plt.figure(figsize=(10, 5))  # Set the figure size
    plt.plot(past_days['datetime'], past_days['tempmax'], label='Max Temp (°C)', marker='o')  # Max Temp with marker
    plt.plot(past_days['datetime'], past_days['tempmin'], label='Min Temp (°C)', marker='o')  # Min Temp with marker

    # Title and labels
    plt.title('Past 7 Days Weather Data')
    plt.xlabel('Date')
    plt.ylabel('Temperature (°C)')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.legend()  # Add legend

    # Save the plot to a file
    plot_path = 'static/weather_plot.png'  # Path for saving plot
    plt.savefig(plot_path)
    plt.close()  # Close the plot to free memory

    return plot_path  # Return the path of the saved plot


@app.route('/irrigation')
def irrigation():
    weather_data = fetch_weather_data()
    
    if weather_data is not None:
        weather_data['datetime'] = pd.to_datetime(weather_data['datetime'])
        today_date = pd.to_datetime("today").normalize()
        past_days = weather_data[weather_data['datetime'] <= today_date].tail(7)

        if today_date in past_days['datetime'].values:
            today_weather = past_days[past_days['datetime'] == today_date].iloc[0]
            precip_today = today_weather['precip']
            wind_speed = today_weather['windspeed']  # Assuming windspeed is a column in the CSV
            humidity = today_weather['humidity']    # Assuming humidity is a column in the CSV
            precip_prob = today_weather['precipprob']  # Assuming precipprob is precipitation probability
            irrigation_amount = irrigation_recommendation(precip_today)
        else:
            today_weather = None
            wind_speed = None
            humidity = None
            precip_prob = None
            irrigation_amount = "No data available for today."
        
        # Plotting the graph
        plot_path = plot_weather_data(past_days)

        return render_template('irrigation.html', 
                               past_days=past_days.iterrows(), 
                               today_weather=today_weather, 
                               irrigation_amount=irrigation_amount,
                               wind_speed=wind_speed,
                               humidity=humidity,
                               precip_prob=precip_prob,
                               plot_path=plot_path)
    else:
        return "Error fetching weather data."
    
# Articles API KEY
API_KEY = '96eb1b2937524f3d97bca9dbd92e422b'



model = joblib.load('fertilizer_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')

@app.route('/disease-prediction')
def disease_prediction():
    return render_template('disease_prediction.html')  # Render Disease Prediction page

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from request
    data = request.get_json(force=True)
    
    # Prepare input features for prediction
    soil_type_encoded = label_encoders['Soil Type'].transform([data['soil_type']])[0]
    crop_type_encoded = label_encoders['Crop Type'].transform([data['crop_type']])[0]
    
    features = np.array([[data['temperature'], data['humidity'], data['moisture'],
                          soil_type_encoded, crop_type_encoded,
                          data['nitrogen'], data['potassium'], data['phosphorous']]])
    
    # Make prediction
    prediction = model.predict(features)
    
    return jsonify({'fertilizer': prediction[0]})


def fetch_articles(query=None):
    if query is None:
        query = 'agriculture OR farming OR technology'
    
    url = f'https://newsapi.org/v2/everything?q={query}&apiKey={API_KEY}'
    
    # Make the API request
    response = requests.get(url)
    
    # Print the raw response text for debugging
    print(f"Response Status Code: {response.status_code}")
    print(f"Response Text: {response.text}")  # Add this line to inspect the response
    
    try:
        # Check if response is successful and return articles
        if response.status_code == 200:
            return response.json().get('articles', [])
        else:
            print(f"Error fetching articles: {response.status_code}, {response.text}")
            return []
    except requests.exceptions.JSONDecodeError:
        print("Error: Unable to parse the response as JSON")
        return []

@app.route('/articles', methods=['GET', 'POST'])
def articles():
    if request.method == 'POST':
        query = request.form.get('search_query')
        articles = fetch_articles(query)
    else:
        articles = fetch_articles()  # Fetch default articles on GET request
    
    return render_template('articles.html', articles=articles)


@app.route('/logout')
def logout():
    session.pop('logged_in', None)  # Clear session
    return redirect(url_for('login'))  # Redirect to login page


if __name__ == '__main__':
    app.run(debug=True)