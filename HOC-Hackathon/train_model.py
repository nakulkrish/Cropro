# In train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
data = pd.read_csv('Fertilizer-Prediction.csv')

# Trim whitespace from all column names
data.columns = data.columns.str.strip()

# Encode categorical variables
label_encoders = {}
for column in ['Soil Type', 'Crop Type']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Features and target variable
X = data[['Temparature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type', 'Nitrogen', 'Potassium', 'Phosphorous']]
y = data['Fertilizer Name']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model and label encoders
joblib.dump(model, 'fertilizer_model.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')