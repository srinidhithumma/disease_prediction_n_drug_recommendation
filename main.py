from flask import Flask, request, render_template, jsonify, redirect, url_for, session, send_file
import numpy as np
import pandas as pd
import pickle
from fuzzywuzzy import process
import ast
import pymongo
from io import StringIO
from io import BytesIO
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from bson.objectid import ObjectId
import os

app = Flask(__name__)
app.secret_key = os.urandom(24).hex()

# MongoDB setup
client = pymongo.MongoClient("mongodb+srv://sitanagapavani65:puppy1334@cluster0.buv6uml.mongodb.net/")
db = client["medicine_recommendation"]
users_collection = db["users"]
feedback_collection = db["feedback"]
activity_collection = db["user_activity"]

# Loading the datasets
sym_des = pd.read_csv("kaggle_dataset/symptoms_df.csv")
precautions = pd.read_csv("kaggle_dataset/precautions_df.csv")
workout = pd.read_csv("kaggle_dataset/workout_df.csv")
description = pd.read_csv("kaggle_dataset/description.csv")
medications = pd.read_csv('kaggle_dataset/medications.csv')
diets = pd.read_csv("kaggle_dataset/diets.csv")

# Load the symptom relationships CSV (your cascading symptoms CSV)
symptom_relationships = pd.read_csv("kaggle_dataset/symptoms_df.csv")

# Load the trained model
Rf = pickle.load(open('model/RandomForest.pkl', 'rb'))

# Symptoms and diseases dictionaries
symptoms_list = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4,
                 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9,
                 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13,
                 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18,
                 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22,
                 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26,
                 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31,
                 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35,
                 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39,
                 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43,
                 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46,
                 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50,
                 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54,
                 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58,
                 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61,
                 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66,
                 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70,
                 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74,
                 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77,
                 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81,
                 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85,
                 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89,
                 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92,
                 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96,
                 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100,
                 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103,
                 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107,
                 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110,
                 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113,
                 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116,
                 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120,
                 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124,
                 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127,
                 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}

diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction',
                 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma',
                 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)',
                 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A',
                 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis',
                 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)',
                 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism',
                 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal Positional Vertigo',
                 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}

symptoms_list_processed = {symptom.replace('_', ' ').lower(): value for symptom, value in symptoms_list.items()}

# New functions for cross-column cascading symptoms
def clean_symptom(symptom):
    """Clean and normalize symptom names"""
    if pd.isna(symptom) or symptom == '':
        return None
    return symptom.strip().replace(' ', '_').lower()

def get_primary_symptoms():
    """Get all unique symptoms from all columns"""
    all_symptoms = []
    for col in ['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4']:
        symptoms = symptom_relationships[col].dropna().unique()
        all_symptoms.extend(symptoms)
    
    cleaned_symptoms = [clean_symptom(s) for s in all_symptoms if clean_symptom(s)]
    return sorted(list(set(cleaned_symptoms)))

def get_secondary_symptoms(primary_symptom):
    """Get available symptoms based on primary symptom selection from any column"""
    if not primary_symptom:
        return []
    
    secondary_symptoms = []
    
    # Find all rows where the primary symptom appears in ANY column
    for col in ['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4']:
        try:
            matching_rows = symptom_relationships[
                symptom_relationships[col].astype(str).str.strip().str.replace(' ', '_').str.lower() == primary_symptom.lower()
            ]
            
            # Get symptoms from OTHER columns in these matching rows
            for other_col in ['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4']:
                if other_col != col:  # Don't include the same column
                    symptoms = matching_rows[other_col].dropna().unique()
                    secondary_symptoms.extend(symptoms)
        except:
            continue
    
    cleaned_symptoms = [clean_symptom(s) for s in secondary_symptoms if clean_symptom(s)]
    # Remove the primary symptom from secondary options
    cleaned_symptoms = [s for s in cleaned_symptoms if s != primary_symptom.lower()]
    return sorted(list(set(cleaned_symptoms)))

def get_tertiary_symptoms(primary_symptom, secondary_symptom):
    """Get available symptoms based on first two selections"""
    if not primary_symptom or not secondary_symptom:
        return []
    
    tertiary_symptoms = []
    selected_symptoms = [primary_symptom.lower(), secondary_symptom.lower()]
    
    # Find rows that contain BOTH selected symptoms in ANY combination of columns
    for index, row in symptom_relationships.iterrows():
        row_symptoms = []
        for col in ['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4']:
            if pd.notna(row[col]):
                cleaned = clean_symptom(row[col])
                if cleaned:
                    row_symptoms.append(cleaned)
        
        # Check if this row contains both selected symptoms
        if all(symptom in row_symptoms for symptom in selected_symptoms):
            # Add the remaining symptoms from this row
            for symptom in row_symptoms:
                if symptom not in selected_symptoms:
                    tertiary_symptoms.append(symptom)
    
    return sorted(list(set(tertiary_symptoms)))

def get_quaternary_symptoms(primary_symptom, secondary_symptom, tertiary_symptom):
    """Get available symptoms based on first three selections"""
    if not primary_symptom or not secondary_symptom or not tertiary_symptom:
        return []
    
    quaternary_symptoms = []
    selected_symptoms = [primary_symptom.lower(), secondary_symptom.lower(), tertiary_symptom.lower()]
    
    # Find rows that contain ALL THREE selected symptoms in ANY combination of columns
    for index, row in symptom_relationships.iterrows():
        row_symptoms = []
        for col in ['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4']:
            if pd.notna(row[col]):
                cleaned = clean_symptom(row[col])
                if cleaned:
                    row_symptoms.append(cleaned)
        
        # Check if this row contains all three selected symptoms
        if all(symptom in row_symptoms for symptom in selected_symptoms):
            # Add the remaining symptom from this row (should be only one left)
            for symptom in row_symptoms:
                if symptom not in selected_symptoms:
                    quaternary_symptoms.append(symptom)
    
    return sorted(list(set(quaternary_symptoms)))

def get_best_matching_disease(selected_symptoms):
    """Get the best matching disease from CSV based on selected symptoms"""
    if not selected_symptoms:
        return None
    
    selected_symptoms_lower = [s.lower() for s in selected_symptoms]
    best_match = None
    max_matches = 0
    
    # Find the row with the most matching symptoms
    for index, row in symptom_relationships.iterrows():
        row_symptoms = []
        for col in ['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4']:
            if pd.notna(row[col]):
                cleaned = clean_symptom(row[col])
                if cleaned:
                    row_symptoms.append(cleaned)
        
        # Count how many selected symptoms match this row
        matches = sum(1 for symptom in selected_symptoms_lower if symptom in row_symptoms)
        
        # If this row has more matches than our current best, and matches all selected symptoms
        if matches > max_matches and matches == len(selected_symptoms_lower):
            max_matches = matches
            if pd.notna(row['Disease']):
                best_match = row['Disease']
    
    return best_match

def get_possible_diseases(selected_symptoms):
    """Get possible diseases based on selected symptoms"""
    if not selected_symptoms:
        return []
    
    possible_diseases = []
    selected_symptoms_lower = [s.lower() for s in selected_symptoms]
    
    # Find rows that contain ALL selected symptoms
    for index, row in symptom_relationships.iterrows():
        row_symptoms = []
        for col in ['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4']:
            if pd.notna(row[col]):
                cleaned = clean_symptom(row[col])
                if cleaned:
                    row_symptoms.append(cleaned)
        
        # Check if this row contains all selected symptoms
        if all(symptom in row_symptoms for symptom in selected_symptoms_lower):
            if pd.notna(row['Disease']):
                possible_diseases.append(row['Disease'])
    
    return list(set(possible_diseases))

def information(predicted_dis):
    """Extract information from datasets based on predicted disease"""
    disease_desciption = description[description['Disease'] == predicted_dis]['Description']
    disease_desciption = " ".join([w for w in disease_desciption])

    disease_precautions = precautions[precautions['Disease'] == predicted_dis][
        ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    disease_precautions = [col for col in disease_precautions.values]

    disease_medications = medications[medications['Disease'] == predicted_dis]['Medication']
    disease_medications = [med for med in disease_medications.values]

    disease_diet = diets[diets['Disease'] == predicted_dis]['Diet']
    disease_diet = [die for die in disease_diet.values]

    disease_workout = workout[workout['disease'] == predicted_dis]['workout']

    return disease_desciption, disease_precautions, disease_medications, disease_diet, disease_workout

def predicted_value(patient_symptoms):
    """Predict disease based on symptoms"""
    i_vector = np.zeros(len(symptoms_list_processed))
    for symptom in patient_symptoms:
        if symptom in symptoms_list_processed:
            i_vector[symptoms_list_processed[symptom]] = 1
    return diseases_list[Rf.predict([i_vector])[0]]

def correct_spelling(symptom):
    """Correct symptom spelling using fuzzy matching"""
    closest_match, score = process.extractOne(symptom, symptoms_list_processed.keys())
    if score >= 80:
        return closest_match
    else:
        return None

@app.route('/', methods=['GET'])
def home_page():
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        # Find user by email
        user = users_collection.find_one({'email': email})
        
        if user and user['password'] == password:  # In production, use password hashing
            session['email'] = email
            session['username'] = user['username']  # Store both email and username
            return redirect(url_for('predict'))
        else:
            return render_template('login.html', message='Invalid username or password')
    
    return render_template('login.html', message=None)

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]

        # Check if username or email already exists
        existing_user = users_collection.find_one({"$or": [{"username": username}, {"email": email}]})

        if existing_user:
            return render_template("signup.html", message="Username or Email already exists!")

        # Insert new user into MongoDB
        users_collection.insert_one({
            "username": username,
            "email": email,
            "password": password,  # In production, hash this password
            "created_at": datetime.now()
        })

        return redirect(url_for("login"))

    return render_template("signup.html", message=None)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home_page'))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # Check if user is logged in
    if 'username' not in session:
        return redirect(url_for('login'))

    if request.method == 'GET':
        primary_symptoms = get_primary_symptoms()
        return render_template('index.html', primary_symptoms=primary_symptoms)

    if request.method == 'POST':
        # Get selected symptoms from the cascading dropdowns
        primary_symptom = request.form.get('primary_symptom')
        secondary_symptom = request.form.get('secondary_symptom')
        tertiary_symptom = request.form.get('tertiary_symptom')
        quaternary_symptom = request.form.get('quaternary_symptom')
        
        # Build the selected symptoms list
        selected_symptoms = []
        if primary_symptom and primary_symptom != '':
            selected_symptoms.append(primary_symptom)
        if secondary_symptom and secondary_symptom != '':
            selected_symptoms.append(secondary_symptom)
        if tertiary_symptom and tertiary_symptom != '':
            selected_symptoms.append(tertiary_symptom)
        if quaternary_symptom and quaternary_symptom != '':
            selected_symptoms.append(quaternary_symptom)
        
        if not selected_symptoms:
            primary_symptoms = get_primary_symptoms()
            message = "Please select at least one symptom."
            return render_template('index.html', message=message, primary_symptoms=primary_symptoms)
        
        try:
            # Get possible diseases from CSV first
            possible_diseases = get_possible_diseases(selected_symptoms)
            
            # Get the best matching disease from CSV
            csv_predicted_disease = get_best_matching_disease(selected_symptoms)
            
            # Use CSV prediction if available, otherwise fall back to ML model
            if csv_predicted_disease:
                predicted_disease = csv_predicted_disease
                prediction_source = "CSV Database"
            else:
                predicted_disease = predicted_value(selected_symptoms)
                prediction_source = "ML Model"
            
            # Also get ML model prediction for comparison
            ml_predicted_disease = predicted_value(selected_symptoms)
            
            dis_des, precautions, medications, rec_diet, workout = information(predicted_disease)
            
            # Process precautions
            my_precautions = []
            if precautions and len(precautions) > 0:
                for i in precautions[0]:
                    if i and str(i).lower() != 'nan':  # Check if precaution is not None, empty, or 'nan'
                        my_precautions.append(i)
            
            # Process medications
            medications_list = []
            if medications and len(medications) > 0:
                try:
                    medication_list = ast.literal_eval(medications[0])
                    for item in medication_list:
                        medications_list.append(item)
                except (ValueError, SyntaxError):
                    # If ast.literal_eval fails, treat it as a simple string
                    medications_list = [medications[0]]
            
            # Process diet
            rec_diet_list = []
            if rec_diet and len(rec_diet) > 0:
                try:
                    diet_list = ast.literal_eval(rec_diet[0])
                    for item in diet_list:
                        rec_diet_list.append(item)
                except (ValueError, SyntaxError):
                    # If ast.literal_eval fails, treat it as a simple string
                    rec_diet_list = [rec_diet[0]]

            # Store user activity
            activity_collection.insert_one({
                'username': session['username'],
                'email': session['email'],
                'timestamp': datetime.now(),
                'symptoms_input': selected_symptoms,
                'predicted_disease': predicted_disease,
                'prediction_source': prediction_source,
                'ml_predicted_disease': ml_predicted_disease,
                'possible_diseases_from_csv': possible_diseases
            })

            primary_symptoms = get_primary_symptoms()
            return render_template('index.html', 
                                 symptoms=selected_symptoms, 
                                 predicted_disease=predicted_disease,
                                 prediction_source=prediction_source,
                                 ml_predicted_disease=ml_predicted_disease,
                                 possible_diseases=possible_diseases,
                                 dis_des=dis_des,
                                 my_precautions=my_precautions, 
                                 medications=medications_list, 
                                 my_diet=rec_diet_list,
                                 workout=workout, 
                                 primary_symptoms=primary_symptoms,
                                 selected_primary=primary_symptom,
                                 selected_secondary=secondary_symptom,
                                 selected_tertiary=tertiary_symptom,
                                 selected_quaternary=quaternary_symptom)
                                 
        except Exception as e:
            primary_symptoms = get_primary_symptoms()
            message = f"An error occurred during prediction: {str(e)}"
            return render_template('index.html', message=message, primary_symptoms=primary_symptoms)

# AJAX routes for cascading dropdowns
@app.route('/get_secondary_symptoms', methods=['POST'])
def get_secondary_symptoms_ajax():
    primary_symptom = request.json.get('primary_symptom')
    secondary_symptoms = get_secondary_symptoms(primary_symptom)
    return jsonify(secondary_symptoms)

@app.route('/get_tertiary_symptoms', methods=['POST'])
def get_tertiary_symptoms_ajax():
    primary_symptom = request.json.get('primary_symptom')
    secondary_symptom = request.json.get('secondary_symptom')
    tertiary_symptoms = get_tertiary_symptoms(primary_symptom, secondary_symptom)
    return jsonify(tertiary_symptoms)

@app.route('/get_quaternary_symptoms', methods=['POST'])
def get_quaternary_symptoms_ajax():
    primary_symptom = request.json.get('primary_symptom')
    secondary_symptom = request.json.get('secondary_symptom')
    tertiary_symptom = request.json.get('tertiary_symptom')
    quaternary_symptoms = get_quaternary_symptoms(primary_symptom, secondary_symptom, tertiary_symptom)
    return jsonify(quaternary_symptoms)

@app.route('/get_possible_diseases', methods=['POST'])
def get_possible_diseases_ajax():
    selected_symptoms = request.json.get('selected_symptoms', [])
    possible_diseases = get_possible_diseases(selected_symptoms)
    return jsonify(possible_diseases)

@app.route('/history')
def history():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    # Get user's prediction history
    user_history = activity_collection.find({'username': session['username']}).sort('timestamp', -1)
    
    return render_template('history.html', history=user_history)

@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        feedback_text = request.form['feedback']
        rating = request.form.get('rating', 0)
        
        feedback_collection.insert_one({
            'username': session['username'],
            'email': session['email'],
            'feedback': feedback_text,
            'rating': int(rating),
            'timestamp': datetime.now()
        })
        
        return render_template('feedback.html', message='Thank you for your feedback!')
    
    return render_template('feedback.html')

@app.route('/download', methods=['POST'])
def download_results():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    # Get form data
    predicted_disease = request.form.get('predicted_disease', '')
    prediction_source = request.form.get('prediction_source', '')
    dis_des = request.form.get('dis_des', '')
    my_precautions = request.form.getlist('my_precautions')
    medications = request.form.getlist('medications')
    my_diet = request.form.getlist('my_diet')
    workout = request.form.getlist('workout')
    
    # Create a text report
    report = f"""
MEDICAL PREDICTION REPORT
========================

User: {session['username']}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PREDICTED DISEASE: {predicted_disease}
PREDICTION SOURCE: {prediction_source}

DESCRIPTION:
{dis_des}

PRECAUTIONS:
"""
    for i, precaution in enumerate(my_precautions, 1):
        report += f"{i}. {precaution}\n"
    
    report += "\nMEDICATIONS:\n"
    for i, medication in enumerate(medications, 1):
        report += f"{i}. {medication}\n"
    
    report += "\nDIET RECOMMENDATIONS:\n"
    for i, diet in enumerate(my_diet, 1):
        report += f"{i}. {diet}\n"
    
    report += "\nWORKOUT RECOMMENDATIONS:\n"
    for i, work in enumerate(workout, 1):
        report += f"{i}. {work}\n"
    
    report += """
DISCLAIMER:
This prediction is generated by an AI system and should not be considered as professional medical advice. 
Please consult with a qualified healthcare professional for proper diagnosis and treatment.
"""
    
    # Create a BytesIO object
    output = BytesIO()
    output.write(report.encode('utf-8'))
    output.seek(0)
    
    return send_file(
        output,
        as_attachment=True,
        download_name=f'medical_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt',
        mimetype='text/plain'
    )

if __name__ == '__main__':
    app.run(debug=True)