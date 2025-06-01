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
# app.secret_key = "d7f3b6017b939599d88314e418b1ccbc"  # Important for session management
app.secret_key=os.urandom(24).hex()
# MongoDB setup
client = pymongo.MongoClient("mongodb://localhost:27017")  # Replace with your MongoDB connection string
db = client["medicine_recommendation"]
users_collection = db["users"]
feedback_collection = db["feedback"]
activity_collection = db["user_activity"] # New collection for user activity

# Loading the datasets from Kaggle website
sym_des = pd.read_csv("kaggle_dataset/symptoms_df.csv")
precautions = pd.read_csv("kaggle_dataset/precautions_df.csv")
workout = pd.read_csv("kaggle_dataset/workout_df.csv")
description = pd.read_csv("kaggle_dataset/description.csv")
medications = pd.read_csv('kaggle_dataset/medications.csv')
diets = pd.read_csv("kaggle_dataset/diets.csv")

Rf = pickle.load(open('model/RandomForest.pkl', 'rb'))

# Here we make a dictionary of symptoms and diseases and preprocess it
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
                 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum':107,
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


# Here we created a function (information) to extract information from all the datasets
def information(predicted_dis):
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

# This is the function that passes the user input symptoms to our Model
def predicted_value(patient_symptoms):
    i_vector = np.zeros(len(symptoms_list_processed))
    for i in patient_symptoms:
        i_vector[symptoms_list_processed[i]] = 1
    return diseases_list[Rf.predict([i_vector])[0]]

# Function to correct the spellings of the symptom (if any)
def correct_spelling(symptom):
    closest_match, score = process.extractOne(symptom, symptoms_list_processed.keys())
    # If the similarity score is above a certain threshold, consider it a match
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
        user = users_collection.find_one({'email': email, 'password': password})
        if user:
            session['email'] = email
            return redirect(url_for('predict')) # Corrected to predict
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

        # Hash the password for security
        hashed_password = generate_password_hash(password)

        # Insert new user into MongoDB
        users_collection.insert_one({
            "username": username,
            "email": email,
            "password": password,
            "created_at": datetime.now()
        })

        session["username"] = username  # log in user
        return redirect(url_for("login"))  # You may change this to your home/dashboard route

    return render_template("signup.html", message=None)

@app.route('/predict', methods=['GET', 'POST'])
def predict(): #Corrected name to predict
    #if 'username' not in session:
    #  return redirect(url_for('login'))

    all_symptoms = sorted(list(symptoms_list.keys())) # Get the list of symptoms for the dropdown

    if request.method == 'POST':
        selected_symptoms = request.form.getlist('symptoms') # Get a list of selected symptoms
        if not selected_symptoms:
            message = "Please select at least one symptom."
            return render_template('index.html', message=message, all_symptoms=all_symptoms)
        else:
            predicted_disease = predicted_value(selected_symptoms)
            dis_des, precautions, medications, rec_diet, workout = information(predicted_disease)
            my_precautions = []
            for i in precautions[0]:
                my_precautions.append(i)
            medication_list = ast.literal_eval(medications[0])
            medications_list = []
            for item in medication_list:
                medications_list.append(item)
            diet_list = ast.literal_eval(rec_diet[0])
            rec_diet_list = []
            for item in diet_list:
                rec_diet_list.append(item)

            # Store user activity
            activity_collection.insert_one({
                'username': session['username'],
                'timestamp': datetime.now(tz=None),
                'symptoms_input': selected_symptoms,
                'corrected_symptoms': selected_symptoms, # No correction needed with dropdown
                'predicted_disease': predicted_disease
            })

            return render_template('index.html', symptoms=selected_symptoms, predicted_disease=predicted_disease, dis_des=dis_des,
                                   my_precautions=my_precautions, medications=medications_list, my_diet=rec_diet_list,
                                   workout=workout, all_symptoms=all_symptoms)
    return render_template('index.html', all_symptoms=all_symptoms)
@app.route('/history')
def history():
    if 'username' not in session:
        return redirect(url_for('login'))
    user_history = activity_collection.find({'username': session['username']}).sort('timestamp', pymongo.DESCENDING)
    history_list = list(user_history)
    return render_template('history.html', history=history_list)
@app.route('/delete_history/<string:item_id>', methods=['POST'])
def delete_history(item_id):
    if 'username' not in session:
        return redirect(url_for('login'))
    try:
        # Convert the item_id string to an ObjectId
        object_id = ObjectId(item_id)
        # Delete the activity from the database
        result = activity_collection.delete_one({'_id': object_id, 'username': session['username']})
        if result.deleted_count > 0:
            # Optionally, you can log the successful deletion
            print(f"Deleted activity with ID: {item_id} for user: {session['username']}")
        else:
            # Handle the case where the activity was not found (optional)
            print(f"Activity with ID: {item_id} not found for user: {session['username']}")
    except Exception as e:
        # Handle any exceptions, such as invalid ObjectId format or database errors
        print(f"Error deleting activity with ID: {item_id}. Error: {e}")
        # You might want to show an error message to the user
        # For simplicity, we'll just pass a message to the history page
        return render_template('history.html', message="Failed to delete activity.")
    # Redirect back to the history page after deletion
    return redirect(url_for('history'))


@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    if 'username' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        feedback_text = request.form['feedback']
        feedback_collection.insert_one({'username': session['username'], 'timestamp': datetime.now(tz=None), 'feedback': feedback_text})
        return render_template('feedback.html', message='Feedback submitted successfully')
    return render_template('feedback.html', message=None)

@app.route('/download', methods=['POST'])
def download():
    predicted_disease = request.form['predicted_disease']
    dis_des = request.form['dis_des']
    my_precautions = request.form.getlist('my_precautions')
    medications = request.form.getlist('medications')
    my_diet = request.form.getlist('my_diet')
    workout = request.form.getlist('workout')

    # Store download activity
    activity_collection.insert_one({
        'username': session['username'],
        'timestamp': datetime.now(tz=None),
        'activity_type': 'download_results',
        'predicted_disease': predicted_disease
    })

    output = BytesIO()
    output.write(f"Predicted Disease: {predicted_disease}\n\n".encode('utf-8'))
    output.write(f"Description: {dis_des}\n\n".encode('utf-8'))
    output.write("Precautions:\n".encode('utf-8'))
    for precaution in my_precautions:
        output.write(f"- {precaution}\n".encode('utf-8'))
    output.write("\nMedications:\n".encode('utf-8'))
    for medication in medications:
        output.write(f"- {medication}\n".encode('utf-8'))
    output.write("\nDiet:\n".encode('utf-8'))
    for diet in my_diet:
        output.write(f"- {diet}\n".encode('utf-8'))
    output.write("\nWorkout:\n".encode('utf-8'))
    for work in workout:
        output.write(f"- {work}\n".encode('utf-8'))

    output.seek(0)
    return send_file(output, as_attachment=True, download_name='result.txt', mimetype='text/plain')

@app.route('/logout')
def logout():
    # Store logout activity
    if 'username' in session:
        activity_collection.insert_one({
            'username': session['username'],
            'timestamp': datetime.now(tz=None),
            'activity_type': 'logout'
        })
        session.pop('username', None)
    return redirect(url_for('home_page'))

if __name__ == '__main__':
    app.run(debug=True)