from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
from werkzeug.utils import secure_filename
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Initialize the Flask app
app = Flask(__name__)

# Folder to store uploaded images
UPLOAD_FOLDER = 'static/uploads/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the saved Keras model
model = tf.keras.models.load_model('models/braintumor.keras')

# Define labels for classification
labels = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']

# Route for the homepage
@app.route('/')
def home():
    return render_template('index.html', prediction=None, image_path=None)

# Route for the resources page
@app.route('/resources')
def resources():
    return render_template('resourses.html')

# Route for the chatbot page
@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    name = request.form.get('name')
    age = request.form.get('age')
    gender = request.form.get('gender')
    symptoms = request.form.getlist('symptoms')

    if not symptoms:
        return render_template('index.html', prediction="Please select at least one symptom", image_path=None)

    if 'file' not in request.files:
        return render_template('index.html', prediction="No file uploaded", image_path=None)

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', prediction="No file selected", image_path=None)

    if file:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # Preprocess the uploaded image
            img = image.load_img(filepath, target_size=(150, 150))  # Adjust size if necessary
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)

            # Make a prediction
            predictions = model.predict(img_array)
            index = np.argmax(predictions[0])
            predicted_label = labels[index]

            # Return the result
            return render_template(
                'result.html',
                name=name,
                age=age,
                gender=gender,
                symptoms=', '.join(symptoms),
                prediction=predicted_label,
                image_path=filepath
            )
        except Exception as e:
            print(f"Error: {e}")
            return render_template('index.html', prediction="Error processing image", image_path=None)

# Route to generate and download the report
@app.route('/generate_report', methods=['POST'])
def generate_report():
    name = request.form['name']
    age = request.form['age']
    gender = request.form['gender']
    diagnosis = request.form['diagnosis']  # You should include 'diagnosis' in your form
    symptoms = request.form['symptoms']  # Symptoms should be passed correctly (a string)
    prediction = request.form['prediction']
    image_path = request.form['image_path']

    # Generate the report content
    report_content = f"""
    Name: {name}
    Age: {age}
    Gender: {gender}
    Previous Diagnosis: {diagnosis}
    Symptoms: {symptoms}
    Prediction: {prediction}
    Image Path: {image_path}
    """

    # Save the report to a text file
    report_filename = f"report_{name}_{age}.txt"
    report_filepath = os.path.join(app.config['UPLOAD_FOLDER'], report_filename)
    with open(report_filepath, 'w') as file:
        file.write(report_content)

    # Provide the link to download the report
    return redirect(url_for('download_report', filename=report_filename))

# Route to download the generated report
@app.route('/download_report/<filename>')
def download_report(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)

