from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
import os
import numpy as np

app = Flask(__name__)
model = load_model("covid19_model_adv.h5")

UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Get form data
        first_name = request.form['firstName']
        last_name = request.form['lastName']
        email = request.form['email']
        age = request.form['age']
        gender = request.form['gender']
        
        # Save the uploaded image
        try:
            x_ray_image = request.files['xRayImage']
            x_ray_image_path = os.path.join(UPLOAD_FOLDER, secure_filename(x_ray_image.filename))
            x_ray_image.save(x_ray_image_path)
        except Exception as e:
            return jsonify({'error': str(e)})
        
        # Process the image with the model
        try:
            img = image.load_img(x_ray_image_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            p = model.predict(x)
            prediction = "Covid" if p[0][0] < 0.5 else "Normal"
        except Exception as e:
            return jsonify({'error': str(e)})
        
        # Delete the uploaded image after processing
        try:
            os.remove(x_ray_image_path)
        except Exception as e:
            return jsonify({'error': str(e)})
        
        # Return prediction and other information
        return render_template('result.html', 
                               firstName=first_name,
                               lastName=last_name,
                               email=email,
                               age=age,
                               gender=gender,
                               prediction=prediction)
    return render_template('login.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/action')
def action():
    return render_template('action.html')

@app.route('/news')
def news():
    return render_template('news.html')

@app.route('/doctores')
def doctores():
    return render_template('doctores.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == "__main__":
    app.run(debug=True)
