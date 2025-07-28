Face Mask Classification
A deep learning project for detecting whether people in images are wearing face masks or not. This project includes a Flask web app, model training script, and a unified launcher for easy use.

Features
Upload images and detect mask/no-mask via web interface
Train your own model using images in the uploads/ folder
Clean, ready-to-deploy Python code
Requirements file for easy setup
Project Structure
Face Mask Classification/
├── app.py                 # Flask web application
├── main.py                # Unified launcher script
├── train_model.py         # Model training script
├── requirements.txt       # Python dependencies
├── model.h5               # Trained Keras model (binary mask/no-mask)
├── static/                # Static files (CSS, JS)
├── templates/             # HTML templates
├── uploads/               # Place your training images here
Getting Started
1. Install requirements
pip install -r requirements.txt
2. Train the model (optional)
Add your mask/no-mask images to the uploads/ folder, then run:

python train_model.py
This will create or update 
model.h5
.

3. Run the web app
python app.py
Visit http://127.0.0.1:5000 in your browser.

4. Use the unified launcher
python main.py
Choose from the menu to train, run the app, etc.

Notes
The model is a simple CNN trained on images in uploads/. For better accuracy, use a larger and more diverse dataset.
The web app expects 
model.h5
 to exist in the project root.
This repo has been cleaned for clarity and ease of use.
