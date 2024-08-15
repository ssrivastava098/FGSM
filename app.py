from flask import Flask, render_template, request, url_for, send_from_directory,redirect
import os
from werkzeug.utils import secure_filename
from utils import createAdversarialExample

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['IMAGE_FOLDER'] = 'images/'

ALLOWED_EXTENSIONS = {'jpeg'}
def allowed_file(filename):
    return ('.' in filename) and (filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS)

check = False

@app.route('/')
def index():
    global check
    return render_template('index.html', check=check)

@app.route('/upload',methods=['POST'])
def upload():
    global check
    if 'image' not in request.files:
        return redirect(url_for('index'))
    file = request.files['image']
    if file.filename == '':
        return redirect(url_for('index'))
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_image.jpeg')
        file.save(filepath)
        createAdversarialExample(filepath)
        check = True
    return redirect(url_for('index'))

@app.route('/images/<filename>')
def serve_image(filename):
    return send_from_directory(app.config['IMAGE_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)

