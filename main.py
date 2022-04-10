import sys
import os

py_file_location = "./"
sys.path.append(os.path.abspath(py_file_location))

templates_path = './templates'
static_path = './static'


from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from object_detection import ObjectDetection as OBJ_DET
from flask_ngrok import run_with_ngrok
import os

UPLOAD_FOLDER = './static/uploads/'

app = Flask(__name__, template_folder=templates_path)
run_with_ngrok(app)
from flask import Flask
app.config['SESSION_TYPE'] = 'memcached'
app.config['SECRET_KEY'] = 'super secret key'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
obj_det = OBJ_DET()

@app.route('/', methods=["GET"])
def index():
    return render_template('index.html', detected_objects=obj_det.read_objects())

@app.route('/', methods=["POST"])
def upload():
    if request.method == 'POST':
        file = request.files['videoInput']
        filename = secure_filename(file.filename)
        file_path =os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        obj_det.video_to_frames(file_path)
        obj_det.detect()
        return redirect('/')

@app.route('/search', methods=['GET', 'POST'])
def search_object():
    search_text = request.form.get("searchInput")
    search_results = obj_det.search_objects(search_text)
    return render_template('search.html', object_frames=search_results, is_list=type(search_results)==list, search_txt=search_text)

@app.route('/search/<text>',  methods=['GET'])
def search_by_id(text):
    search_results = obj_det.search_objects(text)
    return render_template('search.html', object_frames=search_results, is_list=type(search_results) == list, search_txt=text)


app.run()
