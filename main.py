from testing import *
from flask import Flask, jsonify, render_template,request,send_file,Response
from flask_cors import CORS
import os

model = load_models()

app = Flask(__name__)
CORS(app)
@app.route('/',methods=['GET'])
def hello_world():
    return jsonify({'test':'berhasil'})


@app.route('/predict',methods=['POST'])
def coba():
    imagefile = request.files['']
    img_path = 'images/' + imagefile.filename
    imagefile.save(img_path)
    result, peluang = main(img_path, model)
    hasil = report(result)
    peluang = report(peluang)
    return {"Kadar Air": hasil,"peluang": peluang}

if __name__ == '__main__':
    app.run()