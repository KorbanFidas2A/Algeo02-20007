from flask import Flask
import time


app = Flask(__name__, static_folder="../build", static_url_path='/')

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/time')
def get_current_time():
    return {'time' : time.time()}
    #jasonified