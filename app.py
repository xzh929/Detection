from flask import Flask,render_template,url_for,redirect,request
from forms import UploadForm
from flask_wtf.csrf import validate_csrf
import os

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'secret string')

@app.route('/', methods=['GET', 'POST'])
def index():
    form = UploadForm()
    return render_template('index.html', form=form)


if __name__ == '__main__':
    app.run()