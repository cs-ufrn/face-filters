#!/usr/bin/env python
import os
import shutil
from flask import Flask, render_template, request, \
    Response, send_file, redirect, url_for
from camera import Camera
from send_email import Email

app = Flask(__name__)
camera = None
mail_server = None
mail_conf = "static/mail_conf.json"

# Description : This function is going to create an object camera, in case if not exist, otherwise is going to
# return the object camera. 
# Synopsis : get_camera().
# parameters : none.
# return value : It returns an object called camera.

def get_camera():
    global camera
    if not camera:
        camera = Camera()

    return camera

# Description : This function is going to create an object mail_server, in case if not exist, otherwise is going to
# return an object mail_server. 
# Synopsis : get_mail_server().
# parameters : none.
# return value : It returns an object called mail_server.

def get_mail_server():
    global mail_server
    if not mail_server:
        mail_server = Email(mail_conf)

    return mail_server

# Description : This function redirects to index.
# Synopsis : root().
# parameters : none.
# return value : returns a function to redirect to index.

@app.route('/')
def root():
    return redirect(url_for('index', mode='landmark'))

# Description : This function render the template.
# Synopsis : index(mode).
# parameters : mode, the selectede mode. Here mode refers to the sprite.
# return value : returns a function that renders the template and wanted variables.

@app.route('/index/<mode>')
def index(mode):
    return render_template('index.html', mode=mode)

# Description : This function takes a video capture, frame, with the mode.
# Synopsis : gen(camera, mode).
# parameters : mode, the selectede mode. Here mode refers to the sprite. And camera.
# return value : none.

def gen(camera, mode):
    while True:
        frame = camera.get_frame(mode)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Description : This function takes a video capture, frame, with the mode.
# Synopsis : video_feed(mode).
# parameters : mode, the selectede mode. Here mode refers to the sprite. And camera.
# return value : none.
# Eu não entendi a função

@app.route('/video_feed/<mode>')
def video_feed(mode):
    camera = get_camera()
    return Response(gen(camera, mode),
        mimetype='multipart/x-mixed-replace; boundary=frame')

# Description : This function redirects to show_capture, the captured image.
# Synopsis : capture(mode).
# parameters : mode, the selected mode.
# return value : returns a function to redirect to show_capture.

@app.route('/capture/<mode>')
def capture(mode):
    camera = get_camera()
    stamp = camera.capture(mode)
    return redirect(url_for('show_capture', timestamp=stamp))

#Não entendi essas duas ultimas

def stamp_file(timestamp):
    return 'captures/' + timestamp +".jpg"

@app.route('/capture/image/<timestamp>', methods=['POST', 'GET'])
def show_capture(timestamp):
    path = stamp_file(timestamp)

    email_msg = None
    if request.method == 'POST':
        if request.form.get('email'):
            email = get_mail_server()
            email_msg = email.send_email('static/{}'.format(path), 
                request.form['email'])
        else:
            email_msg = "Email field empty!"

    return render_template('capture.html',
        stamp=timestamp, path=path, email_msg=email_msg)

# Main function of web.py #

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)