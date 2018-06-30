import cv2 as cv
import dlib
from time import localtime, strftime
from track import draw_face, get_feature_boundbox, \
    get_inclination, apply_sprite, is_mouth_open

class Camera(object):
    def __init__(self):
        self.video = cv.VideoCapture(0)

        # load face detector
        self.face_detector = dlib.get_frontal_face_detector()

        # load facemark detector
        landmarks_path = "shape_predictor_68_face_landmarks.dat"
        self.facemark_detector = dlib.shape_predictor(landmarks_path)
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self, mode):
        success, frame = self.video.read()
        if not success:
            return

        frame = cv.resize(frame, None, fx=0.8, fy=0.8)
        self.facemark(frame, mode)

        ret, jpeg = cv.imencode('.jpg', frame)
        return jpeg.tobytes()

    def capture(self, mode):
        _, frame = self.video.read()
        frame = cv.resize(frame, None, fx=0.8, fy=0.8)
        self.facemark(frame, mode)
        _, jpeg = cv.imencode('.jpg', frame)
        timestamp = strftime("%d-%m-%Y-%Hh%Mm%Ss", localtime())
        filename = "static/captures/"+ timestamp +".jpg"
        cv.imwrite(filename, frame)
        return timestamp

    def facemark(self, frame, mode):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        faces = self.face_detector(gray, upsample_num_times=0)
        for face in faces:
            shape = self.facemark_detector(gray, face)

            if mode != 'landmark':
                inclination = get_inclination(shape)

            if mode == 'landmark':
                draw_face(frame, shape)
            if mode in ('googly', 'hearts', 'CS', 'crazy'):
                leye = get_feature_boundbox(shape, 'leyebrow')
                reye = get_feature_boundbox(shape, 'reyebrow')

                eye_sprites = {'googly':('googly_left.png', 'googly_right.png'),
                    'hearts':('heart.png', 'heart.png'), 
                    'CS':('CS.png', 'CS.png')}
                eye_sprites['crazy'] = eye_sprites['googly']
                apply_sprite(frame, "sprites/"+eye_sprites[mode][0], leye, inclination)
                apply_sprite(frame, "sprites/"+eye_sprites[mode][1], reye, inclination)
            if mode in ('rainbow', 'crazy') and is_mouth_open(shape, threshold=20):
                mouth = get_feature_boundbox(shape, 'lips')
                apply_sprite(frame, "sprites/rainbow.png", mouth, inclination)
            if mode in ('clown', 'crazy'):
                nose = get_feature_boundbox(shape, 'nose')
                apply_sprite(frame, "sprites/clown_nose.png", nose, inclination)

