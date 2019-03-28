import cv2 as cv
import dlib
from time import localtime, strftime
from track import draw_face, get_feature_boundbox, \
    get_inclination, apply_sprite, is_mouth_open, drawing_frame

class Camera(object):
    CAPTURES_DIR = "static/captures/"
    LANDMARKS_PATH = "shape_predictor_68_face_landmarks.dat"
    EYE_SPRITES = {'googly':('googly_left.png', 'googly_right.png'),
        'hearts':('heart.png', 'heart.png'), 
        'CS':('CS.png', 'CS.png')}
    EYE_SPRITES['crazy'] = EYE_SPRITES['googly']

    RESIZE_RATIO = 1.0

    def __init__(self):
        self.video = cv.VideoCapture(0)

        # load face detector
        self.face_detector = dlib.get_frontal_face_detector()

        # load facemark detector
        self.facemark_detector = dlib.shape_predictor(Camera.LANDMARKS_PATH)
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self, mode):
        success, frame = self.video.read()
        if not success:
            return

        if (Camera.RESIZE_RATIO != 1):
            frame = cv.resize(frame, None, fx=Camera.RESIZE_RATIO, \
                fy=Camera.RESIZE_RATIO)
    
        if mode == "drawing":
            frame = drawing_frame(frame)
        else:
            frame = self.facemark(frame, mode)
        
        ret, jpeg = cv.imencode('.jpg', frame)
        return jpeg.tobytes()

    def capture(self, mode):
        _, frame = self.video.read()
        if (Camera.RESIZE_RATIO != 1):
            frame = cv.resize(frame, None, fx=Camera.RESIZE_RATIO, \
                fy=Camera.RESIZE_RATIO)
        frame = self.facemark(frame, mode)
        _, jpeg = cv.imencode('.jpg', frame)
        timestamp = strftime("%d-%m-%Y-%Hh%Mm%Ss", localtime())
        filename = Camera.CAPTURES_DIR + timestamp +".jpg"
        if not cv.imwrite(filename, frame):
            raise RuntimeError("Unable to capture image "+timestamp)
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
            if mode in Camera.EYE_SPRITES.keys():
                leye = get_feature_boundbox(shape, 'leyebrow')
                reye = get_feature_boundbox(shape, 'reyebrow')

                apply_sprite(frame, "sprites/"+Camera.EYE_SPRITES[mode][0], leye, inclination)
                apply_sprite(frame, "sprites/"+Camera.EYE_SPRITES[mode][1], reye, inclination)
            if mode in ('rainbow', 'crazy') and is_mouth_open(shape, threshold=20):
                mouth = get_feature_boundbox(shape, 'lips')
                apply_sprite(frame, "sprites/rainbow.png", mouth, inclination)
            if mode in ('clown', 'crazy'):
                nose = get_feature_boundbox(shape, 'nose')
                apply_sprite(frame, "sprites/clown_nose.png", nose, inclination)
            if mode in ('anon_mask',"incredibles_mask","who_mask"):
                if mode == "anon_mask":
                    appearance = get_feature_boundbox(shape, 'face')
                    apply_sprite(frame, "sprites/anon_mask.png", appearance, inclination)
                if mode == "incredibles_mask":
                    appearance = get_feature_boundbox(shape, 'face')
                    apply_sprite(frame, "sprites/incredibles_mask.png", appearance, inclination)
                if mode == "who_mask":
                    appearance = get_feature_boundbox(shape, 'face')
                    apply_sprite(frame, "sprites/who_mask.png", appearance, inclination)

        return frame

