import cv2 as cv
import dlib
import numpy as np
import math
from itertools import chain

def draw_face(frame, shape):
    rect = shape.rect
    p1 = (rect.left(), rect.top())
    p2 = (rect.right(), rect.bottom())
    cv.rectangle(frame, p1, p2, (0,255,0))
    for point in shape.parts():
        cv.circle(frame, (point.x, point.y), 2, (0, 0, 255), cv.FILLED)

def make_boundbox(points):
    x = min(p.x for p in points)
    y = min(p.y for p in points)
    w = max(p.x for p in points) - x
    h = max(p.y for p in points) - y
    return (x, y, w, h)

def get_feature_boundbox(shape, face_part):
    part_points = {'leyebrow':[(17,22)], 'reyebrow':[(22,27)],
        'leye':[(36,42)], 'reye':[(42,48)], 'nose':[(29,36)],
        'lips':[(48,68)], 'mouth':[(56,59), (61,64)]}

    part_points['face'] = [(0, 17), *part_points['leyebrow'],
        *part_points['reyebrow']]

    if face_part in part_points:
        all_points = shape.parts()
        points = chain.from_iterable(
            all_points[p1:p2] for p1,p2 in part_points[face_part]
        )
        return make_boundbox(list(points))
    else:
        raise InvalidArgument("No face part named "+str(face_part))

def get_inclination(shape):
    points = shape.parts()
    p1, p2 = points[17], points[26]
    slope = float(p2.y - p1.y) / (p2.x - p1.x)
    return 180 / math.pi*math.atan(slope)

def is_mouth_open(shape, threshold=10):
    points = shape.parts()
    top = points[62]
    bottom = points[66]
    return bottom.y - top.y >= threshold

def rotate_image(image, angle):
    height, width = image.shape[0], image.shape[1]
    cx, cy = width / 2, height / 2

    # clockwise rotation matrix
    rotation = cv.getRotationMatrix2D((cx, cy), -angle, 1.0)
    cos = np.abs(rotation[0, 0])
    sin = np.abs(rotation[0, 1])

    # new bounds
    width_new = int((height * sin) + (width * cos))
    height_new = int((height * cos) + (width * sin))

    # adjust matrix for translation
    rotation[0, 2] += (width_new / 2) - cx
    rotation[1, 2] += (height_new / 2) - cy

    return cv.warpAffine(image, rotation, (width_new, height_new))

def apply_sprite(frame, sprite, boundbox, angle):
    x, y, w, h = boundbox
    sprite = cv.imread(sprite, -1) # -1 for alpha
    sprite = rotate_image(sprite, angle)
    sprite, sprite_y = adjust_sprite(sprite, w, y)
    image = draw_sprite(frame, sprite, x, sprite_y)

def adjust_sprite(sprite, head_width, head_y):
    sprite_height, sprite_width = sprite.shape[0], sprite.shape[1]
    factor = float(head_width/sprite_width)

    # make sprite same width as head
    sprite = cv.resize(sprite, (0,0), fx=factor, fy=factor)
    sprite_height,sprite_width = sprite.shape[0], sprite.shape[1]

    sprite_y = head_y
    return (sprite, sprite_y)

def drawing_frame(frame):
    vertical = cv.Sobel(frame, cv.CV_64F, 1,0, ksize=1)
    horizontal = cv.Sobel(frame, cv.CV_64F,0,1,ksize=1)
    return cv.sqrt(cv.pow(horizontal,2) + cv.pow(vertical,2))

def __shape_to_np__(shape, dtype="int"):
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)
    
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    return coords

def pixelate(frame, face, shape, block_size):
    left = face.left()
    top = face.top()
    right = face.right()
    bottom = face.bottom()

    frame_result = frame.copy()

    for row in range(top, bottom, block_size):
        for col in range(left,right, block_size):

            max_col = min(col+block_size, right)
            max_row = min(row+block_size, bottom)

            frame_result[ row:max_row,col:max_col,0 ] = np.mean( frame[ row:max_row,col:max_col,0 ] )
            frame_result[ row:max_row,col:max_col,1 ] = np.mean( frame[ row:max_row,col:max_col,1 ] )
            frame_result[ row:max_row,col:max_col,2 ] = np.mean( frame[ row:max_row,col:max_col,2 ] )

    points = __shape_to_np__(shape)
    hull = cv.convexHull(points, False)

    mask = np.zeros((frame.shape[0],frame.shape[1]), dtype='uint8')
    cv.drawContours(mask, [hull], 0, (255,255,255), -1)
    
    result_mask = cv.bitwise_and(frame_result, frame_result, mask = mask)
    not_mask = cv.bitwise_not(mask)
    not_frame = cv.bitwise_and(frame, frame, mask = not_mask)

    return cv.bitwise_or(result_mask, not_frame)

def draw_sprite(frame, sprite, x_offset, y_offset):
    sprite_h, sprite_w = sprite.shape[0], sprite.shape[1]
    img_h, img_w = frame.shape[0], frame.shape[1]

    if y_offset < 0:
        sprite = sprite[np.abs(y_offset)::,:,:]
        sprite_h = sprite.shape[0]
        y_offset = 0

    if x_offset < 0:
        sprite = sprite[:,abs(x_offset)::,:]
        sprite_w = sprite.shape[1]
        x_offset = 0

    if y_offset+sprite_h >= img_h:
        sprite = sprite[:img_h-y_offset,:,:]

    if x_offset+sprite_w >= img_w:
        sprite = sprite[:,0:img_w-x_offset,:]

    right = x_offset+sprite_w
    bottom = y_offset+sprite_h
    sprite_alpha = sprite[:,:,3]/255.0
    rgb_channels = 3

    for c in range(rgb_channels):
        frame[y_offset:bottom, x_offset:right, c] = \
            sprite[:,:,c] * sprite_alpha + \
            frame[y_offset:bottom, x_offset:right, c] * (1.0 - sprite_alpha)
    return frame

def apply_blur(frame, shape):
   rect = shape.rect
   x, y = rect.left(), rect.top()
   w, h = rect.width(), rect.height()
   frame[y:y+h, x:x+w] = cv.blur(frame[y:y+h, x:x+w], (50, 50), 70)

   return frame

if __name__ == "__main__":
    # load face detector
    face_detector = dlib.get_frontal_face_detector()

    # load facemark detector
    landmarks_path = "./shape_predictor_68_face_landmarks.dat"
    facemark = dlib.shape_predictor(landmarks_path)

    try:
        video = cv.VideoCapture(0)
        window_name = "Facil landmark detection"
        cv.namedWindow(window_name, cv.WINDOW_NORMAL)

        while True:
            success, frame = video.read()
            if not success:
                break

            frame = cv.resize(frame, None, fx=0.8, fy=0.8)
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            faces = face_detector(gray, upsample_num_times=0)
            for face in faces:
                shape = facemark(gray, face)
                draw_face(frame, shape)
                inclination = get_inclination(shape)

                if is_mouth_open(shape, threshold=20):
                    mouth = get_feature_boundbox(shape, 'lips')
                    apply_sprite(frame, "sprites/rainbow.png", mouth, inclination)

                leye = get_feature_boundbox(shape, 'leyebrow')
                reye = get_feature_boundbox(shape, 'reyebrow')
                apply_sprite(frame, "sprites/googly_left.png", leye, inclination)
                apply_sprite(frame, "sprites/googly_right.png", reye, inclination)

                nose = get_feature_boundbox(shape, 'nose')
                apply_sprite(frame, "sprites/clown_nose.png", nose, inclination)
                
            cv.imshow(window_name, frame)
            key = cv.waitKey(1)

            if key == ord("q"):
                break
    finally:
        video.release()