import cv2 as cv
import dlib
import numpy as np
import math
from itertools import chain


# Description : The function draw_face draws the predicted face on the frame, the image captured.
# Synopsis : draw_face(fram, shape)
# parameters : frame, an image capcture of the video. And shape, the predicted face.
# return value : none.

def draw_face(frame, shape):
    rect = shape.rect
    p1 = (rect.left(), rect.top())
    p2 = (rect.right(), rect.bottom())
    cv.rectangle(frame, p1, p2, (0,255,0))
    for point in shape.parts():
        cv.circle(frame, (point.x, point.y), 2, (0, 0, 255), cv.FILLED)

# Description : 
# Synopsis : _make_boundbox(points)
# parameters : points, an list of point
# return value : x. y. w. h.

def _make_boundbox(points):
    x = min(p.x for p in points)
    y = min(p.y for p in points)
    w = max(p.x for p in points) - x
    h = max(p.y for p in points) - y
    return (x, y, w, h)

# Description : If any value in face_parts exists in any points of our face, the function is goint to take every
# point from shape to create a boundbox.
# Synopsis : get_feature_boundbox(shape, face_part)
# parameters : shape, the predicted face. And face_part
# return value : It returns a function _make_boundbox with a list of points. Or in case of an error, it returns
# a error message.
    

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
        return _make_boundbox(list(points))
    else:
        raise InvalidArgument("No face part named "+str(face_part))

# Description : The function calculates the predicted face inclination angle.
# Synopsis : get_inclination(shape)
# parameters : shape, the predicted face.
# return value : returns an angle. The predicted face's angle inclination.
        
def get_inclination(shape):
    points = shape.parts()
    p1, p2 = points[17], points[26]
    slope = float(p2.y - p1.y) / (p2.x - p1.x)
    return 180 / math.pi*math.atan(slope)

# Description : The function indentifies if the predicted face, shape, has a open mounth.
# Synopsis : get_inclination(shape, thereshold = 20)
# parameters : shape, the predicted face.
# return value : it returns a boolen value. True or False depeding if the mouth is open or not.

def is_mouth_open(shape, threshold=10):
    points = shape.parts()
    top = points[62]
    bottom = points[66]
    return bottom.y - top.y >= threshold

# Description : The function recives a image, and the rotates it. And after rotates the image, the function
# will perform a transformation to get a better result.
# Synopsis : _rotate_image(image, angle)
# parameters : image, an image. And angle, the wanted angle.
# return value : It returns a function that is goint do perform an Affine transformation to an image.

def _rotate_image(image, angle):
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

# Description : The function apply the sprite to the frame, video capture. 
# Synopsis : apply_sprite(frame, sprite, boundbox, angle)
# parameters : frame, is the video capture. sprite, is the image wanted to put the frame. boundbox, . 
# angle, the inclination angle of our head.
# return value : none.

def apply_sprite(frame, sprite, boundbox, angle):
    x, y, w, h = boundbox
    sprite = cv.imread(sprite, -1) # -1 for alpha
    sprite = _rotate_image(sprite, angle)
    sprite, sprite_y = _adjust_sprite(sprite, w, y)
    image = draw_sprite(frame, sprite, x, sprite_y)

# Description : The function adjusts the sprite image to be proportional to the head.
# Synopsis : _adjust_sprite(sprite, head_width, head_y)
# parameters : sprite, the wanted image. head_width, the head's width. And Y from boundboox.
# return value : returns the resized sprite and its Y.

def _adjust_sprite(sprite, head_width, head_y):
    sprite_height, sprite_width = sprite.shape[0], sprite.shape[1]
    factor = float(head_width/sprite_width)

    # make sprite same width as head
    sprite = cv.resize(sprite, (0,0), fx=factor, fy=factor)
    sprite_height,sprite_width = sprite.shape[0], sprite.shape[1]

    sprite_y = head_y
    return (sprite, sprite_y)

# Description : The function calculates and take the horizontal and vertical line from the image and then
# calculates the gradient G = |Gy| + |Gx|.
# Synopsis : drawing_frame(frame)
# parameters : frame, the captured video.
# return value : returns the gradient, square root of horizontal**2 + vertical**2. 

def drawing_frame(frame):
    vertical = cv.Sobel(frame, cv.CV_64F, 1,0, ksize=1)
    horizontal = cv.Sobel(frame, cv.CV_64F,0,1,ksize=1)
    return cv.sqrt(cv.pow(horizontal,2) + cv.pow(vertical,2))

# Description : The function take the predicted face, shape, and then put in a new matrix of zeros. 
# Synopsis : __shape_to_np__(shape, dtype="int")
# parameters : shape, predicted face.
# return value : returns coords, a new matrix.

def __shape_to_np__(shape, dtype="int"):
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)
    
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    return coords

# Description : It does the same as the blur function, but pixalizated.
# Synopsis : pixelate(frame, face, shape, block_size=20)
# parameters : frame, video capcture. face, captured face. shape, predicted face.
# return value : returns the frame with the blur.

def pixelate(frame, face, shape, block_size=20):
    left = face.left()
    top = face.top()
    right = face.right()
    bottom = face.bottom()

    frame_result = frame.copy()

    for row in range(top, bottom, block_size):
        for col in range(left, right, block_size):

            max_col = min(col+block_size, right)
            max_row = min(row+block_size, bottom)

            for c in range(3):
                frame_result[row:max_row, col:max_col, c] = np.mean(
                    frame[row:max_row, col:max_col, c])

    points = __shape_to_np__(shape)
    hull = cv.convexHull(points, False)

    mask = np.zeros(frame.shape[:2], dtype='uint8')
    cv.drawContours(mask, [hull], 0, (255,)*3, -1)
    
    result_mask = cv.bitwise_and(frame_result, frame_result, mask=mask)
    not_mask = cv.bitwise_not(mask)
    not_frame = cv.bitwise_and(frame, frame, mask=not_mask)

    return cv.bitwise_or(result_mask, not_frame)

# Description : The function takes an image, from video capture, and then perform a draw. It draws the sprite
# on the image. 
# Synopsis : draw_sprite(frame, sprite, x_offset, y_offset)
# parameters : frame, video capture. sprite, wanted image that is already adjusted.
# x_offset, X from boundbox. y_offset, sprite's Y.
# return value : returns the new frame with the sprite.

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

# Description : The function take the video capture and the face to apply a blur effect.
# Synopsis : apply_blur(frame, face)
# parameters : frame, video capcture. face, the found face.
# return value : Returns a new image with the blur.

def apply_blur(frame, face):
   x, y = face.left(), face.top()
   w, h = face.width(), face.height()
   frame[y:y+h, x:x+w] = cv.blur(frame[y:y+h, x:x+w], (50, 50), 70)
   return frame

# our main function where everything happens #

if __name__ == "__main__":
    # load face detector
    face_detector = dlib.get_frontal_face_detector()

    # load facemark detector
    landmarks_path = "./shape_predictor_68_face_landmarks.dat"
    facemark = dlib.shape_predictor(landmarks_path)
    #facemark vai receber a previsão feita pela função a cima

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