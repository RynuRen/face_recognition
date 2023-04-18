import cv2
from keras.models import load_model
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import cvlib as cv
import face_recognition

# style
# styles=["Gogh","Kandinsky","Monet","Picasso","Na","Mario"]
# style_transfer=StyleTransfer(1280,720)
# style_transfer.load()
# style_transfer.change_style(styles.index("Na"))
# image_segmentation=ImageSegmentation(1280,720)

# window option
cv2.namedWindow("test", cv2. WINDOW_NORMAL)
cv2.setWindowProperty("test", cv2. WND_PROP_FULLSCREEN, cv2. WINDOW_FULLSCREEN)

# label
class_names_hangul = ['보통_무표정','보통_활짝웃음','보통_찡그림',
                      '일반안경_무표정','일반안경_활짝웃음','일반안경_찡그림',
                      '뿔테안경_무표정','뿔테안경_활짝웃음','뿔테안경_무표정_찡그림',
                      '선글라스_무표정','선글라스_활짝웃음','선글라스_찡그림',
                      '모자_무표정','모자_활짝웃음','모자_찡그림',
                      '모자+뿔테_무표정','모자+뿔테_활짝웃음','모자+뿔테_찡그림',]

fontpath = "fonts/gulim.ttc"
font = ImageFont.truetype(fontpath, 24)
b,g,r,a = 0,255,0,255

# load model
class_model = load_model('model.h5')

#window on
def webon(video, size) :
    # webcam on
    webcam = cv2.VideoCapture(video)
    webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)
    if not webcam.isOpened():
        print("Could not open webcam")
        exit()
    while webcam.isOpened():
        ret, img = webcam.read()
        # style_image = style_transfer.predict(img)
        # seg_mask = image_segmentation.predict(img)
        # seg_mask = cv2.cvtColor(seg_mask, cv2.COLOR_GRAY2RGB)
        # result_image = np.where(seg_mask, style_image, img)
        faces, confidences = cv.detect_face(img)
        glass, hat = 0, 0
        if len(faces) :
            for (x, y, w, h), conf in zip(faces, confidences) :
                face = img[max(0,y-size):h+size,max(0,x-size):w+size]
                face = np.array(cv2.resize(face,(200,300)))/255
                pred = class_model.predict(face.reshape(1,200,300,3))
                cv2.rectangle(img, (max(0, x - size), max(0, y - size)), (w + size, h + size), (255, 0, 0), 2)
                img_pillow = Image.fromarray(img)
                draw = ImageDraw.Draw(img_pillow, 'RGBA')
                draw.text((x, y), class_names_hangul[np.argmax(pred)], font=font, fill=(b, g, r, a))
                if np.argmax(pred) <= 2 :
                    pass
                if 3<= np.argmax(pred) <= 11 or 15<=np.argmax(pred)<=17:
                    glass += 1
                if 12<= np.argmax(pred) <= 17 :
                    hat += 1
                img = np.array(img_pillow)
        img_pillow = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pillow, 'RGBA')
        draw.text((0, 0), f'총 사람 수  : {len(faces)}', font=font, fill=(b, g, r, a))
        draw.text((0, 25), f'모자 쓴 사람 수 : {hat}', font=font, fill=(b, g, r, a))
        draw.text((0, 50), f'안경 쓴 사람 수 : {glass}', font=font, fill=(b, g, r, a))
        img = np.array(img_pillow)
        if ret:
            cv2.imshow("test", img)
        if cv2.waitKey(10) == 27 :
            break
    webcam.release()
    cv2.destroyAllWindows()

webon(0, 30)
