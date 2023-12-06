import cv2
import numpy as np
import tensorflow as tf
from tkinter import filedialog
from tkinter import *
from PIL import Image, ImageTk

# Thiết lập tham số cho mô hình
img_width, img_height = 224, 224
count = 0

def sliding_window(image, stepSize, windowSize):
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def select_image():
    global path_image, img, imgtk
    path_image = filedialog.askopenfilename()
    img = Image.open(path_image)
    img = img.resize((250, 250))  # resize the image without Image.ANTIALIAS
    imgtk = ImageTk.PhotoImage(img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    return None

def detect_objects():
    global path_image, count
    image = cv2.imread(path_image)
    winW, winH = int(entry_window_width.get()), int(entry_window_height.get())
    stepSize = int(entry_step_size.get())

    model = tf.keras.models.load_model(r"C:\Users\ADMIN\cereal_detection_model.h5")

    for (x, y, window) in sliding_window(image, stepSize=stepSize, windowSize=(winW, winH)):
        if window.shape[0] != winH or window.shape[1] != winW:
            continue

        window = cv2.resize(window, (img_width, img_height))
        window = window.astype("float32") / 255.0
        window = np.expand_dims(window, axis=0)

        preds = model.predict(window)
        label = 'Object' if preds[0][0] > 0.5 else 'Not object'
        color = (0, 255, 0) if label == 'Object' else (0, 0, 255)

        clone = image.copy()
        cv2.rectangle(clone, (x, y), (x + winW, y + winH), color, 2)
        cv2.putText(clone, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.imshow("Window", clone)
        cv2.waitKey(1)

        # Save the image after each prediction
        if label == 'Object':
            output_path = f'D:\\OneDrive - tuyenquang.edu.vn\\Docs\\Tài liệu học tập real\\nam4\\xla\\detected_object_{count}.jpg'
            cv2.imwrite(output_path, clone)
            count += 1




root = Tk()

label_step_size = Label(root, text="Step size:")
label_step_size.pack()
entry_step_size = Entry(root)
entry_step_size.pack()

label_window_width = Label(root, text="Window width:")
label_window_width.pack()
entry_window_width = Entry(root)
entry_window_width.pack()

label_window_height = Label(root, text="Window height:")
label_window_height.pack()
entry_window_height = Entry(root)
entry_window_height.pack()

btn_select_image = Button(root, text="Select image", command=select_image)
btn_select_image.pack()

lmain = Label(root)
lmain.pack()

btn_detect_objects = Button(root, text="Detect objects", command=detect_objects)
btn_detect_objects.pack()

root.mainloop()
