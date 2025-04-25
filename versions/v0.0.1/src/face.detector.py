import cv2
import time
import argparse
from imutils.video import VideoStream
import numpy as np
import imutils
import os
import sys

def find_model_files():
    global prototxt_path, model_path
    #Directories to search for the model files
    search_dirs = [os.getcwd(), '/usr/local/bin/app.face.detector']

    prototxt_file = "deploy.prototxt.txt"
    model_file = "res10_300x300_ssd_iter_140000.caffemodel"

    #Search in the directories
    for directory in search_dirs:
        prototxt_path = os.path.join(directory, "neural.net", prototxt_file)
        model_path = os.path.join(directory, "model.with.trained.weights", model_file)

        if os.path.exists(prototxt_path) and os.path.exists(model_path):
            return prototxt_path, model_path

    #If files are not found, print error and exit
    print(f"Error: Model files not found in the current directory or /usr/local/bin/app.face.detector.")
    sys.exit(1)

def get_video_from_webcam():
    return VideoStream(src=0).start()

def get_image_frame(video_stream):
    return video_stream.read()

def resize_image(image, width):
    return imutils.resize(image, width=width)

def image2blob(image):
    return cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))

def detect_faces(image_blob):
    global neural_network
    neural_network.setInput(image_blob)
    return neural_network.forward()

def add_label(image, label, x, y):
    cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

def process_faces(image_frame, faces, do_blur=False):
    (h, w) = image_frame.shape[:2]
    for i in range(0, faces.shape[2]):
        accuracy = faces[0, 0, i, 2]
        if accuracy < 0.5:
            continue

        box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        y = startY - 10
        if y < 10:
            y = startY + 10

        label = "{:.2f}%".format(accuracy * 100)
        add_label(image_frame, label, startX, y)

        if do_blur:
            face = image_frame[startY:endY, startX:endX]
            blurred_face = blur_image(face)
            image_frame[startY:endY, startX:endX] = blurred_face
        else:
            cv2.rectangle(image_frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    return image_frame

def blur_image(image):
    factor = 1.0
    (h, w) = image.shape[:2]
    k_w = int(w / factor)
    k_h = int(h / factor)

    if k_w % 2 == 0:
        k_w -= 1
    if k_h % 2 == 0:
        k_h -= 1

    return cv2.GaussianBlur(image, (k_w, k_h), 0)

def is_exit_key_pressed():
    key = cv2.waitKey(30) & 0xff
    return key == 27

def display_image(image):
    cv2.imshow('Frame', image)

def run_live_mode(do_blur):
    video_stream = get_video_from_webcam()
    time.sleep(1.0)

    try:
        while True:
            image_frame = get_image_frame(video_stream)
            image_frame = resize_image(image_frame, 400)

            blob = image2blob(image_frame)
            faces = detect_faces(blob)

            image_frame = process_faces(image_frame, faces, do_blur=do_blur)
            display_image(image_frame)

            if is_exit_key_pressed():
                break
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user (Ctrl+C).")

    cv2.destroyAllWindows()
    video_stream.stop()

def run_file_mode(image_path, do_blur):
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Could not open image file: {image_path}")
        return

    image = resize_image(image, 400)
    blob = image2blob(image)
    faces = detect_faces(blob)

    output = process_faces(image, faces, do_blur=do_blur)

    # Extract filename only, no path
    filename = os.path.basename(image_path)
    base, ext = os.path.splitext(filename)
    suffix = "_face_blurred" if do_blur else "_face_detected"
    output_filename = f"{base}{suffix}{ext}"

    # Save in current working directory
    output_path = os.path.join(os.getcwd(), output_filename)
    success = cv2.imwrite(output_path, output)

    if success:
        print(f"[INFO] Output saved to: {output_filename}")
    else:
        print(f"[ERROR] Failed to save image to: {output_path}")


def is_exit_key_pressed(window_name='Frame'):
    key = cv2.waitKey(30) & 0xff
    if key == 27 or key == ord('q'):
        return True
    # If the window is closed, getWindowProperty returns a negative value
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        return True
    return False

def print_usage():
    print("Usage:")
    print("  python script.py -f <image_file>         # Detect faces in an image and draw a green box around each face")
    print("  python script.py -f <image_file> -b      # Detect and blur faces in an image")
    print("  python script.py                         # Prompt to access camera and detect faces using webcam")
    print()

def main():
    global prototxt_path, model_path, neural_network

    prototxt_path, model_path = find_model_files()

    # Load the neural network with the found paths
    neural_network = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    print(f"Model loaded successfully from {prototxt_path} and {model_path}")


    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-f', '--file', help='Path to image file (jpg/png)')
    parser.add_argument('-b', '--blur', action='store_true', help='Blur detected faces (optional)')
    args = parser.parse_args()

    if args.file:
        run_file_mode(args.file, do_blur=args.blur)
    else:
        print_usage()
        response = input("\nThis utility is going to access your camera. Are you okay with that? [Y]: ").strip().lower()
        if response == 'y' or response == 'Y':
            run_live_mode(do_blur=args.blur)
        else:
            print("[INFO] Exiting. No operation was performed.")

if __name__ == "__main__":
    main()

