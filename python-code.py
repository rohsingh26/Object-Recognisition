import cv2
import urllib.request
import numpy as np
import cvlib as cv
from cvlib.object_detection import draw_bbox
import concurrent.futures

url = 'http://192.168.148.79/cam-hi.jpg'

def run1():
    try:
        cv2.namedWindow("Live Transmission", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Live Transmission", 640, 480)

        while True:
            img_resp = urllib.request.urlopen(url)
            imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
            im = cv2.imdecode(imgnp, -1)

            cv2.imshow('Live Transmission', im)
            key = cv2.waitKey(5)
            if key == ord('q'):
                break
    except Exception as e:
        print("Error in run1:", e)
    finally:
        cv2.destroyAllWindows()

def run2():
    try:
        cv2.namedWindow("Object Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Object Detection", 640, 480)

        while True:
            img_resp = urllib.request.urlopen(url)
            imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
            im = cv2.imdecode(imgnp, -1)

            try:
                bbox, label, conf = cv.detect_common_objects(im)
                im = draw_bbox(im, bbox, label, conf)

                # Display the names of detected objects
                for l, c in zip(label, conf):
                    print(f"Detected: {l} (Confidence: {c:.2f})")

                cv2.imshow('Object Detection', im)
            except Exception as e:
                print("Error in object detection:", e)

            key = cv2.waitKey(5)
            if key == ord('q'):
                break
    except Exception as e:
        print("Error in run2:", e)
    finally:
        cv2.destroyAllWindows()

if _name_ == '_main_':
    print("Starting...")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.submit(run1)
        executor.submit(run2)
