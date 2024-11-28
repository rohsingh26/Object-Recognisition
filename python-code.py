import cv2
import urllib.request
import numpy as np
import cvlib as cv
from cvlib.object_detection import draw_bbox
import concurrent.futures
from plyer import notification  # For desktop notifications

url = 'http://192.168.148.79/cam-hi.jpg'

# Function to send a desktop notification
def send_notification(detected_object):
    notification.notify(
        title="Object Detected!",
        message=f"A {detected_object} has been detected.",
        timeout=5  # Notification duration in seconds
    )

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

                # Check for a specific object (e.g., 'cat')
                for l, c in zip(label, conf):
                    print(f"Detected: {l} (Confidence: {c:.2f})")
                    if l == 'car' and c > 0.6:  # Threshold confidence for detection
                        send_notification('Car detected in no parking area!')  # Send a notification if a cat is detected

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

if __name__ == '__main__':
    print("Starting...")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.submit(run1)
        executor.submit(run2)
