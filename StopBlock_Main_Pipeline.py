import os
import cv2
import numpy as np
import json
from datetime import datetime
import sys
import time
import onnxruntime as rt
import tensorflow as tf
import uptime
from jetcam.net_camera import NETCamera
from communication import ComMqtt

DEBUG = False
MAX_FILES = 500
CAMERA_USE_FILE = "/home/jetson/CAMERA_IN_USE"

def load_config(config_file):
    """Load configuration from a JSON file."""
    with open(config_file) as f:
        return json.load(f)

def initialize_camera():
    """Initialize and power on the camera."""
    if os.path.isfile(CAMERA_USE_FILE):
        os.remove(CAMERA_USE_FILE)
    
    while os.path.isfile(CAMERA_USE_FILE):
        print("INFO: Camera already in use... waiting 5 seconds")
        time.sleep(5)

    os.system("/usr/sbin/i2cset -y -r 0x0 0x21 0x19 0xf0")
    time.sleep(2)
    os.system("/usr/sbin/i2cset -y -r 0x0 0x21 0x19 0x0f")
    time.sleep(65)
    print("Camera powered on")

def capture_image(camera):
    """Capture an image from the camera."""
    try:
        image = camera.read()
        return image
    except Exception as e:
        print("An exception occurred:", e)
        os.remove(CAMERA_USE_FILE)
        os.system("/usr/sbin/i2cset -y -r 0x0 0x21 0x19 0xf0")
        sys.exit("Error: Could not start the camera")
        return None

def save_image(image, id_sensor):
    """Save the captured image to a file."""
    now = datetime.now()
    current_time = str(now.strftime("%Y_%m_%d_%H_%M_%S"))
    image_file = f"output/{id_sensor}_{current_time}.png"
    cv2.imwrite(image_file, image)
    return image_file

def cleanup_camera():
    """Power down and release the camera."""
    os.system("/usr/sbin/i2cset -y -r 0x0 0x21 0x19 0xf0")
    print("Camera powered down")

def load_detection_model(model_path):
    """Load the object detection ONNX model."""
    return rt.InferenceSession(model_path)

def load_classification_model(model_path):
    """Load the image classification ONNX model."""
    return rt.InferenceSession(model_path)

def preprocess_image(image):
    """Preprocess the captured image for object detection and classification."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_roi = image_rgb[570:828, 1100:1552]
    image_expanded = np.expand_dims(image_roi, axis=0)
    height, width = image_roi.shape[0], image_roi.shape[1]
    return image_roi, image_expanded, height, width

def perform_detection(detection_session, image_expanded, min_score_thresh):
    """Perform object detection on the preprocessed image."""
    outputs = ["detection_boxes:0", "detection_scores:0", "detection_classes:0", "num_detections:0"]
    boxes, scores, classes, num = detection_session.run(outputs, {"image_tensor:0": image_expanded})
    scores_detect = scores[0][scores[0] > min_score_thresh]
    true_boxes = boxes[0][scores[0] > min_score_thresh]
    return true_boxes, scores_detect

def preprocess_and_classify_openings(image_roi, true_boxes, height, width, classification_session, label):
    """Preprocess and classify detected openings."""
    prediction_array = []
    score_array = []

    for i in range(true_boxes.shape[0]):
        ymin = int(true_boxes[i, 0] * height)
        xmin = int(true_boxes[i, 1] * width)
        ymax = int(true_boxes[i, 2] * height)
        xmax = int(true_boxes[i, 3] * width)

        opening_bbox = image_roi[ymin:ymax, xmin:xmax].copy()
        opening_bbox = tf.image.resize(opening_bbox, [224, 224])
        opening_bbox = np.expand_dims(opening_bbox, axis=0)
        pred = classification_session.run(None, {"input_1": opening_bbox})
        y_class = np.argmax(pred)
        prediction_array.append(y_class)
        score_array.append(pred[0][0][y_class])

        if DEBUG:
            cv2.imwrite(f"opening_{i}_{label[y_class]}.png", image_roi[ymin:ymax, xmin:xmax])

    return prediction_array, score_array

def calculate_blockage_statistics(prediction_array, label):
    """Calculate blockage statistics based on classified openings."""
    total_blocked = np.where(np.array(prediction_array) == 0)[0]
    total_clear = np.where(np.array(prediction_array) == 1)[0]
    total_partial = np.where(np.array(prediction_array) == 2)[0]
    total_openings = len(np.array(prediction_array))
    return total_blocked, total_clear, total_partial, total_openings

def is_culvert_blocked(total_blocked, total_openings):
    """Determine if the culvert is visually blocked based on blockage statistics."""
    if len(total_blocked) >= total_openings / 2:
        return True
    else:
        return False

def save_blockage_statistics(total_openings, total_clear, total_partial, total_blocked):
    """Save blockage statistics to a file."""
    with open("culvert_stats.txt", 'w') as f_out:
        f_out.write("opening,label,score_class,score_detect\n")
        for i in range(len(prediction_array)):
            line = f"{i}, {label[prediction_array[i]]}, {score_array[i]}, {scores_detect[i]}\n"
            f_out.write(line)

def transmit_output(client, total_openings, total_clear, total_partial, total_blocked, scores_detect, score_array, prediction_array):
    """Transmit the output to an MQTT broker."""
    payload = dict()
    payload['num_openings'] = total_openings
    payload['num_clear'] = len(total_clear)
    payload['num_partial'] = len(total_partial)
    payload['num_blocked'] = len(total_blocked)
    payload['score_detector'] = scores_detect.tolist()
    payload['score_classifier'] = [s.item() for s in score_array]
    payload['prediction'] = [label[p.item()] for p in prediction_array]
    payload['uptime'] = int(uptime.uptime())
    payload_json = json.dumps(payload)

    client.publish(payload_json)

def main():
    # Load configuration
    config_file = "config.json"
    conf = load_config(config_file)

    # Initialize camera
    initialize_camera()

    # Create camera object
    try:
        camera = NETCamera(device=conf['camera_url'])
    except:
        os.remove(CAMERA_USE_FILE)

    # Capture image
    image = capture_image(camera)

    # Power down and release the camera
    cleanup_camera()

    if image is None:
        sys.exit("Error: Could not start the camera")

    # Save the image
    image_file = save_image(image, conf['id'])

    if DEBUG:
        cv2.imshow("culvert", image)
        cv2.waitKey() & 0xFF == ord('q')

    # Keeping only the 1000 most recent images
    list_of_files = os.listdir('output')
    full_path = ["output/{0}".format(x) for x in list_of_files]

    if len(list_of_files) >= MAX_FILES:
        oldest_file = min(full_path, key=os.path.getctime)
        os.remove(oldest_file)

    # Load detection and classification ONNX models
    sess_detection = load_detection_model("frcnn_resnet50.onnx")
    sess_classification = load_classification_model("ResNet50_Best.onnx")

    # Pre-process and classify openings
    image_roi, image_expanded, height, width = preprocess_image(image)
    true_boxes, scores_detect = perform_detection(sess_detection, image_expanded, conf['min_threshold_detection'])
    prediction_array, score_array = preprocess_and_classify_openings(image_roi, true_boxes, height, width, sess_classification, conf['label'])

    # Calculate blockage statistics
    total_blocked, total_clear, total_partial, total_openings = calculate_blockage_statistics(prediction_array, conf['label'])

    if DEBUG:
        print(f"Total {total_openings} openings were detected - {len(total_clear)} clear, {len(total_partial)} partially blocked, and {len(total_blocked)} blocked.")

        # Determine if the culvert is visually blocked
        if is_culvert_blocked(total_blocked, total_openings):
            print("Culvert is visually blocked")
        else:
            print("Culvert is visually clear")

    # Save blockage statistics to a file
    save_blockage_statistics(total_openings, total_clear, total_partial, total_blocked)

    # Transmit the output
    client = ComMqtt(conf['id'], conf['broker'], conf['port'], conf['application'], conf['user'], conf['password'])
    client.start_listening()

    transmit_output(client, total_openings, total_clear, total_partial, total_blocked, scores_detect, score_array, prediction_array)

    client.stop_listening()

    end_time = time.time()
    print("Processing time: {} seconds".format(end_time - start_time))

if __name__ == "__main__":
    main()
