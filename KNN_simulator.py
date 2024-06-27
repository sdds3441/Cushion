import cv2
import numpy as np
import serial
import threading
import time

# Create KNN object
knn = cv2.ml.KNearest_create()

# Read training data
file = np.genfromtxt('./collected.csv', delimiter=',')
fsr_val = file[:, :-1].astype(np.float32)
label = file[:, -1].astype(np.float32)

# Train KNN model
knn.train(fsr_val, cv2.ml.ROW_SAMPLE, label)

# Set up serial communication with Arduino
ser = serial.Serial('COM8', 9600)
time.sleep(2)  # Wait briefly for the serial communication to stabilize

# Variable to store the latest data
latest_data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# Load image
image_path = 'dataset/cushion2.png'  # Enter the image file path here
image = cv2.imread(image_path)

# Check if the image was loaded properly
if image is None:
    raise FileNotFoundError(f"Cannot load image: {image_path}")

# Set coordinates for points (e.g., (x, y) = (100, 150))
height, width, _ = image.shape
qheight = height // 4
qwidth = width // 4
points = [(qwidth * 2, qheight // 2), (qwidth // 2, qheight * 2), (qwidth * 3 + qwidth // 2, qheight * 2),
          (qwidth * 2, qheight * 3 + qheight // 2)]

# Set window name
window_name = 'Image with Changing Dots'


def read_from_arduino():
    global latest_data
    while True:
        if ser.in_waiting > 0:
            arduino_data = ser.readline().decode('utf-8').strip()
            latest_data = list(map(float, arduino_data.split(',')))
            print(latest_data)


def predict_color():
    global latest_data
    while True:
        if latest_data is not None:
            # Convert the latest data to a numpy array and reshape to 2D array
            test_val = np.array(latest_data).reshape(1, -1).astype(np.float32)

            # KNN prediction
            ret, results, neighbours, dist = knn.findNearest(test_val, 3)

            # Print result values
            #print(f"Results: {results}")

            # Change color based on result values (smaller values -> red, larger values -> blue)


def cal_color(direction):
    mean = np.mean(direction)
    if 750 < mean:
        mean = 750

    normalized = mean / 750
    color = (0, int(255 * (1 - normalized)), int(255 * normalized))
    return color


def display_image():
    while True:
        global latest_data

        left = [latest_data[1], latest_data[2], latest_data[5], latest_data[6]]
        right = [latest_data[0], latest_data[3], latest_data[4], latest_data[7]]
        forward = [latest_data[0], latest_data[1], latest_data[4], latest_data[5]]
        backward = [latest_data[2], latest_data[3], latest_data[6], latest_data[7]]

        left_color = cal_color(left)
        right_color = cal_color(right)
        forward_color = cal_color(forward)
        backward_color = cal_color(backward)

        # Copy the original image to use (to avoid overwriting the original image)
        image_copy = image.copy()

        # Draw dots at each coordinate
        cv2.circle(image_copy, points[0], radius=30, color=forward_color, thickness=-1)
        cv2.circle(image_copy, points[1], radius=30, color=left_color, thickness=-1)
        cv2.circle(image_copy, points[2], radius=30, color=right_color, thickness=-1)
        cv2.circle(image_copy, points[3], radius=30, color=backward_color, thickness=-1)

        # Show the image
        cv2.imshow("window_name", image_copy)

        # Wait for 30ms
        key = cv2.waitKey(30)
        if key == 27:  # Exit loop if ESC key is pressed
            break


# Start thread to read data from serial port
threading.Thread(target=read_from_arduino, daemon=True).start()

# Start thread to perform predictions
threading.Thread(target=predict_color, daemon=True).start()

# Start thread to display image
threading.Thread(target=display_image, daemon=True).start()

# Keep the program running
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Program terminated")
finally:
    ser.close()
    cv2.destroyAllWindows()
