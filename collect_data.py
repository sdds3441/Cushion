import serial
import time
import numpy as np
import tkinter as tk
import threading
import csv

# Flag to control the reading process
reading = False


# Function to start reading data from the serial port
def start_reading():
    global reading
    reading = True
    datas = []

    try:
        ser = serial.Serial("COM8", 9600, timeout=1)
        time.sleep(2)
        print("Serial connection established successfully.")
    except Exception as e:
        print(f"Serial connection error: {e}")
        return

    while reading:
        try:
            get_data = ser.readline()
            if get_data:
                try:
                    get_data = get_data.decode('latin-1').strip()  # Handle decoding errors here
                except Exception as e:
                    print(f"Decoding error: {e}")
                    continue
                split = get_data.split(',')
                split.append('3')
                """ 0 = Good posture,
                    1 = Leaning to the left posture,
                    2 = Leaning to the right posture,
                    3 = Leaning forward posture,
                    4 = Leaning backward posture,
                    5 = Crossed left leg posture,
                    6 = Crossed right leg posture,
                    9 = Don't use"""
                print(split)
                datas.append(split)
        except Exception as e:
            print(f"Data reading error: {e}")
            break

    try:
        # Save data to CSV file with UTF-8 encoding
        with open("dataset/fsr.csv", "w", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(datas)
        print("Data saved to fsr.csv.")
    except Exception as e:
        print(f"Data saving error: {e}")


# Function to run the reading function in a separate thread
def start_thread():
    thread = threading.Thread(target=start_reading)
    thread.start()


# Function to stop reading data
def stop_reading():
    global reading
    reading = False
    print("Reading stopped.")


# Creating the Tkinter window
window = tk.Tk()
window.geometry('500x400')

# Creating buttons and binding them to the respective functions
button = tk.Button(window, text='Start Reading', command=start_thread)
button.place(x=0, y=0)

button2 = tk.Button(window, text='Stop Reading', command=stop_reading)
button2.place(x=200, y=0)

window.mainloop()
