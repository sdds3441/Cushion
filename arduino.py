import serial
import time
import numpy as np
import tkinter as tk

window=tk.Tk()
window.geometry('500x400')

button = tk.Button(window, text='start')
button.place(x=0, y=0)

button2= tk.Button(window, text='start')
button2.place(x=200, y=0)
window.mainloop()




datas = []
ser = serial.Serial("COM8", 9600, timeout=1)
time.sleep(2)
while True:
    get_data = ser.readline()
    if get_data:
        get_data = get_data.decode()
        get_data = get_data.replace("\n", "")
        get_data = get_data.replace("\r", "")

        list(get_data)
        split = get_data.split(',')
        print(split)
        datas.append(split)
        if split[0] == '100':
            np.savetxt("dataset//fsr.csv", datas, delimiter=",", fmt="%s")
            exit()
