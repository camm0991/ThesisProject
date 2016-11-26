#Author: Carlos Armando Martinez Medina
#Experiment: Ingame character movement
#Description: The user will be advised to center his/her attention
#on an in-game character then he/she will try to focus on the character
#movements.

#Imports

from emokit import emotiv
from pykeyboard import PyKeyboard
from sklearn.cross_validation import train_test_split
from time import sleep
from winsound import Beep as beep

import gevent
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import platform
import pyttsx
import time

if platform.system() == "Windows":
    import socket

#Constants
VERY_LOW_SOUND = 400
LOW_SOUND = 1000
HIGH_SOUND = 2000
STANDART_TIME = 500
STANDART_SLEEP = 0.5
SECTION_SLEEP = 2
FILE_NAME = 'EEG_XX.csv'

#Functions
def start_action(action_command):
    k.press_key(action_command)
  
def stop_action(action_command):
    k.release_key(action_command)

def init_experiment_sound():
    beep(LOW_SOUND, STANDART_TIME)
    sleep(STANDART_SLEEP)
    beep(LOW_SOUND, STANDART_TIME)
    sleep(STANDART_SLEEP)
    beep(LOW_SOUND, STANDART_TIME)
    sleep(STANDART_SLEEP)
    beep(HIGH_SOUND, STANDART_TIME)
    sleep(STANDART_SLEEP)

def init_experiment_section():
    beep(HIGH_SOUND, STANDART_TIME)

def end_experiment_section():
    beep(LOW_SOUND, STANDART_TIME)

def say(text):
    reader.say(text)
    reader.runAndWait()

#Main
if __name__ == "__main__":
    #Headset setup
    headset = emotiv.Emotiv(display_output=False)    
    gevent.spawn(headset.setup)
    gevent.sleep(0)
    
    intentions = {0:'.',1:'w',2:'s',3:'q',4:'e'}
    readings_array = []
    readings = 0
    number_of_readings = 150
    experiment = 0
    iteration = 0
    intention = 0
    Class_index = 0
    
    iters = 1000
    k = PyKeyboard()
    init_experiment_sound()
    print('Calibrating')
    epoc_init = 0
    try:
        while epoc_init < 1000:
            packet = headset.dequeue()
            readings_array.append([packet.AF3[0], packet.F7[0], packet.F3[0], packet.FC5[0], packet.T7[0], packet.P7[0], packet.O1[0], packet.O2[0], packet.P8[0], packet.T8[0], packet.FC6[0], packet.F4[0], packet.F8[0], packet.AF4[0], Class_index])
            epoc_init += 1
    except KeyboardInterrupt:
        headset.close()
    readings_array = []

    try:
        print('Real Experiment')
        gevent.sleep(0.5)
        time_record = time.clock()
        init_experiment_section()
        k.press_key(intentions[Class_index])
        while True:
            #Get the package from the headset
            packet = headset.dequeue()
            #Printing the values from tahe headset's package and the class
            readings_array.append([packet.AF3[0], packet.F7[0], packet.F3[0], packet.FC5[0], packet.T7[0], packet.P7[0], packet.O1[0], packet.O2[0], packet.P8[0], packet.T8[0], packet.FC6[0], packet.F4[0], packet.F8[0], packet.AF4[0], Class_index])
            readings += 1
            #Optional delay for reading packages
            #gevent.sleep(0)
            
            if(readings % iters == 0):
                k.release_key(intentions[Class_index])
                end_experiment_section()
                print('Time: {} Class: {}'.format(time.clock() - time_record,Class_index))
                Class_index += 1
                if Class_index < 5:
                    time_record = time.clock()
                    init_experiment_section()
                    k.press_key(intentions[Class_index])

                
            if(readings % (iters * 5) == 0):
                break
            
    #Close headset connection
    except KeyboardInterrupt:
        time.sleep(1)
        headset.close()
    finally:
        time.sleep(1)
        data = pd.DataFrame(readings_array, columns=['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4', 'Class'])
        data.to_csv(FILE_NAME, index=False)
        headset.close()
