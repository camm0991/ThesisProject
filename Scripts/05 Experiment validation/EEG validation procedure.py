#Author: Carlos Armando Martinez Medina
#Experiment: Ingame character movement
#Description: The user will be advised to center his/her attention
#on an in-game character then he/she will try to focus on the character
#movements.

#All packages
#[packet.AF3[0], packet.F7[0], packet.F3[0], packet.FC5[0], packet.T7[0], packet.P7[0], packet.O1[0], packet.O2[0], packet.P8[0], packet.T8[0], packet.FC6[0], packet.F4[0], packet.F8[0], packet.AF4[0]]

#Test 1
#[[0, 4, 4, 4, 1], [1, 0, 2, 2, 2], [3, 3, 4, 3, 1], [0, 3, 1, 1, 2], [2, 2, 0, 0, 0]]
#Test 2
#[[0, 2, 2, 2, 0], [1, 4, 4, 0, 3], [4, 2, 3, 4, 2], [0, 1, 4, 0, 1], [1, 3, 4, 2, 3]]
#Test 3
#[[0, 4, 4, 3, 1], [0, 3, 4, 4, 0], [2, 1, 1, 1, 1], [4, 2, 2, 3, 2], [2, 0, 1, 2, 0]]
#Test 4
#[[1, 1, 2, 2, 3], [3, 1, 0, 0, 4], [1, 0, 1, 4, 1], [0, 4, 0, 2, 0], [0, 2, 1, 3, 0]]
#Test 5
#[[4, 1, 3, 1, 3], [0, 0, 0, 1, 0], [0, 2, 0, 4, 0], [2, 4, 0, 2, 0], [4, 3, 4, 1, 1]]

#All channels No Feature selection (Not filtered, filtered)
#EPOC_channels = ['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4']

#Some channels Feature selection (Not filtered)
#EPOC_channels = ['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4']

#Some channels Feature selection (filtered)
#EPOC_channels = ['F7','F3','FC5','T7','O1','O2','P8','T8','F4','F8','AF4']


#Imports
from collections import Counter
from emokit import emotiv
from pykeyboard import PyKeyboard
from scipy.signal import butter
from scipy.signal import lfilter
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from time import sleep
from winsound import Beep as beep

import gevent
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import platform
import pyttsx
import random



if platform.system() == "Windows":
  import socket

#Constants

#Sound values are specified with a F
#Time values are specified with a T
EXPERIMENT_F_INIT_END = 300
EXPERIMENT_T_INIT_END = 1000

TEST_T = 800
TEST_LOW_F = 1000
TEST_HIG_F = 2000

CAPTURE_F_END = 1200
CAPTURE_F_INI = 1500
CAPTURE_T_SOUND = 800
CAPTURE_T_PAUSE = 5
CAPTURE_T_INTER = 1.5

SHORT_T_SLEEP = 0.5
MEDIUM_T_SLEEP = 1
LONG_T_SLEEP = 2
EXTRA_LONG_T_SLEEP = 3

#Functions
#Sounds
def init_end_experiment_sound():
    beep(EXPERIMENT_F_INIT_END, EXPERIMENT_T_INIT_END)
    sleep(MEDIUM_T_SLEEP)

def init_test_section():
    beep(TEST_LOW_F, TEST_T)
    sleep(SHORT_T_SLEEP)
    beep(TEST_LOW_F, TEST_T)
    sleep(SHORT_T_SLEEP)
    beep(TEST_LOW_F, TEST_T)
    sleep(SHORT_T_SLEEP)
    beep(TEST_HIG_F, TEST_T)

def end_test_section():
    beep(TEST_LOW_F, TEST_T)
    sleep(SHORT_T_SLEEP)
    beep(TEST_LOW_F, TEST_T)
    sleep(SHORT_T_SLEEP)

def init_generate_intention():
    beep(CAPTURE_F_INI, CAPTURE_T_SOUND)

def end_generate_intention():
    beep(CAPTURE_F_END, CAPTURE_T_SOUND)

def get_readings(number_of_readings, std_scaler, filter = None):
    readings = 0
    readings_array = []
    EPOC_channels = ['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4']
    readings_df = pd.DataFrame(columns=EPOC_channels)

    while readings < number_of_readings:
        #Obtain device data
        packet = headset.dequeue()
        #Add readings to array
        
        readings_array.append([packet.AF3[0], packet.F7[0], packet.F3[0], 
                               packet.FC5[0], packet.T7[0], packet.P7[0], 
                               packet.O1[0], packet.O2[0], packet.P8[0], 
                               packet.T8[0], packet.FC6[0], packet.F4[0], 
                               packet.F8[0], packet.AF4[0]])
        readings += 1
        #Optional delay for reading packages
        gevent.sleep(0)
    
    #Load data in data frame
    for i in range(len(EPOC_channels)):
        readings_df[EPOC_channels[i]] = [column[i] for column in readings_array]
    
    #Check if a filter was provided
    if filter is not None:
        for i in range(len(EPOC_channels)):
            readings_df[EPOC_channels[i]] = lfilter(filter[0], filter[1], readings_df[EPOC_channels[i]])

    #Standarize values
    readings_df =  std_scaler.transform(readings_df)
    
    #Return data ready for classification
    return readings_df.mean(axis=0)

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

#Main
#Headset setup
headset = emotiv.Emotiv(display_output=False)    
gevent.spawn(headset.setup)
gevent.sleep(0)

#Mouse and keyboard setup
k = PyKeyboard()

#Define filter
b, a = butter_bandpass(0.5, 30.0, 128.0)

#Data preparation
df = pd.read_csv("SVM_training_input.csv")
X = df.ix[:,[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]]
y = df['Class']

stdsc = StandardScaler()
X = stdsc.fit_transform(X)

#Classifier setup
svm = SVC(C=8, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=2, kernel='rbf',
  max_iter=-1, probability=False, random_state=0, shrinking=True,
  tol=0.001, verbose=False)
print('Training SVM')
#Classifier training
svm.fit(X, y)

commands = {0:'.',1:'w',2:'s',3:'q',4:'e'}
tests_array = [[0, 4, 4, 3, 1], [0, 3, 4, 4, 0], [2, 1, 1, 1, 1], [4, 2, 2, 3, 2], [2, 0, 1, 2, 0]]
predicted_intentions_array = [[] for x in range(len(tests_array))]
number_of_readings = 278
experiment = 0
iteration = 0
intention = 0


try:
    #Start experiment
    init_end_experiment_sound()
    #For each test array in tests
    for i in range(len(tests_array)):
        #Alert of test beginning
        init_test_section()
        sleep(LONG_T_SLEEP)
        #Start test
        for j in range(len(tests_array[i])):
            #Capture intention
            init_generate_intention()
            sleep(CAPTURE_T_INTER)
            #readings_df = get_readings(number_of_readings, stdsc, [b,a])
            readings_df = get_readings(number_of_readings, stdsc)
            sleep(CAPTURE_T_INTER)
            end_generate_intention()
            #Score generated signal
            score_list = svm.predict(readings_df)
            intention =  Counter(score_list).most_common()[0][0]
            #Save predicted intention
            predicted_intentions_array[i].append(intention)
            #Visual feedback
            k.press_key(commands[intention])
            sleep(LONG_T_SLEEP)
            k.release_key(commands[intention])
            sleep(MEDIUM_T_SLEEP)
        #End test alert
        end_test_section()
        sleep(EXTRA_LONG_T_SLEEP)
    #End experiment alert
    init_end_experiment_sound()    
#Close headset connection
except KeyboardInterrupt:
  headset.close()
finally:
  headset.close()

#Final feedback
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix

names = ['Fijo', 'Avanzar', 'Retroceder', 'Izquierda', 'Derecha']

def plot_confusion_matrix(cm, names, title, cmap=plt.cm.Blues):
    #plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(names))
    plt.xticks(tick_marks, names, rotation=45)
    plt.yticks(tick_marks, names)
    plt.tight_layout()
    plt.ylabel('Clasificacion correcta')
    plt.xlabel('Clasificacion estimada')

y_test = []
y_pred = []
precision = 0

for i in zip(tests_array, predicted_intentions_array):
    for j in zip(i[0], i[1]):
        y_test.append(j[0])
        y_pred.append(j[1])
        if j[0] == j[1]:
            precision += 1
print(y_test)
print(y_pred)
print(precision)
print('Accuracy achived in the experiment: {}'.format(precision / 25.0))

cm = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(cm, names, title='Matriz de confusion (Filtrado, sin seleccion de caracteristicas) Presicion')
plt.show()
