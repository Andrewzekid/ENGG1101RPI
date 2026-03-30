from ultralytics import YOLO
import time
import os
import pygame
import pygame.mixer
from gpiozero import AngularServo

from time import sleep
#CONFIGURATIONS
types = ["metal","other","paper","cardboard"]
class_to_audio = {"other":"Other.wav","paper":"Paper.wav","cardboard":"Cardboard.wav","metal":"Metal.wav"}
topservo = AngularServo(14, min_pulse_width=0.0006, max_pulse_width=0.0023,max_angle=110)
baseservo = AngularServo(18, min_pulse_width=0.0006, max_pulse_width=0.0023,max_angle=110)

def reset():
    baseservo.angle = 0
    topservo.angle = -55

def tiltacw():
    #Move the top servo
    topservo.angle = 90
    sleep(2)
    topservo.angle = 0
    sleep(1)

def tiltcw():
    #Move the top servo
    topservo.angle = -90
    sleep(2)
    topservo.angle = 0
    sleep(1)

while True:
    topservo,bottomservo = tuple(map(int,input().split("")))
    topservo.angle = topservo
    bottomservo.angle = bottomservo
    time.sleep(2)