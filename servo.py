from gpiozero import AngularServo

from time import sleep
types = ["alucan","plastic","glass","other"]
# Create an AngularServo object with the specified GPIO pin,

# minimum pulse width, and maximum pulse width

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
reset()
try:
   while True:

    #Four types - alucan, plastic, glass, othe
    identified = input()
    if identified not in types:
        continue

    if(identified == "alucan"):
        baseservo.angle = 0
        tiltacw()
        reset()
    elif(identified == "plastic"):
        baseservo.angle = 0
        tiltcw()
        reset()
    elif(identified == "glass"):
        baseservo.angle = 90
        tiltcw()
        reset()
    else:
        baseservo.angle = 90
        tiltacw()
        reset()



finally:

   # Set the servo angle to 0 degrees before exiting
   reset()

