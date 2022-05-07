#!/usr/bin/python3
# host/RoboPi.py
# Host file for robots using a RoboPi hat

# MIT License
#
# Copyright (c) 2018 BSMRKRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from bsmLib.networking import tcpServer
from adafruit_servokit import ServoKit

#### Global Variables ####
HOST = '0.0.0.0'
PORT = 10000

MAX_THROTTLE = 15

MOTOR = 0
SERVO = 1

######################
## 0. Setup
######################
# Create tcp connection and listen
t = tcpServer(HOST, PORT)
t.listen()

kit = ServoKit(channels=16)

######################
## 1. drive
######################
def drive():
    try:
        d = t.recv()
    except:
        kit.servo[MOTOR].angle = 90
        kit.servo[SERVO].angle = 90

    if d == "stop":
        kit.servo[MOTOR].angle = 90
        kit.servo[SERVO].angle = 90
        t.stop()
        exit()
    d = d.split(' ')

    throttle = 90 + (float(d[0]) * MAX_THROTTLE)
    bearing = (float(d[1]) * 90) + 90

    kit.servo[MOTOR].angle = throttle

    bearing = (bearing - 90) * (1/3) + 90

    if(bearing > 120):
        kit.servo[SERVO].angle = 120
    elif(bearing < 60):
        kit.servo[SERVO].angle = 60
    else:
        kit.servo[SERVO].angle = bearing

    print("%f, %f" % (throttle, bearing))



######################
##      Main        ##
######################
if __name__ == "__main__":
    while(1):
        drive()
