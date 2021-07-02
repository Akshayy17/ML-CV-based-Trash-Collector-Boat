# import curses and GPIO
import pygame
import RPi.GPIO as GPIO
import os #added so we can shut down OK
import time #import time module

#import picamera and datetime (to use in unique file naming)
from picamera import PiCamera
from datetime import datetime

#Open a Pygame window to allow it to detect user events
screen = pygame.display.set_mode([250, 250])##### 240, 160

#setup camera
#camera = PiCamera()
#camera.resolution = (640, 480)###1280, 720
#camera.framerate = (25)

record = 0 #set up a variable to be set to 1 when recording

Lfwd, Lbcwd, L_Enable = 24, 23, 22
Rfwd, Rbcwd, R_Enable = 17, 27, 25

GPIO.setwarnings(False)

#set GPIO numbering mode and define output pins
GPIO.setmode(GPIO.BCM)
GPIO.setup(Rbcwd,GPIO.OUT)
GPIO.setup(Rfwd,GPIO.OUT)
GPIO.setup(Lbcwd,GPIO.OUT)
GPIO.setup(Lfwd,GPIO.OUT)
#####GPIO.setup(29,GPIO.OUT)

#setup PWM control
GPIO.setup(L_Enable,GPIO.OUT)
GPIO.setup(R_Enable,GPIO.OUT)
speedleft = GPIO.PWM(L_Enable, 100)
speedright = GPIO.PWM(R_Enable, 100)
speedleft.start(80)##### do it 0 for safe start
speedright.start(80)##### same as above

#flashes LEDs when all running, and also lets camera settle
#####for x in range(1, 5):
        #####GPIO.output(29,False)
        #####time.sleep(0.5)
        #####GPIO.output(29,True)
        #####time.sleep(0.5)

try:
	while True:
		for event in pygame.event.get():
			if event.type == pygame.KEYDOWN:
				if event.key == pygame.K_q:
					pygame.quit()
				#elif event.key == pygame.K_S:
					#os.system ('sudo shutdown now') # shutdown right now!
				elif event.key == pygame.K_r:
					if record == 0:
						record = 1
						moment = datetime.now()
						#####GPIO.output(29,False)
						camera.start_recording('/home/pi/Videos/vid_%02d_%02d_%02d.mjpg' % (moment.hour, moment.minute, moment.second))
				elif event.key == pygame.K_t:
					if record == 1:
						record = 0
						#####GPIO.output(29,True)
						camera.stop_recording()
				elif event.key == pygame.K_RIGHT:#UP
					GPIO.output(Rbcwd,True)
					GPIO.output(Rfwd,False)
					GPIO.output(Lbcwd,False)
					GPIO.output(Lfwd,True)
					print("Right")
				elif event.key == pygame.K_LEFT:#DOWN
					GPIO.output(Rbcwd,False)
					GPIO.output(Rfwd,True)
					GPIO.output(Lbcwd,True)
					GPIO.output(Lfwd,False)
					print("LEFT")
				elif event.key == pygame.K_UP:
					GPIO.output(Rbcwd,False)
					GPIO.output(Rfwd,True)
					GPIO.output(Lbcwd,False)
					GPIO.output(Lfwd,True)
					print("UP")
				elif event.key == pygame.K_DOWN:
					GPIO.output(Rbcwd,True)
					GPIO.output(Rfwd,False)
					GPIO.output(Lbcwd,True)
					GPIO.output(Lfwd,False)
					print("DOWN")
				elif event.key == pygame.K_1:
					speedleft.ChangeDutyCycle(30)
					speedright.ChangeDutyCycle(30)
					print("PWM-25")
				elif event.key == pygame.K_2:
					speedleft.ChangeDutyCycle(45)
					speedright.ChangeDutyCycle(45)
				elif event.key == pygame.K_3:
					speedleft.ChangeDutyCycle(80)
					speedright.ChangeDutyCycle(80)
					print("PWM-90")
				elif event.key == pygame.K_4:
					speedleft.ChangeDutyCycle(100)
					speedright.ChangeDutyCycle(100)
					print("Full PWM")
			elif event.type == pygame.KEYUP:
				GPIO.output(Rbcwd,False)
				GPIO.output(Rfwd,False)
				GPIO.output(Lbcwd,False)
				GPIO.output(Lfwd,False)
			             
finally:
    #GPIO cleanup
    GPIO.cleanup()
    

