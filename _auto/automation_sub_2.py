import RPi.GPIO as GPIO
import os #added so we can shut down OK
import time #import time module

Lfwd, Lbcwd, L_Enable = 24, 23, 22
Rfwd, Rbcwd, R_Enable = 17, 27, 25
IRsensorLeft, IRsensorRight = 5, 6
_PWM = 100
GPIO.setwarnings(False)

def init_gpio(): #Kamacha nahi ahe. init_gpio can not be executed by 'call in another script' method
    GPIO.setmode(GPIO.BCM)#So, I wrote all these instructions directly in the main 'detect' function of
    
    GPIO.setup(Rbcwd,GPIO.OUT)#main script. This function is currently useless(not in use).
    GPIO.setup(Rfwd,GPIO.OUT)
    GPIO.setup(Lbcwd,GPIO.OUT)
    GPIO.setup(Lfwd,GPIO.OUT)
    GPIO.setup(IRsensorLeft,GPIO.IN)
    GPIO.setup(IRsensorRight,GPIO.IN)
    GPIO.setup(L_Enable,GPIO.OUT)
    GPIO.setup(R_Enable,GPIO.OUT)
    speedLeft = GPIO.PWM(L_Enable, 100)
    speedRight = GPIO.PWM(R_Enable, 100)
    speedLeft.start(55)##### do it 0 for safe start
    speedRight.start(55)

def MoveStraight():
    GPIO.output(Rfwd,True)
    GPIO.output(Lfwd,True)
    GPIO.output(Rbcwd,False)
    GPIO.output(Lbcwd,False)
    time.sleep(0.2) #More the delay more it will turn (0.07 earlier)
    #GPIO.output(Lfwd,False)
    #GPIO.output(Rfwd,False)
    #GPIO.output(Rbcwd,False)
    #GPIO.output(Lbcwd,False)

def MoveLeft():
    GPIO.output(Rbcwd,False)
    GPIO.output(Rfwd,True)
    GPIO.output(Lbcwd,True)
    GPIO.output(Lfwd,False)
    time.sleep(0.07) #More the delay more it will turn
    GPIO.output(Lfwd,False)
    GPIO.output(Rfwd,False)
    GPIO.output(Rbcwd,False)
    GPIO.output(Lbcwd,False)

def MoveRight():
    GPIO.output(Rbcwd,True)
    GPIO.output(Rfwd,False)
    GPIO.output(Lbcwd,False)
    GPIO.output(Lfwd,True)
    time.sleep(0.07) #More the delay more it will turn
    GPIO.output(Lfwd,False)
    GPIO.output(Rfwd,False)
    GPIO.output(Rbcwd,False)
    GPIO.output(Lbcwd,False)

def StopNCheckCoroutine(framePass):
    if framePass>=25:#25 frames passed. 13th will enter the execution

        GPIO.output(Lfwd,False)
        GPIO.output(Rfwd,False)
        GPIO.output(Rbcwd,False)
        GPIO.output(Lbcwd,False)

        time.sleep(0.03)#More the delay more it will turn

        GPIO.output(Rfwd,False)#Turn Right
        GPIO.output(Lfwd,True)
        GPIO.output(Rbcwd,True)
        GPIO.output(Lbcwd,False)

        time.sleep(0.15)#More the delay more it will turn

        GPIO.output(Lfwd,False)
        GPIO.output(Rfwd,False)
        GPIO.output(Rbcwd,False)
        GPIO.output(Lbcwd,False)
        
def StopBot():
    GPIO.output(Lfwd,False)
    GPIO.output(Rfwd,False)
    GPIO.output(Rbcwd,False)
    GPIO.output(Lbcwd,False)

        
def CleanUp():
    GPIO.cleanup()