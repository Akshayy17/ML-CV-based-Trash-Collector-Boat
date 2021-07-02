
import RPi.GPIO as GPIO
import os #added so we can shut down OK
import time #import time module

Lfwd, Lbcwd = 15, 13 #L_Enable = 16
Rfwd, Rbcwd = 11, 7 #R_Enable = 18

def init_gpio():
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(Rbcwd,GPIO.OUT)
    GPIO.setup(Rfwd,GPIO.OUT)
    GPIO.setup(Lbcwd,GPIO.OUT)
    GPIO.setup(Lfwd,GPIO.OUT)


def Move():
    GPIO.output(Rfwd,True)
    GPIO.output(Lfwd,True)
    GPIO.output(Rbcwd,False)
    GPIO.output(Lbcwd,False)

def StopNCheckCoroutine(framePass):
    if framePass>=12:#12 frames passed. 13th will enter the execution

        GPIO.output(Rfwd,False)#Turn Right
        GPIO.output(Lfwd,True)
        GPIO.output(Rbcwd,True)
        GPIO.output(Lbcwd,False)

        time.sleep(0.1)#More the delay more it will turn


'''
class distAndCoordinates:
    def __init__(self, dist, co_Ord):
        self.dist = dist
        self.co_Ord = co_Ord

    def __repr__(self):
        return repr((self.dist, self.co_Ord))


            i=0 #For only one detections
            # Write results
            for *xyxy, conf, cls in reversed(det): #loop for multiple object box drawing in a frame
                if i<=0:  # Add bbox to image
                    i+=1
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)
                    #print(xyxy[0], xyxy[1])
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1)  # normalized xywh
                    im0 = cv2.circle(im0, (int(xywh[0]), int(xywh[1])), 5, (0,0,255), -1)#Circle at centre of object
                    im0 = cv2.circle(im0, (int(im0.shape[1]/2), int(xywh[1])), 5, (51,153,255), -1)#Circle at hypotenus opposite vertice

'''
