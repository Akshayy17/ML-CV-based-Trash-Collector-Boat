import argparse
import time
from pathlib import Path
#import picamera
import RPi.GPIO as GPIO

import cv2
import torch
import yaml
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, non_max_suppression, apply_classifier, scale_coords, \
    xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
#from _auto.automation_sub import distAndCoordinates
from _auto.automation_sub_2 import MoveLeft, MoveRight, MoveStraight, StopNCheckCoroutine, \
                                    init_gpio, StopBot
           
frame_array = []
max_det = 3 #max number of detections drawn by box.
tolerance = 30 #Between 5 to 50. Left-right tolerance. increse to MoveForward scale increment and complexity for bot decreases
_PWM = 80
#max_PWM = 70 #Ranges between 0 to 100. Always keep its value between 40 to 100. more it is more is the speed of bot.
turn_intensity = 1 #Ranges between 1 to 1.5. More this value is more is turning speed of bot
nearbyRange = 25 #If object comes in this vector distance from boat. boat goes straight for below 'delay'
delay = 0.5 #Drive till this delay to capture object inside bot

Lfwd, Lbcwd, L_Enable = 24, 23, 22
Rfwd, Rbcwd, R_Enable = 17, 27, 25
IRsensorLeft, IRsensorRight = 5, 6
GPIO.setwarnings(False)

def CheckForObstacles():
    return GPIO.input(IRsensorLeft), GPIO.input(IRsensorRight)

def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    #set_logging()
    device = 'cpu' #select_device(opt.device)
    #half = device.type != 'cpu'  # half precision only supported on CUDA
    #CleanUp()
    #init_gpio1()
    GPIO.setmode(GPIO.BCM)

    GPIO.setup(Rbcwd,GPIO.OUT)
    GPIO.setup(Rfwd,GPIO.OUT)
    GPIO.setup(Lbcwd,GPIO.OUT)
    GPIO.setup(Lfwd,GPIO.OUT)
    GPIO.setup(IRsensorLeft,GPIO.IN)
    GPIO.setup(IRsensorRight,GPIO.IN)
    GPIO.setup(L_Enable,GPIO.OUT)
    GPIO.setup(R_Enable,GPIO.OUT)
    speedLeft = GPIO.PWM(L_Enable, 100)
    speedRight = GPIO.PWM(R_Enable, 100)
    speedLeft.start(100)##### do it 0 for safe start
    speedRight.start(100)
    #L_Enable, R_Enable = 16, 18
    #GPIO.setmode(GPIO.BCM)
    
    #setup PWM control
    #GPIO.setup(L_Enable,GPIO.OUT)
    #GPIO.setup(R_Enable,GPIO.OUT)
    #speedLeft = GPIO.PWM(L_Enable, _PWM)
    #speedRight = GPIO.PWM(R_Enable, _PWM)
    #speedLeft.start(0)##### do it 0 for safe start
    #speedRight.start(0)##### same as above

    weights = weights[0] if isinstance(weights, list) else weights
    suffix = Path(weights).suffix
    if suffix == '.pt':
        backend = 'pytorch'
        model = attempt_load(weights, map_location=device)  # load FP32 model
        imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
        names = model.module.names if hasattr(model, 'module') else model.names  # class names
    else:
        import tensorflow as tf
        from tensorflow import keras

        with open('data/custom.yaml') as f:
            names = yaml.load(f, Loader=yaml.FullLoader)['names']  # class names (assume COCO)
        
        if suffix == '.pb':
            backend = 'graph_def'
            
            # https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
            # https://github.com/leimao/Frozen_Graph_TensorFlow
            def wrap_frozen_graph(graph_def, inputs, outputs):
                def _imports_graph_def():
                    tf.compat.v1.import_graph_def(graph_def, name="")

                wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
                import_graph = wrapped_import.graph
                return wrapped_import.prune(
                    tf.nest.map_structure(import_graph.as_graph_element, inputs),
                    tf.nest.map_structure(import_graph.as_graph_element, outputs))

            graph = tf.Graph()
            graph_def = graph.as_graph_def()
            graph_def.ParseFromString(open(weights, 'rb').read())
            frozen_func = wrap_frozen_graph(graph_def=graph_def, inputs="x:0", outputs="Identity:0")
            
        elif suffix == '.tflite':
            backend = 'tflite'
            # Load TFLite model and allocate tensors
            #interpreter = tf.lite.Interpreter(model_path=weights)

            import tflite_runtime.interpreter as tflite
            interpreter = tflite.Interpreter(model_path=weights)
            interpreter.allocate_tensors()

            # Get input and output tensors
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

        else:
            backend = 'saved_model'
            model = keras.models.load_model(weights)

    # Set Dataloader
    vid_path, vid_writer = None, None
    view_img = True
    cudnn.benchmark = True  # set True to speed up constant image size inference
    dataset = LoadStreams(source, img_size=imgsz, auto=backend == 'pytorch')


    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    #_ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    framePass = 0 #To pass 2 frames before starting StopCoroutine to ensure no object detected #GlitchProtection
    for path, img, im0s, vid_cap in dataset:
        #print(f'First for loop in detect.py {time.time():.4f} s')

        img = torch.from_numpy(img).to(device)
        img = img.float() #img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        # Inference
        t1 = time_synchronized()
        if backend == 'pytorch':
            pred = model(img, augment=opt.augment)[0]
        else:
            if backend == 'saved_model':
                pred = model(img.permute(0, 2, 3, 1).cpu().numpy(), training=False).numpy()
            elif backend == 'graph_def':
                pred = frozen_func(x=tf.constant(img.permute(0, 2, 3, 1).cpu().numpy())).numpy()
            elif backend == 'tflite':
                input_data = img.permute(0, 2, 3, 1).cpu().numpy()
                if opt.tfl_int8:
                    scale, zero_point = input_details[0]['quantization']
                    input_data = input_data / scale + zero_point
                    input_data = input_data.astype(np.uint8)
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                pred = interpreter.get_tensor(output_details[0]['index'])
                if opt.tfl_int8:
                    scale, zero_point = output_details[0]['quantization']
                    pred = pred.astype(np.float32)
                    pred = (pred - zero_point) * scale
            # Denormalize xywh
            pred[..., :4] *= opt.img_size
            pred = torch.tensor(pred)

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Process detections
        #for i, det in enumerate(pred):  # detections per image per source. enumerate because of number of sources
        det = pred[0] #simplified by diminishing for loop as only one source is used
        p, s, im0, frame = path[0], '%g: ' % 0, im0s[0].copy(), dataset.count

        p = Path(p)  # to Path
        save_path = str(save_dir / p.name)  # img.jpg
        txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
        s += '%gx%g ' % img.shape[2:]  # print string
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

        lft, rgt = CheckForObstacles()
        if not lft:
            im0 = cv2.rectangle(im0, (0,0), (40,480), (74,74,212), -1)#Left
        if not rgt:
            im0 = cv2.rectangle(im0, (600,0), (640,480), (74,74,212), -1)#Right

        if len(det): # no. of predictions found in a frame if any
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f'{n} {names[int(c)]}s, '  # add to string

            #scrHght, scrWid, _ = im0.shape ####In this code img0 is in 640x480 format. maybe change is in plots.py or general.py
            botmX, botmY = int(im0.shape[1]/2), int(im0.shape[0])##Screen mid-bottom
            #im0 = cv2.circle(im0, (botmX, botmY), 5, (51,153,255), -1)#Circle at bottom middle of screen
            #im0 = cv2.circle(im0, (int(im0.shape[1]/2), int(im0.shape[0]/2)), 2, (155,253,155), -1)#Circle at centre of screen

            disList = [] # or =list(). List of Vector dist between Screen mid-bottom and detected object centre
            for *xyxy, conf, cls in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1)  # normalized xywh
                dist = (int(xywh[0]-botmX)**2+(int(xywh[1])-botmY)**2)**0.5
                if(xywh[2]*xywh[3]<160000):
                    disList.append([round(dist, 3), (int(xywh[0]), int(xywh[1])), xyxy])#[vectorDist, objCentre, xyxy]
                #disList.append([round(dist, 2), int(xywh[0]), int(xywh[1]), (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])) ])# too works
                #disList.append(distAndCoordinates(round(dist, 1), (int(xywh[0]), int(xywh[1])) ) )#Too works

            disList.sort(key= lambda x: x[0])
            #disList.sort(key= lambda x: x.dist) #Sorting for distAndCoordinates class listing method
            #print(disList) #==> full list
            #print(disList[0]) ==> 1st element. i.e full list 1st detection parameters
            #print(disList[0][0]) ==> 1st element inside 1st element i.e dist
            
            i = len(disList) if len(disList)<= max_det else max_det
            for _ in range(i):
                label = f'{names[int(0)]} {conf:.2f}'#file contains class names. different index for different class
                plot_one_box(disList[_][2], im0, label=label, color = colors[0], line_thickness=2)#if label=none, no name-box will be drawn
                #im0 = cv2.line(im0, (botmX, botmY), disList[_][1], colors[0], 2)
            #print('DOne1')
            if disList != []:
                if(abs(botmX-disList[0][1][0]) > tolerance) and disList[0][0] > nearbyRange: #if any object detected Move bot there accordingly.
                    sin0 = round((botmX-disList[0][1][0]) / disList[0][0], 3) #Ranges between (-1, 1) excluding both 0 and 1.
                    sin0 = 1 if ((sin0/2)*turn_intensity)>=1 else (sin0/2)*turn_intensity #Ranges between (0, 0.5)*intensity for pwm of bot 50 to 100. Needed for this project only
                    #min_PWM = int(max_PWM * (1-sin0))

                    if(abs(botmX-disList[0][1][0]) > tolerance):
                        if sin0>0:
                            MoveLeft()
                        else:
                            MoveRight()
                    framePass = 0
                else:
                    MoveStraight()#Move Straight

                    if (disList[0][0] <= nearbyRange):# if it is in range go for it other wise object is straight but far
                        time.sleep(delay)
                    print("Move Straight")

                    framePass = 0 #To pass 2 frames before starting StopCoroutine to ensure no object detected #GlitchProtection
            else:
                framePass+=1

        else:
            #StopNCheckCoroutine(framePass)
            StopBot()
            framePass+=1
        # Print time (inference + NMS)
        print(f'{s}Done. ({t2 - t1:.3f}s)')

        # Stream results
        cv2.rectangle(im0, (260,0), (380,20), (255,255,255), -1, cv2.LINE_AA)
        cv2.putText(im0, 'Auto Mode', (263,18), 0, 0.7, color=[100,100,100], thickness=2, lineType=cv2.LINE_AA)
        cv2.imshow('Auto Mode', im0)
        frame_array.append(im0)
        #if cv2.waitKey(1) == ord('q'):
            #CleanUp()
            #cv2.destroyAllWindows()

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/W1-fp16_128.tflite', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=128, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.35, help='IOU threshold for NMS')
    #parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--tfl-int8', action='store_true', help='use int8 quantized TFLite model')
    opt = parser.parse_args()
    #print(opt)
    #check_requirements()

    with torch.no_grad():
        try:
            if opt.update:  # update all models (to fix SourceChangeWarning)
                for opt.weights in ['yolov5s.pt', 'BallsS100.pt', 'BallsS100.pt', 'BallsM.pt', 'PlasticBagsS.pt', 'PlasticBagsM.pt']:
                    detect()
                    strip_optimizer(opt.weights)
            else:
                detect()
        finally:
            GPIO.cleanup()
            
            pOUT = 'video.avi'
            out = cv2.VideoWriter(pOUT, cv2.VideoWriter_fourcc(*'DIVX'), 8, (640,480))
            
            for i in range(len(frame_array)):
                out.write(frame_array[i])
            out.release()