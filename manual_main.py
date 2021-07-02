import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
from numpy import random
import numpy as np
import RPi.GPIO as GPIO

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, non_max_suppression, apply_classifier, scale_coords, \
    xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

frame_array = []
IRsensorLeft, IRsensorRight = 5, 6
GPIO.setwarnings(False)

def CheckForObstacles():
    return GPIO.input(IRsensorLeft), GPIO.input(IRsensorRight)

def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))




    GPIO.setmode(GPIO.BCM)
    GPIO.setup(IRsensorLeft,GPIO.IN)
    GPIO.setup(IRsensorRight,GPIO.IN)
    
    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    weights = weights[0] if isinstance(weights, list) else weights
    suffix = Path(weights).suffix
    if suffix == '.pt':
        backend = 'pytorch'
        model = attempt_load(weights, map_location=device)  # load FP32 model
        imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
        names = model.module.names if hasattr(model, 'module') else model.names  # class names
        if half:
            model.half()  # to FP16
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

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, auto=backend == 'pytorch')
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz, auto=backend == 'pytorch')

    # Get names and colors
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if (device.type != 'cpu' and backend == 'pytorch') else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half and backend == 'pytorch' else img.float()  # uint8 to fp16/32
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

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

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
            
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f'{n} {names[int(c)]}s, '  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            cv2.rectangle(im0, (240,0), (400,20), (255,255,255), -1, cv2.LINE_AA)
            cv2.putText(im0, 'Manual Mode', (244,18), 0, 0.75, color=[100,100,100], thickness=2, lineType=cv2.LINE_AA)
            cv2.imshow('Manual Mode', im0)
            frame_array.append(im0)
            #h, w, l = im0.shape
            #size = (w, h)
            #print(size) === (640, 480)

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/W1-fp16_128.tflite', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=128, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.35, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--tfl-int8', action='store_true', help='use int8 quantized TFLite model')
    opt = parser.parse_args()
    print(opt)
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
            pOUT = 'video.avi'
            out = cv2.VideoWriter(pOUT, cv2.VideoWriter_fourcc(*'DIVX'), 15, (640,480))
            
            for i in range(len(frame_array)):
                out.write(frame_array[i])
            out.release()