import cv2
import base64
import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO
import eventlet
import random
import json
import serial
import time
import argparse
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import math
from decimal import Decimal, ROUND_HALF_UP
import threading
import os
import sys
import subprocess


def _ensure_yolov7_on_path() -> bool:
    """Ensure YOLOv7 source is importable. Try env var, local third_party clone,
    and as a last resort auto-clone into third_party/yolov7.
    """
    yolo_env_path = os.getenv("YOLOV7_PATH")
    candidates = []
    if yolo_env_path:
        candidates.append(yolo_env_path)
    repo_local = os.path.join(os.path.dirname(__file__), "third_party", "yolov7")
    candidates.append(repo_local)

    for path in candidates:
        if path and os.path.isdir(path):
            if path not in sys.path:
                sys.path.insert(0, path)
            return True

    # Attempt to auto-clone
    try:
        os.makedirs(os.path.dirname(repo_local), exist_ok=True)
        subprocess.run([
            "git", "clone", "--depth", "1",
            "https://github.com/WongKinYiu/yolov7.git", repo_local
        ], check=True)
        if repo_local not in sys.path:
            sys.path.insert(0, repo_local)
        print("YOLOv7 cloned into:", repo_local)
        return True
    except Exception as e:
        print("Failed to auto-clone YOLOv7:", e)
        return False


if not _ensure_yolov7_on_path():
    raise ImportError(
        "ไม่สามารถโหลดโมดูล YOLOv7 ได้ กรุณาตั้งค่า YOLOV7_PATH หรือให้มีโฟลเดอร์ third_party/yolov7"
    )

from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")


class Process_system:
    def __init__(self):
        self.processing_enabled = False
        self.savebefore = False
        self.saveafter = False

    def toggle_processing(self):
        self.processing_enabled = not self.processing_enabled

    def get_Realsense(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        self.color_frame = aligned_frames.get_color_frame()
        self.depth_frame = aligned_frames.get_depth_frame()

        if not self.depth_frame or not self.color_frame:
            print("Error: Empty frame or frame size is zero")
            return None, None

        else:
            img = np.asanyarray(self.color_frame.get_data())
            depth_image = np.asanyarray(self.depth_frame.get_data())
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.08), cv2.COLORMAP_JET)
            
            im0 = img.copy()
            
            return True,im0,self.depth_frame


    def get_image(self):
        _, color_frame,depth_frame = self.get_Realsense()

        if color_frame is not None and color_frame.size > 0:
            image_detect = Yolo_detect.detect(color_frame,depth_frame,self.intr)

            # _, color_frame_encoded = cv2.imencode('.jpg', color_frame)
            _, image_detect_encoded = cv2.imencode('.jpg', image_detect)

            return color_frame, image_detect,  base64.b64encode(image_detect_encoded.tobytes()).decode('utf-8')
        else:
            print("Error: Empty frame or frame size is zero")
            return None, None, None

    def Communicate_web(self):
        status_frame,image_detect, image_detect_encode = self.get_image()
        socketio.emit('status', 'Camera: On')

        if self.savebefore:
            self.saveimagebefore(image_detect)
            self.savebefore = False

        if self.saveafter:
            self.saveimageafter(image_detect)
            self.saveafter = False

        
        img2 = cv2.imread('static/img/image2.jpg')
        img3 = cv2.imread('static/img/image3.jpg')
        
        _, image2 = cv2.imencode('.jpg', img2)
        _, image3 = cv2.imencode('.jpg', img3)

        image02 = base64.b64encode(image2.tobytes()).decode('utf-8')
        image03 = base64.b64encode(image3.tobytes()).decode('utf-8')

        tomato = self.read_json()
        if status_frame is not None:
            socketio.emit('image', {'image01': image_detect_encode ,'image02':image02,'image03':image03 , 'tomato': tomato})
        else:
            print("Error: Unable to process image due to empty frame")

    def read_json(self,filename="static/js/tomato.json"):
        with open(filename, encoding='utf8') as json_file:
            data = json.load(json_file)
            return data

    def saveimagebefore(self,img):   
        image_path = "static/img/image2.jpg"
        cv2.imwrite(image_path, img)
        print('Successfully before saved')
        
    def saveimageafter(self,img):
        image_path = "static/img/image3.jpg"
        cv2.imwrite(image_path, img)
        print('Successfully after saved')

    def start_camera(self):
        # โหลดไลบรารี RealSense เมื่อจำเป็น เพื่อให้รันได้แม้ยังไม่ติดตั้ง
        try:
            import pyrealsense2 as rs  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "ไม่พบ pyrealsense2 กรุณาติดตั้ง Intel RealSense SDK (pip install pyrealsense2) และเชื่อมต่อกล้อง"
            ) from e

        # realsense config
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        pipeline = rs.pipeline()
        profile = pipeline.start(config)

        self.profile = profile
        self.pipeline = pipeline

        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)

        self.intr = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
  

    def stop_camera(self):
        self.processing_enabled = False
        self.pipeline.stop()
        socketio.emit('status', 'Camera: Off')

        img = cv2.imread('static/img/image1.jpg')
        img1 = cv2.imread('static/img/image2.jpg')
        img2 = cv2.imread('static/img/image3.jpg')

        _, image1 = cv2.imencode('.jpg', img)
        _, image2 = cv2.imencode('.jpg', img1)
        _, image3 = cv2.imencode('.jpg', img2)

        image01 = base64.b64encode(image1.tobytes()).decode('utf-8')
        image02 = base64.b64encode(image2.tobytes()).decode('utf-8')
        image03 = base64.b64encode(image3.tobytes()).decode('utf-8')

        tomato = self.read_json()

        socketio.emit('image', {'image01': image01 ,'image02':image02,'image03':image03 , 'tomato': tomato})
        

    def start_processing(self):
        while True:
            if self.processing_enabled:
                self.Communicate_web()
            eventlet.sleep(0.1)

    def stop_processing(self):
        try:
            if self.pipeline:
                self.processing_enabled = False
                self.pipeline.stop()
                socketio.emit('status', 'Camera: Off')
                
        except Exception as e:
            print(f"Error stopping the pipeline: {e}")


class Yolov7_AI():
    def __init__(self):
        parser = argparse.ArgumentParser()
        # parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
        parser.add_argument('--weights', nargs='+', type=str,
                            default='tomatos-v7-3.pt', help='model.pt path(s)')
        parser.add_argument('--source', type=str, default='0',
                            help='source')  # file/folder, 0 for webcam
        parser.add_argument('--img-size', type=int,
                            default=640, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float,
                            default=0.35, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float,
                            default=0.45, help='IOU threshold for NMS')
        parser.add_argument('--device', default='',
                            help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument(
            '--view-img', action='store_true', help='display results')
        parser.add_argument('--save-txt', action='store_true',
                            help='save results to *.txt')
        parser.add_argument('--save-conf', action='store_true',
                            help='save confidences in --save-txt labels')
        parser.add_argument('--nosave', action='store_true',
                            help='do not save images/videos')
        parser.add_argument('--classes', nargs='+', type=int,
                            help='filter by class: --class 0, or --class 0 2 3')
        parser.add_argument(
            '--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true',
                            help='augmented inference')
        parser.add_argument('--update', action='store_true',
                            help='update all models')
        parser.add_argument('--project', default='runs/detect',
                            help='save results to project/name')
        parser.add_argument('--name', default='exp',
                            help='save results to project/name')
        parser.add_argument('--exist-ok', action='store_true',
                            help='existing project/name ok, do not increment')
        parser.add_argument('--no-trace', action='store_true',
                            help='don`t trace model')

        self.opt = parser.parse_args()

        print("setting yolov7")
        tt0 = time.time()
        weights, imgsz, trace = self.opt.weights, self.opt.img_size, not self.opt.no_trace

        # Initialize
        set_logging()
        device = select_device(self.opt.device)
        half = device.type != 'cpu'  # half precision only supported on CUDA
        tt1 = time.time()
        ttt = tt1 - tt0
        print("time1 = ", ttt)
        tt0 = time.time()

        # Load model
        print("weights = ", weights)
        model = attempt_load(weights, map_location=device)  # load FP32 model
        tt1 = time.time()
        ttt = tt1 - tt0
        print("time2 = ", ttt)
        tt0 = time.time()

        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
        tt1 = time.time()
        ttt = tt1 - tt0
        print("time3 = ", ttt)
        tt0 = time.time()

        if trace:
            model = TracedModel(model, device, self.opt.img_size)

        if half:
            print("half")
            model.half()  # to FP16
            print("half1")

        tt1 = time.time()
        ttt = tt1 - tt0
        print("time3.5 = ", ttt)
        tt0 = time.time()

        # Second-stage classifier
        classify = False
        if classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load(
                'weights/resnet101.pt', map_location=device)['model']).to(device).eval()

        tt1 = time.time()
        ttt = tt1 - tt0
        print("time4 = ", ttt)
        tt0 = time.time()

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        tt1 = time.time()
        ttt = tt1 - tt0
        print("time4.5 = ", ttt)
        tt0 = time.time()
        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(
                next(model.parameters())))  # run once
        self.old_img_w = self.old_img_h = imgsz
        self.old_img_b = 1

        tt1 = time.time()
        ttt = tt1 - tt0
        print("time5 = ", ttt)

        self.device = device
        self.half = half
        self.model = model
        self.colors = colors       

    def detect(self, img, depth_frame,intr):
        print("run loop detect")

        # Letterbox
        im0 = img.copy()
        im1 = img.copy()

        img = img[np.newaxis, :, :, :]

        # Stack
        img = np.stack(img, 0)

        # Convert
        # BGR to RGB, BHWC to BCHW
        img = img[..., ::-1].transpose((0, 3, 1, 2))
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

            # Warmup
        if self.device.type != 'cpu' and (
            self.old_img_b != img.shape[0] or self.old_img_h != img.shape[2] or self.old_img_w != img.shape[3]
        ):
            self.old_img_b = img.shape[0]
            self.old_img_h = img.shape[2]
            self.old_img_w = img.shape[3]
            for i in range(3):
                self.model(img, augment=self.opt.augment)[0]

            # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = self.model(img, augment=self.opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres,
                                   classes=self.opt.classes, agnostic=self.opt.agnostic_nms)
        t3 = time_synchronized()

        for i, det in enumerate(pred):  # detections per image
            time1 = int(time.time())


            # สร้างรายการเพื่อเก็บข้อมูล xyxy ของแต่ละตัว
            all_data = []

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    # s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    id = 0
                    print("len",len(det))
                    for *xyxy, conf, cls in reversed(det):

                        # Calculate depth
                        # Splitting xyxy* (measurement)
                        xmin = int(xyxy[0])
                        ymin = int(xyxy[1])
                        xmax = int(xyxy[2])
                        ymax = int(xyxy[3])

                        # Calculating measured centroid of the object (in Pixel)
                        xc = int(round(((xmax + xmin) / 2), 0))
                        yc = int(round(((ymax + ymin) / 2), 0))
                        
                        
                        
                        point = (xc, yc)

                        #calculation a,b
                        aa,bb = self.cal_ab(xmin,ymin,xmax,ymax,depth_frame,intr,point)

                        #-2. calculation Percent-#   ##puttext percent
                        #percent = self.cal_AllHSV(im0,xmin,xmax,ymin,ymax)
                        percent = self.cal_HSV(im0, point)
                       
                        a = max(aa,bb)
                        b = min(aa,bb)
                        value = (a,b)
                        

                      
                        #ตรวจสอบว่าขนาดเกินมะเขือเทศรึป่าว a:20-42 mm , b:18-30 mm
                        if a<20 or a>46:
                            # print("ข้าม ขนาด a")
                            continue

                        if b<18 or b>30:
                            # print("ข้าม ขนาด b")
                            continue





                        #-1. calculation xyz-#       ##calculation xyz ### cv2.circle ,puttext xyz ####
                        xxx,yyy,zzz = self.cal_depth(im0, depth_frame, intr, point)

                        # คำนวณตำแหน่งพิกัดโดยอ้างอิงจากแกนตำแหน่งที่ 0 ของแต่ละแกน
                        x = (float(xxx)/10) + 21
                        y = 16 - (float(yyy)/10) - 4
                        z = float(zzz)/10 - 4
                        # เอาทศนิยมแค่ 1 ตำแหน่ง
                        x = round(x, 1)
                        y = round(y, 1)
                        z = round(z, 1)

                        current_xyz = (x, y, z)


                        #ตรวจสอบว่าตรวจจับลูกเดิมรึป่าว
                        is_duplicate = self.check_and_filter(all_data,current_xyz)
                        #ตรวจสอบว่าตรวจจับลูกเดิมรึป่าว ถ้าใช่ให้ข้าม
                        if is_duplicate:
                            continue


                        id = id+1
                        check_value =  "ใช่ได้ id :"+ str(id) +", value:"+ str(value)
                        print(check_value)



                        c = int(cls)  # integer class
                        label = 'id:'+str(id) + "_con:" + f'{conf:.2f}'
                        plot_one_box(xyxy, im1, label=label,
                                     color=self.colors[int(cls)], line_thickness=2)
                        
                        coordinates_text = "("+str(x)+","+str(y)+","+str(z)+")"
                        
                        cv2.circle(im1, point, 5, (0, 255, 125))
                        cv2.putText(im1, text=coordinates_text, org=(int(point[0]), int(
                                    point[1])), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)



                       

                        if  int(percent) >=85.00:
                            result = 1
                        else :
                            result = 0
                            
                        if percent<23.19:
                            level = 1
                        elif percent>=23.19 and percent<53.12:
                            level = 2
                        elif percent>=53.12 and percent<85.00:
                            level = 3
                        elif percent>=85.00:
                            level = 4

                        percent =  "{:.1f}".format(percent)+ "%"
                        
                        # เพิ่มข้อมูลลงใน  all_data
                        all_data.append({
                            "id": id,
                            "point": point,
                            "xyz": (x, y, z),
                            "value": value,
                            "percent": percent,
                            "level": level,
                            "result": result
                        })

        
        # บันทึกข้อมูลในไฟล์ json
        self.save_to_json(all_data,'static/js/tomato.json')
        # print("all data =",all_data)

        return im1
    
    def cal_depth(self,img, depth_frame, intr, point):
        theta = 0
        dist = depth_frame.get_distance(
            int(point[0]), int(point[1]))*1000  # convert to mm

        # calculate real world coordinates
        Xtemp = dist*(point[0] - intr.ppx)/intr.fx
        Ytemp = dist*(point[1] - intr.ppy)/intr.fy
        Ztemp = dist

        Xtarget = Xtemp - 35  # 35 is RGB camera module offset from the center of the realsense
        Ytarget = -(Ztemp*math.sin(theta) + Ytemp*math.cos(theta))
        Ztarget = Ztemp*math.cos(theta) + Ytemp*math.sin(theta)

        coordinates_text = "(" + str(Decimal(str(Xtarget)).quantize(Decimal('0'), rounding=ROUND_HALF_UP)) + \
            ", " + str(Decimal(str(Ytarget)).quantize(Decimal('0'), rounding=ROUND_HALF_UP)) + \
            ", " + \
            str(Decimal(str(Ztarget)).quantize(
                Decimal('0'), rounding=ROUND_HALF_UP)) + ")"

        xxx = Decimal(str(Xtarget)).quantize(
            Decimal('0'), rounding=ROUND_HALF_UP)
        yyy = Decimal(str(Ytarget)).quantize(
            Decimal('0'), rounding=ROUND_HALF_UP)
        zzz = Decimal(str(Ztarget)).quantize(
            Decimal('0'), rounding=ROUND_HALF_UP)

        # cv2.circle(img, point, 5, (0, 255, 125))
        # cv2.putText(img, text=coordinates_text, org=(int(point[0]), int(
        #     point[1])), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

        return int(xxx), int(yyy), int(zzz)

    def cal_HSV(self, img, point):

        # ขนาดของภาพ (640x480)
        image_width = 640
        image_height = 480

        # แปลงรูปภาพเป็น HSV
        x00 = int(point[0])
        y00 = int(point[1])

        x01 = int(point[0])-2
        y01 = int(point[1])

        x02 = int(point[0])+2
        y02 = int(point[1])

        x03 = int(point[0])
        y03 = int(point[1])-2

        x04 = int(point[0])
        y04 = int(point[1])+2

        x0 =max(0,x00)
        x0 = min(640,x00)
        y0 =max(0,y00)
        y0 = min(480,y00)

        x1 =max(0,x01)
        x1 = min(640,x01)
        y1 =max(0,y01)
        y1 = min(480,y01)

        x2 =max(0,x02)
        x2 = min(640,x02)
        y2 =max(0,y02)
        y2 = min(480,y02)

        x3 =max(0,x03)
        x3 = min(640,x03)
        y3 =max(0,y03)
        y3 = min(480,y03)

        x4 =max(0,x04)
        x4 = min(640,x04)
        y4 =max(0,y04)
        y4 = min(480,y04)



        hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hue_value0 = hsv_image[y0, x0, 0]
        hue_value1 = hsv_image[y1, x1, 0]
        hue_value2 = hsv_image[y2, x2, 0]
        hue_value3 = hsv_image[y3, x3, 0]
        hue_value4 = hsv_image[y4, x4, 0]
        hue_averge = (hue_value0 + hue_value1 + hue_value2 + hue_value3 + hue_value4)/5
        print("Average Hue point:", hue_averge)
        percent1 = self.percent(hue_averge)
        text_percent1 = "{:.1f}".format(percent1) + "%"
        text_percent2 = int(percent1)

        return text_percent2

    def cal_AllHSV(self, img, xmin, xmax,ymin, ymax):
        #  ค่าMin-Max สี 1เขียว 2เหลือง 3ส้ม 4แดง
        hsvMin = np.array([5, 164, 0])
        hsvMax = np.array([18, 255, 255])

        cropped_image = img[ymin:ymax, xmin:xmax]

        # แปลงรูปภาพเป็น HSV
        hsv_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)

        # กรองสีอื่นออกจากภาพ
        mask = cv2.inRange(hsv_image, hsvMin, hsvMax)
        hsv_image_filtered = cv2.bitwise_and(hsv_image, hsv_image, mask=mask)


        # คำนวณค่าสีเฉลี่ยในทุกๆ พิกเซลที่ไม่เป็นสีดำ
        total_pixels = np.sum(mask > 0)
        if total_pixels<=0:
            total_pixels = 1
            
        average_hue = np.sum(hsv_image_filtered[:, :, 0]) / total_pixels

        # พิมพ์ค่าสีเฉลี่ย HSV
        # print("Average Hue:", int(average_hue))
        print("Average Hue all:", average_hue)

        # cv2.imshow('image',cropped_image)
        # cv2.waitKey(0)

        percent1 = self.percent(int(average_hue))
        text_percent1 =  "{:.1f}".format(percent1)+ "%"
        text_percent2 = int(percent1)


        return text_percent2


    def percent(self,hue):
        percents = (-0.178*math.pow(hue, 3)+8.662 *
                    math.pow(hue, 2)-108.4*hue+982.0)/(hue+1.036)
        return percents

    def cal_ab(self,xmin, ymin, xmax, ymax, depth_frame, intr, point):
        theta = 0
        dist = depth_frame.get_distance(
            int(point[0]), int(point[1]))*1000  # convert to mm

        # calculate real world coordinates
        Xtemp0 = dist*(xmin - intr.ppx)/intr.fx
        Xtemp1 = dist*(xmax - intr.ppx)/intr.fx

        Ytemp0 = dist*(ymin - intr.ppy)/intr.fy
        Ytemp1 = dist*(ymax - intr.ppy)/intr.fy

        Ztemp = dist

        Xtarget0 = Xtemp0 - 35  # 35 is RGB camera module offset from the center of the realsense
        Ytarget0 = -(Ztemp*math.sin(theta) + Ytemp0*math.cos(theta))

        Xtarget1 = Xtemp1 - 35  # 35 is RGB camera module offset from the center of the realsense
        Ytarget1 = -(Ztemp*math.sin(theta) + Ytemp1*math.cos(theta))

        xxx0 = Decimal(str(Xtarget0)).quantize(
            Decimal('0'), rounding=ROUND_HALF_UP)
        yyy0 = Decimal(str(Ytarget0)).quantize(
            Decimal('0'), rounding=ROUND_HALF_UP)
        xxx1 = Decimal(str(Xtarget1)).quantize(
            Decimal('0'), rounding=ROUND_HALF_UP)
        yyy1 = Decimal(str(Ytarget1)).quantize(
            Decimal('0'), rounding=ROUND_HALF_UP)

        # print("x1:"+str(xxx1)+"x0:"+str(xxx0))
        # print("y1:"+str(yyy1)+"y0:"+str(yyy0))

        a = abs(xxx1-xxx0)
        b = abs(yyy1-yyy0)
        # print("a:",a)
        # rint("b:",b)

        return int(a), int(b)

    def check_and_filter(self,all_data,current_xyz):
        is_duplicate = False
     
        for data in all_data:
            is_duplicate = False
            existing_xyz = data['xyz']
            
            #ถ้า check_point เป็น True คือซ้ำ
            check_point = self.calculate_distance(current_xyz, existing_xyz)

            if check_point:
                is_duplicate = True
                break
        print("is_duplicate = ",is_duplicate)
        return is_duplicate

    def calculate_distance(self,point1, point2):
        threshold = 2  # ค่า threshold ที่ต้องการ
        distance_x = abs(point1[0] - point2[0])
        distance_y = abs(point1[1] - point2[1])
        distance_z = abs(point1[2] - point2[2])

        div_x = round(distance_x, 3)
        div_y = round(distance_y, 3)
        div_z = round(distance_z, 3)        

        # ตรวจสอบว่าค่าห่างเกิน x y z ทั้งสามห่างกันน้อยกว่า 2 หรือไม่ หรือไม่
        if div_x < threshold and div_y < threshold and div_z < threshold:
            # print(f'div_x ={div_x},div_y ={div_y},div_z ={div_z}')
            # print(f'point1 {point1}')
            # print(f'point2 {point2}')
            return True  # ซ้ำ
           
        else:
            # print(f'div_x ={distance_x},div_y ={distance_y},div_z ={distance_z}')
            return False  # ไม่ซ้ำ
    
    def save_img_before(self,img):
        # Image path
        # print(type(img))
        img_array = np.frombuffer(img, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        image_path = "static/img/image2.jpg"
        cv2.imwrite(image_path, img)
        print('Successfully before saved')

    def save_img_after(self,img):
        # Image path
        img_array = np.frombuffer(img, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        image_path = "static/img/image3.jpg"
        cv2.imwrite(image_path, img)
        print('Successfully after saved')

    def save_to_json(self,data, filename):
        with open(filename, 'w') as json_file:
            json.dump(data, json_file, indent=2)

processor = Process_system()
Yolo_detect = Yolov7_AI()


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/admin')
def admin():
    return render_template('admin.html')


@app.route('/control')
def control():
    return render_template('control.html')


@socketio.on('start_camera')
def handle_start_camera():
    print('Received start_camera command')
    processor.processing_enabled = True  # เปิดการประมวลผล
    processor.start_camera()

  


@socketio.on('stop_camera')
def handle_stop_camera():
    print('Received stop_camera command')
    processor.processing_enabled = False  # ปิดการประมวลผล
    processor.stop_camera()


@socketio.on('savebefore')
def handle_save_before():
    processor.savebefore = True
    print("saveimage before")

@socketio.on('saveafter')
def handle_save_after():
    processor.saveafter = True
    print("saveimage after")
   


# กำหนดตัวแปรเก็บค่าการเชื่อมต่อ arduino
arduino_serial = None
# ตัวแปรสถานะการเชื่อต่อ arduino
setting_arduino = False
# ไม่อัพเดทตำแหน่งหุ่น
current_xxx = 10
current_yyy = 20
current_zzz = 0
# ตัวแปรสถานะ (state variable) เพื่อตรวจสอบว่าฟังก์ชันกำลังทำงานหรือไม่
is_function_running = False

# สร้าง Lock เพื่อป้องกัน data race condition
lock = threading.Lock()

# สร้าง Flag สำหรับบอกให้ Thread หยุดทำงาน
stop_thread = False

@socketio.on('stoprobot')
def handle_thread_stop():
    global stop_thread
    print("stop all thread")
    # ตั้งค่า Flag เพื่อให้ Thread หยุดทำงาน
    with lock:
        stop_thread = True

@socketio.on('unlock')
def handle_thread_stop():
    global stop_thread
    print("unlock thread")
    # ตั้งค่า Flag เพื่อให้ Thread หยุดทำงาน
    with lock:
        stop_thread = False
   

@socketio.on('control')
def handle_control_arduino(msg):
    global setting_arduino, arduino_serial,is_function_running
    if not setting_arduino:
        arduino_serial = serial.Serial(arduino_port, baud_rate, timeout=1)
        setting_arduino = True  
    # แปลง JSON เป็น Object
    data = json.loads(msg)
    if not is_function_running:   
        print("Move to position ",msg)
        control_thread = threading.Thread(target=Control_arduino, args=(data,))
        control_thread.start()
    else:
        print("arduino is not ready!!!")
   
@socketio.on('setrobot')
def handle_set_robot():
    global setting_arduino, arduino_serial,is_function_running
    if not setting_arduino:
        arduino_serial = serial.Serial(arduino_port, baud_rate, timeout=1)
        setting_arduino = True  
    if not is_function_running:   
        print("set Robot")
        Set_Robot = threading.Thread(target=SetRobot, args=())
        Set_Robot.start()
    else:
        print("arduino is not ready!!!")

def SetRobot():
    print("setrobot")
    
    global is_function_running,stop_thread

    if not is_function_running:  
         # ใช้หยุดการทำงานของ Thread
        with lock:
            if stop_thread:
                print("Thread is stop")
                return 
        print("set Robot")
        # sethome
        data = {'status':'h','x':'0','y':'0','z':'0'}
        print("Move to position ",data)
        SetRobot_thread = threading.Thread(target=Control_arduino, args=(data,))
        SetRobot_thread.start()
        # รอ thread ทำงานเสร็จสิ้น
        SetRobot_thread.join()
         # ใช้หยุดการทำงานของ Thread
        with lock:
            if stop_thread:
                print("Thread is stop")
                return
        # setposition
        data = {'status':'o','x':'10','y':'20','z':'0'}
        print("Move to position ",data)
        SetRobot_thread = threading.Thread(target=Control_arduino, args=(data,))
        SetRobot_thread.start()
        # รอ thread ทำงานเสร็จสิ้น
        SetRobot_thread.join()

@socketio.on('farming')
def handle_control_farming():
    processor.savebefore = True
    print("saveimage before")
    global setting_arduino, arduino_serial,is_function_running
    if not setting_arduino:
        arduino_serial = serial.Serial(arduino_port, baud_rate, timeout=1)
        setting_arduino = True

    if not is_function_running:   
        print("Farming ")
        control_thread = threading.Thread(target=Farming, args=())
        control_thread.start()
    else:
        print("arduino is not ready!!!")

def Control_arduino(data):
    global is_function_running,stop_thread
    print("Sending to ardiuno")

     # ตรวจสอบว่าฟังก์ชันกำลังทำงานหรือไม่
    if is_function_running:
        print("Function is already running. Skipping this call.")
        return
    
    try:
        # ใช้หยุดการทำงานของ Thread
        with lock:
            if stop_thread:
                print("Thread is stop")
                return
        is_function_running = True
        # ทำงานกับ Arduino หรืออย่างอื่น ๆ ที่คุณต้องการ
        # Extract data from the input dictionary
        status = data['status']
        x = int(data['x'])
        y = int(data['y'])
        z = int(data['z'])


        #รับค่าการเชื่อมต่อ arduino
        global arduino_serial
        #รับค่า current_xxx, current_yyy, current_zzz เพื่อคำนวณ Delay
        global current_xxx, current_yyy, current_zzz
        # คำนวณเวลาจาก ตำแหน่งที่ไกลที่สุด  0.7 s/cm
        max_value = max(abs(x-current_xxx), abs(y-current_yyy), abs(z-current_zzz))
       
        # อัพเดทค่าตำแหน่งปัจจุบัน
        current_xxx = x
        current_yyy = y
        current_zzz = z

        delay_time = abs(current_zzz-7)
        if status == 'h':
            delay_all = 10
            print("เวลาการทำงานของหุ่น(s) = 10")
            
        else:
            delay_all = (delay_time+max_value)*0.7
            print("เวลาการทำงานของหุ่น(s)) = ",delay_all)
        
        time_of_delay = 20
        delay_part = delay_all-(delay_time*0.7)/time_of_delay
        

        # ตรวจสอบว่า Arduino ถูกเชื่อมต่อ
        if arduino_serial is None or not arduino_serial.is_open:
            print("Arduino not connected. Reconnecting...")
            arduino_serial = serial.Serial(arduino_port, baud_rate, timeout=1)
            print("Arduino reconnected successfully")

        while True:
            data = arduino_serial.readline().decode('utf-8')
             # ใช้หยุดการทำงานของ Thread
            with lock:
                if stop_thread:
                    print("Thread is stop")
                    return
            if not data:
                break
            print(f'Receive from arduino : {data.strip()}')

        # เลือกฟังก์ชั่นในการสั่ง h หรือ O
        if status == "h":
            Str = "h"+"\r" + str(x)+"\r" + str(y)+"\r" + str(z)+"\r"
    
        else:
            Str = "o"+"\r" + str(x)+"\r" + str(y)+"\r" + str(z)+"\r"

        # ตำแหน่งหลังจากตัดแล้วให้ถอยกลับมา
        Str1 = "o"+"\r" + str(x)+"\r" + str(y)+"\r" + str(0)+"\r"

        # ตัดที่ลูก
        arduino_serial.write(Str.encode())
        
        if status == 'h':
            for i in range(10):
                with lock:
                    if stop_thread:
                        print("Thread is stop")
                        return
                time.sleep(1)
        else:
            for i in range(20):
                with lock:
                    if stop_thread:
                        print("Thread is stop")
                        return
            time.sleep(delay_part)
            # time.sleep(max_value*0.7)


        # จะสั่งหุ่นกลับเมื่อมันตัด
        if z >= 8:
            # delay เวลากรรไกรตัด 3 วินาที
            for i in range(3):
                with lock:
                    if stop_thread:
                        print("Thread is stop")
                        return
                time.sleep(1)
            arduino_serial.write(Str1.encode())  # worked

            # delay แกน Z ให้ถอยกลับมา
            for i in range(10):
                with lock:
                    if stop_thread:
                        print("Thread is stop")
                        return
                time.sleep((delay_time*0.7)/10)


        # ทำงานเสร็จสิ้น
        print("Robot worked")

    finally:
        # ปรับปรุงตัวแปรสถานะเมื่อฟังก์ชันทำงานเสร็จ
        is_function_running = False

def Farming():
    with open("static/js/tomato.json", encoding='utf8') as json_file:
            data_list = json.load(json_file)
            tomatos=[]
            for item in data_list:
                # ใช้หยุดการทำงานของ Thread
                global stop_thread
                with lock:
                    if stop_thread:
                        print("Thread is stop")
                        return
                if int(item['result']) == 1: 
                    # print(f"ID: {item['id']}")
                    status = "o"
                    x = int(item['xyz'][0]) + 2
                    # ขยับจากกลางลูกขึ้นไป 2 cm โดยลบ 2
                    y = int(item['xyz'][1]) + 4
                    z = int(item['xyz'][2]) + 2

                    # ทำให้แน่ใจว่าเป็นค่า + และ สั่งไปเกินที่หุ่นจะขยับไปได้คือ 38
                    xxx = min(38,abs(x))
                    yyy = min(38,abs(y))
                    zzz = min(38,abs(z))
                    if xxx>38 or yyy>38 or zzz>38:
                        print("x or y or z is over 38")

                    data = {'status':status,'x':xxx,'y':yyy,'z':zzz}
                    tomatos.append(data)
    
            print("เก็บมะเขือเทศจำนวน",len(tomatos))
            for i,tomato in enumerate(tomatos):
                # ใช้หยุดการทำงานของ Thread
                with lock:
                    if stop_thread:
                        print("Thread is stop")
                        return
                print("information of tomato",tomato)
                print("เก็บมะเขือเทศลูกที่",i+1)
                # สั่งเก็บมะเขือเทศ
                farming_thread = threading.Thread(target=Control_arduino, args=(tomato,))
                farming_thread.start()
                # รอ thread ทำงานเสร็จสิ้น
                farming_thread.join()

            # หลังจากตัดเสร็จ setposition
            data = {'status':'o','x':'10','y':'20','z':'0'}
            SetRobot_thread = threading.Thread(target=Control_arduino, args=(data,))
            SetRobot_thread.start()
            # รอ thread ทำงานเสร็จสิ้น
            SetRobot_thread.join()
    processor.saveafter = True
    print("saveimage after")

            

    



        
# ใส่ตัวแปรเพิ่มเติมเพื่อตรวจสอบสถานะการเชื่อมต่อของ clients
connected_clients = 0


@socketio.on('connect')
def handle_connect():
    global connected_clients
    connected_clients += 1
    print(f"Client connected. Total connected clients: {connected_clients}")

    if connected_clients == 1:
        processor_thread = eventlet.spawn(processor.start_processing)


@socketio.on('disconnect')
def handle_disconnect():
    eventlet.sleep(1)
    global connected_clients
    connected_clients -= 1
    print(f"Client disconnected. Total connected clients: {connected_clients}")

    if connected_clients == 0:
        processor.stop_processing()


if __name__ == '__main__':
    # ตั้งค่าพอร์ตและความเร็ว Serial จากตัวแปรแวดล้อม (มีค่าเริ่มต้น)
    arduino_port = os.getenv('ARDUINO_PORT', 'COM8')
    try:
        baud_rate = int(os.getenv('BAUD_RATE', '9600'))
    except ValueError:
        baud_rate = 9600

    # โฮสต์และพอร์ตของเว็บเซิร์ฟเวอร์
    host = os.getenv('HOST', '0.0.0.0')
    try:
        port = int(os.getenv('PORT', '5000'))
    except ValueError:
        port = 5000

    # arduino_serial = serial.Serial(arduino_port, baud_rate, timeout=1)
    socketio.run(app, host=host, port=port, debug=True)
