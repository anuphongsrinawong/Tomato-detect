# โปรเจกต์ตรวจจับความแดงของมะเขือเทศ

โปรเจกต์นี้เป็นระบบตรวจจับความแดงของมะเขือเทศ (Tomato Ripeness Detection) โดยใช้ YOLOv7 (You Only Look Once version 7) เป็นโมเดลที่ใช้ในการตรวจจับวัตถุ ซึ่งในที่นี้คือการตรวจจับความแดงของมะเขือเทศเพื่อประเมินความสุกและคุณภาพของผลผลิต.

## สารบัญ

1. [การติดตั้ง](#การติดตั้ง)
2. [การใช้งาน](#การใช้งาน)
3. [ตัวอย่าง](#ตัวอย่าง)
4. [รายละเอียดเพิ่มเติม](#รายละเอียดเพิ่มเติม)
5. [การเป็นนักพัฒนา](#การเป็นนักพัฒนา)
6. [การให้สิ่งที่สามารถเพิ่มเติม](#การให้สิ่งที่สามารถเพิ่มเติม)
7. [การร่วมมือ](#การร่วมมือ)
8. [ลิขสิทธิ์และใบอนุญาต](#ลิขสิทธิ์และใบอนุญาต)

## การติดตั้ง

อธิบายขั้นตอนการติดตั้งโปรเจ็กต์ของคุณ และระบุความขึ้นต้นที่ผู้ใช้จำเป็นต้องมี

## Key Features

1. **YOLOv7 Model:** Utilizes YOLOv7, a state-of-the-art object detection model, adapted for the specific task of detecting the ripeness of tomatoes.

2. **Flask Web Control:** Implements a web control interface using Flask, allowing users to configure and monitor detection results through a web interface.

3. **Data and Coordinates to Arduino:** Sends the detection results and relevant data (such as the coordinates of detected tomatoes) to an Arduino for additional processing and control operations.

## Installation and Usage

1. **Install YOLOv7:**

   - Follow the installation instructions for YOLOv7 as specified in the [YOLOv7 repository](https://github.com/WongKinYiu/yolov7).

2. **Install Flask:**

   - Install Flask to create the web control interface:
     ```bash
     pip install Flask
     ```

3. **Install and Connect Arduino:**

   - Install the necessary libraries for communication with Arduino and establish a connection between Arduino and your computer.

4. **Start the System:**
   - Start the system by running your main program and access the control interface through the Flask web application.

## Folder Structure

|- yolov7_tomato_detection/
|- yolo/ # Folder containing the YOLOv7 model
|- flask_app/ # Folder containing the Flask application
|- arduino_code/ # Folder containing the Arduino code
|- README.md # README file for the project
