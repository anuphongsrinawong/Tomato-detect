# โปรเจกต์ตรวจจับความแดงของมะเขือเทศ

โปรเจกต์นี้เป็นระบบตรวจจับความแดงของมะเขือเทศ (Tomato Ripeness Detection) โดยใช้ YOLOv7 (You Only Look Once version 7) เป็นโมเดลที่ใช้ในการตรวจจับวัตถุ ซึ่งในที่นี้คือการตรวจจับความแดงของมะเขือเทศเพื่อประเมินความสุกและคุณภาพของผลผลิต.

## ฟีเจอร์

1. **YOLOv7 Model:** ใช้ YOLOv7 เป็นโมเดลที่ถูกพัฒนาขึ้นเพื่อการตรวจจับวัตถุทั่วไป และได้ถูกปรับให้เหมาะสมกับการตรวจจับความแดงของมะเขือเทศในที่นี้.

2. **Flask Web Control:** ใช้ Flask เป็นเฟรมเวิร์กเพื่อสร้างหน้าควบคุมที่ให้ผู้ใช้สามารถตั้งค่าและติดตามผลการตรวจจับได้ผ่านทางเว็บไซต์.

3. **Data and Coordinates to Arduino:** ผลลัพธ์จากการตรวจจับและข้อมูลที่เกี่ยวข้อง (เช่น พิกัดของมะเขือเทศที่ตรวจจับได้) จะถูกส่งไปยัง Arduino เพื่อการประมวลผลและการปฏิบัติการต่าง ๆ ที่เกี่ยวข้องกับระบบ (เช่น การควบคุมการรดน้ำหรือการเก็บเกี่ยว).

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

อธิบายขั้นตอนการติดตั้งโปรเจ็กต์ของคุณและระบุความขึ้นต้นที่ผู้ใช้จำเป็นต้องมี

### โครงสร้างโฟลเดอร์

```bash
Project/
- yolov7/
  - static
  - templates
  - appRS.py
```

1.  **ติดตั้ง YOLOv7:**

    - ปฏิบัติตามคำแนะนำในการติดตั้งสำหรับ YOLOv7 ตามที่ระบุไว้ใน [YOLOv7 repository](https://github.com/WongKinYiu/yolov7) หรือ

    ```bash
     git clone https://github.com/WongKinYiu/yolov7.git
    ```

2.  **นำเข้าไฟล์โปรเจค:**
    - เข้าไปยังโฟลเดอร์ และ นำเข้าไฟล์:
      ```bash
      cd yolov7
      git clone https://github.com/anuphongsrinawong/Tomato-detect.git
      ```
3.  **ติดตั้ง Library:**
4.  - ติดตั้ง:
      ```bash
      pip install opencv-python==4.5.3
      Flask 2.3.3
      Flask-SocketIO 5.3.6
      pyrealsense2 2.53.1.4623
      serial 0.0.97
      torch 1.13.1
      numpy 1.24.1
           ```

5.  **ติดตั้ง and เชื่อมต่อ Arduino:**

    - Install the necessary libraries for communication with Arduino and establish a connection between Arduino and your computer.

6.  **Start the System:**
    - Start the system by running your main program and access the control interface through the Flask web application.

## การใช้งาน

อธิบายขั้นตอนการติดตั้งโปรเจ็กต์ของคุณและระบุความขึ้นต้นที่ผู้ใช้จำเป็นต้องมี

1. **เริ่มการทำงานโปรแกรม**
   ```bash
     cd yolov7
     python appRS.py
   ```

## รายละเอียดโฟลเดอร์
