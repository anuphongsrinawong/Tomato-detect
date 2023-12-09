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

1. **ติดตั้ง YOLOv7:**

   - ปฏิบัติตามคำแนะนำในการติดตั้งสำหรับ YOLOv7 ตามที่ระบุไว้ใน [YOLOv7 repository]  (https://github.com/WongKinYiu/yolov7). หรือ
  
     
    ```bash
     git clone https://github.com/WongKinYiu/yolov7.git
     ```

2. **ติดตั้ง Flask:**

   - Install Flask to create the web control interface:
     ```bash
     pip install Flask
     ```

     ```bash
     pip install Flask
     cd yolov7
     ```
     

3. **ติดตั้ง and เชื่อมต่อ Arduino:**

   - Install the necessary libraries for communication with Arduino and establish a connection between Arduino and your computer.

4. **Start the System:**
   - Start the system by running your main program and access the control interface through the Flask web application.

## Folder Structure

|- project-root/
  |- AppRS.py/ # โค้ดหลักของโปรเจ็กต์
  |- templates/ # templates
  |- static/ # static


## รายละเอียดโฟลเดอร์
