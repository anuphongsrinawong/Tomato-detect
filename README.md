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
4. [โครงสร้างโฟลเดอร์](#โครงสร้างโฟลเดอร์)
5. [รายละเอียดโฟลเดอร์](#รายละเอียดโฟลเดอร์)

## การติดตั้ง

- ข้อกำหนด python เวอร์ชั่น 3.11.6

ดำเนินการทำตามขั้นตอนดังนี้

1.  **ติดตั้ง YOLOv7:**

    ปฏิบัติตามคำแนะนำในการติดตั้งสำหรับ YOLOv7 ตามที่ระบุไว้ใน [YOLOv7 repository](https://github.com/WongKinYiu/yolov7)
    
      หรือ

    ```bash
     git clone https://github.com/WongKinYiu/yolov7.git
    ```

2.  **นำเข้าไฟล์โปรเจค:**
    นำเข้าไฟล์:
      ```bash
      cd yolov7
      git clone https://github.com/anuphongsrinawong/Tomato-detect.git
      ```
3.  **ติดตั้ง Library:**
    ติดตั้ง Library ที่จำเป็น:
      ```bash
      pip install -r requirements.txt
      ```

## การใช้งาน

ขั้นตอนการใช้งานมีดังนี้

1. **เริ่มการทำงานโปรแกรม**
   ```bash
     cd yolov7
     python appRS.py
   ```

2. **การแชร์เว็บไซต์ให้สามารถเข้าถึงผ่านอินเตอร์เน็ต**
   - 1.กดRun ngrok.exe
   - 2.พิมพ์คำสั่ง
   
   ```bash
     ngrok http 5000
   ```

## ตัวอย่าง
1.การติดตั้ง
![Image](https://images.unsplash.com/photo-1501780392773-287d506245a5?auto=format&fit=crop&w=1950&q=80&ixid=dW5zcGxhc2guY29tOzs7Ozs%3D)


2.การใช้งาน
3.การแชร์เว็บไซต์
   
## โครงสร้างโฟลเดอร์

```bash
Project/
- yolov7/
  - static
  - templates
  - appRS.py
  - tomatos-v7-3.pt
```

## รายละเอียดโฟลเดอร์
อธิบายขั้นตอนการติดตั้งโปรเจ็กต์ของคุณและระบุความขึ้นต้นที่ผู้ใช้จำเป็นต้องมี

