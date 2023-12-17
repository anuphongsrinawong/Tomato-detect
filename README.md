# โปรเจกต์ตรวจจับความแดงของมะเขือเทศ

โปรเจกต์นี้เป็นระบบตรวจจับความแดงของมะเขือเทศ (Tomato Ripeness Detection) โดยใช้ YOLOv7 (You Only Look Once version 7) เป็นโมเดลที่ใช้ในการตรวจจับวัตถุ ซึ่งในที่นี้คือการตรวจจับความแดงของมะเขือเทศเพื่อประเมินความสุกและคุณภาพของผลผลิต.

## ฟีเจอร์

1. **YOLOv7 Model:** ใช้ YOLOv7 เป็นโมเดลที่ถูกพัฒนาขึ้นเพื่อการตรวจจับวัตถุทั่วไป และได้ถูกปรับให้เหมาะสมกับการตรวจจับความแดงของมะเขือเทศในที่นี้.

2. **Flask Web Control:** ใช้ Flask เป็นเฟรมเวิร์กเพื่อสร้างหน้าควบคุมที่ให้ผู้ใช้สามารถตั้งค่าและติดตามผลการตรวจจับได้ผ่านทางเว็บไซต์.

3. **Data and Coordinates to Arduino:** ผลลัพธ์จากการตรวจจับและข้อมูลที่เกี่ยวข้อง (เช่น พิกัดของมะเขือเทศที่ตรวจจับได้) จะถูกส่งไปยัง Arduino เพื่อการประมวลผลและการปฏิบัติการต่าง ๆ ที่เกี่ยวข้องกับระบบ (เช่น การควบคุมการรดน้ำหรือการเก็บเกี่ยว).

## สารบัญ

1. [การติดตั้ง](#การติดตั้ง)
2. [การใช้งาน](#การใช้งาน)

## การติดตั้ง

- ข้อกำหนด python เวอร์ชั่น 3.11.6

ดำเนินการทำตามขั้นตอนดังนี้

1.  **ติดตั้ง YOLOv7:**

    ปฏิบัติตามคำแนะนำในการติดตั้งสำหรับ YOLOv7 ตามที่ระบุไว้ใน [YOLOv7 repository](https://github.com/WongKinYiu/yolov7)

    หรือ
```bash
git clone https://github.com/WongKinYiu/yolov7.git
```
    เข้าถึงโฟรเดอร์ yolov7 และ ติดตั้ง library ที่จำเป็น
```bash
cd yolov7
pip install -r requirements.txt
```
    ทดลองรันโปรแกรม
```bash
python detect.py
```

2.  **นำเข้าไฟล์โปรเจค:**
   
    ออกจากโฟรเดอร์ก่อนหน้า
```bash
cd ../
```
    โคลนโปรเจค
```bash
git clone https://github.com/anuphongsrinawong/Tomato-detect.git
```
    เข้าถึงโฟลเดอร์ Tomato-detect และ ติดตั้ง library ที่จำเป็น
```bash
cd Tomato-detect
pip install -r requirements.txt
```
3.  **ย้ายไฟล์จากโปรเจค Tomato-detect ไปยังโฟลเดอร์ yolov7 ตามโครงสร้าง:**
```bash
Project/ #โฟลเดอร์
    - yolov7/ #โฟลเดอร์
        - static  #โฟลเดอร์
        - templates #โฟลเดอร์
        - appRS.py #ไฟล์
         - tomatos-v7-3.pt #ไฟล์
-ngrok.exe #ไฟล์ สำหรับแชร์เว็บไซต์
```
3.  **สิ่งที่ต้องแก้ไข**
    - ตัวแปร arduino_port  ที่ไฟล์ appRS.py บรรทัด 1076 เช่น
```bash
arduino_port = 'COM8'  # แก้ตามพอร์ตที่ Arduino ต่อ
```


## การใช้งาน

ขั้นตอนการใช้งานมีดังนี้

1. **เริ่มการทำงานโปรแกรม**
```bash
cd yolov7
python appRS.py
```

2. **การเปิดเว็บไซต์**
```bash
http://127.0.0.1:5000/
```
3. **การสั่งงาน**
    - หน้า admin เป็นหน้าควบคุมหลัก
         - กดปุ่ม Start Camera เพื่อเปิดกล้อง
         - กดปุ่ม Stop Camera เพื่อปิดกล้อง
         - กดปุ่ม Set Robot เพื่อตั้งค่าตำแหน่งหุ่นให้พร้อมใช้งาน
         - กดปุ่ม Stop เพื่อปิดกล้อง
         - กดปุ่ม Cut เพื่อสั่งตัดตามจำนวนลูกที่สุกจากข้อมูลในตาราง
         - กดปุ่ม lock Cut เพื่อล็อกคำสั่งการตัดลูกต่อๆไป และควรกด Set Robot เพื่อกลับตำแหน่งเริ่มต้นของกล้อง
         - กดปุ่ม unlock Cut เพื่อยกเลิกการสั่งตัด
         - กดปุ่ม Saveimage before เพื่อบันทึกรูปตรงกลาง
         - กดปุ่ม Saveimage after เพื่อบันทึกรูปทางขวา
    
    - หน้า Control เป็นหน้าควบคุมตำแหน่งหุ่น
         - กดปุ่ม Sethome เพื่อกลับมาตำแหน่ง 0 ทั้งสามเเกน
         - กดปุ่ม SetPosition เพื่อกลับเริ่มต้นของกล้อง
         - กรอกตำแหน่ง ทั้ง 3 แกน และกดปุ่ม Submit เพื่อไปยังตำแหน่งที่กรอก
   





