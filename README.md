# Tomato-detect (YOLOv7 + Flask + Intel RealSense + Arduino)

ระบบตรวจจับความแดง/ความสุกของมะเขือเทศด้วย YOLOv7 พร้อมเว็บ UI ควบคุม และส่งพิกัดไปยัง Arduino สำหรับควบคุมหุ่นยนต์เก็บเกี่ยว ใช้งานได้ “พกพา” ไม่ต้องย้ายไฟล์เข้าโฟลเดอร์ YOLOv7 อีกต่อไป แอปจะดึงซอร์ส YOLOv7 ให้อัตโนมัติเมื่อรันครั้งแรก

## ฟีเจอร์

- YOLOv7 inference แบบเรียลไทม์จากกล้อง Intel RealSense (RGB+D)
- เว็บแอป Flask + Socket.IO สำหรับดูภาพ, ตารางผลตรวจ, และสั่งงานหุ่น
- ส่งพิกัด (x, y, z) และสั่งโหมด sethome/setposition/เก็บผลผลิต ไปยัง Arduino ผ่าน Serial
- ตั้งค่าด้วยตัวแปรแวดล้อม: พอร์ต/บอดเรต Arduino, โฮสต์/พอร์ตเว็บเซิร์ฟเวอร์, ตำแหน่ง YOLOv7

## โครงสร้างโปรเจกต์ (สำคัญ)

```
Tomato-detect/
  appRS.py
  tomatos-v7-3.pt
  static/
  templates/
  requirements.txt
  README.md
```

ไม่ต้องมีโฟลเดอร์ `yolov7/` ข้างเคียงอีกแล้ว ตัวโปรแกรมจะคลอน YOLOv7 มาไว้ที่ `third_party/yolov7` โดยอัตโนมัติเมื่อรันครั้งแรก (หรือกำหนดเองด้วย `YOLOV7_PATH`)

## ข้อกำหนดระบบ

- Python 3.11.x แนะนำ 3.11.6
- กล้อง Intel RealSense (เช่น D435/D455) และไดร์เวอร์/SDK ติดตั้งพร้อมใช้งาน
- CUDA (ถ้ามี GPU) เพื่อความเร็ว, หากไม่มีจะรันโหมด CPU ได้

## การติดตั้งแบบรวดเร็ว

```bash
git clone https://github.com/anuphongsrinawong/Tomato-detect.git
cd Tomato-detect
python -m venv .venv
.venv\Scripts\activate  # Windows
# หรือ source .venv/bin/activate  # Linux/macOS
pip install --upgrade pip
pip install -r requirements.txt

# ติดตั้ง pyrealsense2 เฉพาะหากใช้กล้อง Intel RealSense
pip install pyrealsense2
```

ถ้าต้องระบุที่อยู่ซอร์ส YOLOv7 เอง ให้ตั้งค่าตัวแปรแวดล้อม `YOLOV7_PATH` ชี้ไปยังโฟลเดอร์ที่มีไฟล์ YOLOv7 (เช่นที่โคลนไว้ก่อนหน้า)

## ตัวแปรแวดล้อมที่รองรับ

- `YOLOV7_PATH` เส้นทางโฟลเดอร์ YOLOv7 (ไม่จำเป็น หากให้แอปคลอนอัตโนมัติได้)
- `ARDUINO_PORT` พอร์ต Serial ของ Arduino (เช่น Windows: COM8, Linux: /dev/ttyACM0)
- `BAUD_RATE` ความเร็วบอดเรต Serial (ดีฟอลต์ 9600)
- `HOST` โฮสต์ของเว็บเซิร์ฟเวอร์ (ดีฟอลต์ 0.0.0.0)
- `PORT` พอร์ตของเว็บเซิร์ฟเวอร์ (ดีฟอลต์ 5000)

ตัวอย่างบน Windows PowerShell:

```powershell
$env:ARDUINO_PORT="COM8"
$env:BAUD_RATE="9600"
$env:HOST="0.0.0.0"
$env:PORT="5000"
python appRS.py
```

## การใช้งาน

1) เปิดเซิร์ฟเวอร์

```bash
python appRS.py
```

2) เปิดเว็บเบราว์เซอร์

```
http://127.0.0.1:5000/
```

3) หน้าจอหลัก

- หน้า Home: ดูภาพสตรีม/ผลตรวจ
- หน้า Control: ป้อนพิกัดและสั่งให้หุ่นไปยังตำแหน่ง
- หน้า Admin: ควบคุม Start/Stop กล้อง, Set Robot, Farming, บันทึกรูปก่อน/หลัง, ล็อก/ปลดล็อกคำสั่ง

หมายเหตุ: หากยังไม่ติดตั้ง `pyrealsense2` หรือไม่ได้เสียบกล้อง Intel RealSense แอปจะฟ้องและหยุด ให้ติดตั้งและเชื่อมต่อก่อน

## โมเดล

ไฟล์โมเดล `tomatos-v7-3.pt` อยู่ในรากโปรเจกต์ ใช้เป็นค่าเริ่มต้น คุณสามารถเปลี่ยนผ่านอาร์กิวเมนต์ `--weights` ขณะรันได้ด้วย

```bash
python appRS.py --weights path/to/your_model.pt
```

## ปัญหาที่พบบ่อย

- ImportError YOLOv7: ตั้ง `YOLOV7_PATH` ให้ถูก หรือให้แอปคลอนอัตโนมัติ (ต้องมี git)
- ImportError pyrealsense2: ติดตั้งด้วย `pip install pyrealsense2` และติดตั้ง SDK/ไดร์เวอร์
- เปิดพอร์ต COM ไม่ได้: ตรวจสอบพอร์ต `ARDUINO_PORT` และสิทธิ์การใช้งานอุปกรณ์

## การพัฒนา/ดีพลอยต่อ

แนะนำให้ใช้ Python virtualenv แยกโปรเจกต์, pin เวอร์ชันใน `requirements.txt`, และใช้ GPU/CUDA หากต้องการความเร็วสูงสุด

