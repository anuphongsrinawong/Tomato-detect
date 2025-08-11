param(
  [string]$PythonExe = "python",
  [string]$Port = "5000",
  [string]$WebHost = "0.0.0.0"
)

Write-Host "[bootstrap] สร้าง virtualenv และติดตั้ง dependencies..."

if (-Not (Test-Path .venv)) {
  & $PythonExe -m venv .venv
}

$venvPython = Join-Path ".venv" "Scripts\python.exe"

& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r requirements.txt

Write-Host "[bootstrap] หากใช้ Intel RealSense ให้ติดตั้งเพิ่ม: pip install pyrealsense2"

$env:HOST=$WebHost
$env:PORT=$Port

Write-Host "[bootstrap] เริ่มรันแอป..."
& $venvPython appRS.py

