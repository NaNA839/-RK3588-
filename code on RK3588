import cv2
import numpy as np
import time
import threading
import serial
import asyncio
import base64
import requests
import json
import re
from bleak import BleakClient, BleakScanner
from rknnlite.api import RKNNLite

# ==============================================================================
# 配置参数区
# ==============================================================================
API_KEY = "sk-ws-H.RXYMYHE.Gy1q.MEQCIAVzk7OEdEBRnQpaNdKfYtHER7lcp6iKT7JiLob-fw-HAiADc5NLuBTXHdjRZ3_ENBDf5MOMuj7-u8W0L6BJmIjpsQ"
URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"

MODEL_PATH = "/home/cat/fire_detect/bestfire_fp.rknn"
IMG_SIZE = 640
CONF_THRES = 0.25
CLASSES = ["danger", "gas stove", "knob_OFF", "knob_ON"]

SERIAL_PORT = '/dev/ttyUSB2'  
BAUD_RATE = 115200            
TARGET_PHONE = "18132108008"
DEVICE_NAME = "Genshin Impact"
TX_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8" 
CAMERA_DEV = '/dev/video20'

THRESHOLDS = {
    "MQ2": {"normal": 15, "warning": 25}, 
    "MQ4": {"normal": 20, "warning": 50}, 
    "MQ7": {"normal": 50,   "warning": 10}   
}

# ==============================================================================
# 全局状态管理（异步解耦容器）
# ==============================================================================
class SystemState:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None
        
        # 气体传感器状态
        self.gas_data = {"MQ2": 0, "MQ4": 0, "MQ7": 0}
        self.gas_status = {"MQ2": "normal", "MQ4": "normal", "MQ7": "normal"}
        
        # AI 复核及判定
        self.flame_dangerous = False
        self.flame_normal = False
        self.person_present = True
        self.person_last_seen = time.time()
        self.last_alarm_time = 0
        
        # 【关键异步同步变量】专门存放后台 NPU 计算出来的最新框信息，供主线 30FPS 实时绘制
        self.latest_boxes = []  # 格式: [(x1, y1, x2, y2, label, score), ...]
        
        # 联动云端的事件通知机制
        self.cloud_trigger_event = threading.Event()

state = SystemState()

def evaluate_gas_levels(mq2, mq4, mq7):
    status = {}
    values = {"MQ2": mq2, "MQ4": mq4, "MQ7": mq7}
    for key in ["MQ2", "MQ4", "MQ7"]:
        val = values[key]
        if val >= THRESHOLDS[key]["warning"]: status[key] = "alarm"
        elif val >= THRESHOLDS[key]["normal"]: status[key] = "warning"
        else: status[key] = "normal"
    return status

def preprocess(frame):
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.expand_dims(img.astype(np.uint8), 0)

def xywh2xyxy(x):
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

# ==============================================================================
# 线程 1：主采集与显示线程 (全速 30 FPS 彻底解决卡顿)
# ==============================================================================
def camera_capture_and_show_thread():
    print("✅ [Camera] 正在以 30FPS 硬解格式拉起罗技 C920 监控流...")
    cap = cv2.VideoCapture(CAMERA_DEV, cv2.CAP_V4L2)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while True:
        try:
            if not cap.isOpened():
                cap.open(CAMERA_DEV, cv2.CAP_V4L2)
                time.sleep(1); continue
            
            ret, frame = cap.read()
            if not ret: 
                time.sleep(0.01); continue

            # 1. 迅速将高清原始帧存入内存，供后台 NPU 和云端 AI 随时抓取最新一帧
            with state.lock:
                state.frame = frame.copy()
                current_boxes = state.latest_boxes.copy() # 获取副线程刚刚算好的最新框层

            # 2. 核心异步图层叠加：用当前主线流利的 frame 叠加副线跳动的 boxes
            for x1, y1, x2, y2, label, score in current_boxes:
                color = (0, 0, 255) if label in ["danger", "gas stove"] else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {score:.2f}", (x1, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # 3. 极速显示窗口，由于没有 Sleep 阻塞，这里将保持 30FPS 的极致丝滑
            cv2.imshow("RK3588 Kitchen Monitor (Smooth Mode)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            
        except Exception as e:
            print(f"[Camera UI Loop Error] {e}")
            time.sleep(1)
            
    cap.release()
    cv2.destroyAllWindows()

# ==============================================================================
# 线程 1-Sub：后台 NPU 独立计算节拍器 (严格锁定 2 FPS，完全不拖累 UI)
# ==============================================================================
def rknn_inference_loop():
    print("⏳ [RKNN] 正在加载本地 NPU 模型...")
    rknn = RKNNLite()
    if rknn.load_rknn(MODEL_PATH) != 0 or rknn.init_runtime() != 0:
        print("❌ [RKNN] 初始化失败，请核准模型或驱动状态！"); return
    print("✅ [RKNN] 边缘加速计算引擎加载成功")

    while True:
        start_time = time.time()
        try:
            img_to_process = None
            with state.lock:
                if state.frame is not None:
                    img_to_process = state.frame.copy() # 从主内存抓取最新的那一帧
            
            if img_to_process is None:
                time.sleep(0.1); continue

            # 在后台做矩阵预处理与 NPU 推理
            h, w = img_to_process.shape[:2]
            inp = preprocess(img_to_process)
            outputs = rknn.inference(inputs=[inp], data_format=["nhwc"])
            
            new_boxes_layer = []
            suspected_flame = False
            
            if outputs is not None:
                out = outputs[0][0]
                if out.shape[0] < out.shape[1]: out = out.T
                
                boxes = out[:, :4]
                scores = out[:, 4:]
                cls_ids = np.argmax(scores, axis=1)
                confs = np.max(scores, axis=1)
                
                mask = confs > CONF_THRES
                boxes = boxes[mask]
                confs = confs[mask]
                cls_ids = cls_ids[mask]
                
                if len(boxes) > 0:
                    boxes = xywh2xyxy(boxes)
                    if boxes.max() <= 1.5: boxes *= IMG_SIZE
                    
                    for box, score, cls_id in zip(boxes, confs, cls_ids):
                        x1, y1, x2, y2 = map(int, box)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        label = CLASSES[int(cls_id)]
                        
                        if label in ["danger", "gas stove"]:
                            suspected_flame = True
                            
                        # 压入准备送往主 UI 线程的临时图层容器
                        new_boxes_layer.append((x1, y1, x2, y2, label, score))

            # 计算完成，用互斥锁无缝更新至全局，供主线 30FPS 刷新显示
            with state.lock:
                state.latest_boxes = new_boxes_layer
            
            if suspected_flame:
                state.cloud_trigger_event.set()

            # 严格控制后台 NPU 推理周期为 0.5s（即 2 FPS 节拍）
            elapsed = time.time() - start_time
            sleep_time = max(0.01, 0.5 - elapsed)
            time.sleep(sleep_time)
            
        except Exception as e:
            print(f"[RKNN Background Error] {e}")
            time.sleep(1)
            
    rknn.release()

# ==============================================================================
# 线程 2：BLE 通信层 (无修改)
# ==============================================================================
def ble_notification_handler(sender, data):
    try:
        msg = data.decode('utf-8', errors='ignore').strip()
        mq2_match = re.search(r'MQ2\s*[:=]\s*(\d+)', msg, re.IGNORECASE)
        mq4_match = re.search(r'MQ4\s*[:=]\s*(\d+)', msg, re.IGNORECASE)
        mq7_match = re.search(r'MQ7\s*[:=]\s*(\d+)', msg, re.IGNORECASE)
        
        mq2 = int(mq2_match.group(1)) if mq2_match else 0
        mq4 = int(mq4_match.group(1)) if mq4_match else 0
        mq7 = int(mq7_match.group(1)) if mq7_match else 0
        
        status = evaluate_gas_levels(mq2, mq4, mq7)
        with state.lock:
            state.gas_data = {"MQ2": mq2, "MQ4": mq4, "MQ7": mq7}
            state.gas_status = status
    except Exception:
        pass

def ble_disconnected_callback(client):
    print("⚠️ [BLE] 状态: 已断开连接，准备自动重连...")

async def ble_async_loop():
    while True:
        try:
            device = await BleakScanner.find_device_by_name(DEVICE_NAME, timeout=5.0)
            if device:
                async with BleakClient(device, disconnected_callback=ble_disconnected_callback) as client:
                    await client.start_notify(TX_UUID, ble_notification_handler)
                    print(f"🔗 [BLE] 状态: 已成功连接 ESP32 设备 ({device.address})")
                    while client.is_connected: await asyncio.sleep(2)
        except Exception: pass
        await asyncio.sleep(3)

def ble_thread():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(ble_async_loop())

# ==============================================================================
# 线程 3：Qwen-VL 云端 AI (无修改)
# ==============================================================================
def ai_thread():
    prompt_text = "你是厨房安全监控AI。请观察图片并判断情况。仅返回 JSON：\n{\"flame_detected\": true/false, \"is_dangerous_flame\": true/false, \"person_present\": true/false}"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    last_cloud_call = 0
    
    while True:
        try:
            triggered = state.cloud_trigger_event.wait(timeout=10.0)
            state.cloud_trigger_event.clear()
            
            now = time.time()
            if now - last_cloud_call < 5.0:
                time.sleep(5.0 - (now - last_cloud_call))
            last_cloud_call = time.time()
            
            img = None
            with state.lock:
                if state.frame is not None: img = state.frame.copy()
            if img is None: continue

            _, jpg = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 60])
            img_base64 = "data:image/jpeg;base64," + base64.b64encode(jpg).decode("utf-8")
            
            payload = {
                "model": "qwen-vl-plus",
                "input": {"messages": [{"role": "user", "content": [{"text": prompt_text}, {"image": img_base64}]}]}
            }
            res = requests.post(URL, json=payload, headers=headers, timeout=15)
            if res.status_code == 200:
                data = res.json()
                ai_text = data['output']['choices'][0]['message']['content'][0]['text']
                match = re.search(r'\{.*\}', ai_text, re.DOTALL)
                if match:
                    ai_result = json.loads(match.group(0))
                    with state.lock:
                        state.flame_dangerous = ai_result.get("is_dangerous_flame", False)
                        state.flame_normal = ai_result.get("flame_detected", False) and not state.flame_dangerous
                        person = ai_result.get("person_present", True)
                        state.person_present = person
                        if person: state.person_last_seen = time.time()
                    print(f"🤖 [API] 云端复核成功 | 危险火:{state.flame_dangerous} | 正常火:{state.flame_normal} | 有人:{state.person_present}")
        except Exception as e:
            print(f"❌ [API] 线程异常 | {e}")

# ==============================================================================
# 线程 4：GSM 报警控制与逻辑判断 (无修改)
# ==============================================================================
def send_at(ser, command, expected="OK", timeout=3):
    ser.write((command + '\r\n').encode('utf-8'))
    start_time = time.time()
    response = ""
    while (time.time() - start_time) < timeout:
        if ser.in_waiting > 0:
            response += ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
            if expected in response: return True
            if "ERROR" in response: return False
        time.sleep(0.1)
    return False

def execute_alarm(reason):
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        send_at(ser, "AT"); send_at(ser, "AT+CMGF=1") 
        print(f"\n🚨 [ALARM TRIGGERED] 触发紧急报警!!\n原因: {reason}\n")
        ser.write((f'AT+CMGS="{TARGET_PHONE}"\r\n').encode('utf-8'))
        time.sleep(0.5); ser.write(f"Kitchen Guard Alert: {reason}".encode('utf-8'))
        time.sleep(0.5); ser.write(bytes([26])); time.sleep(3) 
        if send_at(ser, f"ATD{TARGET_PHONE};", "OK"):
            time.sleep(15); send_at(ser, "ATH") 
        ser.close()
    except Exception as e:
        print(f"❌ 报警通信模块异常: {e}")

def gsm_alarm_thread():
    debounce_counter = 0 
    while True:
        time.sleep(1)
        now = time.time()
        if now - state.last_alarm_time < 60:
            debounce_counter = 0; continue 
            
        with state.lock:
            gas = state.gas_status.copy()
            flame_danger = state.flame_dangerous
            flame_norm = state.flame_normal
            person_away_time = now - state.person_last_seen
        
        alarm_count = sum(1 for v in gas.values() if v == "alarm")
        warning_count = sum(1 for v in gas.values() if v == "warning")
        rule_triggered = False
        reason = ""
        
        if alarm_count > 0:
            rule_triggered = True; reason = f"任意气体达到报警阈值: {gas}"
        elif warning_count >= 2:
            rule_triggered = True; reason = f"两类及以上气体处于预警: {gas}"
        elif flame_danger:
            rule_triggered = True; reason = "云端复核判定火焰危险失控"
        elif flame_norm and person_away_time > 1800:
            rule_triggered = True; reason = "检测到普通火焰且人员离场超时(>30分钟)"
        
        if rule_triggered:
            debounce_counter += 1
            if debounce_counter >= 5: 
                execute_alarm(reason)
                state.last_alarm_time = time.time()
                debounce_counter = 0 
        else: debounce_counter = 0 

# ==============================================================================
# 系统入口引导程序 (5核心子任务并联)
# ==============================================================================
if __name__ == '__main__':
    print("🚀 厨房立体安防一体化系统（解耦高流畅版）全面启动...")
    threads = [
        threading.Thread(target=camera_capture_and_show_thread, name="UI_Stream"), # 主图像屏显线
        threading.Thread(target=rknn_inference_loop, daemon=True, name="NPU_Engine"), # 后台NPU运算线
        threading.Thread(target=ble_thread, daemon=True),
        threading.Thread(target=ai_thread, daemon=True),
        threading.Thread(target=gsm_alarm_thread, daemon=True)
    ]
    for t in threads: t.start()
    while True: time.sleep(10)
