# main.py - 天花板定位系统主服务器（支持多建筑物 + FLANN + RANSAC + 时序检查）
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import json
import time
import os
from pathlib import Path
from collections import deque

# ==================== 创建FastAPI应用 ====================
app = FastAPI(
    title="天花板定位系统",
    description="基于计算机视觉的室内定位系统，使用手机摄像头拍摄天花板图像",
    version="2.1"  # 版本号增加
)

# ==================== 允许网页访问（CORS设置） ====================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 开发时允许所有来源，生产环境要限制
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有HTTP方法
    allow_headers=["*"],  # 允许所有HTTP头
)

# ==================== 挂载静态文件 ====================
app.mount("/static", StaticFiles(directory="static"), name="static")

# ==================== 全局变量 ====================
# 数据库结构：{ building_name: { location_id: { keypoints, descriptors, ... } } }
database = {}

# ORB特征检测器（与建库时参数一致）
orb = cv2.ORB_create(nfeatures=1500)

# FLANN参数（针对ORB二进制描述符）
FLANN_INDEX_LSH = 6
index_params = dict(
    algorithm=FLANN_INDEX_LSH,
    table_number=6,
    key_size=12,
    multi_probe_level=1
)
search_params = dict(checks=50)

# ==================== 新增：时序检查相关全局变量 ====================
# 会话历史存储结构：key = session_id 或 client_ip, value = deque 每个元素为 (timestamp, building, floor, number)
session_history = {}
HISTORY_MAXLEN = 5      # 每个会话最多保留最近5次记录
TIME_WINDOW = 60        # 300秒内认为连续移动（超过此时间视为新会话）
# =============================================================
# ==================== 新增：时序检查相关全局变量 ====================
# 会话历史存储结构：key = session_id 或 client_ip, value = deque 每个元素为 (timestamp, building, floor, number)
session_history = {}
HISTORY_MAXLEN = 5      # 每个会话最多保留最近5次记录
TIME_WINDOW = 300       # 300秒内认为连续移动（超过此时间视为新会话）

# 🎯 核心升级：拓扑图连接字典 (只记录合法相连的交叉路口)
# 格式: "建筑物_走廊名_步数"
VALID_JUNCTIONS = {
    "Earl Mountbatten Building_CorridorA_0": ["Earl Mountbatten Building_CorridorB_0"],
    "Earl Mountbatten Building_CorridorB_0": ["Earl Mountbatten Building_CorridorA_0"],
    "Colin Maclaurin Building_CorridorC_0": ["Colin Maclaurin Building_CorridorD_0"],
    "Colin Maclaurin Building_CorridorD_0": ["Colin Maclaurin Building_CorridorC_0"]
}
# =============================================================

def deserialize_keypoint(kp_data):
    """将字典转换为OpenCV KeyPoint对象"""
    kp = cv2.KeyPoint()
    kp.pt = (kp_data["pt"][0], kp_data["pt"][1])
    kp.size = kp_data["size"]
    kp.angle = kp_data["angle"]
    kp.response = kp_data["response"]
    return kp


def match_with_flann_ransac(query_desc, query_kp, db_desc, db_kp):
    """
    使用FLANN匹配 + 适配ORB的距离测试 + RANSAC几何验证
    """
    if query_desc is None or db_desc is None or len(query_desc) < 2 or len(db_desc) < 2:
        return 0, 0, 0.0

    # FLANN匹配器
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    knn_matches = flann.knnMatch(query_desc, db_desc, k=2)

    good_matches = []
    for match_pair in knn_matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.85 * n.distance or m.distance < 50:
                good_matches.append(m)
        elif len(match_pair) == 1:
            if match_pair[0].distance < 50:
                good_matches.append(match_pair[0])

    total_matches = len(good_matches)
    if total_matches < 10:  # 匹配点太少，无法进行RANSAC
        return 0, total_matches, 0.0

    # 提取匹配点坐标
    src_pts = np.float32([query_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([db_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # RANSAC计算单应性矩阵
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if mask is None:
        return 0, total_matches, 0.0

    inliers = np.sum(mask)  # 内点数量
    inlier_ratio = inliers / total_matches

    # 计算匹配分数：综合考虑内点数量和比例
    score = inliers * inlier_ratio

    return int(inliers), total_matches, score


def parse_location_id(location_id):
    if "_pos_" in location_id:
        import re
        match = re.search(r'(.*?)_pos_0*(\d+)', location_id, re.IGNORECASE)
        if match:
            building_and_corridor = match.group(1)
            number = int(match.group(2))
            floor = "1F"
            return building_and_corridor, floor, number

    return None, None, None


def check_temporal_consistency(session_key, new_building, new_floor, new_number):
    """
    Check if the new location matches a continuous movement pattern
    Upgraded with Topological Graph validation for junctions.
    """
    now = time.time()
    history = session_history.get(session_key, [])
    history = [h for h in history if now - h[0] < TIME_WINDOW]
    session_history[session_key] = history

    if not history:
        return True, ""

    last_time, last_building, last_floor, last_number = history[-1]

    # 1. 检查是否在同一条走廊内直走 (执行一维动态步数校验)
    if new_building == last_building:
        if last_number is not None and new_number is not None:
            time_elapsed = now - last_time
            dynamic_step_limit = max(2, int(time_elapsed * 0.5))

            if abs(new_number - last_number) > dynamic_step_limit:
                return False, f"Unrealistic jump: pos_{last_number:03d} to pos_{new_number:03d} over {int(time_elapsed)}s."
        return True, ""

    # 2. 检查是否在合法的交叉路口拐弯 (执行拓扑图校验)
    # 注意：这里的 new_building 实际上包含 "Building_Corridor"
    last_node = f"{last_building}_{last_number}"
    new_node = f"{new_building}_{new_number}"

    valid_neighbors = VALID_JUNCTIONS.get(last_node, [])

    if new_node in valid_neighbors:
        return True, ""  # 属于合法的拐弯

    # 3. 如果名字不一样，且不在合法邻居字典里，说明是非法跳跃 (跨楼宇/穿墙)
    return False, f"Warning: Building changed (from {last_building} to {new_building})."


# ==================== 服务器启动事件 ====================
@app.on_event("startup")
async def startup_event():
    """服务器启动时自动运行"""
    global database
    print("=" * 60)
    print("The ceiling positioning system is activated")
    print("=" * 60)
    print("The database is being loaded...")

    # 检查是否有数据库文件
    db_file = Path("database.json")
    if db_file.exists():
        try:
            with open("database.json", "r", encoding='utf-8') as f:
                raw_data = json.load(f)

            # 重建数据库，将描述符转回numpy数组，关键点转为OpenCV对象
            for building_name, locations in raw_data.items():
                database[building_name] = {}
                for loc_id, loc_data in locations.items():
                    # 转换描述符
                    desc = np.array(loc_data["descriptors"], dtype=np.uint8)
                    # 转换关键点
                    kp_list = [deserialize_keypoint(k) for k in loc_data["keypoints"]]
                    database[building_name][loc_id] = {
                        "filename": loc_data["filename"],
                        "keypoints": kp_list,
                        "descriptors": desc,
                        "keypoints_count": loc_data["keypoints_count"]
                    }

            total_locations = sum(len(v) for v in database.values())
            print(f"The database has been loaded successfully!")
            print(f"   The number of buildings：{len(database)}")
            print(f"   Total number of positions：{total_locations}")
        except Exception as e:
            print(f"数据库加载失败: {e}")
            print("   请先运行 database.py 建立数据库")
    else:
        print("找不到数据库文件 database.json")
        print("   请先运行 database.py 建立数据库")

    print("")
    print("server information：")
    print("   Access address: http://localhost:8000")
    print("   API document: http://localhost:8000/docs")
    print("   Check-up: http://localhost:8000/api/health")
    print("")
    print("method of application：")
    print("   1. Open it with a browser http://localhost:8000")
    print("   2. Select the photo of the ceiling for upload")
    print("   3. View the positioning result")
    print("=" * 60)


# ==================== API端点定义 ====================

@app.get("/")
async def home():
    """返回系统主页（HTML页面）"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Ceiling Localization System</title>
        <style>
            /* =========================================
               基础变量与全局设定 (日间/夜间模式)
               ========================================= */
            :root {
                --bg-color: #fafafa;
                --bg-radial: radial-gradient(circle at 50% 100%, rgba(244, 114, 182, 0.25) 0%, transparent 50%),
                             radial-gradient(circle at 0% 0%, rgba(167, 139, 250, 0.15) 0%, transparent 50%);
                --card-bg: rgba(255, 255, 255, 0.85);
                --card-border: rgba(255, 255, 255, 1);
                --shadow: 0 10px 40px -10px rgba(0, 0, 0, 0.08);
                
                --text-main: #475569;
                --text-sub: #64748b;
                --accent-gradient: linear-gradient(90deg, #c026d3, #8b5cf6);
                --btn-gradient: linear-gradient(90deg, #ec4899, #8b5cf6);
                --border-color: #cbd5e1;
            }

            body.dark-mode {
                --bg-color: #0f172a;
                --bg-radial: radial-gradient(circle at 50% 100%, rgba(190, 24, 93, 0.2) 0%, transparent 50%),
                             radial-gradient(circle at 0% 0%, rgba(76, 29, 149, 0.2) 0%, transparent 50%);
                --card-bg: rgba(30, 41, 59, 0.85);
                --card-border: rgba(255, 255, 255, 0.05);
                --shadow: 0 10px 40px -10px rgba(0, 0, 0, 0.4);
                
                --text-main: #cbd5e1;
                --text-sub: #94a3b8;
                --accent-gradient: linear-gradient(90deg, #e879f9, #a78bfa);
                --border-color: #475569;
            }

            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                font-family: 'Inter', system-ui, -apple-system, sans-serif; 
                background-color: var(--bg-color); 
                background-image: var(--bg-radial);
                background-attachment: fixed;
                color: var(--text-main);
                min-height: 100vh; 
                padding: 20px; 
                transition: all 0.4s ease;
            }

            .container { max-width: 850px; margin: 20px auto; }

            /* 玻璃质感圆角卡片 */
            .ui-card {
                background: var(--card-bg);
                border-radius: 24px;
                box-shadow: var(--shadow);
                border: 1px solid var(--card-border);
                padding: 35px;
                margin-bottom: 20px;
                backdrop-filter: blur(12px);
                -webkit-backdrop-filter: blur(12px);
                transition: all 0.4s ease;
            }

            /* =========================================
               标题与文本样式
               ========================================= */
            .header-section { text-align: center; position: relative; }
            .gradient-title {
                font-size: 2.6rem;
                font-weight: 800;
                background: var(--accent-gradient);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 5px;
            }
            .subtitle { font-size: 1.15rem; font-weight: 600; color: var(--text-main); margin-top: 5px; }
            .dyn-subtext { color: var(--text-sub); }
            .section-title { font-size: 1.5em; color: var(--text-main); margin-bottom: 20px; padding-bottom: 10px; border-bottom: 2px solid var(--border-color); transition: color 0.4s; }

            /* 深色模式切换按钮 */
            .theme-btn {
                position: absolute; right: 0; top: 0;
                border-radius: 12px; background: transparent; border: 1px solid var(--border-color);
                color: var(--text-main); cursor: pointer; padding: 8px 16px; font-weight: 600;
                transition: all 0.3s ease;
            }
            .theme-btn:hover { background: rgba(148, 163, 184, 0.1); }
            body.dark-mode .theme-btn { background: rgba(30, 41, 59, 0.5); }

            /* =========================================
               对齐交互区 (按钮与输入)
               ========================================= */
            .upload-box { text-align: center; }
            .file-input { display: none; }
            
            .action-row {
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 15px;
                margin-top: 25px;
            }

            .file-label {
                height: 48px;
                display: flex;
                align-items: center;
                padding: 0 25px;
                background: transparent;
                color: var(--text-main);
                border: 2px dashed var(--border-color);
                border-radius: 12px;
                cursor: pointer;
                font-size: 1.05em;
                font-weight: 600;
                transition: all 0.3s;
            }
            .file-label:hover { border-color: #8b5cf6; color: #8b5cf6; background: rgba(139, 92, 246, 0.05); }

            .upload-btn {
                height: 48px;
                display: flex;
                align-items: center;
                padding: 0 35px;
                background: var(--btn-gradient);
                color: white;
                border: none;
                border-radius: 12px;
                font-size: 1.05em;
                font-weight: 600;
                cursor: pointer;
                transition: transform 0.2s ease, box-shadow 0.2s ease, opacity 0.3s;
            }
            .upload-btn:hover:not(:disabled) { transform: translateY(-2px); box-shadow: 0 8px 20px -5px rgba(236, 72, 153, 0.5); }
            .upload-btn:disabled { opacity: 0.5; cursor: not-allowed; }

            #fileName { margin-top: 15px; color: var(--text-sub); font-style: italic; font-size: 0.95em; }

            /* =========================================
               结果展示区
               ========================================= */
            .result-box { display: none; animation: fadeIn 0.5s; margin-top: 0; }
            .result-box.show { display: block; }
            @keyframes fadeIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }

            /* 状态提示框适配深色模式 */
            .status-box { padding: 15px 20px; border-radius: 12px; margin-bottom: 25px; font-weight: 600; border: 1px solid transparent; }
            .success { color: #155724; background: rgba(212, 237, 218, 0.8); border-color: #c3e6cb; }
            body.dark-mode .success { color: #86efac; background: rgba(21, 128, 61, 0.2); border-color: #166534; }
            .warning { color: #856404; background: rgba(255, 243, 205, 0.8); border-color: #ffeeba; }
            body.dark-mode .warning { color: #fde047; background: rgba(161, 98, 7, 0.2); border-color: #854d0e; }
            .error { color: #721c24; background: rgba(248, 215, 218, 0.8); border-color: #f5c6cb; }
            body.dark-mode .error { color: #fca5a5; background: rgba(153, 27, 27, 0.2); border-color: #991b1b; }
            .processing { color: #0c5460; background: rgba(209, 236, 241, 0.8); border-color: #bee5eb; margin-top: 20px; }
            body.dark-mode .processing { color: #7dd3fc; background: rgba(3, 105, 161, 0.2); border-color: #0c4a6e; }

            /* 详情网格卡片 */
            .result-details { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }
            .inner-card { 
                background: rgba(255, 255, 255, 0.4);
                padding: 20px; 
                border-radius: 16px; 
                border: 1px solid var(--border-color);
                border-left: 5px solid #8b5cf6; 
                transition: all 0.4s ease;
            }
            body.dark-mode .inner-card { background: rgba(0, 0, 0, 0.2); border-left-color: #a78bfa; }
            
            .trajectory-card { grid-column: 1 / -1; border-left-color: #ec4899; }
            body.dark-mode .trajectory-card { border-left-color: #f472b6; }
            
            .detail-label { font-size: 0.85em; color: var(--text-sub); margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600; }
            .detail-value { font-size: 1.3em; font-weight: 700; color: var(--text-main); }
            .trajectory-value { font-size: 1.1em; font-family: monospace; color: #ec4899; letter-spacing: 0.5px; }
            body.dark-mode .trajectory-value { color: #f472b6; }

            footer { text-align: center; padding: 20px; color: var(--text-sub); margin-top: 20px; font-size: 0.9em; }
            .links a { color: #8b5cf6; text-decoration: none; margin: 0 10px; font-weight: 600; transition: opacity 0.3s; }
            .links a:hover { opacity: 0.7; }
        </style>
    </head>
    <body>
        <div class="container">
            <header class="ui-card header-section">
                <button id="themeToggle" class="theme-btn">🌓 Theme</button>
                <h1 class="gradient-title">Ceiling Localization</h1>
                <p class="subtitle">Computer Vision Based Indoor Navigation | BSc Robotics</p>
                <p class="dyn-subtext" style="margin-top: 8px; font-size: 0.95em;">Algorithm: ORB + FLANN + RANSAC + Sequence Check</p>
            </header>

            <div class="ui-card">
                <h2 class="section-title">Image Upload & Localization</h2>
                <div class="upload-box">
                    <p class="dyn-subtext" style="font-size: 1.05em; line-height: 1.6; max-width: 600px; margin: 0 auto;">
                        Upload a ceiling photo to identify your current building and specific location.<br>
                        <span style="font-size: 0.9em; opacity: 0.8;">💡 Tip: Upload photos sequentially while walking to test the Reality Model.</span>
                    </p>
                    
                    <div class="action-row">
                        <input type="file" id="fileInput" class="file-input" accept="image/*" capture="environment">
                        <label for="fileInput" class="file-label">📁 Choose Photo</label>
                        <button id="uploadBtn" class="upload-btn" disabled>📤 Upload & Locate</button>
                    </div>
                    
                    <div id="fileName"></div>
                    
                    <div id="status" class="status-box processing" style="display: none;">
                        <span style="display: inline-block; animation: spin 2s linear infinite;">⚙️</span> Processing image...<br>
                        <small>Matching via FLANN+RANSAC & checking temporal sequence...</small>
                    </div>
                    <style>@keyframes spin { 100% { transform: rotate(360deg); } }</style>
                </div>
            </div>

            <div id="resultBox" class="ui-card result-box">
                <div id="resultMessage" class="status-box"></div>
                
                <div class="result-details">
                    <div class="inner-card trajectory-card">
                        <div class="detail-label">Walking Trajectory (Reality Model)</div>
                        <div id="trajectory" class="trajectory-value">None</div>
                    </div>
                    <div class="inner-card">
                        <div class="detail-label">Building</div>
                        <div id="building" class="detail-value">--</div>
                    </div>
                    <div class="inner-card">
                        <div class="detail-label">Exact Location</div>
                        <div id="location" class="detail-value">--</div>
                    </div>
                    <div class="inner-card">
                        <div class="detail-label">Match Score</div>
                        <div id="score" class="detail-value">--</div>
                    </div>
                    <div class="inner-card">
                        <div class="detail-label">RANSAC Inliers</div>
                        <div id="inliers" class="detail-value">--</div>
                    </div>
                    <div class="inner-card">
                        <div class="detail-label">Features</div>
                        <div id="features" class="detail-value">--</div>
                    </div>
                    <div class="inner-card">
                        <div class="detail-label">Processing Time</div>
                        <div id="time" class="detail-value">--</div>
                    </div>
                </div>
            </div>

            <footer>
                <p>Heriot-Watt University | Supervised by Prof. Bartie, Phil</p>
                <div class="links" style="margin-top: 15px;">
                    <a href="/docs" target="_blank">📖 API Docs</a>
                    <a href="/api/health" target="_blank">💚 System Health</a>
                </div>
            </footer>
        </div>

        <script>
            // 核心逻辑保持完全不变
            const fileInput = document.getElementById('fileInput');
            const uploadBtn = document.getElementById('uploadBtn');
            const fileName = document.getElementById('fileName');
            const resultBox = document.getElementById('resultBox');
            const resultMessage = document.getElementById('resultMessage');
            const statusDiv = document.getElementById('status');
            const themeToggle = document.getElementById('themeToggle');

            // 主题切换逻辑
            themeToggle.addEventListener('click', () => {
                document.body.classList.toggle('dark-mode');
                const isDark = document.body.classList.contains('dark-mode');
                localStorage.setItem('theme', isDark ? 'dark' : 'light');
            });
            // 记住用户的主题选择
            if (localStorage.getItem('theme') === 'dark') {
                document.body.classList.add('dark-mode');
            }

            let sessionId = localStorage.getItem('ceiling_session_id') || 'session_' + Date.now();
            localStorage.setItem('ceiling_session_id', sessionId);

            let locationPath = [];

            fileInput.addEventListener('change', function() {
                if (this.files && this.files[0]) {
                    const file = this.files[0];
                    fileName.textContent = `Selected: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
                    uploadBtn.disabled = false;
                }
            });

            uploadBtn.addEventListener('click', async function() {
                const file = fileInput.files[0];
                statusDiv.style.display = 'block';
                resultBox.classList.remove('show');

                const formData = new FormData();
                formData.append('file', file);
                formData.append('session_id', sessionId);

                try {
                    const response = await fetch('/upload', { method: 'POST', body: formData });
                    const data = await response.json();
                    statusDiv.style.display = 'none';

                    if (data.status === 'success' || data.status === 'warning') {
                        resultMessage.className = 'status-box ' + data.status;
                        resultMessage.innerHTML = data.status === 'success' ? `✅ ${data.message}` : `⚠️ ${data.message}`;

                        document.getElementById('building').textContent = data.building || '--';
                        document.getElementById('location').textContent = data.location || '--';
                        document.getElementById('score').textContent = data.score ? data.score.toFixed(2) : '--';
                        document.getElementById('inliers').textContent = data.inliers || '--';
                        document.getElementById('features').textContent = data.features_detected || '--';
                        document.getElementById('time').textContent = data.processing_time_ms ? `${data.processing_time_ms} ms` : '--';

                        if (data.location) {
                            locationPath.push(data.location);
                            if(locationPath.length > 5) locationPath.shift(); 
                            document.getElementById('trajectory').textContent = locationPath.join(' ➔ ');
                        }

                    } else {
                        resultMessage.className = 'status-box error';
                        resultMessage.innerHTML = `❌ ${data.message}`;
                    }
                    resultBox.classList.add('show');
                } catch (error) {
                    statusDiv.style.display = 'none';
                    resultMessage.className = 'status-box error';
                    resultMessage.innerHTML = `❌ Network Error: ${error.message}`;
                    resultBox.classList.add('show');
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.post("/upload")
async def upload_image(
        request: Request,
        file: UploadFile = File(...),
        session_id: str = Form(None)
):
    """Process uploaded ceiling image for localization"""
    start_time = time.time()

    try:
        if not file.content_type.startswith("image/"):
            return JSONResponse(status_code=400,
                                content={"status": "error", "message": "Please upload an image file (JPEG, PNG)."})

        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        contents = await file.read()

        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

        if img is None:
            return JSONResponse(status_code=400,
                                content={"status": "error", "message": "Cannot read image. Invalid format."})

        # 图像预处理 (缩放)
        height, width = img.shape
        if height > 800 or width > 800:
            scale = 800 / max(height, width)
            img = cv2.resize(img, (int(width * scale), int(height * scale)))

        img = cv2.equalizeHist(img)
        query_kp, query_desc = orb.detectAndCompute(img, None)

        if query_desc is None or len(query_kp) < 10:
            return JSONResponse(status_code=400, content={"status": "error",
                                                          "message": "Not enough features detected. Please take a clearer photo."})

        if not database:
            return JSONResponse(status_code=500,
                                content={"status": "error", "message": "Database is empty. Run database.py first."})

        best_match, best_score, best_building, best_location_id, best_inliers = None, -1, None, None, 0
        match_details = []

        for building_name, locations in database.items():
            for loc_id, loc_data in locations.items():
                inliers, total_matches, score = match_with_flann_ransac(query_desc, query_kp, loc_data["descriptors"],
                                                                        loc_data["keypoints"])
                if score > best_score:
                    best_score, best_building, best_location_id, best_inliers = score, building_name, loc_id, inliers

        if best_building and best_location_id:
            building, floor, number = parse_location_id(best_location_id)
        else:
            building, floor, number = None, None, None

        session_key = session_id if session_id else request.client.host
        temporal_pass = True
        temporal_message = ""

        if building and number is not None:
            temporal_pass, temporal_message = check_temporal_consistency(session_key, building, floor, number)
        else:
            temporal_pass = False

        processing_time = (time.time() - start_time) * 1000
        THRESHOLD = 4.0

        if best_score > THRESHOLD:
            # 更新历史记录
            now = time.time()
            history = session_history.get(session_key, [])
            history.append((now, building, floor, number))
            if len(history) > HISTORY_MAXLEN: history.pop(0)
            session_history[session_key] = history

            # 终极修复：精简显示，自动去除重复的建筑物名称
            parts = best_location_id.split('_pos_')
            if len(parts) > 1:
                # parts[0] 此时是 "Earl Mountbatten Building_CorridorA"
                # best_building 是 "Earl Mountbatten Building"
                # 我们利用字符串替换，把建筑物名字和连接的下划线自动删掉，只留下走廊名
                corridor_raw = parts[0].replace(f"{best_building}_", "")
                corridor_name = corridor_raw.replace('_', ' ')

                # parts[1] 是位置和房间号 (例如 "04_G39")
                friendly_suffix = parts[1].replace('_', ' ')
                # 将它们拼接在一起
                friendly_location = f"{corridor_name} - Pos {friendly_suffix}"
            else:
                friendly_location = best_location_id

            # 如果时序检查没通过，给一个警告，但不阻止结果显示
            status_msg = "Localization Successful" if temporal_pass else f"Warning: {temporal_message}"
            status_type = "success" if temporal_pass else "warning"

            return JSONResponse({
                "status": status_type,
                "building": best_building.replace('_', ' '),
                "location": friendly_location,
                "location_id": best_location_id,
                "score": round(best_score, 2),
                "inliers": int(best_inliers),
                "processing_time_ms": round(processing_time, 2),
                "features_detected": len(query_kp),
                "message": status_msg
            })
        else:
            return JSONResponse({
                "status": "error",
                "message": "No reliable match found. Please try another location.",
                "processing_time_ms": round(processing_time, 2)
            })

    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": f"Server Error: {str(e)}"})


@app.get("/api/health")
async def health_check():
    """健康检查端点，用于监控系统状态"""
    status = {
        "status": "healthy",
        "service": "ceiling-localization",
        "buildings": list(database.keys()),
        "total_locations": sum(len(v) for v in database.values()),
        "algorithm": "ORB + FLANN + RANSAC + 时序检查",
        "timestamp": time.time()
    }

    if not database:
        status["status"] = "warning"
        status["message"] = "数据库为空，请运行 database.py"

    return status


@app.get("/api/database")
async def get_database():
    """获取数据库信息"""
    result = {}
    for building_name, locations in database.items():
        result[building_name] = {}
        for loc_id, loc_data in locations.items():
            result[building_name][loc_id] = {
                "filename": loc_data["filename"],
                "keypoints_count": loc_data["keypoints_count"]
            }

    return {
        "total_buildings": len(database),
        "total_locations": sum(len(v) for v in database.values()),
        "details": result
    }


# ==================== 服务器启动 ====================
if __name__ == "__main__":
    import uvicorn

    print("The ceiling positioning server is being started...")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)


    