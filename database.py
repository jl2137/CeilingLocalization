# database.py - 建立天花板图像特征数据库（支持多建筑物）
import cv2
import numpy as np
import json
from pathlib import Path

def build_database():
    """建立多建筑物天花板图像数据库"""
    print("=" * 60)
    print("🏢 多建筑物天花板图像数据库构建工具")
    print("=" * 60)

    # 初始化ORB特征检测器（增加特征点数量，便于匹配）
    orb = cv2.ORB_create(nfeatures=1500)

    database = {}
    database_dir = Path("database")

    if not database_dir.exists():
        print("❌ 错误：找不到 database 文件夹！")
        print("请先创建 database 文件夹，并在其中按建筑物组织图片。")
        return

    # 获取所有建筑物子文件夹
    building_dirs = [d for d in database_dir.iterdir() if d.is_dir()]
    if not building_dirs:
        print("❌ 错误：database 文件夹下没有建筑物子文件夹！")
        print("请创建如 Mary_Burton、Colin_Maclaurin 等子文件夹。")
        return

    print(f"找到 {len(building_dirs)} 个建筑物：{[d.name for d in building_dirs]}")

    # 遍历每个建筑物
    for building_dir in building_dirs:
        building_name = building_dir.name
        print(f"\n📁 处理建筑物：{building_name}")
        database[building_name] = {}

        # 递归查找所有图片文件
        image_files = list(building_dir.rglob("*.jpg")) + list(building_dir.rglob("*.png"))
        if not image_files:
            print(f"   ⚠ 该建筑物下没有图片")
            continue

        print(f"   找到 {len(image_files)} 张图片")

        # 遍历该建筑物下的所有图片
        for img_path in image_files:
            # 读取灰度图
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"   ⚠ 无法读取：{img_path.name}")
                continue


            height, width = img.shape
            if height > 800 or width > 800:
                scale = 800 / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                img = cv2.resize(img, (new_width, new_height))
            img = cv2.equalizeHist(img)

            # 提取ORB特征
            keypoints, descriptors = orb.detectAndCompute(img, None)

            if descriptors is None or len(keypoints) < 10:
                print(f"   ⚠ 特征点太少：{img_path.name}")
                continue

            # 生成位置ID：相对路径（相对于 database 文件夹）去掉扩展名，用下划线连接
            # 例如：Mary_Burton/floor_1/pos_001 -> Mary_Burton_floor_1_pos_001
            relative_path = img_path.relative_to(database_dir)
            location_id = str(relative_path.with_suffix('')).replace('/', '_').replace('\\', '_')
            print(f"   ✅ 位置ID：{location_id} 特征点：{len(keypoints)}")

            # 存储关键点坐标（用于RANSAC）
            keypoints_data = []
            for kp in keypoints:
                keypoints_data.append({
                    "pt": (float(kp.pt[0]), float(kp.pt[1])),
                    "size": float(kp.size),
                    "angle": float(kp.angle),
                    "response": float(kp.response)
                })

            # 存储到数据库
            database[building_name][location_id] = {
                "filename": str(img_path),
                "keypoints": keypoints_data,
                "descriptors": descriptors.tolist(),
                "keypoints_count": len(keypoints)
            }

    # 保存数据库到JSON文件
    output_file = "database.json"
    with open(output_file, "w") as f:
        json.dump(database, f, indent=2)

    print("\n" + "=" * 60)
    print("✅ 数据库构建完成！")
    print(f"   输出文件：{output_file}")
    print("=" * 60)
    return database

if __name__ == "__main__":
    build_database()