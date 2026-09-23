# import os
# from dotenv import load_dotenv
# from openai import OpenAI
#
# # 读取.env配置
# load_dotenv()
#
# client = OpenAI(
#     base_url="https://ark.cn-beijing.volces.com/api/v3",
#     api_key=os.getenv("ARK_API_KEY"),
# )
#
# resp = client.chat.completions.create(
#     model=os.getenv("ARK_EP_ID"),
#     messages=[{"role":"user","content":"你好，请简短回复一句话"}],
#     temperature=0
# )
# print("🎉 API调用成功！返回内容：")
# print(resp.choices[0].message.content)
import os
import json
import torch
from dotenv import load_dotenv
from openai import OpenAI
from ultralytics import YOLO

# 加载环境变量
load_dotenv()

# 选择设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"CUDA是否可用：{torch.cuda.is_available()}")
print(f"YOLO使用设备：{device}")

# 加载YOLO模型
model = YOLO("yolov8n.pt").to(device)

# 初始化火山方舟客户端
client = OpenAI(
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    api_key=os.getenv("ARK_API_KEY")
)
ep_id = os.getenv("ARK_EP_ID")

def detect_image(img_path: str, conf_thresh=0.25):
    """YOLO检测，返回结构化结果"""
    results = model.predict(img_path, device=device, conf=conf_thresh)
    output = []
    for res in results:
        if res.boxes is None:
            continue
        for box in res.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1,y1,x2,y2 = map(float, box.xyxy[0])
            cls_name = res.names[cls_id]
            output.append({
                "class_name": cls_name,
                "confidence": round(conf,2),
                "bbox": [x1,y1,x2,y2]
            })
    return output

def build_inspect_report(detect_list, scene_name="1号商铺北向"):
    """调用Seed大模型生成巡检报告"""
    sys_prompt = """
你是厂区安全巡检专家。接收YOLO目标检测结果，输出巡检分析报告。
规则：
1. 结合场景（商铺外墙），评估检测到物体是否存在安全隐患；
2. clock（时钟）属于正常设施，一般无危险；
3. 如果出现消防栓、灭火器、堆放杂物、破损构件才判定隐患；
4. 输出结构：【巡检概况】【隐患判定】【整改建议】，语言简洁正式。
不要输出多余闲聊。
"""
    user_msg = f"""
巡检场景：{scene_name}
YOLO检测结果：
{json.dumps(detect_list, ensure_ascii=False, indent=2)}
请生成安全巡检报告。
"""
    resp = client.chat.completions.create(
        model=ep_id,
        messages=[
            {"role":"system","content":sys_prompt},
            {"role":"user","content":user_msg}
        ],
        temperature=0.1
    )
    return resp.choices[0].message.content

if __name__ == "__main__":
    img_file = r"img\shop1_north_01.png"
    det_result = detect_image(img_file)
    print("\n==== YOLO检测结果 ====")
    print(json.dumps(det_result, ensure_ascii=False, indent=2))

    report = build_inspect_report(det_result)
    print("\n==== 智能巡检报告（Seed-2.1-turbo生成） ====")
    print(report)

