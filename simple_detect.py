import os
import cv2
import torch
import torch.nn as nn
import torchvision.models as tvm
from torchvision import transforms
from PIL import Image

EMO = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear"]

EMO_COLOR = {
    "neutral":    (180, 180, 180),
    "happiness":  (0, 255, 100),
    "surprise":   (255, 200, 0),
    "sadness":    (200, 80, 0),
    "anger":      (0, 0, 240),
    "disgust":    (0, 140, 60),
    "fear":       (120, 0, 180),
}

def load_model(device):
    # 使用绝对路径，确保不管你在哪个文件夹运行都不会报错找不到模型
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = os.path.join(BASE_DIR, "runs", "classmate_model", "best.pt")
    
    if not os.path.exists(ckpt_path):
        print(f"[Error] 找不到模型文件: {ckpt_path}")
        exit(1)
        
    ckpt_data = torch.load(ckpt_path, map_location=device)
    m = tvm.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 7)
    m.load_state_dict(ckpt_data["model_state"])
    return m.to(device).eval()

def preprocess_face(face_bgr, img_size=224):
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_pil = Image.fromarray(face_rgb)
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return tf(face_pil).unsqueeze(0)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"正在加载模型到 {device}...")
    model = load_model(device)
    print("模型加载成功！")

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(0)

    print("正在打开摄像头... 按下 'q' 键退出。")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1) # 镜像翻转画面
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 人脸检测
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            if face_img.size == 0: continue
            
            # 情绪识别
            with torch.no_grad():
                tensor = preprocess_face(face_img).to(device)
                logits = model(tensor)
                probs = torch.softmax(logits, 1).cpu().numpy()[0]
                
            top_idx = int(probs.argmax())
            top_emo = EMO[top_idx]
            conf = probs[top_idx]
            
            # 画框与写字
            color = EMO_COLOR.get(top_emo, (0, 255, 0))
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            label = f"{top_emo} {conf:.0%}"
            cv2.putText(frame, label, (x, max(10, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Simple Emotion Detect", frame)
        
        # 按 q 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
