#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证新的人脸识别模型与原有ONNX模型的一致性
"""

import os
import sys
import cv2
import torch
import numpy as np
import pickle

# 添加项目路径
current_file_path = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(os.path.dirname(current_file_path), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入模型定义
from vision_tools.face_recognition_pytorch import IR_101


class NewFaceRecognitionModel:
    """使用PyTorch模型的人脸识别"""
    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device)
        
        # 构建模型
        self.model = IR_101(input_size=(112, 112))
        
        # 加载权重
        if os.path.exists(model_path):
            print(f"[INFO] 加载模型权重: {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # 处理不同格式的checkpoint
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            self.model.load_state_dict(state_dict, strict=True)
            print("[INFO] 模型权重加载成功")
        else:
            raise FileNotFoundError(f"模型权重文件不存在: {model_path}")
        
        self.model.to(self.device)
        self.model.eval()
    
    def preprocess(self, img):
        """预处理图像，与ONNX模型保持一致"""
        # BGR to RGB
        img = img[:, :, ::-1]
        # Resize to 112x112
        img = cv2.resize(img, (112, 112))
        # Transpose to CHW
        img = np.transpose(img, axes=(2, 0, 1))
        # Normalize: (img / 255. - 0.5) / 0.5
        img = (img / 255.0 - 0.5) / 0.5
        # Add batch dimension
        img = np.expand_dims(img.astype(np.float32), 0)
        return img
    
    def __call__(self, img):
        """提取人脸特征，流程与之前相同"""
        # 预处理
        img_tensor = self.preprocess(img)
        img_tensor = torch.from_numpy(img_tensor).to(self.device)
        
        # 前向推理
        with torch.no_grad():
            emb = self.model(img_tensor)
            emb = emb.cpu().numpy()
        
        # L2归一化
        # NOTE: Need to consider using or not when use mlp head
        emb = emb / np.sqrt(np.sum(emb**2, axis=-1, keepdims=True))
        
        return emb[0]


def load_face_image(image_path):
    """加载人脸图像"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像文件不存在: {image_path}")
    
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    print(f"[INFO] 成功加载图像: {image_path}")
    print(f"[INFO] 图像尺寸: {img.shape}")
    return img


# 新增：从视频中提取指定人脸
def extract_face_from_video(video_path, pkl_path, subseg_id, face_idx):
    """
    根据pkl文件信息，从视频中提取指定subseg_id和face_idx的人脸区域。
    返回该人脸的BGR图像（与直接保存图片一致）。
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件不存在: {video_path}")
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"PKL文件不存在: {pkl_path}")
    import pickle
    import cv2
    import numpy as np
    # 读取pkl，找到对应帧和bbox
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    found = False
    for i in range(len(data['audio_seg_id'])):
        if data['audio_seg_id'][i] == subseg_id and data['face_idx'][i] == face_idx:
            mid_time = data['times'][i]
            bbox = data['bbox'][i]  # [x1, y1, x2, y2]
            found = True
            break
    if not found:
        raise ValueError(f"未找到匹配的embedding: subseg_id={subseg_id}, face_idx={face_idx}")
    # 打开视频，定位到对应帧
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        cap.release()
        raise ValueError(f"无法获取视频帧率: {video_path}")
    mid_frame_idx = int(mid_time * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"无法读取视频帧: {mid_frame_idx} @ {video_path}")
    # 裁剪人脸区域
    x1, y1, x2, y2 = bbox
    face_img = frame[y1:y2, x1:x2]
    if face_img.size == 0:
        raise ValueError(f"裁剪后人脸为空: bbox={bbox}, frame shape={frame.shape}")
    print(f"[INFO] 从视频帧 {mid_frame_idx} 裁剪人脸: bbox={bbox}, 裁剪尺寸: {face_img.shape}")
    return face_img


def load_pkl_embedding(pkl_path, subseg_id, face_idx):
    """从pkl文件中加载对应的embedding"""
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"PKL文件不存在: {pkl_path}")
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    # 查找对应的embedding
    for i in range(len(data['audio_seg_id'])):
        if data['audio_seg_id'][i] == subseg_id and data['face_idx'][i] == face_idx:
            print(f"[INFO] 找到匹配的embedding: subseg_id={subseg_id}, face_idx={face_idx}")
            print(f"[INFO] 时间戳: {data['times'][i]:.2f}s")
            print(f"[INFO] 边界框: {data['bbox'][i]}")
            return data['feat'][i]
    
    raise ValueError(f"未找到匹配的embedding: subseg_id={subseg_id}, face_idx={face_idx}")


def compare_embeddings(emb1, emb2):
    """比较两个embedding的一致性"""
    # 确保维度一致
    if emb1.shape != emb2.shape:
        print(f"[WARNING] Embedding维度不一致: {emb1.shape} vs {emb2.shape}")
        # 如果维度不同，尝试压缩
        emb1 = emb1.flatten()
        emb2 = emb2.flatten()
    # 计算范数
    norm1 = np.sqrt(np.sum(emb1**2, axis=-1, keepdims=True))
    norm2 = np.sqrt(np.sum(emb2**2, axis=-1, keepdims=True))

    # 计算余弦相似度
    cosine_sim = np.dot(emb1.flatten(), emb2.flatten()) / (
        np.linalg.norm(emb1) * np.linalg.norm(emb2)
    )
    
    # 计算欧氏距离
    euclidean_dist = np.linalg.norm(emb1 - emb2)
    
    # 计算L2距离
    l2_dist = np.sqrt(np.sum((emb1 - emb2) ** 2))
    
    # 计算逐元素差异
    abs_diff = np.abs(emb1 - emb2)
    mean_abs_diff = np.mean(abs_diff)
    max_abs_diff = np.max(abs_diff)
    
    print("\n" + "="*20)
    print("Embedding 比较结果:")
    print("="*20)
    print(f"Embedding 1 shape: {emb1.shape}")
    print(f"Embedding 2 shape: {emb2.shape}")
    print(f"Embedding 1 norm: {float(norm1):.6f}")
    print(f"Embedding 2 norm: {float(norm2):.6f}")
    print(f"余弦相似度: {cosine_sim:.6f}")
    print(f"欧氏距离: {euclidean_dist:.6f}")
    print(f"L2距离: {l2_dist:.6f}")
    print(f"平均绝对差异: {mean_abs_diff:.6f}")
    print(f"最大绝对差异: {max_abs_diff:.6f}")
    print("="*20)
    
    # 判断一致性
    if cosine_sim > 0.99:
        print("✅ 两个模型高度一致 (余弦相似度 > 0.99)")
    elif cosine_sim > 0.95:
        print("⚠️  两个模型基本一致 (余弦相似度 > 0.95)")
    elif cosine_sim > 0.90:
        print("⚠️  两个模型存在一定差异 (余弦相似度 > 0.90)")
    else:
        print("❌ 两个模型差异较大 (余弦相似度 <= 0.90)")


def main():
    # 配置路径
    PROJECT_ROOT = "/Users/wangchen/文档/Research/tv_series_plus/3D-Speaker"
    
    # 人脸图像路径 (根据实际情况调整)
    face_image_path = os.path.join(
        PROJECT_ROOT,
        "dataset/the big bang theory/midframe_faces/E01/E01-1_0.jpg"
    )

    video_path = os.path.join(
        PROJECT_ROOT,
        "dataset/the big bang theory/raw/E01.mp4"
    )
    
    # 新模型权重路径
    model_weight_path = os.path.join(
        PROJECT_ROOT,
        "egs/3dspeaker/speaker-diarization/pretrained_models/pytorch_model_tongyi.bin"
    )
    
    # PKL文件路径 (根据实际情况调整)
    pkl_path = os.path.join(
        PROJECT_ROOT,
        "egs/3dspeaker/speaker-diarization/runs/the big bang theory/exp_video/embs_video/E01_midframe.pkl"
    )
    
    # 从 face_image_path 解析 subseg_id 和 face_idx
    # 假设文件名格式为: .../E01-1_0.jpg
    subseg_id, face_idx_str = os.path.splitext(os.path.basename(face_image_path))[0].rsplit("_", 1)
    face_idx = int(face_idx_str)
    
    print("="*20)
    print("人脸识别模型一致性验证")
    print("="*20)
    print(f"人脸图像: {face_image_path}")
    print(f"模型权重: {model_weight_path}")
    print(f"PKL文件: {pkl_path}")
    print(f"Subseg ID: {subseg_id}")
    print(f"Face Index: {face_idx}")
    print("="*20)
    
    try:
        # 1. 使用新模型提取embedding
        print("\n[步骤 1] 使用新模型提取embedding...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[INFO] 使用设备: {device}")
        
        new_model = NewFaceRecognitionModel(model_weight_path, device=device)
        face_img = extract_face_from_video(video_path, pkl_path, subseg_id, face_idx)
        # face_img = load_face_image(face_image_path)
        new_embedding = new_model(face_img)
        
        print(f"[INFO] 新模型提取的embedding shape: {new_embedding.shape}")
        print(f"[INFO] Embedding前5个值: {new_embedding.flatten()[:5]}")
        
        # 2. 从PKL文件加载原有embedding
        print("\n[步骤 2] 从PKL文件加载原有embedding...")
        old_embedding = load_pkl_embedding(pkl_path, subseg_id, face_idx)
        
        print(f"[INFO] 原有embedding shape: {old_embedding.shape}")
        print(f"[INFO] Embedding前5个值: {old_embedding.flatten()[:5]}")
        
        # 3. 比较两个embedding
        print("\n[步骤 3] 比较两个embedding...")
        compare_embeddings(new_embedding, old_embedding)
        print("\n✅ 验证完成!")
        
    except Exception as e:
        print(f"\n❌ 发生错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
