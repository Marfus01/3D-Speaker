import torch
import sys

# 导入三种IRSE模型实现
from model_irse_ms import IR_101 as IR_101_ms
from model_irse_HuangYG123 import IR_101 as IR_101_huang
from model_irse_TFace import IR_101 as IR_101_tface

def try_load_model(model_class, bin_path, model_name):
	try:
		model = model_class((112, 112))
		state_dict = torch.load(bin_path, map_location='cpu')
		# 兼容部分bin文件保存为dict结构
		if 'state_dict' in state_dict:
			state_dict = state_dict['state_dict']
		model.load_state_dict(state_dict, strict=True)
		print(f"成功用 {model_name} 加载权重！")
		return model
	except Exception as e:
		print(f"用 {model_name} 加载权重失败: {e}")
		return None

if __name__ == "__main__":
	bin_path = "/Users/wangchen/文档/Research/tv_series_plus/3D-Speaker/runtime/models/pytorch_model_tongyi.bin"
	print("尝试用 model_irse_ms 加载...")
	model = try_load_model(IR_101_ms, bin_path, "model_irse_ms")
	if model is None:
		print("尝试用 model_irse_HuangYG123 加载...")
		model = try_load_model(IR_101_huang, bin_path, "model_irse_HuangYG123")
	if model is None:
		print("尝试用 model_irse_TFace 加载...")
		model = try_load_model(IR_101_tface, bin_path, "model_irse_TFace")
	if model is None:
		print("所有模型加载均失败，请检查权重文件与模型定义是否匹配。")
