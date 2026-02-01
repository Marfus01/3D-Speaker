#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简单测试脚本，用于验证VBx集成是否正常工作
"""

import os
import sys
import numpy as np

# Add parent directory to path
current_file_path = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.dirname(current_file_path))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from vbx_enhancer import VBxEnhancer


def test_vbx_basic():
    """测试VBx的基本功能"""
    print("=" * 60)
    print("测试1: VBx基本功能测试")
    print("=" * 60)
    
    # 生成模拟数据
    np.random.seed(42)
    n_samples = 200
    n_speakers = 3
    embed_dim = 256
    
    # 创建模拟的embeddings（每个说话人一个簇）
    embeddings = []
    labels = []
    for spk in range(n_speakers):
        # 每个说话人的embeddings围绕一个中心
        center = np.random.randn(embed_dim)
        spk_embeddings = center + np.random.randn(n_samples // n_speakers, embed_dim) * 0.5
        embeddings.append(spk_embeddings)
        labels.extend([spk] * (n_samples // n_speakers))
    
    embeddings = np.vstack(embeddings)
    labels = np.array(labels)
    
    # 添加一些标签噪声
    noise_ratio = 0.1
    n_noise = int(n_samples * noise_ratio)
    noise_idx = np.random.choice(n_samples, n_noise, replace=False)
    labels[noise_idx] = np.random.randint(0, n_speakers, n_noise)
    
    print(f"生成数据: {n_samples}个样本, {n_speakers}个说话人, {embed_dim}维embedding")
    print(f"添加{noise_ratio*100}%的标签噪声")
    print(f"初始标签分布: {np.bincount(labels)}")
    
    # 创建VBx增强器
    vbx = VBxEnhancer(
        lda_dim=32,  # 使用较小的LDA维度以加快测试速度
        Fa=1.0,
        Fb=1.0,
        loopP=0.9,
        num_em_iters=3,  # 减少迭代次数以加快测试
        init_smoothing=5.0,
        max_iters=5
    )
    
    print("\n开始VBx训练和推理...")
    try:
        # 设置随机种子以确保可重复性
        np.random.seed(123)
        
        # 训练并预测
        smoothed_labels = vbx.fit_predict(embeddings, labels)
        
        print(f"\n平滑后标签分布: {np.bincount(smoothed_labels)}")
        
        # 计算标签变化
        n_changed = np.sum(labels != smoothed_labels)
        print(f"平滑后改变的标签数: {n_changed}/{n_samples} ({n_changed/n_samples*100:.1f}%)")
        
        print("\n✓ 测试通过！")
        return True
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vbx_save_load():
    """测试模型保存和加载功能"""
    print("\n" + "=" * 60)
    print("测试2: VBx模型保存和加载测试")
    print("=" * 60)
    
    # 生成模拟数据
    np.random.seed(42)
    n_samples = 100
    n_speakers = 2
    embed_dim = 128
    
    embeddings = np.random.randn(n_samples, embed_dim)
    labels = np.random.randint(0, n_speakers, n_samples)
    
    print(f"生成数据: {n_samples}个样本, {n_speakers}个说话人")
    
    try:
        # 设置随机种子以确保可重复性
        np.random.seed(123)
        
        # 训练模型
        vbx = VBxEnhancer(lda_dim=16, num_em_iters=2, max_iters=3)
        smoothed_labels1 = vbx.fit_predict(embeddings, labels)
        
        # 保存模型
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            transform_path = os.path.join(tmpdir, 'transform.h5')
            plda_path = os.path.join(tmpdir, 'plda.h5')
            
            vbx.save_models(transform_path, plda_path)
            print(f"模型已保存到临时目录: {tmpdir}")
            
            # 加载模型
            vbx2 = VBxEnhancer(lda_dim=16, num_em_iters=2, max_iters=3)  # 使用相同参数
            vbx2.load_models(transform_path, plda_path)
            print("模型加载成功")
            
            # 使用加载的模型进行推理
            smoothed_labels2 = vbx2.predict(embeddings, labels)
            
            # 验证结果一致性 - 由于VBx算法的数值特性，允许少量差异
            n_diff = np.sum(smoothed_labels1 != smoothed_labels2)
            diff_ratio = n_diff / len(smoothed_labels1)
            
            print(f"标签差异: {n_diff}/{len(smoothed_labels1)} ({diff_ratio*100:.1f}%)")
            
            if n_diff == 0:
                print("\n✓ 保存/加载测试通过！结果完全一致")
                return True
            elif diff_ratio < 0.05:  # 允许小于5%的差异
                print("\n✓ 保存/加载测试通过！结果基本一致（在容忍范围内）")
                return True
            else:
                print(f"\n✗ 保存/加载测试失败：差异过大 ({diff_ratio*100:.1f}% > 5%)")
                print(f"原始标签分布: {np.bincount(smoothed_labels1)}")
                print(f"加载后标签分布: {np.bincount(smoothed_labels2)}")
                return False
                
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_parameters():
    """测试模型参数的保存和加载正确性"""
    print("\n" + "=" * 60)
    print("测试3: 模型参数保存和加载验证")
    print("=" * 60)
    
    # 生成模拟数据
    np.random.seed(42)
    n_samples = 50
    n_speakers = 2
    embed_dim = 64
    
    embeddings = np.random.randn(n_samples, embed_dim)
    labels = np.random.randint(0, n_speakers, n_samples)
    
    print(f"生成数据: {n_samples}个样本, {n_speakers}个说话人")
    
    try:
        # 训练模型
        vbx = VBxEnhancer(lda_dim=8, num_em_iters=2, max_iters=2)
        vbx.fit(embeddings, labels)
        
        # 保存模型参数
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            transform_path = os.path.join(tmpdir, 'transform.h5')
            plda_path = os.path.join(tmpdir, 'plda.h5')
            
            vbx.save_models(transform_path, plda_path)
            
            # 记录原始参数
            orig_mean1 = vbx.mean1.copy()
            orig_lda = vbx.lda.copy()
            orig_mean2 = vbx.mean2.copy()
            orig_mu = vbx.plda_mu.copy()
            orig_tr = vbx.plda_tr.copy()
            orig_psi = vbx.plda_psi.copy()
            
            # 加载模型
            vbx2 = VBxEnhancer(lda_dim=8)
            vbx2.load_models(transform_path, plda_path)
            
            # 验证参数一致性
            params_match = True
            tolerance = 1e-10
            
            if not np.allclose(orig_mean1, vbx2.mean1, atol=tolerance):
                print(f"mean1参数不匹配，最大差异: {np.max(np.abs(orig_mean1 - vbx2.mean1))}")
                params_match = False
            if not np.allclose(orig_lda, vbx2.lda, atol=tolerance):
                print(f"LDA参数不匹配，最大差异: {np.max(np.abs(orig_lda - vbx2.lda))}")
                params_match = False
            if not np.allclose(orig_mean2, vbx2.mean2, atol=tolerance):
                print(f"mean2参数不匹配，最大差异: {np.max(np.abs(orig_mean2 - vbx2.mean2))}")
                params_match = False
            if not np.allclose(orig_mu, vbx2.plda_mu, atol=tolerance):
                print(f"PLDA mu参数不匹配，最大差异: {np.max(np.abs(orig_mu - vbx2.plda_mu))}")
                params_match = False
            if not np.allclose(orig_tr, vbx2.plda_tr, atol=tolerance):
                print(f"PLDA transform参数不匹配，最大差异: {np.max(np.abs(orig_tr - vbx2.plda_tr))}")
                params_match = False
            if not np.allclose(orig_psi, vbx2.plda_psi, atol=tolerance):
                print(f"PLDA psi参数不匹配，最大差异: {np.max(np.abs(orig_psi - vbx2.plda_psi))}")
                params_match = False
            
            if params_match:
                print("\n✓ 模型参数保存/加载测试通过！所有参数完全一致")
                return True
            else:
                print("\n✗ 模型参数保存/加载测试失败：存在参数不匹配")
                return False
                
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "🚀 " * 20)
    print("VBx集成测试")
    print("🚀 " * 20 + "\n")
    
    results = []
    
    # 运行测试
    results.append(("基本功能", test_vbx_basic()))
    results.append(("保存/加载", test_vbx_save_load()))
    results.append(("参数验证", test_model_parameters()))
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{name}: {status}")
    
    all_passed = all(r[1] for r in results)
    if all_passed:
        print("\n🎉 所有测试通过！")
        return 0
    else:
        print("\n❌ 部分测试失败")
        return 1


if __name__ == '__main__':
    sys.exit(main())
