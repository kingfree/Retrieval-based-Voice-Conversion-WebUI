#!/usr/bin/env python3
"""
Python RVC功能测试脚本
用于与Rust版本进行功能对比
"""

import os
import sys
import time
import numpy as np
import torch
import traceback
from pathlib import Path

# 添加项目路径
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

try:
    from infer.lib import rtrvc
    from configs.config import Config
    from i18n.i18n import I18nAuto
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("请确保在Retrieval-based-Voice-Conversion-WebUI目录下运行此脚本")
    sys.exit(1)


def create_test_config():
    """创建测试配置"""
    config = Config()

    return {
        'pth_path': 'assets/pretrained/f0G40k.pth',
        'index_path': 'logs/added_index.index',
        'pitch': 0.0,
        'formant': 0.0,
        'index_rate': 0.5,
        'n_cpu': 4,
        'f0method': 'rmvpe'
    }


def print_config(config):
    """打印配置信息"""
    print("配置信息:")
    print(f"  模型路径: {config['pth_path']}")
    print(f"  索引路径: {config['index_path']}")
    print(f"  音高调整: {config['pitch']}")
    print(f"  共振峰: {config['formant']}")
    print(f"  索引率: {config['index_rate']}")
    print(f"  CPU核心: {config['n_cpu']}")
    print(f"  F0方法: {config['f0method']}")
    print()


def test_basic_functionality(config):
    """测试基础功能"""
    print("🔧 测试基础功能...")

    try:
        # 创建虚拟队列用于测试
        from multiprocessing import Queue
        inp_q = Queue()
        opt_q = Queue()

        # 创建RVC实例
        rvc_instance = rtrvc.RVC(
            key=config['pitch'],
            formant=config['formant'],
            pth_path=config['pth_path'],
            index_path=config['index_path'],
            index_rate=config['index_rate'],
            n_cpu=config['n_cpu'],
            inp_q=inp_q,
            opt_q=opt_q,
            config=Config()
        )

        print("  - RVC实例创建成功")

        # 测试参数更新
        print("  - 测试参数更新")
        rvc_instance.change_key(12.0)  # 升高一个八度
        rvc_instance.change_formant(2.0)  # 改变共振峰

        # 检查设备和模型状态
        print("  - 模型状态:")
        print(f"    设备: {rvc_instance.device}")
        print(f"    半精度: {rvc_instance.is_half}")
        print(f"    目标采样率: {rvc_instance.tgt_sr}")
        print(f"    F0条件: {rvc_instance.if_f0}")
        print(f"    版本: {rvc_instance.version}")

        # 检查索引状态
        index_loaded = hasattr(rvc_instance, 'index')
        print(f"    索引已加载: {index_loaded}")

        print("✅ 基础功能测试完成\n")
        return rvc_instance

    except Exception as e:
        print(f"❌ 基础功能测试失败: {e}")
        traceback.print_exc()
        return None


def test_audio_processing(rvc_instance, config):
    """测试音频处理流程"""
    print("🎵 测试音频处理流程...")

    if rvc_instance is None:
        print("  ⚠️ RVC实例未创建，跳过音频处理测试\n")
        return

    try:
        # 创建测试音频数据 (1秒的正弦波，16kHz)
        sample_rate = 16000
        duration = 1.0  # 1秒
        frequency = 440.0  # A4音符

        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
        test_audio = np.sin(2 * np.pi * frequency * t) * 0.5

        print(f"  - 创建测试音频: {frequency}Hz正弦波，{duration}秒")
        print(f"  - 音频长度: {len(test_audio)} 样本")

        # 转换为torch tensor
        input_wav = torch.from_numpy(test_audio).to(rvc_instance.device)

        # 测试推理 (如果模型已加载)
        model_loaded = hasattr(rvc_instance, 'net_g') and rvc_instance.net_g is not None
        if model_loaded:
            print("  - 执行音频推理...")

            try:
                # 设置推理参数
                block_frame_16k = 4000
                skip_head = 1600
                return_length = 2400

                start_time = time.time()
                output = rvc_instance.infer(
                    input_wav,
                    block_frame_16k,
                    skip_head,
                    return_length,
                    config['f0method']
                )
                end_time = time.time()

                print(f"  - 推理成功! 耗时: {end_time - start_time:.3f}s")
                print(f"  - 输出长度: {len(output)} 样本")

                # 输出统计信息
                output_np = output.cpu().numpy() if isinstance(output, torch.Tensor) else output
                max_val = np.max(output_np)
                min_val = np.min(output_np)
                mean_val = np.mean(output_np)

                print(f"  - 输出统计: 最大值={max_val:.4f}, 最小值={min_val:.4f}, 均值={mean_val:.4f}")

            except Exception as e:
                print(f"  ⚠️ 推理失败: {e}")
        else:
            print("  ⚠️ 模型未加载，跳过音频推理测试")

        # 测试F0提取
        print("  - 测试F0提取功能")
        test_f0_extraction(rvc_instance, test_audio, config)

        print("✅ 音频处理测试完成\n")

    except Exception as e:
        print(f"❌ 音频处理测试失败: {e}")
        traceback.print_exc()


def test_f0_extraction(rvc_instance, audio, config):
    """测试F0提取功能"""
    methods = ['pm', 'harvest', 'rmvpe']

    for method in methods:
        print(f"    - 测试{method.upper()}方法")
        try:
            start_time = time.time()
            pitch, pitchf = rvc_instance.get_f0(
                torch.from_numpy(audio).to(rvc_instance.device),
                f0_up_key=0.0,
                n_cpu=config['n_cpu'],
                f0method=method
            )
            end_time = time.time()

            print(f"      F0提取完成，耗时: {end_time - start_time:.3f}s")
            print(f"      pitch长度={len(pitch)}, pitchf长度={len(pitchf)}")

            # 计算基础统计
            if len(pitchf) > 0:
                pitchf_np = pitchf.cpu().numpy() if isinstance(pitchf, torch.Tensor) else pitchf
                non_zero_pitch = pitchf_np[pitchf_np > 0.0]

                if len(non_zero_pitch) > 0:
                    mean_pitch = np.mean(non_zero_pitch)
                    print(f"      平均音高: {mean_pitch:.2f}Hz")
                else:
                    print("      未检测到音高信息")

        except Exception as e:
            print(f"      ⚠️ {method.upper()}方法失败: {e}")


def create_test_signal(frequency, duration, sample_rate):
    """创建测试音频信号"""
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    return np.sin(2 * np.pi * frequency * t)


def calculate_rms(signal):
    """计算音频信号的RMS"""
    return np.sqrt(np.mean(signal ** 2))


def main():
    """主函数"""
    print("RVC Python实时推理功能测试")
    print("===========================")

    # 创建测试配置
    config = create_test_config()

    # 显示配置信息
    print_config(config)

    # 检查模型文件是否存在
    pth_exists = os.path.exists(config['pth_path'])
    index_exists = os.path.exists(config['index_path'])

    print("文件状态检查:")
    print(f"  模型文件 (.pth): {'✅' if pth_exists else '❌'} {config['pth_path']}")
    print(f"  索引文件 (.index): {'✅' if index_exists else '❌'} {config['index_path']}")
    print()

    if not pth_exists:
        print("⚠️ 警告: 模型文件不存在，某些功能将无法测试")

    # 测试基础功能
    rvc_instance = test_basic_functionality(config)

    # 测试音频处理
    test_audio_processing(rvc_instance, config)

    # 性能测试
    if rvc_instance is not None:
        print("⚡ 性能基准测试...")
        performance_test(rvc_instance, config)

    print("\n✅ 所有测试完成！")


def performance_test(rvc_instance, config):
    """性能基准测试"""
    try:
        # 创建不同长度的测试信号
        test_cases = [
            (440.0, 0.5, "短音频"),   # 0.5秒
            (440.0, 1.0, "中等音频"), # 1.0秒
            (440.0, 2.0, "长音频"),   # 2.0秒
        ]

        print("  - F0提取性能测试:")
        for freq, duration, desc in test_cases:
            audio = create_test_signal(freq, duration, 16000)

            print(f"    {desc} ({duration}s):")
            for method in ['pm', 'harvest', 'rmvpe']:
                try:
                    start_time = time.time()
                    rvc_instance.get_f0(
                        torch.from_numpy(audio).to(rvc_instance.device),
                        f0_up_key=0.0,
                        n_cpu=config['n_cpu'],
                        f0method=method
                    )
                    end_time = time.time()

                    processing_time = end_time - start_time
                    real_time_factor = duration / processing_time

                    print(f"      {method.upper()}: {processing_time:.3f}s (实时率: {real_time_factor:.1f}x)")
                except Exception as e:
                    print(f"      {method.upper()}: 失败 ({e})")

        print("✅ 性能测试完成")

    except Exception as e:
        print(f"❌ 性能测试失败: {e}")


if __name__ == "__main__":
    main()
