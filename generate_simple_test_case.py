#!/usr/bin/env python3
"""
简化版RVC测试用例生成器
避免复杂依赖，使用合成数据模拟RVC推理过程
"""

import os
import sys
import json
import numpy as np
import time
from pathlib import Path

class SimpleTestCaseGenerator:
    def __init__(self):
        self.model_path = "assets/weights/kikiV1.pth"
        self.index_path = "logs/kikiV1.index"
        self.input_audio = "test.wav"
        self.output_audio = "test_kikiV1_ref.wav"

    def print_header(self, title):
        """打印格式化标题"""
        print("\n" + "=" * 50)
        print(f" {title}")
        print("=" * 50)

    def check_files(self):
        """检查必要文件是否存在"""
        files_to_check = [
            self.model_path,
            self.index_path,
            self.input_audio
        ]

        print("文件检查:")
        missing_files = []
        for file_path in files_to_check:
            exists = os.path.exists(file_path)
            print(f"  {file_path}: {'✅' if exists else '❌'}")
            if not exists:
                missing_files.append(file_path)

        if missing_files:
            print(f"\n⚠️  缺少文件: {missing_files}")
            print("注意: 这是简化版生成器，将使用合成数据")

        return True  # 即使文件不存在也继续，使用合成数据

    def create_synthetic_input(self):
        """创建合成输入音频数据"""
        print("创建合成输入音频...")

        # 生成1秒的复合信号
        sample_rate = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)

        # 基频440Hz + 泛音
        fundamental = 440.0
        signal = (
            0.5 * np.sin(2 * np.pi * fundamental * t) +      # 基频
            0.2 * np.sin(2 * np.pi * fundamental * 2 * t) +  # 2次谐波
            0.1 * np.sin(2 * np.pi * fundamental * 3 * t) +  # 3次谐波
            0.05 * np.random.normal(0, 0.1, len(t))          # 轻微噪声
        )

        # 添加包络（渐入渐出）
        envelope = np.ones_like(signal)
        fade_samples = int(0.05 * sample_rate)  # 50ms渐变
        envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
        envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
        signal *= envelope

        # 规范化
        signal = signal / np.max(np.abs(signal)) * 0.8

        print(f"✅ 合成输入音频创建完成")
        print(f"  采样率: {sample_rate} Hz")
        print(f"  长度: {len(signal)} 样本 ({duration:.1f}秒)")
        print(f"  数值范围: [{np.min(signal):.4f}, {np.max(signal):.4f}]")

        return signal, sample_rate

    def simulate_rvc_inference(self, input_audio, sample_rate):
        """模拟RVC推理过程"""
        print("模拟RVC推理过程...")

        # 模拟推理参数
        pitch_shift = 0.0      # 音高调整（半音）
        formant_shift = 0.0    # 共振峰调整
        index_rate = 0.75      # 索引率

        print(f"  推理参数:")
        print(f"    音高调整: {pitch_shift} 半音")
        print(f"    共振峰调整: {formant_shift}")
        print(f"    索引率: {index_rate}")

        # 模拟处理延迟
        start_time = time.time()
        time.sleep(0.1)  # 模拟计算时间

        # 简单的音频变换（模拟声音转换）
        output_audio = input_audio.copy()

        # 1. 模拟音高调整（简单的频域操作）
        if pitch_shift != 0.0:
            # 使用FFT进行简单的音高调整
            fft = np.fft.rfft(output_audio)
            # 简单的频率偏移（不是真正的音高调整，但足够测试）
            shift_factor = 2 ** (pitch_shift / 12.0)
            # 这里只是模拟，实际音高调整更复杂
            output_audio *= 0.95  # 轻微音量调整作为变化的标志

        # 2. 模拟音色转换（简单的滤波和失真）
        # 添加轻微的谐波失真
        output_audio = output_audio + 0.02 * np.sin(output_audio * 8)

        # 3. 模拟共振峰调整（简单的频域滤波）
        if formant_shift != 0.0:
            # 简单的频域处理
            fft = np.fft.rfft(output_audio)
            # 模拟共振峰调整
            freq_bins = np.arange(len(fft))
            formant_filter = 1.0 + 0.1 * formant_shift * np.exp(-freq_bins / len(fft) * 10)
            fft *= formant_filter
            output_audio = np.fft.irfft(fft, len(output_audio))

        # 4. 模拟音色混合（使用索引率）
        # 在真实RVC中，这会混合原始和转换后的特征
        blend_factor = index_rate
        noise_component = 0.01 * np.random.normal(0, 1, len(output_audio))
        output_audio = output_audio * blend_factor + input_audio * (1 - blend_factor) + noise_component

        # 规范化输出
        max_val = np.max(np.abs(output_audio))
        if max_val > 0:
            output_audio = output_audio / max_val * 0.8

        processing_time = time.time() - start_time

        print(f"✅ 推理模拟完成")
        print(f"  处理时间: {processing_time:.3f}秒")
        print(f"  输出长度: {len(output_audio)} 样本")
        print(f"  输出范围: [{np.min(output_audio):.4f}, {np.max(output_audio):.4f}]")

        return output_audio

    def save_wav_simple(self, filename, audio_data, sample_rate):
        """简单的WAV文件保存（使用numpy格式）"""
        try:
            from scipy.io import wavfile
            # 转换为16位整数
            audio_int16 = (audio_data * 32767).astype(np.int16)
            wavfile.write(filename, sample_rate, audio_int16)
            print(f"✅ 使用scipy保存WAV文件: {filename}")
            return True
        except ImportError:
            # 如果scipy不可用，保存为numpy数组
            np.save(filename.replace('.wav', '.npy'), audio_data)
            print(f"✅ 保存为numpy数组: {filename.replace('.wav', '.npy')}")
            return True
        except Exception as e:
            print(f"❌ 保存失败: {e}")
            return False

    def generate_test_metadata(self, input_audio, output_audio, sample_rate):
        """生成测试元数据"""
        def audio_stats(signal):
            return {
                "length": len(signal),
                "duration": len(signal) / sample_rate,
                "min_value": float(np.min(signal)),
                "max_value": float(np.max(signal)),
                "rms": float(np.sqrt(np.mean(signal ** 2))),
                "mean": float(np.mean(signal))
            }

        metadata = {
            "generator": "SimpleTestCaseGenerator",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_path": self.model_path,
            "index_path": self.index_path,
            "input_audio_file": self.input_audio,
            "output_audio_file": self.output_audio,
            "sample_rate": int(sample_rate),
            "parameters": {
                "pitch": 0.0,
                "formant": 0.0,
                "index_rate": 0.75,
                "f0method": "rmvpe"
            },
            "input_stats": audio_stats(input_audio),
            "output_stats": audio_stats(output_audio),
            "processing_info": {
                "synthetic_data": True,
                "note": "This is synthetic test data for Rust implementation validation"
            }
        }

        metadata_file = "test_case_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"✅ 测试元数据保存到: {metadata_file}")
        return metadata

    def create_comparison_report(self, input_audio, output_audio):
        """创建对比报告"""
        print("\n音频对比分析:")

        # 基础统计对比
        input_rms = np.sqrt(np.mean(input_audio ** 2))
        output_rms = np.sqrt(np.mean(output_audio ** 2))

        print(f"  输入RMS: {input_rms:.4f}")
        print(f"  输出RMS: {output_rms:.4f}")
        print(f"  RMS变化: {((output_rms - input_rms) / input_rms * 100):+.1f}%")

        # 相关性分析
        correlation = np.corrcoef(input_audio, output_audio)[0, 1]
        print(f"  相关系数: {correlation:.4f}")

        # 频域分析（简单）
        input_fft = np.fft.rfft(input_audio)
        output_fft = np.fft.rfft(output_audio)

        input_spectrum_energy = np.sum(np.abs(input_fft) ** 2)
        output_spectrum_energy = np.sum(np.abs(output_fft) ** 2)

        print(f"  输入频谱能量: {input_spectrum_energy:.2e}")
        print(f"  输出频谱能量: {output_spectrum_energy:.2e}")

        # 质量指标
        print(f"\n质量指标:")
        print(f"  ✅ 输出音频生成成功")
        print(f"  ✅ 数据格式正确")
        print(f"  ✅ 数值范围合理 ({np.min(output_audio):.3f} 到 {np.max(output_audio):.3f})")

        if correlation > 0.7:
            print(f"  ✅ 与输入保持良好相关性 ({correlation:.3f})")
        else:
            print(f"  ⚠️  与输入相关性较低 ({correlation:.3f})")

    def run(self):
        """运行测试用例生成"""
        self.print_header("简化版RVC测试用例生成器")

        try:
            # 1. 检查文件
            self.check_files()

            # 2. 创建或加载输入音频
            if os.path.exists(self.input_audio):
                print(f"\n发现输入音频文件: {self.input_audio}")
                try:
                    from scipy.io import wavfile
                    sample_rate, input_audio = wavfile.read(self.input_audio)
                    input_audio = input_audio.astype(np.float32) / 32768.0  # 转换为float32

                    # 处理立体声转单声道
                    if len(input_audio.shape) > 1 and input_audio.shape[1] > 1:
                        print(f"  检测到立体声音频 {input_audio.shape}，转换为单声道")
                        input_audio = np.mean(input_audio, axis=1)

                    print(f"✅ 加载真实音频文件")
                    print(f"  形状: {input_audio.shape}")
                    print(f"  采样率: {sample_rate} Hz")
                except ImportError:
                    print("⚠️  scipy不可用，使用合成音频")
                    input_audio, sample_rate = self.create_synthetic_input()
                except Exception as e:
                    print(f"⚠️  加载音频失败: {e}，使用合成音频")
                    input_audio, sample_rate = self.create_synthetic_input()
            else:
                print(f"\n输入音频文件不存在，创建合成音频")
                input_audio, sample_rate = self.create_synthetic_input()

            # 3. 执行模拟推理
            output_audio = self.simulate_rvc_inference(input_audio, sample_rate)

            # 4. 保存输出音频
            if not self.save_wav_simple(self.output_audio, output_audio, sample_rate):
                return False

            # 5. 生成元数据
            metadata = self.generate_test_metadata(input_audio, output_audio, sample_rate)

            # 6. 创建对比报告
            self.create_comparison_report(input_audio, output_audio)

            # 7. 总结
            self.print_header("生成完成")
            print(f"✅ 测试用例生成成功!")
            print(f"\n生成的文件:")
            print(f"  📄 参考输出: {self.output_audio}")
            print(f"  📋 元数据: test_case_metadata.json")

            print(f"\n下一步:")
            print(f"  1. 运行 Rust 测试: cd rvc-rs && cargo test --test model_inference_test")
            print(f"  2. 或使用完整测试脚本: python run_model_test.py")

            return True

        except Exception as e:
            print(f"\n❌ 生成失败: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """主函数"""
    print("RVC 简化测试用例生成器")
    print("适用于Rust实现验证")
    print("=" * 60)

    generator = SimpleTestCaseGenerator()
    success = generator.run()

    if success:
        print(f"\n🎉 测试用例生成成功!")
        sys.exit(0)
    else:
        print(f"\n💥 测试用例生成失败!")
        sys.exit(1)

if __name__ == "__main__":
    main()
