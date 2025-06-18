#!/usr/bin/env python3
"""
生成RVC测试用例
使用kikiV1模型对test.wav进行推理，生成参考输出用于Rust端验证
"""

import os
import sys
import json
import torch
import numpy as np
import scipy.io.wavfile as wavfile
from pathlib import Path
import traceback

# 添加项目路径
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

try:
    from infer.lib.rtrvc import RVC
    from configs.config import Config
    from infer.lib.audio import load_audio
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("请确保在Retrieval-based-Voice-Conversion-WebUI目录下运行此脚本")
    sys.exit(1)


class TestCaseGenerator:
    def __init__(self):
        self.config = Config()
        self.model_path = "assets/weights/kikiV1.pth"
        self.index_path = "logs/kikiV1.index"
        self.input_audio = "test.wav"
        self.output_audio = "test_kikiV1_ref.wav"
        self.rvc_instance = None

    def check_files(self):
        """检查必要文件是否存在"""
        files_to_check = [
            self.model_path,
            self.index_path,
            self.input_audio
        ]

        print("文件检查:")
        for file_path in files_to_check:
            exists = os.path.exists(file_path)
            print(f"  {file_path}: {'✅' if exists else '❌'}")
            if not exists:
                raise FileNotFoundError(f"文件不存在: {file_path}")

    def load_model(self):
        """加载RVC模型"""
        print(f"正在加载模型...")
        print(f"  模型文件: {self.model_path}")
        print(f"  索引文件: {self.index_path}")

        try:
            # 创建虚拟队列
            from multiprocessing import Queue
            inp_q = Queue()
            opt_q = Queue()

            # 创建RVC实例
            self.rvc_instance = RVC(
                key=0,                    # 音高调整
                formant=0,               # 共振峰调整
                pth_path=self.model_path,
                index_path=self.index_path,
                index_rate=0.75,         # 索引率
                n_cpu=4,                 # CPU核心数
                inp_q=inp_q,
                opt_q=opt_q,
                config=self.config
            )

            print("✅ 模型加载成功")
            print(f"  设备: {self.rvc_instance.device}")
            print(f"  半精度: {self.rvc_instance.is_half}")
            print(f"  目标采样率: {self.rvc_instance.tgt_sr}")
            print(f"  F0条件: {self.rvc_instance.if_f0}")
            print(f"  版本: {self.rvc_instance.version}")

            return True

        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            traceback.print_exc()
            return False

    def load_input_audio(self):
        """加载输入音频"""
        print(f"正在加载输入音频: {self.input_audio}")

        try:
            # 使用RVC的音频加载函数
            audio = load_audio(self.input_audio, 16000)

            print(f"✅ 音频加载成功")
            print(f"  采样率: 16000 Hz")
            print(f"  长度: {len(audio)} 样本 ({len(audio)/16000:.2f}秒)")
            print(f"  数据类型: {audio.dtype}")
            print(f"  数值范围: [{np.min(audio):.4f}, {np.max(audio):.4f}]")

            return audio

        except Exception as e:
            print(f"❌ 音频加载失败: {e}")
            traceback.print_exc()
            return None

    def perform_inference(self, audio):
        """执行推理"""
        print("开始推理...")

        try:
            # 转换为torch tensor
            input_wav = torch.from_numpy(audio).to(self.rvc_instance.device)

            # 推理参数
            block_frame_16k = 4000  # 块大小
            skip_head = 1600        # 跳过头部
            return_length = 2400    # 返回长度
            f0method = "rmvpe"      # F0提取方法

            print(f"  推理参数:")
            print(f"    块大小: {block_frame_16k}")
            print(f"    跳过头部: {skip_head}")
            print(f"    返回长度: {return_length}")
            print(f"    F0方法: {f0method}")

            # 执行推理
            import time
            start_time = time.time()

            output = self.rvc_instance.infer(
                input_wav,
                block_frame_16k,
                skip_head,
                return_length,
                f0method
            )

            end_time = time.time()
            processing_time = end_time - start_time

            print(f"✅ 推理完成")
            print(f"  处理时间: {processing_time:.3f}秒")
            print(f"  实时率: {len(audio)/16000/processing_time:.1f}x")

            # 转换为numpy数组
            if isinstance(output, torch.Tensor):
                output = output.cpu().numpy()

            print(f"  输出长度: {len(output)} 样本 ({len(output)/16000:.2f}秒)")
            print(f"  输出范围: [{np.min(output):.4f}, {np.max(output):.4f}]")

            return output

        except Exception as e:
            print(f"❌ 推理失败: {e}")
            traceback.print_exc()
            return None

    def save_output(self, output):
        """保存输出音频"""
        print(f"保存输出音频: {self.output_audio}")

        try:
            # 确保输出在合理范围内
            output = np.clip(output, -1.0, 1.0)

            # 转换为16位整数格式
            output_int16 = (output * 32767).astype(np.int16)

            # 保存为WAV文件
            wavfile.write(self.output_audio, 16000, output_int16)

            print(f"✅ 输出保存成功")
            print(f"  文件: {self.output_audio}")
            print(f"  采样率: 16000 Hz")
            print(f"  长度: {len(output)} 样本")

            return True

        except Exception as e:
            print(f"❌ 保存失败: {e}")
            traceback.print_exc()
            return False

    def generate_test_metadata(self, input_audio, output_audio):
        """生成测试元数据"""
        metadata = {
            "model_path": self.model_path,
            "index_path": self.index_path,
            "input_audio": self.input_audio,
            "output_audio": self.output_audio,
            "parameters": {
                "pitch": 0,
                "formant": 0,
                "index_rate": 0.75,
                "f0method": "rmvpe"
            },
            "input_stats": {
                "length": len(input_audio),
                "duration": len(input_audio) / 16000,
                "min_value": float(np.min(input_audio)),
                "max_value": float(np.max(input_audio)),
                "rms": float(np.sqrt(np.mean(input_audio ** 2)))
            },
            "output_stats": {
                "length": len(output_audio),
                "duration": len(output_audio) / 16000,
                "min_value": float(np.min(output_audio)),
                "max_value": float(np.max(output_audio)),
                "rms": float(np.sqrt(np.mean(output_audio ** 2)))
            }
        }

        metadata_file = "test_case_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"✅ 测试元数据保存到: {metadata_file}")
        return metadata

    def run(self):
        """运行测试用例生成"""
        print("RVC测试用例生成器")
        print("=" * 50)

        try:
            # 1. 检查文件
            self.check_files()

            # 2. 加载模型
            if not self.load_model():
                return False

            # 3. 加载输入音频
            input_audio = self.load_input_audio()
            if input_audio is None:
                return False

            # 4. 执行推理
            output_audio = self.perform_inference(input_audio)
            if output_audio is None:
                return False

            # 5. 保存输出
            if not self.save_output(output_audio):
                return False

            # 6. 生成元数据
            metadata = self.generate_test_metadata(input_audio, output_audio)

            print("\n" + "=" * 50)
            print("✅ 测试用例生成成功!")
            print(f"输入文件: {self.input_audio}")
            print(f"输出文件: {self.output_audio}")
            print(f"元数据文件: test_case_metadata.json")

            return True

        except Exception as e:
            print(f"❌ 测试用例生成失败: {e}")
            traceback.print_exc()
            return False


def main():
    """主函数"""
    generator = TestCaseGenerator()
    success = generator.run()

    if success:
        print("\n下一步: 在Rust端实现相同的推理逻辑并验证输出")
        sys.exit(0)
    else:
        print("\n测试用例生成失败")
        sys.exit(1)


if __name__ == "__main__":
    main()
