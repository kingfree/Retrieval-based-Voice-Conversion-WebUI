#!/usr/bin/env python3
"""
RVC模型测试运行脚本
自动执行Python测试用例生成和Rust端验证

使用方法:
python run_model_test.py
"""

import os
import sys
import subprocess
import time
from pathlib import Path

class ModelTestRunner:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.required_files = {
            "model": "assets/weights/kikiV1.pth",
            "index": "logs/kikiV1.index",
            "audio": "test.wav",
            "python_generator": "generate_simple_test_case.py",
            "rust_test": "rvc-rs/rvc-lib/tests/model_inference_test.rs"
        }

    def print_header(self, title):
        """打印格式化的标题"""
        print("\n" + "=" * 60)
        print(f" {title}")
        print("=" * 60)

    def print_step(self, step, description):
        """打印步骤信息"""
        print(f"\n[步骤 {step}] {description}")
        print("-" * 40)

    def check_prerequisites(self):
        """检查必要的文件和环境"""
        self.print_step(1, "检查必要文件和环境")

        missing_files = []

        for name, path in self.required_files.items():
            full_path = self.project_root / path
            exists = full_path.exists()
            status = "✅" if exists else "❌"
            print(f"  {status} {name}: {path}")

            if not exists:
                missing_files.append((name, path))

        if missing_files:
            print(f"\n❌ 缺少必要文件:")
            for name, path in missing_files:
                print(f"  - {name}: {path}")

            if any(name in ["model", "index", "audio"] for name, _ in missing_files):
                print("\n提示:")
                print("  - kikiV1.pth: 请确保模型文件在 assets/weights/ 目录下")
                print("  - kikiV1.index: 请确保索引文件在 logs/ 目录下")
                print("  - test.wav: 请确保测试音频文件在项目根目录下")
                return False

        # 检查Python环境
        print(f"\n  检查Python环境:")
        try:
            import torch
            print(f"  ✅ PyTorch: {torch.__version__}")
        except ImportError:
            print(f"  ❌ PyTorch 未安装")
            return False

        try:
            import scipy.io
            print(f"  ✅ scipy 已安装")
        except ImportError:
            print(f"  ⚠️  scipy 未安装，将使用合成音频数据")

        # 检查Rust环境
        print(f"\n  检查Rust环境:")
        try:
            result = subprocess.run(["cargo", "--version"],
                                  capture_output=True, text=True, cwd=self.project_root)
            if result.returncode == 0:
                print(f"  ✅ Cargo: {result.stdout.strip()}")
            else:
                print(f"  ❌ Cargo 未找到")
                return False
        except FileNotFoundError:
            print(f"  ❌ Cargo 未找到，请安装Rust")
            return False

        print(f"\n✅ 所有必要条件检查通过")
        return True

    def run_python_generator(self):
        """运行Python测试用例生成器"""
        self.print_step(2, "运行Python测试用例生成器")

        script_path = self.project_root / "generate_simple_test_case.py"

        print(f"执行: python {script_path}")
        start_time = time.time()

        try:
            result = subprocess.run([
                sys.executable, str(script_path)
            ], cwd=self.project_root, capture_output=True, text=True)

            end_time = time.time()
            execution_time = end_time - start_time

            print(f"执行时间: {execution_time:.2f}秒")

            if result.returncode == 0:
                print("✅ Python测试用例生成成功")
                print("\n输出:")
                print(result.stdout)
                return True
            else:
                print("❌ Python测试用例生成失败")
                print("\n错误输出:")
                print(result.stderr)
                if result.stdout:
                    print("\n标准输出:")
                    print(result.stdout)
                return False

        except Exception as e:
            print(f"❌ 执行Python生成器时出错: {e}")
            return False

    def check_generated_files(self):
        """检查生成的文件"""
        self.print_step(3, "检查生成的文件")

        expected_files = [
            "test_kikiV1_ref.wav",
            "test_case_metadata.json"
        ]

        all_exist = True
        for filename in expected_files:
            file_path = self.project_root / filename
            exists = file_path.exists()
            status = "✅" if exists else "❌"
            print(f"  {status} {filename}")

            if exists and filename.endswith('.json'):
                # 显示元数据信息
                try:
                    import json
                    with open(file_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)

                    print(f"    模型: {metadata.get('model_path', 'N/A')}")
                    print(f"    索引: {metadata.get('index_path', 'N/A')}")

                    if 'input_stats' in metadata:
                        input_stats = metadata['input_stats']
                        print(f"    输入长度: {input_stats.get('length', 'N/A')} 样本")
                        print(f"    输入时长: {input_stats.get('duration', 'N/A'):.2f}秒")

                    if 'output_stats' in metadata:
                        output_stats = metadata['output_stats']
                        print(f"    输出长度: {output_stats.get('length', 'N/A')} 样本")
                        print(f"    输出时长: {output_stats.get('duration', 'N/A'):.2f}秒")

                except Exception as e:
                    print(f"    警告: 无法读取元数据 - {e}")

            if not exists:
                all_exist = False

        return all_exist

    def run_rust_tests(self):
        """运行Rust端验证测试"""
        self.print_step(4, "运行Rust端验证测试")

        rust_dir = self.project_root / "rvc-rs"

        print(f"切换到目录: {rust_dir}")
        print(f"执行: cargo test --test model_inference_test -- --nocapture")

        start_time = time.time()

        try:
            result = subprocess.run([
                "cargo", "test", "--test", "model_inference_test", "--", "--nocapture"
            ], cwd=rust_dir, capture_output=True, text=True)

            end_time = time.time()
            execution_time = end_time - start_time

            print(f"执行时间: {execution_time:.2f}秒")

            if result.returncode == 0:
                print("✅ Rust测试验证成功")
                print("\n测试输出:")
                print(result.stdout)
                return True
            else:
                print("❌ Rust测试验证失败")
                print("\n错误输出:")
                print(result.stderr)
                if result.stdout:
                    print("\n标准输出:")
                    print(result.stdout)
                return False

        except Exception as e:
            print(f"❌ 执行Rust测试时出错: {e}")
            return False

    def run_rust_check(self):
        """运行Rust代码检查"""
        print(f"\n[额外检查] 运行Rust代码检查")
        print("-" * 40)

        rust_dir = self.project_root / "rvc-rs"

        try:
            result = subprocess.run([
                "cargo", "check", "-p", "rvc-lib", "--manifest-path", "Cargo.toml"
            ], cwd=rust_dir, capture_output=True, text=True)

            if result.returncode == 0:
                print("✅ Rust代码检查通过")
                return True
            else:
                print("❌ Rust代码检查失败")
                print(result.stderr)
                return False

        except Exception as e:
            print(f"❌ Rust代码检查出错: {e}")
            return False

    def cleanup_generated_files(self, keep_files=True):
        """清理生成的文件"""
        if not keep_files:
            print(f"\n[清理] 删除生成的文件")
            files_to_clean = [
                "test_kikiV1_ref.wav",
                "test_case_metadata.json",
                "test_kikiV1_rust.wav"
            ]

            for filename in files_to_clean:
                file_path = self.project_root / filename
                if file_path.exists():
                    file_path.unlink()
                    print(f"  删除: {filename}")

    def print_summary(self, success_steps):
        """打印测试总结"""
        self.print_header("测试总结")

        total_steps = 4
        success_count = len(success_steps)

        print(f"完成步骤: {success_count}/{total_steps}")

        step_names = [
            "环境检查",
            "Python测试用例生成",
            "文件生成验证",
            "Rust端验证"
        ]

        for i, name in enumerate(step_names, 1):
            status = "✅" if i in success_steps else "❌"
            print(f"  {status} 步骤{i}: {name}")

        if success_count == total_steps:
            print(f"\n🎉 所有测试步骤完成! 模型加载和推理验证成功!")
            print(f"\n生成的文件:")
            print(f"  - test_kikiV1_ref.wav (Python参考输出)")
            print(f"  - test_case_metadata.json (测试元数据)")
            print(f"  - test_kikiV1_rust.wav (Rust输出,如果生成)")

        else:
            print(f"\n⚠️  有 {total_steps - success_count} 个步骤失败")
            print(f"请检查上面的错误信息并解决相关问题")

    def run(self):
        """运行完整的测试流程"""
        self.print_header("RVC模型测试流程")

        success_steps = []

        try:
            # 步骤1: 检查必要条件
            if self.check_prerequisites():
                success_steps.append(1)
            else:
                self.print_summary(success_steps)
                return False

            # 步骤2: 运行Python生成器
            if self.run_python_generator():
                success_steps.append(2)
            else:
                self.print_summary(success_steps)
                return False

            # 步骤3: 检查生成的文件
            if self.check_generated_files():
                success_steps.append(3)
            else:
                print("❌ 生成的文件不完整，但继续进行Rust测试")

            # 额外: Rust代码检查
            self.run_rust_check()

            # 步骤4: 运行Rust测试
            if self.run_rust_tests():
                success_steps.append(4)

            # 打印总结
            self.print_summary(success_steps)

            return len(success_steps) == 4

        except KeyboardInterrupt:
            print(f"\n\n⚠️  用户中断测试")
            self.print_summary(success_steps)
            return False

        except Exception as e:
            print(f"\n\n❌ 测试过程中出现意外错误: {e}")
            self.print_summary(success_steps)
            return False

def main():
    """主函数"""
    print("RVC Rust重写项目 - 模型测试验证")
    print("作者: AI Assistant")
    print("=" * 60)

    runner = ModelTestRunner()
    success = runner.run()

    print(f"\n" + "=" * 60)
    if success:
        print("✅ 测试完成: 成功!")
        sys.exit(0)
    else:
        print("❌ 测试完成: 失败!")
        sys.exit(1)

if __name__ == "__main__":
    main()
