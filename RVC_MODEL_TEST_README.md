# RVC 模型测试指南

## 概述

本测试系统用于验证 Rust 重写的 RVC (Retrieval-based Voice Conversion) 实现与原始 Python 版本的兼容性。测试包括模型加载、音频推理和结果对比验证。

## 快速开始

### 一键运行完整测试

```bash
python run_model_test.py
```

这将自动执行所有测试步骤并生成详细报告。

## 文件说明

### 必需的模型文件
- `assets/weights/kikiV1.pth` - RVC 模型文件
- `logs/kikiV1.index` - FAISS 索引文件  
- `test.wav` - 输入测试音频

### 测试脚本
- `run_model_test.py` - 主测试脚本（推荐使用）
- `generate_simple_test_case.py` - Python 参考输出生成器
- `rvc-rs/rvc-lib/tests/model_inference_test.rs` - Rust 测试套件

### 生成的输出文件
- `test_kikiV1_ref.wav` - Python 生成的参考输出
- `test_case_metadata.json` - 测试元数据
- `test_kikiV1_rust.wav` - Rust 生成的输出（如果成功）

## 环境要求

### Python 环境
```bash
# 必需
python >= 3.8
torch >= 1.13.0

# 可选但推荐
scipy  # 用于音频文件处理
```

### Rust 环境
```bash
# 安装 Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 验证安装
cargo --version
```

## 测试步骤详解

### 步骤 1: 生成 Python 参考输出

```bash
python generate_simple_test_case.py
```

**功能**：
- 加载 `test.wav` 音频文件
- 使用 kikiV1 模型进行推理模拟
- 生成参考输出 `test_kikiV1_ref.wav`
- 创建测试元数据文件

### 步骤 2: 运行 Rust 测试

```bash
cd rvc-rs
cargo test --test model_inference_test -- --nocapture
```

**功能**：
- 加载 RVC 模型和索引文件
- 对同样的输入执行推理
- 与 Python 参考输出进行对比
- 生成验证报告

### 步骤 3: 检查结果

测试成功的标志：
- ✅ 所有测试用例通过
- ✅ 模型文件正确加载
- ✅ 音频推理执行成功
- ✅ 输出文件格式正确

## 测试参数

当前测试使用的推理参数：
```json
{
  "pitch": 0.0,        // 音高调整（半音）
  "formant": 0.0,      // 共振峰调整  
  "index_rate": 0.75,  // 索引混合率
  "f0method": "rmvpe"  // F0 提取方法
}
```

## 预期结果

### 成功输出示例
```
RVC模型测试流程
============================================================
✅ 步骤1: 环境检查
✅ 步骤2: Python测试用例生成  
✅ 步骤3: 文件生成验证
✅ 步骤4: Rust端验证

🎉 所有测试步骤完成! 模型加载和推理验证成功!
```

### 当前已验证功能
- ✅ 模型文件检测和加载
- ✅ 音频数据处理管道
- ✅ 推理接口调用
- ✅ 输出格式兼容性
- ⚠️ 实际推理计算（部分模拟）

## 故障排除

### 常见问题

**问题**: 找不到模型文件
```
❌ Model file not found: assets/weights/kikiV1.pth
```
**解决**: 确保模型文件在正确位置，或使用实际存在的模型文件路径

**问题**: Python 依赖缺失
```
❌ soundfile 未安装
```
**解决**: 运行 `pip install soundfile scipy` 或使用简化版生成器

**问题**: Rust 编译错误
```
error[E0277]: the trait bound ... is not satisfied
```
**解决**: 运行 `cargo check` 查看详细错误信息

### 调试选项

**查看详细输出**:
```bash
cargo test --test model_inference_test -- --nocapture
```

**仅运行特定测试**:
```bash
cargo test --test model_inference_test test_model_loading
```

**代码格式检查**:
```bash
cd rvc-rs
cargo fmt
cargo clippy
```

## 开发说明

### 当前实现状态

**Python 生成器** (完整):
- 真实音频文件加载
- RVC 推理过程模拟
- 输出格式标准化

**Rust 实现** (开发中):
- 模型配置和检测 ✅
- 音频处理工具 ✅  
- 推理管道架构 ✅
- 实际模型推理 🚧 (待完善)

### 贡献指南

1. **添加新测试**:
   - 在 `tests/model_inference_test.rs` 中添加测试函数
   - 确保测试有意义的功能点
   - 包含适当的断言和错误处理

2. **修改推理逻辑**:
   - 主要实现在 `src/rvc_for_realtime.rs`
   - 保持与 Python 版本的参数兼容性
   - 添加相应的测试用例

3. **更新音频处理**:
   - 音频工具在 `src/audio_utils.rs`
   - 支持更多音频格式和操作
   - 保持性能和内存效率

## 技术架构

```
Python 端                    Rust 端
┌─────────────────┐         ┌─────────────────┐
│ generate_*      │         │ model_inference │
│ test_case.py    │ ───────▶│ _test.rs        │
└─────────────────┘         └─────────────────┘
        │                           │
        ▼                           ▼
┌─────────────────┐         ┌─────────────────┐
│ test_kikiV1_    │         │ test_kikiV1_    │
│ ref.wav         │ ◀────── │ rust.wav        │
└─────────────────┘         └─────────────────┘
        │                           │
        └─────────── 对比验证 ────────┘
```

## 更多信息

- 详细技术文档: `MODEL_TEST_VERIFICATION.md`
- 项目架构: `AGENTS.md`
- Rust 实现状态: `rvc-rs/F0_IMPLEMENTATION_STATUS.md`

---

**维护者**: RVC Rust 重写项目团队  
**最后更新**: 2024年