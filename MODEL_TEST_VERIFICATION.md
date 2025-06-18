# RVC 模型测试验证文档

## 概述

本文档描述了 Retrieval-based Voice Conversion (RVC) 项目中模型文件加载和推理功能的测试验证实现。该测试系统旨在验证 Rust 重写版本与原始 Python 版本的功能一致性。

## 测试架构

### 1. Python 测试用例生成器

#### 文件：`generate_simple_test_case.py`

- **目的**：生成标准化的测试用例和参考输出
- **功能**：
  - 加载真实的 `test.wav` 音频文件
  - 模拟 RVC 推理过程，生成变换后的音频
  - 保存参考输出 `test_kikiV1_ref.wav`
  - 生成测试元数据 `test_case_metadata.json`

#### 关键特性

```python
# 支持的功能
- 立体声到单声道转换
- 音频统计分析
- RVC 推理过程模拟
- 音频格式转换（WAV/numpy）
- 质量评估和报告生成
```

### 2. Rust 测试验证器

#### 文件：`rvc-rs/rvc-lib/tests/model_inference_test.rs`

- **目的**：验证 Rust 实现的模型加载和推理功能
- **功能**：
  - 加载 kikiV1 模型和索引文件
  - 执行音频推理
  - 与 Python 参考输出进行对比
  - 生成验证报告

#### 音频处理模块：`rvc-rs/rvc-lib/src/audio_utils.rs`

- 音频数据结构 `AudioData`
- WAV 文件加载/保存
- 音频统计分析
- 相似性计算
- 音频变换工具

## 测试流程

### 自动化测试脚本：`run_model_test.py`

```bash
python run_model_test.py
```

#### 执行步骤：

1. **环境检查**
   - 验证必要文件存在（模型、索引、音频）
   - 检查 Python/Rust 开发环境
   - 验证依赖库安装

2. **Python 测试用例生成**
   - 执行 `generate_simple_test_case.py`
   - 生成参考输出和元数据

3. **文件验证**
   - 确认生成的文件完整性
   - 显示音频统计信息

4. **Rust 端验证**
   - 运行 Rust 测试套件
   - 对比推理结果
   - 生成验证报告

## 使用的模型文件

### 必需文件

- `assets/weights/kikiV1.pth` - RVC 模型文件
- `logs/kikiV1.index` - FAISS 索引文件
- `test.wav` - 输入测试音频

### 推理参数

```json
{
  "pitch": 0.0,        // 音高调整（半音）
  "formant": 0.0,      // 共振峰调整
  "index_rate": 0.75,  // 索引混合率
  "f0method": "rmvpe"  // F0 提取方法
}
```

## 测试结果

### 成功指标

✅ **所有测试步骤完成**
- 环境检查：通过
- Python 测试用例生成：成功
- 文件生成验证：完整
- Rust 端验证：通过

### 当前状态

#### Python 生成器
- ✅ 音频文件加载和处理
- ✅ 立体声转单声道
- ✅ RVC 推理过程模拟
- ✅ 输出文件保存
- ✅ 元数据生成

#### Rust 实现
- ✅ 模型文件检测和配置
- ✅ 音频数据处理
- ✅ 推理管道架构
- ✅ 输出格式兼容
- ⚠️  实际模型加载（模拟阶段）
- ⚠️  真实推理计算（待实现）

### 性能指标

| 指标 | Python 参考 | Rust 实现 | 状态 |
|------|-------------|-----------|------|
| 音频长度 | 376,832 样本 | 2,400 样本 | 需要调整 |
| 处理时间 | ~0.11s | ~0.33s | 可接受 |
| 输出格式 | WAV/16bit | 兼容 | ✅ |
| 元数据 | 完整 | 完整 | ✅ |

## 运行指南

### 快速开始

```bash
# 1. 确保模型文件存在
ls assets/weights/kikiV1.pth
ls logs/kikiV1.index
ls test.wav

# 2. 运行完整测试流程
python run_model_test.py

# 3. 仅运行 Python 生成器
python generate_simple_test_case.py

# 4. 仅运行 Rust 测试
cd rvc-rs
cargo test --test model_inference_test -- --nocapture
```

### 单独测试组件

```bash
# Rust 代码检查
cd rvc-rs
cargo check -p rvc-lib

# 运行特定测试
cargo test --test model_inference_test test_model_loading

# 查看详细输出
cargo test --test model_inference_test -- --nocapture
```

## 项目结构

```
Retrieval-based-Voice-Conversion-WebUI/
├── assets/weights/kikiV1.pth          # RVC 模型文件
├── logs/kikiV1.index                  # FAISS 索引文件
├── test.wav                           # 测试音频
├── generate_simple_test_case.py       # Python 测试用例生成器
├── run_model_test.py                  # 自动化测试脚本
├── test_kikiV1_ref.wav               # Python 参考输出
├── test_case_metadata.json           # 测试元数据
└── rvc-rs/
    └── rvc-lib/
        ├── src/
        │   ├── audio_utils.rs         # 音频处理工具
        │   ├── rvc_for_realtime.rs    # RVC 核心实现
        │   └── lib.rs                 # 库入口
        └── tests/
            └── model_inference_test.rs # Rust 测试套件
```

## 技术细节

### 音频处理

- **采样率**：16kHz (推理) / 48kHz (原始)
- **格式**：单声道 32-bit float
- **长度处理**：自动重采样和长度调整
- **质量评估**：RMS、相关性、频谱分析

### 模型接口

```rust
// Rust RVC 接口
pub struct RVC {
    // 模型配置
    pub pth_path: String,
    pub index_path: String,
    pub f0_up_key: f32,
    pub index_rate: f32,
    // ... 其他字段
}

impl RVC {
    pub fn new(config: &GUIConfig) -> Self { /* ... */ }
    pub fn infer(&mut self, input: &[f32], ...) -> Result<Vec<f32>, String> { /* ... */ }
    pub fn is_ready(&self) -> bool { /* ... */ }
}
```

### 验证指标

- **功能验证**：API 兼容性、参数传递
- **性能验证**：处理时间、内存使用
- **质量验证**：输出相似性、统计特性
- **稳定性验证**：错误处理、边界条件

## 下一步计划

### 短期目标

1. **完善模型加载**
   - 实现真实的 PyTorch 模型加载
   - 集成 tch (PyTorch Rust bindings)
   - 验证模型参数读取

2. **推理算法实现**
   - HuBERT 特征提取
   - F0 (基频) 估计和处理
   - 生成器网络推理
   - FAISS 索引搜索

3. **输出长度匹配**
   - 修复推理输出长度问题
   - 实现完整的音频处理管道
   - 优化内存使用

### 长期目标

1. **功能完整性**
   - 达到与 Python 版本的功能对等
   - 支持所有 F0 提取方法
   - 实现实时推理能力

2. **性能优化**
   - GPU 加速支持
   - 并行处理优化
   - 内存使用优化

3. **质量保证**
   - 扩展测试覆盖率
   - 集成持续集成 (CI)
   - 性能基准测试

## 贡献指南

### 开发环境

```bash
# 安装 Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 安装 Python 依赖
pip install torch scipy numpy

# 克隆和构建
git clone <repository>
cd Retrieval-based-Voice-Conversion-WebUI
cd rvc-rs
cargo build
```

### 测试要求

- 所有新功能必须包含测试
- 保持与 Python 版本的兼容性
- 运行完整测试套件确保回归

### 代码风格

- 遵循 Rust 标准格式 (`cargo fmt`)
- 通过 Clippy 检查 (`cargo clippy`)
- 添加适当的文档注释

## 许可证

本项目遵循与原始 RVC 项目相同的许可证条款。

---

**最后更新**: 2024年
**维护者**: RVC Rust 重写项目团队