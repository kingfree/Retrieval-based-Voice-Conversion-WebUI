# CHANGELOG

## [0.2.0] - 2024-12-19

### Major Architecture Cleanup
- **删除重复的推理系统**: 移除了 `RVCInference` 和相关的 `inference.rs` 模块
  - 该模块与 `rvc_for_realtime.rs` 中的 `RVC` 类功能重复，违反了 AGENTS.md 中的"禁止逻辑重复"原则
  - 保留符合原始 Python 实现的 `RVC` 类作为唯一推理接口
- **删除不必要的音频流水线**: 移除了依赖已删除推理模块的 `audio_pipeline.rs`
- **清理模块依赖**: 更新 `lib.rs` 和相关文件，移除对已删除模块的引用
- **修复编译错误**: 更新 `model_loader.rs` 中的兼容性检查方法，移除对已删除配置类型的依赖

### 推理系统对齐
- **修正实时音频回调**: 更新 `realtime.rs` 中的音频处理逻辑
  - 从使用简化的 `infer_simple` 方法改为使用完整的 `RVC::infer` 方法
  - 与原始 Python 实现的 `audio_callback` 逻辑保持一致
  - 正确传递推理参数：`block_frame_16k`, `skip_head`, `return_length`, `f0method`
- **确保调用链完整**: 验证从前端点击"开始转换"到实际调用 `RVC::infer` 的完整路径
  - 前端 → `start_voice_conversion` (Tauri) → `start_vc_with_callback` → 音频回调 → `RVC::infer`

### 与原始 Python 实现对比
- **已实现功能**:
  - ✅ 核心 `RVC` 类及其 `infer` 方法（忠实按照 Python 实现）
  - ✅ 实时音频流处理和回调机制
  - ✅ 模型加载、特征提取、F0估计、生成器推理
  - ✅ FAISS 索引搜索和特征检索
  - ✅ 音频预处理和后处理
  - ✅ 设备管理和配置系统

- **架构一致性**:
  - ✅ 单一 `RVC` 类负责所有推理逻辑（匹配 Python `rvc_for_realtime.py`）
  - ✅ `infer` 方法签名与 Python 版本完全一致
  - ✅ 实时音频处理流程与 `gui_v1.py` 的 `audio_callback` 对应
  - ✅ 移除了 Python 中不存在的冗余推理系统

### 文件变更
- **删除的文件**:
  - `rvc-rs/rvc-lib/src/inference.rs` - 冗余的推理模块
  - `rvc-rs/rvc-lib/src/audio_pipeline.rs` - 依赖已删除模块的流水线
- **修改的文件**:
  - `rvc-rs/rvc-lib/src/lib.rs` - 移除已删除模块的引用
  - `rvc-rs/rvc-lib/src/model_loader.rs` - 简化兼容性检查方法
  - `rvc-rs/rvc-lib/src/realtime.rs` - 修正音频回调使用正确的 `infer` 方法

### 编译状态
- ✅ `rvc-lib` 编译成功（仅有无害的 dead_code 警告）
- ✅ Tauri 应用编译成功
- ✅ 所有测试通过

### 下一步计划
- 完善 F0 方法配置的传递（目前硬编码为 "harvest"）
- 添加更多实时参数的动态配置
- 优化音频缓冲和延迟管理
- 增强错误处理和恢复机制

## [0.1.0] - 2024-12-19

### Added
- **进度日志功能**: 在音频转换过程中每隔1秒输出一行进度日志，告知用户后端正在处理
  - 在 `audio_pipeline.rs` 中的 `perform_voice_conversion` 方法添加定时日志输出
  - 在 `inference.rs` 中添加带进度日志的特征提取、F0估计和音频生成方法
  - 在 `rvc_for_realtime.rs` 中为实时推理添加详细的进度日志和时间统计
- **处理时间统计**: 显示各个处理阶段的耗时和实时倍率
- **中文日志消息**: 将日志消息本地化为中文，提供更好的用户体验

### Changed
- **音频转换日志**: 改进了音频转换过程中的日志输出格式和内容
- **实时推理反馈**: 增强了实时语音转换的用户反馈，包括输入时长、处理进度和性能指标

### Technical Details
- 添加了 `Duration` 和定时器相关的导入
- 使用 `tokio::spawn` 和 `std::thread::spawn` 实现异步和同步的定时日志输出
- 在关键处理步骤中添加了开始/结束时间戳记录
- 计算并显示实时倍率（输入时长 / 处理时长）

### Files Modified
- `rvc-rs/rvc-lib/src/audio_pipeline.rs`: 添加语音转换过程的定时日志
- `rvc-rs/rvc-lib/src/inference.rs`: 添加各处理阶段的进度日志方法
- `rvc-rs/rvc-lib/src/rvc_for_realtime.rs`: 增强实时推理的进度反馈和时间统计

### Performance Impact
- 日志输出对性能影响极小
- 定时器线程在长时间处理时提供有用的进度反馈
- 不影响核心音频处理逻辑的性能

### 功能验证
- ✅ **测试通过**: 创建了专门的测试示例 `test_progress_logging.rs`
- ✅ **短音频处理**: 3秒音频处理快速完成，无需进度日志
- ✅ **中等长度音频**: 10秒音频显示适当的进度日志
- ✅ **长音频处理**: 30秒音频每隔1秒输出进度更新
- ✅ **各处理阶段**: HuBERT特征提取、F0估计、语音转换、音频生成都有独立的进度追踪
- ✅ **时间统计**: 显示每个阶段的耗时和总体实时倍率
- ✅ **中文本地化**: 所有日志消息都使用中文，提供更好的用户体验

### 测试示例输出
```
🎵 开始语音转换推理 (输入时长: 30.00s)
🧠 提取 HuBERT 特征...
   ⏱️  HuBERT 特征提取进行中... (1.0s)
   ⏱️  HuBERT 特征提取进行中... (2.0s)
✅ HuBERT 特征提取完成 (耗时: 2.97s)
🎼 估计基频 (F0)...
   ⏱️  F0 估计中... (1.0s)
✅ F0 估计完成 (耗时: 1.38s)
🎨 开始语音转换推理
   ⏱️  语音转换推理进行中... (1.0s)
   ⏱️  语音转换推理进行中... (2.0s)
✅ 语音转换完成 (总耗时: 8.45s, 实时倍率: 3.5x)
```

### 如何使用
运行进度日志测试示例：
```bash
cargo run --example test_progress_logging --manifest-path rvc-rs/rvc-lib/Cargo.toml
```

### 与Python实现对比状态
- **当前 Rust 实现是否已覆盖 Python 的全部功能？**
  - 核心音频转换功能：✅ 已实现
  - 进度监控功能：✅ 已实现并增强（每秒进度日志）
  - 模型加载：✅ 已实现
  - 特征提取：✅ 已实现
  - F0估计：✅ 已实现
  - FAISS索引：✅ 已实现

- **下一步需要补充的功能：**
  - 实时语音转换的WebUI集成
  - 批量处理优化
  - GPU加速的完整测试
  - 错误处理和恢复机制的完善