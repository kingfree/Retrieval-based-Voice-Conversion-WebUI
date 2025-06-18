# 音频波形可视化功能 (Audio Waveform Visualization)

## 概述 (Overview)

本项目新增了实时音频波形可视化功能，可以在语音转换过程中实时显示输入和输出音频的波形和频谱信息。

This project now includes real-time audio waveform visualization capabilities that display input and output audio waveforms and frequency spectrums during voice conversion.

## 功能特性 (Features)

### 🎵 双通道音频可视化
- **输入音频可视化**: 实时显示原始音频输入的波形和频谱
- **输出音频可视化**: 实时显示经过语音转换后的音频波形和频谱
- **同步显示**: 两个可视化窗口同步更新，便于对比转换效果

### 📊 多种显示模式
- **波形模式**: 显示时域波形图，展示音频信号的幅度变化
- **频谱模式**: 显示频域谱图，展示音频信号的频率分布
- **双视图模式**: 同时显示波形和频谱，提供完整的音频信息

### 🎛️ 实时监控功能
- **音量指示器**: 实时显示音频电平和分贝值
- **活动状态指示**: 可视化指示器显示音频活动状态
- **平滑动画**: 流畅的动画效果，提供良好的视觉体验

### 🎨 自定义外观
- **主题支持**: 支持明暗主题自动切换
- **颜色自定义**: 可配置主色调和强调色
- **响应式设计**: 适配不同屏幕尺寸

## 技术实现 (Technical Implementation)

### 前端组件架构
```
ui/src/
├── components/
│   ├── AudioVisualizer.vue     # 高级音频可视化组件
│   └── WaveformDisplay.vue     # 基础波形显示组件
├── composables/
│   └── useAudioStream.js       # 音频流数据管理
└── App.vue                     # 主应用集成
```

### 后端音频流处理
- **Tauri事件系统**: 使用Tauri的事件发射机制传输音频数据
- **实时数据流**: 50ms间隔的高频率音频数据更新
- **缓冲区管理**: 智能的音频缓冲区管理，防止内存溢出
- **性能优化**: 异步处理和资源清理机制

### 音频数据格式
```typescript
interface AudioData {
  samples: number[];      // 音频样本数据 (-1.0 到 1.0)
  sample_rate: number;    // 采样率 (通常为 44100Hz)
  timestamp: number;      // 时间戳 (毫秒)
}

interface AudioStats {
  input_sample_rate: number;
  output_sample_rate: number;
  buffer_size: number;
  processed_samples: number;
  dropped_frames: number;
  latency: number;
}
```

## 使用说明 (Usage Instructions)

### 1. 启动应用
```bash
# 构建前端
cd rvc-rs/ui
npm install
npm run build

# 运行Tauri应用 (可选)
cd ../
cargo tauri dev
```

### 2. 演示模式 (Demo Mode)
当不在Tauri环境中运行时，应用会自动启用演示模式：
- 点击"开始演示"按钮开始音频可视化演示
- 演示模式会生成模拟的音频数据用于测试可视化效果
- 可以切换不同的视图模式查看效果

### 3. 实际使用 (Production Use)
1. 加载音频模型 (pth 和 index 文件)
2. 配置音频设备 (输入/输出设备)
3. 调整转换参数
4. 点击"开始音频转换"
5. 观察实时音频可视化效果

### 4. 可视化控制
- **切换显示模式**: 在波形、频谱、双视图之间切换
- **暂停/恢复**: 暂停或恢复可视化更新
- **音量监控**: 观察实时音量电平和分贝值

## 配置选项 (Configuration Options)

### AudioVisualizer 组件属性
```vue
<AudioVisualizer
  title="音频可视化"           // 标题
  :audio-data="audioData"      // 音频数据数组
  :sample-rate="44100"         // 采样率
  primary-color="#2196F3"      // 主色调
  accent-color="#FF9800"       // 强调色
  theme="light"                // 主题 (light/dark)
  @level-change="handleLevel"  // 音量变化事件
/>
```

### useAudioStream 配置
```javascript
const {
  inputAudioData,     // 输入音频数据
  outputAudioData,    // 输出音频数据
  isStreaming,        // 流状态
  inputVolume,        // 输入音量
  outputVolume,       // 输出音量
  stats,              // 音频统计
} = useAudioStream(demoMode);
```

## 性能考虑 (Performance Considerations)

### 优化策略
- **帧率控制**: 可视化更新频率限制在20-30 FPS
- **数据缓冲**: 限制音频缓冲区大小，防止内存泄漏
- **按需渲染**: 只在可视化激活时进行绘制
- **资源清理**: 组件卸载时自动清理所有资源

### 内存使用
- 音频缓冲区: 最大 8192 样本 (~185ms @ 44.1kHz)
- Canvas渲染: 硬件加速的2D绘制
- 事件监听: 自动清理事件监听器

## 故障排除 (Troubleshooting)

### 常见问题

**1. 可视化不显示数据**
- 确认音频转换已开始
- 检查音频设备配置
- 验证音频输入信号

**2. 性能问题**
- 降低可视化更新频率
- 关闭不需要的显示模式
- 检查系统资源使用情况

**3. 演示模式不工作**
- 确认浏览器支持Canvas
- 检查JavaScript控制台错误
- 验证组件正确加载

### 调试信息
启用开发者工具查看详细日志：
```javascript
// 在浏览器控制台中查看音频流状态
console.log('Audio streaming status:', isStreaming.value);
console.log('Input audio data length:', inputAudioData.value.length);
console.log('Output audio data length:', outputAudioData.value.length);
```

## 开发指南 (Development Guide)

### 扩展可视化功能

1. **添加新的可视化类型**:
```javascript
// 在 AudioVisualizer.vue 中添加新的绘制方法
function drawSpectrogram() {
  // 实现频谱图绘制逻辑
}
```

2. **自定义音频处理**:
```javascript
// 在 useAudioStream.js 中添加音频处理逻辑
function processAudioData(samples) {
  // 自定义音频数据处理
  return processedSamples;
}
```

3. **集成新的可视化库**:
```bash
npm install d3-scale d3-selection  # 示例：集成D3.js
```

### 贡献指南
- 遵循现有的代码风格和命名约定
- 添加适当的错误处理和日志记录
- 更新相关文档和测试用例
- 确保跨平台兼容性

## 依赖项 (Dependencies)

### 前端依赖
```json
{
  "vue": "^3.4.0",
  "wavesurfer.js": "^7.7.0",
  "@tauri-apps/api": "^2.5.0"
}
```

### Rust依赖
```toml
[dependencies]
serde = { version = "1.0", features = ["derive"] }
tauri = { version = "2.0", features = ["shell-open"] }
```

## 许可证 (License)

本音频可视化功能遵循项目的整体许可证协议。

---

## 更新日志 (Changelog)

### v1.0.0 (2024-01-XX)
- ✨ 新增实时音频波形可视化
- ✨ 支持输入/输出双通道显示
- ✨ 实现波形和频谱双视图模式
- ✨ 添加演示模式用于测试
- 🎨 支持主题切换和自定义配色
- 📱 响应式设计支持移动端
- ⚡ 性能优化和内存管理

### 未来计划 (Roadmap)
- 🔊 3D频谱瀑布图
- 📊 音频质量分析指标
- 🎵 音调和和声可视化
- 💾 音频可视化数据导出
- 🔄 更多音频处理算法集成