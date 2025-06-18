# AGENTS

## 项目愿景

本仓库的目标是用 Rust 重写原版基于检索的语音转换 WebUI。新的 Rust 工作区（`rvc-rs`）包含：

- `rvc-lib`：核心功能的 Rust 库，基于 `tch` 实现
- `ui`：使用 Vue 3 和 Vite 构建的前端，打包进 Tauri 应用中（位于 `ui/src-tauri`）

本项目旨在实现与 `gui_v1.py` 功能等效，同时借助 Rust 与 Tauri 提供更好的性能与可维护性。

## 架构原则

### 职责分离

- **`rvc-lib`**：包含**所有核心功能**，包括：
  - 模型加载与管理
  - 音频处理流程
  - 参数的加载与保存
  - 配置管理
  - 实时语音转换逻辑
  - 设备管理
  - 所有业务逻辑与数据处理

- **Tauri (`ui/src-tauri`)**：**仅作为前端与后端的桥梁**：
  - 接收来自 Vue 前端的请求
  - 调用 `rvc-lib` 中对应的方法
  - 将结果返回前端
  - 向前端推送事件或进度更新
  - **不应包含任何实际功能或业务逻辑**
  - **不得重复实现 `rvc-lib` 中已有的逻辑**

### 核心规则

1. **禁止逻辑重复**：若功能已存在于 `rvc-lib`，Tauri 不得重新实现
2. **Tauri 仅作中介**：Tauri 的函数应为薄包装，仅用于调用 `rvc-lib`
3. **状态管理归属**：应用状态必须由 `rvc-lib` 管理，不能放在 Tauri 中
4. **参数处理归属**：所有参数的校验、加载、保存必须在 `rvc-lib` 中完成

## 开发指南

- 修改 Rust 代码后，请运行：

  ```bash
  cargo check -p rvc-lib --manifest-path rvc-rs/Cargo.toml
  ```

* 修改 Tauri 代码后，请运行：

  ```bash
  cargo check --manifest-path rvc-rs/ui/src-tauri/Cargo.toml
  ```

* 修改 Vue 前端代码后，请在 `rvc-rs/ui` 中运行：

  ```bash
  npm run build
  ```

  若有依赖变动，请先运行：

  ```bash
  npm install
  ```

* 是否运行 Tauri 应用程序可选，但它依赖于系统组件如 `glib`

* 添加新功能时，**请先在 `rvc-lib` 中实现**，然后创建 Tauri 的简薄封装接口

* **避免** `rvc-lib` 中的函数名与 Tauri 命令函数发生命名冲突
好的，以下是在原说明基础上**增加的要求**，以完整中文 Markdown 格式呈现：

## 开发附加要求

### 变更记录（CHANGELOG）

- 每次进行功能添加、修改或重构后，**必须在 `CHANGELOG.md` 文件中记录改动内容**，包括但不限于：
  - 添加的新功能或命令
  - 修改的参数结构或默认值
  - 修复的 Bug
  - 性能优化点
- 版本号每次增加 0.1.0，记录提交时间而非日期

### 功能对照检查

* 每次提交代码后，必须手动或脚本方式**对比 Rust 实现与原始 Python 实现（`gui_v1.py`）之间的功能差异**

  * 对照内容包括参数支持情况、模型行为、输出格式、错误处理逻辑等

* 对比完成后，请回答以下两个问题以指导后续开发：

  1. **当前 Rust 实现是否已覆盖 Python 的全部功能？**
  2. **若有缺失，请明确指出下一步该补充哪些功能？**

## Python Code Index

### api_231006.py
- Variable `logger`: lines 22-22
- Variable `app`: lines 25-25
- Variable `audio_api`: lines 355-355
- Class `GUIConfig`: lines 27-44
  - Method `__init__`: lines 28-44
- Class `ConfigData`: lines 46-60
- Class `AudioAPI`: lines 62-353
  - Method `__init__`: lines 63-69
  - Method `load`: lines 71-101
  - Method `set_values`: lines 103-124
  - Method `start_vc`: lines 126-204
  - Method `soundinput`: lines 206-220
  - Method `audio_callback`: lines 222-292
  - Method `get_devices`: lines 294-328
  - Method `set_devices`: lines 330-353
- Function `get_input_devices`: lines 358-364
- Function `get_output_devices`: lines 367-373
- Function `configure_audio`: lines 376-392
- Function `start_conversion`: lines 395-408
- Function `stop_conversion`: lines 411-426

### api_240604.py
- Variable `logger`: lines 23-23
- Variable `app`: lines 26-26
- Variable `audio_api`: lines 481-481
- Class `GUIConfig`: lines 28-47
  - Method `__init__`: lines 29-47
- Class `ConfigData`: lines 49-66
- Class `Harvest`: lines 68-88
  - Method `__init__`: lines 69-72
  - Method `run`: lines 74-88
- Class `AudioAPI`: lines 90-479
  - Method `__init__`: lines 91-100
  - Method `initialize_queues`: lines 102-108
  - Method `load`: lines 110-141
  - Method `set_values`: lines 143-166
  - Method `start_vc`: lines 168-277
  - Method `soundinput`: lines 279-293
  - Method `audio_callback`: lines 295-418
  - Method `get_devices`: lines 420-454
  - Method `set_devices`: lines 456-479
- Function `get_input_devices`: lines 484-490
- Function `get_output_devices`: lines 493-499
- Function `configure_audio`: lines 502-517
- Function `start_conversion`: lines 520-533
- Function `stop_conversion`: lines 536-551

### configs/config.py
- Variable `logger`: lines 21-21
- Variable `version_config_list`: lines 24-30
- Class `Config`: lines 44-254
  - Method `__init__`: lines 45-63
  - Method `load_config_json`: lines 66-74
  - Method `arg_parse`: lines 77-107
  - Method `has_mps`: lines 112-119
  - Method `has_xpu`: lines 122-126
  - Method `use_fp32_config`: lines 128-137
  - Method `device_config`: lines 139-254
- Function `singleton_variable`: lines 33-40

### gui_v1.py
- Variable `logger`: lines 7-7
- Variable `now_dir`: lines 15-15
- Variable `flag_vc`: lines 19-19
- Class `Harvest`: lines 53-74
  - Method `__init__`: lines 54-57
  - Method `run`: lines 59-74
- Function `printt`: lines 22-26
- Function `phase_vocoder`: lines 29-50

### i18n/i18n.py
- Class `I18nAuto`: lines 12-28
  - Method `__init__`: lines 13-22
  - Method `__call__`: lines 24-25
  - Method `__repr__`: lines 27-28
- Function `load_language_list`: lines 6-9

### i18n/locale_diff.py
- Variable `standard_file`: lines 6-6
- Variable `dir_path`: lines 9-9
- Variable `languages`: lines 10-14

### i18n/scan_i18n.py
- Variable `strings`: lines 29-29
- Variable `code_keys`: lines 38-38
- Variable `standard_file`: lines 52-52
- Variable `standard_keys`: lines 55-55
- Variable `unused_keys`: lines 58-58
- Variable `missing_keys`: lines 63-63
- Variable `code_keys_dict`: lines 68-68
- Function `extract_i18n_strings`: lines 7-22

### infer-web.py
- Variable `now_dir`: lines 5-5
- Variable `logger`: lines 39-39
- Variable `tmp`: lines 41-41
- Variable `config`: lines 53-53
- Variable `vc`: lines 54-54
- Variable `i18n`: lines 65-65
- Variable `ngpu`: lines 68-68
- Variable `gpu_infos`: lines 69-69
- Variable `mem`: lines 70-70
- Variable `if_gpu_ok`: lines 71-71
- Variable `gpus`: lines 120-120
- Variable `weight_root`: lines 133-133
- Variable `weight_uvr5_root`: lines 134-134
- Variable `index_root`: lines 135-135
- Variable `outside_index_root`: lines 136-136
- Variable `names`: lines 138-138
- Variable `index_paths`: lines 142-142
- Variable `uvr5_names`: lines 155-155
- Variable `sr_dict`: lines 187-191
- Variable `F0GPUVisible`: lines 798-798
- Class `ToolButton`: lines 123-130
  - Method `__init__`: lines 126-127
  - Method `get_block_name`: lines 129-130
- Function `lookup_indices`: lines 145-150
- Function `change_choices`: lines 161-174
- Function `clean`: lines 177-178
- Function `export_onnx`: lines 181-184
- Function `if_done`: lines 194-200
- Function `if_done_multi`: lines 203-215
- Function `preprocess_dataset`: lines 218-254
- Function `extract_f0_feature`: lines 258-395
- Function `get_pretrained_models`: lines 398-430
- Function `change_sr2`: lines 433-436
- Function `change_version19`: lines 439-452
- Function `change_f0`: lines 455-461
- Function `click_train`: lines 465-612
- Function `train_index`: lines 616-711
- Function `train1key`: lines 715-778
- Function `change_info_`: lines 782-795
- Function `change_f0_method`: lines 801-806

### infer/lib/audio.py
- Function `wav2`: lines 10-30
- Function `load_audio`: lines 33-52
- Function `clean_path`: lines 56-60

### infer/lib/infer_pack/attentions.py
- Class `Encoder`: lines 14-77
  - Method `__init__`: lines 15-60
  - Method `forward`: lines 62-77
- Class `Decoder`: lines 80-163
  - Method `__init__`: lines 81-138
  - Method `forward`: lines 140-163
- Class `MultiHeadAttention`: lines 166-385
  - Method `__init__`: lines 167-218
  - Method `forward`: lines 220-230
  - Method `attention`: lines 232-288
  - Method `_matmul_with_relative_values`: lines 290-297
  - Method `_matmul_with_relative_keys`: lines 299-306
  - Method `_get_relative_embeddings`: lines 308-325
  - Method `_relative_position_to_absolute_position`: lines 327-352
  - Method `_absolute_position_to_relative_position`: lines 354-374
  - Method `_attention_bias_proximal`: lines 376-385
- Class `FFN`: lines 388-459
  - Method `__init__`: lines 389-415
  - Method `padding`: lines 417-422
  - Method `forward`: lines 424-433
  - Method `_causal_padding`: lines 435-446
  - Method `_same_padding`: lines 448-459

### infer/lib/infer_pack/attentions_onnx.py
- Class `Encoder`: lines 22-85
  - Method `__init__`: lines 23-68
  - Method `forward`: lines 70-85
- Class `Decoder`: lines 88-171
  - Method `__init__`: lines 89-146
  - Method `forward`: lines 148-171
- Class `MultiHeadAttention`: lines 174-385
  - Method `__init__`: lines 175-226
  - Method `forward`: lines 228-238
  - Method `attention`: lines 240-293
  - Method `_matmul_with_relative_values`: lines 295-302
  - Method `_matmul_with_relative_keys`: lines 304-311
  - Method `_get_relative_embeddings`: lines 313-328
  - Method `_relative_position_to_absolute_position`: lines 330-354
  - Method `_absolute_position_to_relative_position`: lines 356-374
  - Method `_attention_bias_proximal`: lines 376-385
- Class `FFN`: lines 388-459
  - Method `__init__`: lines 389-415
  - Method `padding`: lines 417-422
  - Method `forward`: lines 424-433
  - Method `_causal_padding`: lines 435-446
  - Method `_same_padding`: lines 448-459

### infer/lib/infer_pack/commons.py
- Function `init_weights`: lines 10-13
- Function `get_padding`: lines 16-17
- Function `kl_divergence`: lines 26-32
- Function `rand_gumbel`: lines 35-38
- Function `rand_gumbel_like`: lines 41-43
- Function `slice_segments`: lines 46-52
- Function `slice_segments2`: lines 55-61
- Function `rand_slice_segments`: lines 64-71
- Function `get_timing_signal_1d`: lines 74-87
- Function `add_timing_signal_1d`: lines 90-93
- Function `cat_timing_signal_1d`: lines 96-99
- Function `subsequent_mask`: lines 102-104
- Function `fused_add_tanh_sigmoid_multiply`: lines 108-114
- Function `convert_pad_shape`: lines 123-124
- Function `shift_1d`: lines 127-129
- Function `sequence_mask`: lines 132-136
- Function `generate_path`: lines 139-154
- Function `clip_grad_value_`: lines 157-172

### infer/lib/infer_pack/models.py
- Variable `logger`: lines 5-5
- Variable `has_xpu`: lines 16-16
- Variable `sr2sr`: lines 595-599
- Class `TextEncoder`: lines 19-79
  - Method `__init__`: lines 20-52
  - Method `forward`: lines 54-79
- Class `ResidualCouplingBlock`: lines 82-145
  - Method `__init__`: lines 83-115
  - Method `forward`: lines 117-130
  - Method `remove_weight_norm`: lines 132-134
  - Method `__prepare_scriptable__`: lines 136-145
- Class `PosteriorEncoder`: lines 148-201
  - Method `__init__`: lines 149-176
  - Method `forward`: lines 178-189
  - Method `remove_weight_norm`: lines 191-192
  - Method `__prepare_scriptable__`: lines 194-201
- Class `Generator`: lines 204-309
  - Method `__init__`: lines 205-250
  - Method `forward`: lines 252-281
  - Method `__prepare_scriptable__`: lines 283-303
  - Method `remove_weight_norm`: lines 305-309
- Class `SineGen`: lines 312-388
  - Method `__init__`: lines 328-343
  - Method `_f02uv`: lines 345-351
  - Method `_f02sine`: lines 353-369
  - Method `forward`: lines 371-388
- Class `SourceModuleHnNSF`: lines 391-445
  - Method `__init__`: lines 409-430
  - Method `forward`: lines 433-445
- Class `GeneratorNSF`: lines 448-592
  - Method `__init__`: lines 449-520
  - Method `forward`: lines 522-565
  - Method `remove_weight_norm`: lines 567-571
  - Method `__prepare_scriptable__`: lines 573-592
- Class `SynthesizerTrnMs256NSFsid`: lines 602-776
  - Method `__init__`: lines 603-686
  - Method `remove_weight_norm`: lines 688-692
  - Method `__prepare_scriptable__`: lines 694-718
  - Method `forward`: lines 721-743
  - Method `infer`: lines 746-776
- Class `SynthesizerTrnMs768NSFsid`: lines 779-833
  - Method `__init__`: lines 780-833
- Class `SynthesizerTrnMs256NSFsid_nono`: lines 836-991
  - Method `__init__`: lines 837-917
  - Method `remove_weight_norm`: lines 919-923
  - Method `__prepare_scriptable__`: lines 925-949
  - Method `forward`: lines 952-961
  - Method `infer`: lines 964-991
- Class `SynthesizerTrnMs768NSFsid_nono`: lines 994-1049
  - Method `__init__`: lines 995-1049
- Class `MultiPeriodDiscriminator`: lines 1052-1079
  - Method `__init__`: lines 1053-1062
  - Method `forward`: lines 1064-1079
- Class `MultiPeriodDiscriminatorV2`: lines 1082-1109
  - Method `__init__`: lines 1083-1092
  - Method `forward`: lines 1094-1109
- Class `DiscriminatorS`: lines 1112-1139
  - Method `__init__`: lines 1113-1126
  - Method `forward`: lines 1128-1139
- Class `DiscriminatorP`: lines 1142-1223
  - Method `__init__`: lines 1143-1197
  - Method `forward`: lines 1199-1223

### infer/lib/infer_pack/models_onnx.py
- Variable `logger`: lines 13-13
- Variable `sr2sr`: lines 522-526
- Class `TextEncoder256`: lines 27-71
  - Method `__init__`: lines 28-54
  - Method `forward`: lines 56-71
- Class `TextEncoder768`: lines 74-118
  - Method `__init__`: lines 75-101
  - Method `forward`: lines 103-118
- Class `ResidualCouplingBlock`: lines 121-167
  - Method `__init__`: lines 122-154
  - Method `forward`: lines 156-163
  - Method `remove_weight_norm`: lines 165-167
- Class `PosteriorEncoder`: lines 170-212
  - Method `__init__`: lines 171-198
  - Method `forward`: lines 200-209
  - Method `remove_weight_norm`: lines 211-212
- Class `Generator`: lines 215-288
  - Method `__init__`: lines 216-261
  - Method `forward`: lines 263-282
  - Method `remove_weight_norm`: lines 284-288
- Class `SineGen`: lines 291-367
  - Method `__init__`: lines 307-322
  - Method `_f02uv`: lines 324-330
  - Method `_f02sine`: lines 332-348
  - Method `forward`: lines 350-367
- Class `SourceModuleHnNSF`: lines 370-416
  - Method `__init__`: lines 388-409
  - Method `forward`: lines 411-416
- Class `GeneratorNSF`: lines 419-519
  - Method `__init__`: lines 420-489
  - Method `forward`: lines 491-513
  - Method `remove_weight_norm`: lines 515-519
- Class `SynthesizerTrnMsNSFsidM`: lines 529-649
  - Method `__init__`: lines 530-622
  - Method `remove_weight_norm`: lines 624-627
  - Method `construct_spkmixmap`: lines 629-633
  - Method `forward`: lines 635-649
- Class `MultiPeriodDiscriminator`: lines 652-679
  - Method `__init__`: lines 653-662
  - Method `forward`: lines 664-679
- Class `MultiPeriodDiscriminatorV2`: lines 682-709
  - Method `__init__`: lines 683-692
  - Method `forward`: lines 694-709
- Class `DiscriminatorS`: lines 712-739
  - Method `__init__`: lines 713-726
  - Method `forward`: lines 728-739
- Class `DiscriminatorP`: lines 742-818
  - Method `__init__`: lines 743-797
  - Method `forward`: lines 799-818

### infer/lib/infer_pack/modules.py
- Variable `LRELU_SLOPE`: lines 17-17
- Class `LayerNorm`: lines 20-32
  - Method `__init__`: lines 21-27
  - Method `forward`: lines 29-32
- Class `ConvReluNorm`: lines 35-84
  - Method `__init__`: lines 36-75
  - Method `forward`: lines 77-84
- Class `DDSConv`: lines 87-133
  - Method `__init__`: lines 92-119
  - Method `forward`: lines 121-133
- Class `WN`: lines 136-249
  - Method `__init__`: lines 137-186
  - Method `forward`: lines 188-217
  - Method `remove_weight_norm`: lines 219-225
  - Method `__prepare_scriptable__`: lines 227-249
- Class `ResBlock1`: lines 252-364
  - Method `__init__`: lines 253-326
  - Method `forward`: lines 328-341
  - Method `remove_weight_norm`: lines 343-347
  - Method `__prepare_scriptable__`: lines 349-364
- Class `ResBlock2`: lines 367-420
  - Method `__init__`: lines 368-395
  - Method `forward`: lines 397-406
  - Method `remove_weight_norm`: lines 408-410
  - Method `__prepare_scriptable__`: lines 412-420
- Class `Log`: lines 423-437
  - Method `forward`: lines 424-437
- Class `Flip`: lines 440-456
  - Method `forward`: lines 444-456
- Class `ElementwiseAffine`: lines 459-474
  - Method `__init__`: lines 460-464
  - Method `forward`: lines 466-474
- Class `ResidualCouplingLayer`: lines 477-549
  - Method `__init__`: lines 478-510
  - Method `forward`: lines 512-537
  - Method `remove_weight_norm`: lines 539-540
  - Method `__prepare_scriptable__`: lines 542-549
- Class `ConvFlow`: lines 552-615
  - Method `__init__`: lines 553-577
  - Method `forward`: lines 579-615

### infer/lib/infer_pack/modules/F0Predictor/DioF0Predictor.py
- Class `DioF0Predictor`: lines 7-91
  - Method `__init__`: lines 8-12
  - Method `interpolate_f0`: lines 14-50
  - Method `resize_f0`: lines 52-61
  - Method `compute_f0`: lines 63-76
  - Method `compute_f0_uv`: lines 78-91

### infer/lib/infer_pack/modules/F0Predictor/F0Predictor.py
- Class `F0Predictor`: lines 1-16
  - Method `compute_f0`: lines 2-8
  - Method `compute_f0_uv`: lines 10-16

### infer/lib/infer_pack/modules/F0Predictor/HarvestF0Predictor.py
- Class `HarvestF0Predictor`: lines 7-87
  - Method `__init__`: lines 8-12
  - Method `interpolate_f0`: lines 14-50
  - Method `resize_f0`: lines 52-61
  - Method `compute_f0`: lines 63-74
  - Method `compute_f0_uv`: lines 76-87

### infer/lib/infer_pack/modules/F0Predictor/PMF0Predictor.py
- Class `PMF0Predictor`: lines 7-98
  - Method `__init__`: lines 8-12
  - Method `interpolate_f0`: lines 14-50
  - Method `compute_f0`: lines 52-74
  - Method `compute_f0_uv`: lines 76-98

### infer/lib/infer_pack/modules/F0Predictor/__init__.py

### infer/lib/infer_pack/onnx_inference.py
- Variable `logger`: lines 8-8
- Class `ContentVec`: lines 11-35
  - Method `__init__`: lines 12-22
  - Method `__call__`: lines 24-25
  - Method `forward`: lines 27-35
- Class `OnnxRVC`: lines 64-149
  - Method `__init__`: lines 65-85
  - Method `forward`: lines 87-96
  - Method `inference`: lines 98-149
- Function `get_f0_predictor`: lines 38-61

### infer/lib/infer_pack/transforms.py
- Variable `DEFAULT_MIN_BIN_WIDTH`: lines 5-5
- Variable `DEFAULT_MIN_BIN_HEIGHT`: lines 6-6
- Variable `DEFAULT_MIN_DERIVATIVE`: lines 7-7
- Function `piecewise_rational_quadratic_transform`: lines 10-40
- Function `searchsorted`: lines 43-45
- Function `unconstrained_rational_quadratic_spline`: lines 48-95
- Function `rational_quadratic_spline`: lines 98-207

### infer/lib/jit/__init__.py
- Function `load_inputs`: lines 9-17
- Function `benchmark`: lines 20-30
- Function `jit_warm_up`: lines 33-34
- Function `to_jit_model`: lines 37-73
- Function `export`: lines 76-99
- Function `load`: lines 102-104
- Function `save`: lines 107-109
- Function `rmvpe_jit_export`: lines 112-134
- Function `synthesizer_jit_export`: lines 137-163

### infer/lib/jit/get_hubert.py
- Function `pad_to_multiple`: lines 14-25
- Function `extract_features`: lines 28-92
- Function `compute_mask_indices`: lines 95-224
- Function `apply_mask`: lines 227-263
- Function `get_hubert_model`: lines 266-342

### infer/lib/jit/get_rmvpe.py
- Function `get_rmvpe`: lines 4-12

### infer/lib/jit/get_synthesizer.py
- Function `get_synthesizer`: lines 4-38

### infer/lib/rmvpe.py
- Variable `logger`: lines 26-26
- Class `STFT`: lines 29-156
  - Method `__init__`: lines 30-76
  - Method `transform`: lines 78-107
  - Method `inverse`: lines 109-142
  - Method `forward`: lines 144-156
- Class `BiGRU`: lines 162-174
  - Method `__init__`: lines 163-171
  - Method `forward`: lines 173-174
- Class `ConvBlockRes`: lines 177-210
  - Method `__init__`: lines 178-204
  - Method `forward`: lines 206-210
- Class `Encoder`: lines 213-248
  - Method `__init__`: lines 214-240
  - Method `forward`: lines 242-248
- Class `ResEncoderBlock`: lines 251-271
  - Method `__init__`: lines 252-263
  - Method `forward`: lines 265-271
- Class `Intermediate`: lines 274-290
  - Method `__init__`: lines 275-285
  - Method `forward`: lines 287-290
- Class `ResDecoderBlock`: lines 293-321
  - Method `__init__`: lines 294-314
  - Method `forward`: lines 316-321
- Class `Decoder`: lines 324-339
  - Method `__init__`: lines 325-334
  - Method `forward`: lines 336-339
- Class `DeepUnet`: lines 342-370
  - Method `__init__`: lines 343-364
  - Method `forward`: lines 366-370
- Class `E2E`: lines 373-412
  - Method `__init__`: lines 374-404
  - Method `forward`: lines 406-412
- Class `MelSpectrogram`: lines 418-492
  - Method `__init__`: lines 419-450
  - Method `forward`: lines 452-492
- Class `RMVPE`: lines 495-646
  - Method `__init__`: lines 496-567
  - Method `mel2hidden`: lines 569-585
  - Method `decode`: lines 587-592
  - Method `infer_from_audio`: lines 594-620
  - Method `to_local_average_cents`: lines 622-646

### infer/lib/rtrvc.py
- Variable `now_dir`: lines 20-20
- Variable `mm`: lines 28-28
- Class `RVC`: lines 40-465
  - Method `__init__`: lines 41-194
  - Method `change_key`: lines 196-197
  - Method `change_formant`: lines 199-200
  - Method `change_index_rate`: lines 202-207
  - Method `get_f0_post`: lines 209-220
  - Method `get_f0`: lines 222-291
  - Method `get_f0_crepe`: lines 293-315
  - Method `get_f0_rmvpe`: lines 317-330
  - Method `get_f0_fcpe`: lines 332-349
  - Method `infer`: lines 351-465
- Function `printt`: lines 31-35

### infer/lib/slicer2.py
- Class `Slicer`: lines 38-179
  - Method `__init__`: lines 39-62
  - Method `_apply_slice`: lines 64-72
  - Method `slice`: lines 75-179
- Function `get_rms`: lines 5-35
- Function `main`: lines 182-256

### infer/lib/train/data_utils.py
- Variable `logger`: lines 5-5
- Class `TextAudioLoaderMultiNSFsid`: lines 15-144
  - Method `__init__`: lines 22-32
  - Method `_filter`: lines 34-48
  - Method `get_sid`: lines 50-52
  - Method `get_audio_text_pair`: lines 54-81
  - Method `get_labels`: lines 83-96
  - Method `get_audio`: lines 98-138
  - Method `__getitem__`: lines 140-141
  - Method `__len__`: lines 143-144
- Class `TextAudioCollateMultiNSFsid`: lines 147-220
  - Method `__init__`: lines 150-151
  - Method `__call__`: lines 153-220
- Class `TextAudioLoader`: lines 223-336
  - Method `__init__`: lines 230-240
  - Method `_filter`: lines 242-256
  - Method `get_sid`: lines 258-260
  - Method `get_audio_text_pair`: lines 262-280
  - Method `get_labels`: lines 282-288
  - Method `get_audio`: lines 290-330
  - Method `__getitem__`: lines 332-333
  - Method `__len__`: lines 335-336
- Class `TextAudioCollate`: lines 339-398
  - Method `__init__`: lines 342-343
  - Method `__call__`: lines 345-398
- Class `DistributedBucketSampler`: lines 401-517
  - Method `__init__`: lines 411-427
  - Method `_create_buckets`: lines 429-450
  - Method `__iter__`: lines 452-499
  - Method `_bisect`: lines 501-514
  - Method `__len__`: lines 516-517

### infer/lib/train/losses.py
- Function `feature_loss`: lines 4-12
- Function `discriminator_loss`: lines 15-28
- Function `generator_loss`: lines 31-40
- Function `kl_loss`: lines 43-58

### infer/lib/train/mel_processing.py
- Variable `logger`: lines 6-6
- Variable `MAX_WAV_VALUE`: lines 8-8
- Variable `mel_basis`: lines 38-38
- Variable `hann_window`: lines 39-39
- Function `dynamic_range_compression_torch`: lines 11-17
- Function `dynamic_range_decompression_torch`: lines 20-26
- Function `spectral_normalize_torch`: lines 29-30
- Function `spectral_de_normalize_torch`: lines 33-34
- Function `spectrogram_torch`: lines 42-89
- Function `spec_to_mel_torch`: lines 92-108
- Function `mel_spectrogram_torch`: lines 111-127

### infer/lib/train/process_ckpt.py
- Variable `i18n`: lines 10-10
- Function `savee`: lines 13-48
- Function `show_info`: lines 51-61
- Function `extract_small_model`: lines 64-191
- Function `change_info`: lines 194-203
- Function `merge`: lines 206-261

### infer/lib/train/utils.py
- Variable `MATPLOTLIB_FLAG`: lines 14-14
- Variable `logger`: lines 17-17
- Class `HParams`: lines 454-483
  - Method `__init__`: lines 455-459
  - Method `keys`: lines 461-462
  - Method `items`: lines 464-465
  - Method `values`: lines 467-468
  - Method `__len__`: lines 470-471
  - Method `__getitem__`: lines 473-474
  - Method `__setitem__`: lines 476-477
  - Method `__contains__`: lines 479-480
  - Method `__repr__`: lines 482-483
- Function `load_checkpoint_d`: lines 20-68
- Function `load_checkpoint`: lines 100-141
- Function `save_checkpoint`: lines 144-162
- Function `save_checkpoint_d`: lines 165-188
- Function `summarize`: lines 191-207
- Function `latest_checkpoint_path`: lines 210-215
- Function `plot_spectrogram_to_numpy`: lines 218-241
- Function `plot_alignment_to_numpy`: lines 244-272
- Function `load_wav_to_torch`: lines 275-277
- Function `load_filepaths_and_text`: lines 280-288
- Function `get_hparams`: lines 291-391
- Function `get_hparams_from_dir`: lines 394-402
- Function `get_hparams_from_file`: lines 405-411
- Function `check_git_hash`: lines 414-436
- Function `get_logger`: lines 439-451

### infer/lib/uvr5_pack/lib_v5/dataset.py
- Class `VocalRemoverValidationSet`: lines 12-28
  - Method `__init__`: lines 13-14
  - Method `__len__`: lines 16-17
  - Method `__getitem__`: lines 19-28
- Function `make_pair`: lines 31-51
- Function `train_val_split`: lines 54-87
- Function `augment`: lines 90-115
- Function `make_padding`: lines 118-125
- Function `make_training_set`: lines 128-150
- Function `make_validation_set`: lines 153-183

### infer/lib/uvr5_pack/lib_v5/layers.py
- Class `Conv2DBNActiv`: lines 8-26
  - Method `__init__`: lines 9-23
  - Method `__call__`: lines 25-26
- Class `SeperableConv2DBNActiv`: lines 29-49
  - Method `__init__`: lines 30-46
  - Method `__call__`: lines 48-49
- Class `Encoder`: lines 52-62
  - Method `__init__`: lines 53-56
  - Method `__call__`: lines 58-62
- Class `Decoder`: lines 65-83
  - Method `__init__`: lines 66-71
  - Method `__call__`: lines 73-83
- Class `ASPPModule`: lines 86-118
  - Method `__init__`: lines 87-105
  - Method `forward`: lines 107-118

### infer/lib/uvr5_pack/lib_v5/layers_123812KB .py
- Class `Conv2DBNActiv`: lines 8-26
  - Method `__init__`: lines 9-23
  - Method `__call__`: lines 25-26
- Class `SeperableConv2DBNActiv`: lines 29-49
  - Method `__init__`: lines 30-46
  - Method `__call__`: lines 48-49
- Class `Encoder`: lines 52-62
  - Method `__init__`: lines 53-56
  - Method `__call__`: lines 58-62
- Class `Decoder`: lines 65-83
  - Method `__init__`: lines 66-71
  - Method `__call__`: lines 73-83
- Class `ASPPModule`: lines 86-118
  - Method `__init__`: lines 87-105
  - Method `forward`: lines 107-118

### infer/lib/uvr5_pack/lib_v5/layers_123821KB.py
- Class `Conv2DBNActiv`: lines 8-26
  - Method `__init__`: lines 9-23
  - Method `__call__`: lines 25-26
- Class `SeperableConv2DBNActiv`: lines 29-49
  - Method `__init__`: lines 30-46
  - Method `__call__`: lines 48-49
- Class `Encoder`: lines 52-62
  - Method `__init__`: lines 53-56
  - Method `__call__`: lines 58-62
- Class `Decoder`: lines 65-83
  - Method `__init__`: lines 66-71
  - Method `__call__`: lines 73-83
- Class `ASPPModule`: lines 86-118
  - Method `__init__`: lines 87-105
  - Method `forward`: lines 107-118

### infer/lib/uvr5_pack/lib_v5/layers_33966KB.py
- Class `Conv2DBNActiv`: lines 8-26
  - Method `__init__`: lines 9-23
  - Method `__call__`: lines 25-26
- Class `SeperableConv2DBNActiv`: lines 29-49
  - Method `__init__`: lines 30-46
  - Method `__call__`: lines 48-49
- Class `Encoder`: lines 52-62
  - Method `__init__`: lines 53-56
  - Method `__call__`: lines 58-62
- Class `Decoder`: lines 65-83
  - Method `__init__`: lines 66-71
  - Method `__call__`: lines 73-83
- Class `ASPPModule`: lines 86-126
  - Method `__init__`: lines 87-111
  - Method `forward`: lines 113-126

### infer/lib/uvr5_pack/lib_v5/layers_537227KB.py
- Class `Conv2DBNActiv`: lines 8-26
  - Method `__init__`: lines 9-23
  - Method `__call__`: lines 25-26
- Class `SeperableConv2DBNActiv`: lines 29-49
  - Method `__init__`: lines 30-46
  - Method `__call__`: lines 48-49
- Class `Encoder`: lines 52-62
  - Method `__init__`: lines 53-56
  - Method `__call__`: lines 58-62
- Class `Decoder`: lines 65-83
  - Method `__init__`: lines 66-71
  - Method `__call__`: lines 73-83
- Class `ASPPModule`: lines 86-126
  - Method `__init__`: lines 87-111
  - Method `forward`: lines 113-126

### infer/lib/uvr5_pack/lib_v5/layers_537238KB.py
- Class `Conv2DBNActiv`: lines 8-26
  - Method `__init__`: lines 9-23
  - Method `__call__`: lines 25-26
- Class `SeperableConv2DBNActiv`: lines 29-49
  - Method `__init__`: lines 30-46
  - Method `__call__`: lines 48-49
- Class `Encoder`: lines 52-62
  - Method `__init__`: lines 53-56
  - Method `__call__`: lines 58-62
- Class `Decoder`: lines 65-83
  - Method `__init__`: lines 66-71
  - Method `__call__`: lines 73-83
- Class `ASPPModule`: lines 86-126
  - Method `__init__`: lines 87-111
  - Method `forward`: lines 113-126

### infer/lib/uvr5_pack/lib_v5/layers_new.py
- Class `Conv2DBNActiv`: lines 8-26
  - Method `__init__`: lines 9-23
  - Method `__call__`: lines 25-26
- Class `Encoder`: lines 29-39
  - Method `__init__`: lines 30-33
  - Method `__call__`: lines 35-39
- Class `Decoder`: lines 42-64
  - Method `__init__`: lines 43-49
  - Method `__call__`: lines 51-64
- Class `ASPPModule`: lines 67-102
  - Method `__init__`: lines 68-85
  - Method `forward`: lines 87-102
- Class `LSTMModule`: lines 105-125
  - Method `__init__`: lines 106-114
  - Method `forward`: lines 116-125

### infer/lib/uvr5_pack/lib_v5/model_param_init.py
- Variable `default_param`: lines 5-5
- Class `ModelParameters`: lines 45-69
  - Method `__init__`: lines 46-69
- Function `int_keys`: lines 36-42

### infer/lib/uvr5_pack/lib_v5/nets.py
- Class `BaseASPPNet`: lines 9-37
  - Method `__init__`: lines 10-22
  - Method `__call__`: lines 24-37
- Class `CascadedASPPNet`: lines 40-123
  - Method `__init__`: lines 41-59
  - Method `forward`: lines 61-114
  - Method `predict`: lines 116-123

### infer/lib/uvr5_pack/lib_v5/nets_123812KB.py
- Class `BaseASPPNet`: lines 8-36
  - Method `__init__`: lines 9-21
  - Method `__call__`: lines 23-36
- Class `CascadedASPPNet`: lines 39-122
  - Method `__init__`: lines 40-58
  - Method `forward`: lines 60-113
  - Method `predict`: lines 115-122

### infer/lib/uvr5_pack/lib_v5/nets_123821KB.py
- Class `BaseASPPNet`: lines 8-36
  - Method `__init__`: lines 9-21
  - Method `__call__`: lines 23-36
- Class `CascadedASPPNet`: lines 39-122
  - Method `__init__`: lines 40-58
  - Method `forward`: lines 60-113
  - Method `predict`: lines 115-122

### infer/lib/uvr5_pack/lib_v5/nets_33966KB.py
- Class `BaseASPPNet`: lines 8-36
  - Method `__init__`: lines 9-21
  - Method `__call__`: lines 23-36
- Class `CascadedASPPNet`: lines 39-122
  - Method `__init__`: lines 40-58
  - Method `forward`: lines 60-113
  - Method `predict`: lines 115-122

### infer/lib/uvr5_pack/lib_v5/nets_537227KB.py
- Class `BaseASPPNet`: lines 9-37
  - Method `__init__`: lines 10-22
  - Method `__call__`: lines 24-37
- Class `CascadedASPPNet`: lines 40-123
  - Method `__init__`: lines 41-59
  - Method `forward`: lines 61-114
  - Method `predict`: lines 116-123

### infer/lib/uvr5_pack/lib_v5/nets_537238KB.py
- Class `BaseASPPNet`: lines 9-37
  - Method `__init__`: lines 10-22
  - Method `__call__`: lines 24-37
- Class `CascadedASPPNet`: lines 40-123
  - Method `__init__`: lines 41-59
  - Method `forward`: lines 61-114
  - Method `predict`: lines 116-123

### infer/lib/uvr5_pack/lib_v5/nets_61968KB.py
- Class `BaseASPPNet`: lines 8-36
  - Method `__init__`: lines 9-21
  - Method `__call__`: lines 23-36
- Class `CascadedASPPNet`: lines 39-122
  - Method `__init__`: lines 40-58
  - Method `forward`: lines 60-113
  - Method `predict`: lines 115-122

### infer/lib/uvr5_pack/lib_v5/nets_new.py
- Class `BaseNet`: lines 8-42
  - Method `__init__`: lines 9-25
  - Method `__call__`: lines 27-42
- Class `CascadedNet`: lines 45-133
  - Method `__init__`: lines 46-76
  - Method `forward`: lines 78-114
  - Method `predict_mask`: lines 116-123
  - Method `predict`: lines 125-133

### infer/lib/uvr5_pack/lib_v5/spec_utils.py
- Function `crop_center`: lines 12-27
- Function `wave_to_spectrogram`: lines 30-51
- Function `wave_to_spectrogram_mt`: lines 54-86
- Function `combine_spectrograms`: lines 89-124
- Function `spectrogram_to_image`: lines 127-148
- Function `reduce_vocal_aggressively`: lines 151-159
- Function `mask_silence`: lines 162-197
- Function `align_wave_head_and_tail`: lines 200-203
- Function `cache_or_load`: lines 206-292
- Function `spectrogram_to_wave`: lines 295-316
- Function `spectrogram_to_wave_mt`: lines 319-350
- Function `cmb_spectrogram_to_wave`: lines 353-428
- Function `fft_lp_filter`: lines 431-439
- Function `fft_hp_filter`: lines 442-450
- Function `mirroring`: lines 453-490
- Function `ensembling`: lines 493-507
- Function `stft`: lines 510-517
- Function `istft`: lines 520-526

### infer/lib/uvr5_pack/utils.py
- Function `load_data`: lines 8-12
- Function `make_padding`: lines 15-22
- Function `inference`: lines 25-99
- Function `_get_name_params`: lines 102-121

### infer/modules/ipex/__init__.py
- Function `ipex_init`: lines 12-190

### infer/modules/ipex/attention.py
- Variable `original_torch_bmm`: lines 6-6
- Variable `original_scaled_dot_product_attention`: lines 81-81
- Function `torch_bmm`: lines 9-78
- Function `scaled_dot_product_attention`: lines 84-212
- Function `attention_init`: lines 215-218

### infer/modules/ipex/gradscaler.py
- Variable `OptState`: lines 8-8
- Variable `_MultiDeviceReplicator`: lines 9-9
- Variable `_refresh_per_optimizer_state`: lines 10-12
- Function `_unscale_grads_`: lines 15-63
- Function `unscale_`: lines 66-113
- Function `update`: lines 116-179
- Function `gradscaler_init`: lines 182-187

### infer/modules/ipex/hijacks.py
- Variable `_utils`: lines 43-43
- Variable `original_autocast`: lines 121-121
- Variable `original_torch_cat`: lines 131-131
- Variable `original_interpolate`: lines 147-147
- Variable `original_linalg_solve`: lines 183-183
- Class `CondFunc`: lines 9-40
  - Method `__new__`: lines 10-29
  - Method `__init__`: lines 31-34
  - Method `__call__`: lines 36-40
- Class `DummyDataParallel`: lines 80-88
  - Method `__new__`: lines 83-88
- Function `_shutdown_workers`: lines 46-77
- Function `return_null_context`: lines 91-92
- Function `check_device`: lines 95-100
- Function `return_xpu`: lines 103-112
- Function `ipex_no_cuda`: lines 115-118
- Function `ipex_autocast`: lines 124-128
- Function `torch_cat`: lines 134-144
- Function `interpolate`: lines 150-180
- Function `linalg_solve`: lines 186-193
- Function `ipex_hijacks`: lines 196-365

### infer/modules/onnx/export.py
- Function `export_onnx`: lines 6-54

### infer/modules/train/extract/extract_f0_print.py
- Variable `now_dir`: lines 7-7
- Variable `exp_dir`: lines 19-19
- Variable `f`: lines 20-20
- Variable `n_p`: lines 29-29
- Variable `f0method`: lines 30-30
- Class `FeatureInput`: lines 33-139
  - Method `__init__`: lines 34-42
  - Method `compute_f0`: lines 44-93
  - Method `coarse_f0`: lines 95-109
  - Method `go`: lines 111-139
- Function `printt`: lines 23-26

### infer/modules/train/extract/extract_f0_rmvpe.py
- Variable `now_dir`: lines 7-7
- Variable `n_part`: lines 18-18
- Variable `i_part`: lines 19-19
- Variable `i_gpu`: lines 20-20
- Variable `exp_dir`: lines 22-22
- Variable `is_half`: lines 23-23
- Variable `f`: lines 24-24
- Class `FeatureInput`: lines 33-102
  - Method `__init__`: lines 34-42
  - Method `compute_f0`: lines 44-56
  - Method `coarse_f0`: lines 58-72
  - Method `go`: lines 74-102
- Function `printt`: lines 27-30

### infer/modules/train/extract/extract_f0_rmvpe_dml.py
- Variable `now_dir`: lines 7-7
- Variable `exp_dir`: lines 18-18
- Variable `device`: lines 21-21
- Variable `f`: lines 22-22
- Class `FeatureInput`: lines 31-100
  - Method `__init__`: lines 32-40
  - Method `compute_f0`: lines 42-54
  - Method `coarse_f0`: lines 56-70
  - Method `go`: lines 72-100
- Function `printt`: lines 25-28

### infer/modules/train/extract_feature_print.py
- Variable `device`: lines 8-8
- Variable `n_part`: lines 9-9
- Variable `i_part`: lines 10-10
- Variable `f`: lines 45-45
- Variable `model_path`: lines 55-55
- Variable `wavPath`: lines 58-58
- Variable `outPath`: lines 59-61
- Variable `model`: lines 93-93
- Variable `model`: lines 94-94
- Variable `todo`: lines 101-101
- Variable `n`: lines 102-102
- Function `printt`: lines 48-51
- Function `readwave`: lines 66-77

### infer/modules/train/preprocess.py
- Variable `now_dir`: lines 7-7
- Variable `inp_root`: lines 10-10
- Variable `sr`: lines 11-11
- Variable `n_p`: lines 12-12
- Variable `exp_dir`: lines 13-13
- Variable `noparallel`: lines 14-14
- Variable `per`: lines 15-15
- Variable `f`: lines 26-26
- Class `PreProcess`: lines 35-131
  - Method `__init__`: lines 36-57
  - Method `norm_write`: lines 59-79
  - Method `pipeline`: lines 81-105
  - Method `pipeline_mp`: lines 107-109
  - Method `pipeline_mp_inp_dir`: lines 111-131
- Function `println`: lines 29-32
- Function `preprocess_trainset`: lines 134-138

### infer/modules/train/train.py
- Variable `logger`: lines 5-5
- Variable `now_dir`: lines 7-7
- Variable `hps`: lines 14-14
- Variable `n_gpus`: lines 16-16
- Variable `global_step`: lines 79-79
- Class `EpochRecorder`: lines 82-92
  - Method `__init__`: lines 83-84
  - Method `record`: lines 86-92
- Function `main`: lines 95-117
- Function `run`: lines 120-296
- Function `train_and_evaluate`: lines 299-635

### infer/modules/uvr5/mdxnet.py
- Variable `logger`: lines 4-4
- Variable `cpu`: lines 12-12
- Class `ConvTDFNetTrim`: lines 15-75
  - Method `__init__`: lines 16-39
  - Method `stft`: lines 41-56
  - Method `istft`: lines 58-75
- Class `Predictor`: lines 90-238
  - Method `__init__`: lines 91-107
  - Method `demix`: lines 109-141
  - Method `demix_base`: lines 143-197
  - Method `prediction`: lines 199-238
- Class `MDXNetDereverb`: lines 241-256
  - Method `__init__`: lines 242-253
  - Method `_path_audio_`: lines 255-256
- Function `get_models`: lines 78-87

### infer/modules/uvr5/modules.py
- Variable `logger`: lines 5-5
- Variable `config`: lines 14-14
- Function `uvr`: lines 17-108

### infer/modules/uvr5/vr.py
- Variable `logger`: lines 4-4
- Class `AudioPre`: lines 18-195
  - Method `__init__`: lines 19-42
  - Method `_path_audio_`: lines 44-195
- Class `AudioPreDeEcho`: lines 198-368
  - Method `__init__`: lines 199-223
  - Method `_path_audio_`: lines 225-368

### infer/modules/vc/__init__.py

### infer/modules/vc/modules.py
- Variable `logger`: lines 4-4
- Class `VC`: lines 22-304
  - Method `__init__`: lines 23-34
  - Method `get_vc`: lines 36-144
  - Method `vc_single`: lines 146-225
  - Method `vc_multi`: lines 227-304

### infer/modules/vc/pipeline.py
- Variable `logger`: lines 6-6
- Variable `now_dir`: lines 21-21
- Variable `input_audio_path2wav`: lines 26-26
- Class `Pipeline`: lines 65-457
  - Method `__init__`: lines 66-82
  - Method `get_f0`: lines 84-184
  - Method `vc`: lines 186-279
  - Method `pipeline`: lines 281-457
- Function `cache_harvest_f0`: lines 30-40
- Function `change_rms`: lines 43-62

### infer/modules/vc/utils.py
- Function `get_index_path_from_model`: lines 6-19
- Function `load_hubert`: lines 22-33

### launch_rvc_rust.py
- Class `Colors`: lines 29-39
- Class `RVCLauncher`: lines 42-333
  - Method `__init__`: lines 45-49
  - Method `log`: lines 51-60
  - Method `setup_logging`: lines 62-74
  - Method `check_prerequisites`: lines 76-115
  - Method `build_frontend`: lines 117-159
  - Method `compile_rust`: lines 161-185
  - Method `start_application`: lines 187-218
  - Method `monitor_logs`: lines 220-242
  - Method `signal_handler`: lines 244-247
  - Method `shutdown`: lines 249-277
  - Method `run`: lines 279-333
- Function `main`: lines 336-339

### run_model_test.py
- Class `ModelTestRunner`: lines 16-335
  - Method `__init__`: lines 17-25
  - Method `print_header`: lines 27-31
  - Method `print_step`: lines 33-36
  - Method `check_prerequisites`: lines 38-95
  - Method `run_python_generator`: lines 97-132
  - Method `check_generated_files`: lines 134-176
  - Method `run_rust_tests`: lines 178-215
  - Method `run_rust_check`: lines 217-239
  - Method `cleanup_generated_files`: lines 241-255
  - Method `print_summary`: lines 257-286
  - Method `run`: lines 288-335
- Function `main`: lines 337-352

### rvc-rs/rvc-lib/tests/data/gen_phase_vocoder.py
- Variable `a`: lines 3-6
- Variable `b`: lines 7-10
- Variable `fade_out`: lines 11-14
- Variable `fade_in`: lines 15-18
- Variable `n`: lines 20-20
- Variable `a_t`: lines 21-21
- Variable `b_t`: lines 22-22
- Variable `fo_t`: lines 23-23
- Variable `fi_t`: lines 24-24
- Variable `window`: lines 25-25
- Variable `fa`: lines 26-26
- Variable `fb`: lines 27-27
- Variable `absab`: lines 28-28
- Variable `phia`: lines 33-33
- Variable `phib`: lines 34-34
- Variable `deltaphase`: lines 35-35
- Variable `w`: lines 37-37
- Variable `case`: lines 48-54
- Function `compute`: lines 39-46

### rvc-rs/rvc-lib/tests/data/gen_test_data.py
- Variable `fs`: lines 3-3
- Variable `samples`: lines 14-14
- Variable `wave`: lines 15-15
- Function `save_case`: lines 5-8

### rvc-rs/rvc-lib/tests/data/generate_f0_test_cases.py
- Function `save_test_case`: lines 14-47
- Function `generate_sine_wave`: lines 49-53
- Function `generate_complex_harmonic`: lines 55-68
- Function `generate_chirp`: lines 70-90
- Function `generate_noisy_sine`: lines 92-97
- Function `generate_voiced_unvoiced_segments`: lines 99-133
- Function `compute_expected_f0_frames`: lines 135-149
- Function `main`: lines 151-305

### scripts/gen_python_index.py
- Function `parse_file`: lines 6-33
- Function `gather_python_files`: lines 36-42
- Function `generate_index`: lines 45-61

### test_python_rvc.py
- Variable `current_dir`: lines 16-16
- Function `create_test_config`: lines 29-41
- Function `print_config`: lines 44-54
- Function `test_basic_functionality`: lines 57-105
- Function `test_audio_processing`: lines 108-176
- Function `test_f0_extraction`: lines 179-210
- Function `create_test_signal`: lines 213-216
- Function `calculate_rms`: lines 219-221
- Function `main`: lines 224-258
- Function `performance_test`: lines 261-297

### test_rvc_config.py
- Function `test_file_existence`: lines 14-36
- Function `test_config_creation`: lines 38-90
- Function `test_rust_compilation`: lines 92-118
- Function `test_frontend_build`: lines 120-153
- Function `print_system_info`: lines 155-174
- Function `main`: lines 176-216

### tools/app.py
- Variable `logger`: lines 16-16
- Variable `i18n`: lines 18-18
- Variable `config`: lines 22-22
- Variable `vc`: lines 23-23
- Variable `weight_root`: lines 25-25
- Variable `weight_uvr5_root`: lines 26-26
- Variable `index_root`: lines 27-27
- Variable `names`: lines 28-28
- Variable `hubert_model`: lines 29-29
- Variable `index_paths`: lines 33-33
- Variable `app`: lines 40-40

### tools/calc_rvc_model_similarity.py
- Variable `logger`: lines 6-6
- Function `cal_cross_attn`: lines 13-29
- Function `model_hash`: lines 32-43
- Function `eval`: lines 46-53
- Function `main`: lines 56-90

### tools/download_models.py
- Variable `RVC_DOWNLOAD_LINK`: lines 5-5
- Variable `BASE_DIR`: lines 7-7
- Function `dl_model`: lines 10-16

### tools/export_onnx.py

### tools/infer/infer-pm-index256.py
- Variable `logger`: lines 9-9
- Variable `device`: lines 37-37
- Variable `model_path`: lines 38-38
- Variable `model`: lines 44-44
- Variable `model`: lines 45-45
- Variable `model`: lines 46-46
- Variable `net_g`: lines 51-70
- Variable `weights`: lines 81-81
- Variable `index`: lines 125-125
- Variable `big_npy`: lines 126-126
- Variable `ta0`: lines 127-127
- Variable `ta1`: lines 127-127
- Variable `ta2`: lines 127-127
- Function `get_f0`: lines 88-120

### tools/infer/train-index-v2.py
- Variable `logger`: lines 9-9
- Variable `n_cpu`: lines 18-18
- Variable `inp_root`: lines 21-21
- Variable `npys`: lines 22-22
- Variable `listdir_res`: lines 23-23
- Variable `big_npy`: lines 27-27
- Variable `big_npy_idx`: lines 28-28
- Variable `big_npy`: lines 30-30
- Variable `n_ivf`: lines 56-56
- Variable `index`: lines 57-57
- Variable `index_ivf`: lines 59-59
- Variable `batch_size_add`: lines 66-66

### tools/infer/train-index.py
- Variable `logger`: lines 8-8
- Variable `inp_root`: lines 14-14
- Variable `npys`: lines 15-15
- Variable `big_npy`: lines 19-19
- Variable `index`: lines 26-26
- Variable `index_ivf`: lines 28-28

### tools/infer/trans_weights.py
- Variable `a`: lines 9-13

### tools/infer_batch_rvc.py
- Variable `now_dir`: lines 7-7
- Function `arg_parse`: lines 19-38
- Function `main`: lines 41-68

### tools/infer_cli.py
- Variable `now_dir`: lines 5-5
- Function `arg_parse`: lines 19-38
- Function `main`: lines 41-63

### tools/onnx_inference_demo.py
- Variable `hop_size`: lines 5-5
- Variable `sampling_rate`: lines 6-6
- Variable `f0_up_key`: lines 7-7
- Variable `sid`: lines 8-8
- Variable `f0_method`: lines 9-9
- Variable `model_path`: lines 10-10
- Variable `vec_name`: lines 11-13
- Variable `wav_path`: lines 14-14
- Variable `out_path`: lines 15-15
- Variable `model`: lines 17-19
- Variable `audio`: lines 21-21

### tools/rvc_for_realtime.py
- Variable `now_dir`: lines 27-27
- Variable `mm`: lines 35-35
- Class `RVC`: lines 47-446
  - Method `__init__`: lines 48-194
  - Method `change_key`: lines 196-197
  - Method `change_index_rate`: lines 199-204
  - Method `get_f0_post`: lines 206-217
  - Method `get_f0`: lines 219-288
  - Method `get_f0_crepe`: lines 290-312
  - Method `get_f0_rmvpe`: lines 314-327
  - Method `get_f0_fcpe`: lines 329-346
  - Method `infer`: lines 348-446
- Function `printt`: lines 38-42

### tools/torchgate/__init__.py

### tools/torchgate/torchgate.py
- Class `TorchGate`: lines 8-280
  - Method `__init__`: lines 33-72
  - Method `_generate_mask_smoothing_filter`: lines 75-125
  - Method `_stationary_mask`: lines 128-175
  - Method `_nonstationary_mask`: lines 178-208
  - Method `forward`: lines 210-280

### tools/torchgate/utils.py
- Function `amp_to_db`: lines 6-25
- Function `temperature_sigmoid`: lines 29-41
- Function `linspace`: lines 45-70

