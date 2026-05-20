# 目标说话人 ASR API

`/transcribe-target-streaming` 用于目标说话人语音识别（Target-Speaker ASR）。客户端先提交一段目标说话人的注册音频，再持续发送混合音频流；服务端只返回目标说话人的转写结果。

REST 上传版本见本文末尾的 `/api/tsasr/upload`。

## 接口信息

| 项目 | 说明 |
|---|---|
| 协议 | WebSocket |
| 路径 | `/transcribe-target-streaming` |
| 完整 URL | `ws://172.16.0.3:8080/transcribe-target-streaming?language=<lang>` |
| 鉴权 | 无内置鉴权，无需自定义请求头 |
| 注册音频 | `start.enrollment_audio`，base64 编码 WAV |
| 混合音频输入 | 二进制 PCM 帧，16 kHz、mono、signed 16-bit little-endian |
| 分段策略 | 服务端 VAD 自动切段 |
| 中间结果 | 默认关闭，可通过 `tsasr_enable_partial=true` 开启 |
| 最终结果 | 每个目标说话人语音段一条 `final` |

### Query 参数

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `language` | string | 否 | 语言代码，如 `zh`、`en`、`id`、`th` |

## 注册音频要求

| 项目 | 要求 |
|---|---|
| 格式 | WAV，base64 编码后放入 `enrollment_audio` |
| 内容 | 目标说话人单人语音，尽量无背景人声 |
| 默认最短时长 | 1 秒 |
| 默认最长时长 | 8 秒；超长音频服务端会按配置裁剪 |
| 采样率/声道 | 可不是 16 kHz mono，服务端会解码并转换 |

## 调用流程

```text
Client                                      Server
  | ---- WebSocket connect --------------> |
  | <---------------- ready -------------- |
  | ---- start(enrollment_audio) --------> |
  | <---------------- enrollment_ok ------ |
  | ---- binary PCM chunk ---------------> |
  | <---------------- partial ----------- |  optional
  | <---------------- final ------------- |  one speech segment
  | ---- binary PCM chunk ---------------> |
  | ---- stop ---------------------------> |
  | <---------------- final ------------- |  trailing audio, if any
  | ---- close --------------------------> |
```

如果注册音频缺失或校验失败，服务端返回 `error`，通常不会主动关闭 WebSocket。客户端应关闭连接并重新录制注册音频。

## 客户端消息

### start

```json
{
  "type": "start",
  "format": "pcm_s16le",
  "sample_rate_hz": 16000,
  "channels": 1,
  "language": "zh",
  "enrollment_audio": "UklGR...",
  "enrollment_format": "wav",
  "hotwords": ["产品名", "人名"],
  "config": {
    "tsasr_enable_partial": false
  }
}
```

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `type` | string | 是 | 固定为 `start` |
| `format` | string | 否 | 固定为 `pcm_s16le` |
| `sample_rate_hz` | integer | 否 | 固定为 `16000` |
| `channels` | integer | 否 | 固定为 `1` |
| `language` | string | 否 | 语言代码；与 query 参数二选一即可 |
| `enrollment_audio` | string | 是 | base64 WAV 注册音频 |
| `enrollment_format` | string | 否 | 固定为 `wav` |
| `hotwords` | string[] | 否 | 热词列表 |
| `voice_traits` | string | 否 | 兼容旧客户端字段，当前不建议使用 |
| `config` | object | 否 | 当前连接的服务端配置覆写 |

### update_hotwords

```json
{
  "type": "update_hotwords",
  "hotwords": ["新热词"],
  "src_lang": "zh"
}
```

当 `tsasr_enable_hotwords=true` 时，中途更新的热词会影响后续推理。

### extract_hotwords

如果服务端配置了外部长文本热词抽取模型，可发送：

```json
{
  "type": "extract_hotwords",
  "request_id": "req-001",
  "text": "需要抽取热词的长文本"
}
```

服务端返回 `extract_hotwords_result` 或 `extract_hotwords_error`，并回填同一个 `request_id`。

### 二进制音频帧

注册成功后持续发送混合音频 PCM bytes。格式固定为 16 kHz、mono、s16le，建议每帧 30-80 ms。

### stop

```json
{"type":"stop"}
```

服务端收到后处理尾部残余音频，并尽量返回最后一条 `final`。

## 服务端消息

### ready

```json
{"type":"ready"}
```

### enrollment_ok

```json
{
  "type": "enrollment_ok",
  "duration_sec": 4.12
}
```

表示注册音频校验通过，可以开始发送混合音频。

### partial

```json
{
  "type": "partial",
  "id": "seg-001",
  "text": "目标说话人正在",
  "task": "tsasr"
}
```

默认不返回 partial。需要时可在 `start.config` 中设置 `tsasr_enable_partial=true`。

### final

```json
{
  "type": "final",
  "id": "seg-001",
  "task": "tsasr",
  "text": "目标说话人的转写文本",
  "text_secondary": "通用 ASR 的参考文本",
  "language": "zh",
  "duration_sec": 5.12,
  "audio_b64": "UklGR..."
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `type` | string | 固定为 `final` |
| `id` | string | 语音段 ID；WebSocket 流式结果中可能出现 |
| `task` | string | 固定为 `tsasr` |
| `text` | string | 目标说话人转写文本 |
| `text_secondary` | string | 可选，通用 ASR 参考文本 |
| `language` | string | 检测或传入的语言 |
| `duration_sec` | number | 当前语音段或上传音频的推理时长 |
| `audio_b64` | string | 可选，当前段音频 WAV base64，便于回放或排查 |

### 热词抽取结果

```json
{
  "type": "extract_hotwords_result",
  "request_id": "req-001",
  "hotwords": ["产品名", "人名"]
}
```

失败时：

```json
{
  "type": "extract_hotwords_error",
  "request_id": "req-001",
  "message": "hotword extraction failed"
}
```

### error

```json
{
  "type": "error",
  "code": "enrollment_missing",
  "message": "enrollment_audio is required"
}
```

注册音频相关错误码：

| code | 说明 |
|---|---|
| `enrollment_missing` | `start` 未提供 `enrollment_audio` |
| `enrollment_empty` | 注册音频解码后为空 |
| `enrollment_too_short` | 注册音频短于 `tsasr_enrollment_min_sec` |
| `enrollment_too_long` | 注册音频超过 `tsasr_enrollment_max_sec` 且无法裁剪到有效范围 |
| `enrollment_decode_failed` | base64 或 WAV 解码失败 |
| `enrollment_unsupported_format` | `enrollment_format` 不是 `wav` |

## 可覆写配置

| 字段 | 类型 | 说明 |
|---|---|---|
| `tsasr_request_timeout` | number | 单次 TS-ASR 模型请求超时秒数 |
| `tsasr_enrollment_min_sec` | number | 注册音频最短秒数 |
| `tsasr_enrollment_max_sec` | number | 注册音频最长秒数 |
| `tsasr_max_audio_seconds` | number | 每段混合音频最大推理秒数 |
| `tsasr_enable_partial` | boolean | 是否返回 partial |
| `tsasr_enable_hotwords` | boolean | 是否把热词注入 TS-ASR prompt |
| `tsasr_enable_secondary_gate` | boolean | 是否启用通用 ASR 空语音门控 |
| `tsasr_show_secondary_text` | boolean | 是否返回通用 ASR 参考文本 |
| `vad_threshold` | number | VAD 判定阈值 |
| `silence_duration_ms` | integer | 静音持续多久后切段 |
| `min_segment_duration_ms` | integer | 短于该值的语音段会被丢弃 |

## Python WebSocket 示例

完整可运行脚本见 [examples/ws_tsasr.py](examples/ws_tsasr.py)。

```bash
pip install websockets numpy

python docs/examples/ws_tsasr.py mixed.wav \
  --url ws://172.16.0.3:8080/transcribe-target-streaming \
  --enrollment enrollment.wav \
  --language zh \
  --hotwords "产品名,人名" \
```

## REST 上传接口

`POST /api/tsasr/upload` 适合一次性上传注册音频和混合音频，获得一条目标说话人转写结果。

### 请求

| 表单字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `audio` | file | 是 | 混合音频 WAV 文件 |
| `enrollment_wav_base64` | string | 是 | base64 WAV 注册音频 |
| `language` | string | 否 | 语言代码 |
| `hotwords` | string | 否 | 逗号分隔热词 |
| `voice_traits` | string | 否 | 兼容字段，当前不建议使用 |

### 响应

```json
{
  "type": "final",
  "task": "tsasr",
  "text": "目标说话人的转写文本",
  "text_secondary": "通用 ASR 的参考文本",
  "language": "zh",
  "duration_sec": 5.12,
  "audio_b64": "UklGR..."
}
```

### Python REST 示例

完整可运行脚本见 [examples/rest_upload.py](examples/rest_upload.py)。

```bash
pip install requests

python docs/examples/rest_upload.py tsasr mixed.wav \
  --base-url http://172.16.0.3:8080 \
  --enrollment enrollment.wav \
  --language zh \
  --hotwords "产品名,人名" \
```

## 相关文档

- [API 总览](api-reference.md)
- [通用流式 ASR](transcribe-streaming-protocol.md)
- [整段情感识别](emotion-streaming-protocol.md)
- [分段情感识别](emotion-segmented-streaming-protocol.md)
