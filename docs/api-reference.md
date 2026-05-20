# AudioLLM API Reference

本文档面向外部系统集成方，说明如何远程调用 AudioLLM 服务完成语音转写、目标说话人转写和情感识别测试。

## 基础信息

| 项目 | 说明 |
|---|---|
| Base URL | `http://172.16.0.3:8080`（systemd 生产部署） |
| WebSocket Base URL | `ws://172.16.0.3:8080` |
| 鉴权 | 当前服务不要求 API Key、Token 或自定义请求头 |
| 音频格式 | WebSocket 发送原始 PCM：16 kHz、mono、signed 16-bit little-endian |
| REST 上传 | 使用 multipart/form-data 上传 WAV 文件，服务端会解码为 16 kHz mono |
| 默认端口 | systemd 生产部署 `8080`（HTTP）；`start.sh` 开发启动为 `8443`（HTTPS） |

生产环境如需访问控制、限流或 IP 白名单，应在 API 网关、负载均衡或反向代理层配置。

## 接口总览

### WebSocket 任务接口

| 接口 | 任务 | 适用场景 | 结果消息 |
|---|---|---|---|
| `/transcribe-streaming` | 通用流式 ASR | 实时语音转写、带热词的对话转写 | `partial` / `partial_asr`、`final` / `final_asr` |
| `/transcribe-target-streaming` | 目标说话人 ASR | 给定注册音频，只转写目标说话人 | `enrollment_ok`、`partial`、`final` |
| `/emotion-segmented-streaming` | 分段情感识别 | 长连接中按 VAD 语音段持续返回情感 | 多条 `final_emotion` |

`/ws/audio` 是浏览器 Demo 使用的调试接口，包含前端专用消息和双模型调试视图。第三方系统集成建议优先使用上表中的任务接口。

### REST 上传接口

| 方法 | 路径 | 任务 | 表单字段 |
|---|---|---|---|
| POST | `/api/asr/upload` | 上传整段音频做 ASR | `audio`、`language`、`hotwords` |
| POST | `/api/emotion/jobs` | 异步整段情感识别（202 + 轮询） | `audio`、`mode`、`language` |
| GET | `/api/emotion/jobs/{job_id}` | 查询情感任务状态与结果 | — |
| POST | `/api/tsasr/upload` | 上传注册音频和混合音频做目标说话人 ASR | `audio`、`enrollment_wav_base64`、`language`、`hotwords`、`voice_traits` |
| POST | `/api/audio/analyze` | 非实时聚合分析：ASR 原始结果、文本清洗、情感标签和情感描述 | `audio`、`language`、`hotwords` |

## WebSocket 调用流程

所有任务型 WebSocket 接口共享同一条基本流程：

1. 连接 `ws://172.16.0.3:8080/<endpoint>`。
2. 等待服务端发送 `{"type":"ready"}`。
3. 发送一条 `start` JSON 消息，声明音频格式和任务参数。
4. 持续发送二进制 PCM 音频帧。
5. 接收中间结果或最终结果。
6. 发送 `{"type":"stop"}` 结束本次音频输入。
7. 等待服务端处理尾部音频并返回最终结果，然后关闭连接。

推荐每帧 30-80 ms PCM。16 kHz、mono、s16le 的字节数计算为：

```text
bytes_per_ms = 16000 * 1 * 2 / 1000 = 32
80 ms chunk = 2560 bytes
```

### 通用 start 消息

```json
{
  "type": "start",
  "format": "pcm_s16le",
  "sample_rate_hz": 16000,
  "channels": 1
}
```

各任务可以在此基础上增加字段，例如 ASR 的 `language` / `hotwords`、情感识别的 `mode`、TS-ASR 的 `enrollment_audio`。

### 临时配置覆写

`start.config` 可覆写服务端允许的配置字段，仅对当前连接生效。常用字段如下：

| 类别 | 字段 |
|---|---|
| VAD / 分段 | `vad_threshold`、`silence_duration_ms`、`min_segment_duration_ms` |
| ASR | `enable_pseudo_stream`、`pseudo_stream_interval_ms`、`asr_request_timeout` |
| 情感 | `emotion_task_mode`、`emotion_request_timeout`、`emotion_max_audio_seconds` |
| TS-ASR | `tsasr_enable_partial`、`tsasr_enable_hotwords`、`tsasr_request_timeout` |

示例：

```json
{
  "type": "start",
  "format": "pcm_s16le",
  "sample_rate_hz": 16000,
  "channels": 1,
  "config": {
    "vad_threshold": 0.45,
    "min_segment_duration_ms": 500
  }
}
```

## REST 上传调用

REST 接口适合离线测试或一次性上传完整录音。请求使用 `multipart/form-data`，音频字段名固定为 `audio`。

### 非实时聚合分析

`POST /api/audio/analyze` 会对同一段音频执行 ASR、文本清洗和情感理解。情感理解默认同时返回分类标签和文本描述。

`hotwords` 只传给 ASR 模型，用于 ASR 原生热词识别；文本清洗阶段不会接收热词，也不会根据热词做事后替换。

```bash
python docs/examples/rest_upload.py analyze sample.wav \
  --base-url http://172.16.0.3:8080 \
  --language zh \
  --hotwords "挚音科技,张硕"
```

响应示例：

```json
{
  "type": "audio_analysis",
  "duration_sec": 8.24,
  "language": "zh",
  "hotwords": ["挚音科技", "张硕"],
  "asr": {
    "text": "原始融合转写文本",
    "language": "zh"
  },
  "cleaned_asr": {
    "text": "清洗后的转写文本。"
  },
  "emotion": {
    "type": "final_emotion_pair",
    "mode": "both",
    "ser": {
      "type": "final_emotion",
      "mode": "ser",
      "label": "Neutral",
      "text": "Neutral",
      "duration_sec": 8.24
    },
    "sec": {
      "type": "final_emotion",
      "mode": "sec",
      "label": "Neutral",
      "text": "The speaker sounds calm and neutral.",
      "duration_sec": 8.24
    }
  }
}
```

### ASR 上传

```bash
python docs/examples/rest_upload.py asr sample.wav \
  --base-url http://172.16.0.3:8080 \
  --language zh \
  --hotwords "挚音科技,张硕"
```

响应示例：

```json
{
  "type": "final",
  "text": "你好，欢迎使用语音识别服务。",
  "language": "zh",
  "duration_sec": 3.42
}
```

### 情感上传

```bash
python docs/examples/rest_upload.py emotion sample.wav \
  --base-url http://172.16.0.3:8080 \
  --mode ser \
  --language zh
```

响应示例：

```json
{
  "type": "final_emotion",
  "mode": "ser",
  "label": "Happy",
  "text": "Happy",
  "duration_sec": 3.42,
  "language": "zh"
}
```

### TS-ASR 上传

```bash
python docs/examples/rest_upload.py tsasr mixed.wav \
  --base-url http://172.16.0.3:8080 \
  --enrollment enrollment.wav \
  --language zh \
  --hotwords "产品名,人名"
```

响应示例：

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

## Python 示例

先安装依赖：

```bash
pip install websockets requests numpy
```

运行 WebSocket ASR：

```bash
python docs/examples/ws_transcribe.py sample.wav \
  --url ws://172.16.0.3:8080/transcribe-streaming \
  --language zh
```

运行整段情感识别（异步 HTTP）：

```bash
python docs/examples/http_emotion_job.py sample.wav \
  --base-url http://172.16.0.3:8080 \
  --mode ser
```

或：

```bash
python docs/examples/rest_upload.py emotion sample.wav \
  --base-url http://172.16.0.3:8080 \
  --mode ser
```

运行分段情感识别（WebSocket）：

```bash
python tests/test_emotion_ws_client.py sample.wav \
  --url ws://172.16.0.3:8080/emotion-segmented-streaming \
  --segmented \
  --language zh
```

运行目标说话人 ASR：

```bash
python docs/examples/ws_tsasr.py mixed.wav \
  --url ws://172.16.0.3:8080/transcribe-target-streaming \
  --enrollment enrollment.wav \
  --language zh
```

使用 `bash start.sh`（`https://172.16.0.3:8443`）时，示例脚本可加 `--insecure` 跳过自签证书校验。

## 错误处理

### WebSocket 错误消息

服务端遇到可恢复错误时会发送：

```json
{
  "type": "error",
  "code": "enrollment_missing",
  "message": "enrollment_audio is required"
}
```

部分接口只返回 `message`。客户端应至少记录完整错误 payload，并在收到错误后停止发送音频或主动关闭连接。

### REST 错误响应

REST 接口使用标准 HTTP 状态码：

| 状态码 | 含义 |
|---|---|
| 400 | 请求字段缺失、音频为空、音频无法解码、注册音频校验失败 |
| 413 | 上传文件超过服务端大小限制 |
| 422 | multipart 字段类型或必填字段不符合 FastAPI 校验 |
| 502 | 后端模型服务推理失败 |
| 502 | `/api/audio/analyze` 的 ASR、情感或文本清洗模型调用失败 |
| 202 | `POST /api/emotion/jobs` 已受理（需轮询 GET） |
| 503 | 情感任务队列已满（`Retry-After`） |
| 404 | `GET /api/emotion/jobs/{id}` 任务不存在或已过期 |

错误体示例：

```json
{
  "detail": "audio file is empty"
}
```

TS-ASR 注册音频错误会返回结构化 detail：

```json
{
  "detail": {
    "code": "enrollment_too_short",
    "message": "enrollment audio is too short"
  }
}
```

## 相关文档

- [公网非实时音频分析 API](public-audio-analyze-api.md)
- [非实时音频分析 API](audio-analyze-api.md)
- [通用流式 ASR WebSocket](transcribe-streaming-protocol.md)
- [目标说话人 ASR WebSocket](tsasr.md)
- [整段情感识别 HTTP（异步）](emotion-streaming-protocol.md)
- [分段情感识别 WebSocket](emotion-segmented-streaming-protocol.md)
