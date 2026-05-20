# 整段情感识别 API

`/emotion-streaming` 用于对一段完整语音进行一次情感识别。客户端通过 WebSocket 发送 PCM 音频流，发送 `stop` 后服务端对整段音频做推理并返回一条 `final_emotion`。

如果需要长连接中按句或按语音段持续返回情感结果，请使用 [分段情感识别 API](emotion-segmented-streaming-protocol.md)。

REST 上传版本见本文末尾的 `/api/emotion/upload`。

## 接口信息

| 项目 | 说明 |
|---|---|
| 协议 | WebSocket |
| 路径 | `/emotion-streaming` |
| 完整 URL | `ws://172.16.0.3:8080/emotion-streaming` |
| 鉴权 | 无内置鉴权，无需自定义请求头 |
| 音频输入 | 二进制 PCM 帧，16 kHz、mono、signed 16-bit little-endian |
| 分段策略 | 不启用 VAD 切段，整段缓存到 `stop` |
| 中间结果 | 不支持 |
| 最终结果 | 每个 start/stop 周期返回一条 `final_emotion` |

## 模式

| mode | 说明 | 输出 |
|---|---|---|
| `ser` | Speech Emotion Recognition | 8 分类情感标签 |
| `sec` | Speech Emotion Captioning | 自由文本情感描述，并尽量给出匹配标签 |

`mode` 不传时使用服务端配置 `emotion_task_mode`，通常为 `ser`。

### SER 标签集

| 标签 | 含义 |
|---|---|
| `Angry` | 愤怒 |
| `Sad` | 悲伤 |
| `Happy` | 高兴 |
| `Surprise` | 惊讶 |
| `Fear` | 害怕 |
| `Disgust` | 厌恶 |
| `Contempt` | 轻蔑 |
| `Neutral` | 中性 |

## 调用流程

```text
Client                                      Server
  | ---- WebSocket connect --------------> |
  | <---------------- ready -------------- |
  | ---- start --------------------------> |
  | ---- binary PCM chunk ---------------> |
  | ---- binary PCM chunk ---------------> |
  | ---- stop ---------------------------> |
  | <---------------- final_emotion ------ |
  | ---- close --------------------------> |
```

## 客户端消息

### start

```json
{
  "type": "start",
  "format": "pcm_s16le",
  "sample_rate_hz": 16000,
  "channels": 1,
  "mode": "ser",
  "language": "zh",
  "config": {
    "emotion_request_timeout": 30
  }
}
```

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `type` | string | 是 | 固定为 `start` |
| `format` | string | 是 | 固定为 `pcm_s16le` |
| `sample_rate_hz` | integer | 是 | 固定为 `16000` |
| `channels` | integer | 是 | 固定为 `1` |
| `mode` | string | 否 | `ser` 或 `sec` |
| `language` | string | 否 | 透传字段；如提供，响应会回填 |
| `config` | object | 否 | 当前连接的服务端配置覆写 |

### 二进制音频帧

`start` 后持续发送 PCM bytes。建议每帧 80 ms，即 2560 bytes。

### stop

```json
{"type":"stop"}
```

服务端收到后对累计音频做一次推理，并返回一条 `final_emotion`。如果没有有效音频，也会返回一条空结果。

## 服务端消息

### ready

```json
{"type":"ready"}
```

### final_emotion

SER 响应：

```json
{
  "type": "final_emotion",
  "mode": "ser",
  "label": "Happy",
  "text": "Happy",
  "duration_sec": 3.21,
  "language": "zh"
}
```

SEC 响应：

```json
{
  "type": "final_emotion",
  "mode": "sec",
  "label": "Happy",
  "text": "The speaker sounds excited and cheerful.",
  "duration_sec": 3.21,
  "language": "zh"
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `type` | string | 固定为 `final_emotion` |
| `mode` | string | 本次推理模式：`ser` 或 `sec` |
| `label` | string | SER 主标签；SEC 下为文本中匹配到的参考标签，可能为空 |
| `text` | string | SER 下通常与 `label` 一致；SEC 下为自由文本描述 |
| `raw_text` | string | 可选，模型原始输出与 `text` 不一致时返回 |
| `duration_sec` | number | 实际推理使用的音频时长 |
| `language` | string | 可选，客户端传入 language 时回填 |

### error

```json
{
  "type": "error",
  "message": "emotion inference failed"
}
```

## 可覆写配置

| 字段 | 类型 | 说明 |
|---|---|---|
| `emotion_task_mode` | string | 默认情感任务模式：`ser` 或 `sec` |
| `emotion_request_timeout` | number | 单次情感模型请求超时秒数 |
| `emotion_max_audio_seconds` | number | 单次推理最大音频秒数；超出保留尾部 |

## Python WebSocket 示例

完整可运行脚本见 [examples/ws_emotion.py](examples/ws_emotion.py)。

```bash
pip install websockets numpy

python docs/examples/ws_emotion.py sample.wav \
  --url ws://172.16.0.3:8080/emotion-streaming \
  --mode ser \
```

## REST 上传接口

`POST /api/emotion/upload` 适合上传完整音频并获得一次情感识别结果。

### 请求

| 表单字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `audio` | file | 是 | WAV 音频文件 |
| `mode` | string | 否 | `ser` 或 `sec` |
| `language` | string | 否 | 透传语言字段 |

### 响应

```json
{
  "type": "final_emotion",
  "mode": "ser",
  "label": "Happy",
  "text": "Happy",
  "duration_sec": 3.21,
  "language": "zh"
}
```

### Python REST 示例

完整可运行脚本见 [examples/rest_upload.py](examples/rest_upload.py)。

```bash
pip install requests

python docs/examples/rest_upload.py emotion sample.wav \
  --base-url http://172.16.0.3:8080 \
  --mode ser \
  --language zh \
```

## 相关文档

- [API 总览](api-reference.md)
- [通用流式 ASR](transcribe-streaming-protocol.md)
- [分段情感识别](emotion-segmented-streaming-protocol.md)
- [目标说话人 ASR](tsasr.md)
