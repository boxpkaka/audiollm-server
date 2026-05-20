# 通用流式 ASR API

`/transcribe-streaming` 用于实时语音转写。客户端通过 WebSocket 发送 16 kHz mono s16le PCM 音频流，服务端按 VAD 语音段返回中间结果和最终结果。

REST 上传版本见本文末尾的 `/api/asr/upload`。

## 接口信息

| 项目 | 说明 |
|---|---|
| 协议 | WebSocket |
| 路径 | `/transcribe-streaming` |
| 完整 URL | `ws://172.16.0.3:8080/transcribe-streaming?language=<lang>` |
| 鉴权 | 无内置鉴权，无需自定义请求头 |
| 音频输入 | 二进制 PCM 帧，16 kHz、mono、signed 16-bit little-endian |
| 分段策略 | 服务端 VAD 自动切段 |
| 中间结果 | 支持，受 `enable_pseudo_stream` 与 `pseudo_stream_interval_ms` 影响 |
| 最终结果 | 每个语音段一条；`stop` 后会 flush 尾部残余音频 |

### Query 参数

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `language` | string | 否 | 语言代码，如 `zh`、`en`、`id`、`th`。不传表示自动检测或使用服务端默认策略 |

## 调用流程

```text
Client                                      Server
  | ---- WebSocket connect --------------> |
  | <---------------- ready -------------- |
  | ---- start --------------------------> |
  | ---- binary PCM chunk ---------------> |
  | <---------------- partial ----------- |
  | ---- binary PCM chunk ---------------> |
  | <---------------- final ------------- |  one speech segment
  | ---- binary PCM chunk ---------------> |
  | ---- stop ---------------------------> |
  | <---------------- final ------------- |  trailing audio, if any
  | ---- close --------------------------> |
```

服务端历史版本可能返回 `partial_asr` / `final_asr`，当前第三方客户端建议同时兼容 `partial` / `partial_asr` 与 `final` / `final_asr`。

## 客户端消息

### start

收到 `ready` 后、发送任何 PCM 前必须先发送 `start`。

```json
{
  "type": "start",
  "format": "pcm_s16le",
  "sample_rate_hz": 16000,
  "channels": 1,
  "language": "zh",
  "hotwords": ["挚音科技", "张硕"],
  "config": {
    "vad_threshold": 0.45
  }
}
```

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `type` | string | 是 | 固定为 `start` |
| `format` | string | 是 | 固定为 `pcm_s16le` |
| `sample_rate_hz` | integer | 是 | 固定为 `16000` |
| `channels` | integer | 是 | 固定为 `1` |
| `language` | string | 否 | 语言代码；与 query 参数二选一即可 |
| `hotwords` | string[] | 否 | 热词列表 |
| `config` | object | 否 | 当前连接的服务端配置覆写 |

### update_hotwords

会话中可随时更新热词。

```json
{
  "type": "update_hotwords",
  "hotwords": ["产品名", "人名"],
  "src_lang": "zh"
}
```

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `type` | string | 是 | 固定为 `update_hotwords` |
| `hotwords` | string[] | 是 | 新热词列表；空数组表示清空 |
| `src_lang` | string | 否 | 语言代码或语言名称 |

### 二进制音频帧

`start` 后持续发送 PCM bytes。

| 项目 | 要求 |
|---|---|
| 编码 | signed 16-bit little-endian PCM |
| 采样率 | 16000 Hz |
| 声道 | 1 |
| 推荐 chunk | 30-80 ms |
| 80 ms 字节数 | 2560 bytes |

### stop

```json
{"type":"stop"}
```

`stop` 表示本次音频输入结束。服务端会处理剩余音频并返回最终结果。

## 服务端消息

### ready

```json
{"type":"ready"}
```

### partial / partial_asr

```json
{
  "type": "partial",
  "id": "seg-001",
  "text": "你好",
  "language": "zh"
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `type` | string | `partial` 或 `partial_asr` |
| `id` | string | 语音段 ID；可能不存在 |
| `text` | string | 当前语音段的临时转写文本 |
| `language` | string | 检测或传入的语言 |

### final / final_asr

```json
{
  "type": "final",
  "id": "seg-001",
  "text": "你好，欢迎使用语音识别服务。",
  "language": "zh",
  "duration_sec": 3.42
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `type` | string | `final` 或 `final_asr` |
| `id` | string | 语音段 ID；可能不存在 |
| `text` | string | 最终转写文本 |
| `language` | string | 检测或传入的语言 |
| `duration_sec` | number | 本次推理使用的音频时长；部分流式消息可能不带 |

### error

```json
{
  "type": "error",
  "message": "invalid start message"
}
```

客户端收到 `error` 后应记录完整 payload，并根据业务需要停止发送音频或关闭连接。

## 可覆写配置

常用 `start.config` 字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `vad_threshold` | number | VAD 判定阈值 |
| `silence_duration_ms` | integer | 静音持续多久后切段 |
| `min_segment_duration_ms` | integer | 短于该值的语音段会被丢弃 |
| `enable_pseudo_stream` | boolean | 是否输出伪流式中间结果 |
| `pseudo_stream_interval_ms` | integer | 伪流式中间结果间隔 |
| `asr_request_timeout` | number | 单次 ASR 模型请求超时秒数 |

## Python WebSocket 示例

完整可运行脚本见 [examples/ws_transcribe.py](examples/ws_transcribe.py)。

```bash
pip install websockets numpy

python docs/examples/ws_transcribe.py sample.wav \
  --url ws://172.16.0.3:8080/transcribe-streaming \
  --language zh \
  --hotwords "挚音科技,张硕" \
```

使用 `bash start.sh`（`wss://172.16.0.3:8443/...`）时，示例脚本可加 `--insecure` 跳过自签证书校验。

## REST 上传接口

`POST /api/asr/upload` 适合上传完整音频文件并获得一次最终转写结果。

### 请求

| 表单字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `audio` | file | 是 | WAV 音频文件 |
| `language` | string | 否 | 语言代码 |
| `hotwords` | string | 否 | 逗号分隔热词，如 `挚音科技,张硕` |

### 响应

```json
{
  "type": "final",
  "text": "你好，欢迎使用语音识别服务。",
  "language": "zh",
  "duration_sec": 3.42
}
```

### Python REST 示例

完整可运行脚本见 [examples/rest_upload.py](examples/rest_upload.py)。

```bash
pip install requests

python docs/examples/rest_upload.py asr sample.wav \
  --base-url http://172.16.0.3:8080 \
  --language zh \
  --hotwords "挚音科技,张硕" \
```

## 相关文档

- [API 总览](api-reference.md)
- [目标说话人 ASR](tsasr.md)
- [整段情感识别](emotion-streaming-protocol.md)
- [分段情感识别](emotion-segmented-streaming-protocol.md)
