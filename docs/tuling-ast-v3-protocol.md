# 实时转写 AST v3 WebSocket API

`/tuling/ast/v3` 以讯飞图灵 AST v3 信封协议对外提供实时语音转写。它复用与 `/transcribe-streaming` 相同的 VAD 分段 + 双模型 ASR 流水线，区别仅在于线上协议：音频以 base64 编码放在 JSON 帧的 `payload.audio.audio` 中，`header.status`（0/1/2）驱动开始/中间/结束状态机，结果按 `payload.result` 词图结构返回。

该端点与现有 `/transcribe-streaming`、`/ws/audio` 并存，互不影响。客户端可使用讯飞 `tuling-ast-sdk`（Java）或任意 WebSocket 客户端按本协议对接。

## 接口信息

| 项目 | 说明 |
|---|---|
| 协议 | WebSocket |
| 路径 | `/tuling/ast/v3` |
| 完整 URL | `ws(wss)://[ip]:[port]/tuling/ast/v3` |
| 鉴权 | 无内置鉴权 |
| 音频输入 | base64 编码的 16 kHz、mono、signed 16-bit little-endian PCM；首帧若为带 RIFF/WAVE 头的整段音频会自动剥离文件头 |
| 分段策略 | 服务端 VAD 自动切段 |
| 中间结果 | 支持，msgtype 为 Progressive，受 `enable_pseudo_stream` 影响，可经 `parameter.asr_config` 覆写 |
| 最终结果 | 每个语音段一条，msgtype 为 sentence；尾帧（status=2）后 flush 残余音频 |

## 调用流程

```text
Client                                      Server
  | ---- WebSocket connect --------------> |
  | ---- frame status=0 (首帧 + 音频) ----> |
  | <---------------- Progressive -------- |  中间结果（可选）
  | ---- frame status=1 (音频) ----------> |
  | <---------------- sentence ----------- |  一个语音段
  | ---- frame status=1 (音频) ----------> |
  | ---- frame status=2 (尾帧 + 音频) ----> |
  | <---------------- sentence ----------- |  尾部音频
  | <---------------- status=2 ----------- |  会话结束标志
```

与 `/transcribe-streaming` 不同，AST v3 没有 `ready` 握手和 `start`/`stop` 控制消息：会话生命周期完全由 `header.status` 表达。

## 客户端请求

每一帧都是一个完整的 JSON 信封：

```json
{
  "header": {
    "traceId": "traceId123456",
    "appId": "123456",
    "bizId": "39769795890",
    "status": 0,
    "resIdList": ["234567", "345678"]
  },
  "parameter": {
    "engine": {
      "wdec_param_LanguageTypeChoice": "3"
    }
  },
  "payload": {
    "audio": {
      "audio": "JiuY3iK9AAB..."
    }
  }
}
```

### header

| 字段 | 类型 | 必传 | 说明 |
|---|---|---|---|
| traceId | String | 是 | 日志追踪 id，原样回显到响应 header.traceId |
| appId | String | 否 | 应用系统 id，仅记录日志 |
| bizId | String | 是 | 业务 id，仅记录日志 |
| resIdList | List<String> | 否 | 目标说话人 enrollment id 列表，取 resIdList[0] 作为目标说话人（TS-ASR），需先经 REST 注册获取；仅用第一个，不做多说话人分离（见“目标说话人”章节） |
| status | int | 是 | 请求状态：0 首帧，1 中间帧，2 尾帧 |

### parameter

| 字段 | 类型 | 必传 | 说明 |
|---|---|---|---|
| engine | Map | 否 | 引擎透传参数，仅记录日志，当前不映射到任何行为（见已知限制）。兼容 SDK 使用的 parameter.service |
| asr_config | Map | 否 | 本服务扩展的 per-connection 配置覆写，仅首帧（status=0）读取，仅当前连接生效、不落盘。详见“配置覆写”章节 |

### payload

| 字段 | 类型 | 必传 | 说明 |
|---|---|---|---|
| payload.audio.audio | String | 是 | base64 编码的 PCM 音频分片 |
| payload.text.text | String | 否 | 文本类型热词，仅在首帧（status=0）生效，按逗号/顿号/分号/换行切分为热词列表 |

### 状态机与音频

| status | 含义 | 服务端处理 |
|---|---|---|
| 0 | 首帧 | 建立会话、捕获 traceId、生成 sid、读取热词；若本帧带音频则同时送入 VAD |
| 1 | 中间帧 | 解码音频送入 VAD |
| 2 | 尾帧 | 先送本帧音频，再 flush 残余音频结束会话 |

音频建议：每帧约 4096 字节（讯飞 SDK 默认 `32 * 128`），原则上单帧不超过 16 KB，建议至少 40 ms 语音。正常处理过程中客户端不要主动断开。首帧允许携带带 WAV 头的整段音频前缀，服务端会一次性剥离文件头后按裸 PCM 处理。

## 配置覆写（parameter.asr_config）

`parameter.asr_config` 是本服务在 AST v3 信封上的扩展槽位，用于按连接临时调参，与讯飞 `parameter.engine`（仅记录日志、不映射行为）并列、互不影响。仅在首帧（status=0）读取，仅对当前连接生效、不落盘；新连接或服务重启都回到服务端默认。

取值优先级（后者覆盖前者）：`backend/config.py` 内置默认 → `backend/config.json` 服务端默认 → `parameter.asr_config` 客户端临时覆写。

只接受白名单内的扁平字段名；未知字段、受限字段（模型地址、密钥、连接池/队列等基础设施项）以及非法值会被忽略并保持服务端默认，不会中断连接。`language` 是特例：它不是配置字段，会被用作本次会话语言（等价于 `/transcribe-streaming` 的 `start.language`）。

可覆写字段与 `/transcribe-streaming` 的 `start.config` 共用同一白名单。下表为对本端点有效的 ASR 相关字段，每个字段语义见 [通用流式 ASR WebSocket](transcribe-streaming-protocol.md) 与 [API 总览](api-reference.md) 的“临时配置覆写”：

| 类别 | 字段 |
|---|---|
| VAD / 分段 | vad_threshold、silence_duration_ms、vad_smoothing_alpha、vad_start_frames、vad_pre_speech_ms、vad_end_frames、vad_keep_tail_ms、min_segment_duration_ms |
| 伪流式 | enable_pseudo_stream、pseudo_stream_interval_ms |
| ASR 模型组合 / 超时 | enable_primary_asr、enable_secondary_asr、enable_dual_asr_fusion、primary_asr_timeout、asr_request_timeout、debug_show_dual_asr |
| 融合阈值 | fusion_similarity_threshold、fusion_min_primary_score、fusion_max_repetition_ratio、fusion_disagreement_threshold、fusion_hotword_boost、fusion_primary_score_margin |
| TS-ASR | asr_enrollment_min_sec、asr_enrollment_max_sec、asr_enrollment_ttl_sec |

ASR 模型组合开关存在不变量：`enable_dual_asr_fusion=true` 但 `enable_secondary_asr=false` 会被自动降级为 false（行为矩阵见 [API 总览](api-reference.md)）。共享白名单还包含情感类字段（`emotion_*`），对本 ASR 端点无效，完整清单见 [API 总览](api-reference.md)。

首帧示例（指定语言、关闭伪流式中间结果、放宽 VAD 切段）：

```json
{
  "header": {
    "traceId": "traceId123456",
    "bizId": "39769795890",
    "status": 0
  },
  "parameter": {
    "asr_config": {
      "language": "zh",
      "enable_pseudo_stream": false,
      "vad_threshold": 0.45,
      "silence_duration_ms": 300
    }
  },
  "payload": {
    "audio": { "audio": "JiuY3iK9AAB..." }
  }
}
```

## 目标说话人（TS-ASR）

支持只转写指定说话人的语音，复用与 `/transcribe-streaming` 相同的注册机制，分两步：

1. 注册：通过 `POST /api/asr/enrollment` 上传 1-8 秒目标说话人音频，拿到 `enrollment_id`（见 [API 总览](api-reference.md) 的注册接口）。
2. 携带：在首帧（status=0）把该 id 放进 `header.resIdList`，服务端取 `resIdList[0]` 作为目标说话人。

```json
{
  "header": {
    "traceId": "traceId123456",
    "bizId": "39769795890",
    "status": 0,
    "resIdList": ["ule8QilVjZql30Q9oy9kiQ"]
  },
  "parameter": { "engine": {} },
  "payload": { "audio": { "audio": "JiuY3iK9AAB..." } }
}
```

说明：

- enrollment_id 仅在首帧读取，整段会话沿用同一目标说话人。
- 若 resIdList[0] 未注册或已过期，服务端静默回退为普通 ASR（仅记 WARN，不返回 error），避免长连接因陈旧 id 中断。
- resIdList 含多个 id 时只用第一个，不做多说话人分离。
- 未携带 resIdList 时为普通 ASR。

## 服务端响应

每条响应也是一个信封：

```json
{
  "header": {
    "code": 0,
    "message": "success",
    "sid": "AST_MKMZO0WX2SLZ4",
    "traceId": "traceId123456",
    "status": 1
  },
  "payload": {
    "result": { }
  }
}
```

### header

| 字段 | 类型 | 说明 |
|---|---|---|
| code | int | 错误码，0 表示成功，非 0（当前实现为 -1）表示错误 |
| message | String | 描述信息，成功为 success |
| sid | String | 会话唯一标识，服务端按 AST_ 前缀生成 |
| traceId | String | 回显请求的 traceId |
| status | int | 结果状态：1 识别中，2 识别结束（终止帧） |

错误响应只含 header（code 非 0），不含 payload；客户端应在 code 非 0 时停止处理。

### payload.result

| 字段 | 类型 | 说明 |
|---|---|---|
| segId | int | 段 id，从 0 递增 |
| bg | int | 段开始时间，单位 ms，msgtype 为 sentence 时给出 |
| ed | int | 段结束时间，单位 ms，msgtype 为 sentence 时给出 |
| ei | int | 暂未使用，固定 0 |
| ls | Bool | 最后结果标志；逐段结果为 false，仅终止帧为 true |
| metadata | String | 暂未使用，固定空串 |
| msgtype | String | sentence 为最终结果，Progressive 为中间结果 |
| sn | int | 结果序号，从 1 递增，msgtype 为 sentence 时给出 |
| pa | int | 暂未使用，固定 0 |
| vad | Object | 句子级 VAD 信息，vad.ws[].bg/ed 为句子起止时间，单位 10 ms 帧 |
| ws | Array | 转写词图，见下表 |

### payload.result.ws

| 字段 | 类型 | 说明 |
|---|---|---|
| ws[].bg | int | 词语开始时间，单位 10 ms 帧 |
| ws[].cw | Array | 词语识别候选 |
| cw[].w | String | 识别文本 |
| cw[].lg | String | 语种，如 zh |
| cw[].wb | int | 词开始位置，单位 10 ms 帧（数值 ×10 为毫秒） |
| cw[].we | int | 词结束位置，单位 10 ms 帧 |
| cw[].wp | String | 顺滑词类型：s 顺滑词，n 普通字符，p 标点，g 语义分段标志 |
| cw[].sc | String | 词置信度 |
| cw[].wc | String | 词置信度 |
| cw[].ng | String | 噪声分 |
| cw[].ph | String | 音素信息 |

result 示例（最终结果）：

```json
{
  "segId": 0,
  "bg": 140,
  "ed": 3230,
  "ei": 0,
  "ls": false,
  "metadata": "",
  "msgtype": "sentence",
  "sn": 1,
  "pa": 0,
  "vad": { "ws": [{ "bg": 14, "ed": 323 }] },
  "ws": [
    {
      "bg": 14,
      "cw": [
        {
          "lg": "zh", "ng": "0.00", "ph": "phone", "sc": "0.00",
          "w": "你好兄弟", "wb": 14, "wc": "0.00", "we": 323, "wp": "n"
        }
      ]
    }
  ]
}
```

## 降级说明（重要）

当前 ASR 模型只输出整段文本，不产出词级对齐、词级语种或置信度。因此本端点对 ws/cw 词图采用降级填充，集成方需知悉：

| 字段 | 降级行为 |
|---|---|
| ws / cw 结构 | 每个 sentence 段只产生一个 ws、一个 cw，cw.w 为整段文本，不做逐词切分 |
| bg / ed（段级） | 真实值，基于会话累计已消费音频估算，单位 ms |
| vad.ws[].bg/ed | 取段级起止，单位 10 ms 帧 |
| cw.wb / cw.we | 取段级起止帧，并非逐词时间戳 |
| cw.wp | 固定 n |
| cw.sc / cw.wc / cw.ng | 固定字符串 0.00 |
| cw.ph | 固定字符串 phone |
| cw.lg | 取段级检测/传入语种映射的短码 |

段级 bg/ed 为近似值：它基于流累计消费的样本数，会忽略 VAD 静音判定延迟与尾部裁剪，误差通常在百毫秒量级。

## 已知限制

| 限制 | 说明 |
|---|---|
| resIdList 多说话人 | resIdList 仅取首个作目标说话人（TS-ASR），其余忽略；当前单路 ASR 不做多说话人分离 |
| parameter.engine | 讯飞引擎透传参数（如 wdec_param_LanguageTypeChoice、wrec_param_language_name）在本服务无对应能力，仅记录日志，不影响识别；如需按连接调参请改用 parameter.asr_config（见配置覆写章节） |
| 词级时间戳 | 见降级说明，非逐词真实值 |
| 鉴权 | 无内置鉴权，需在网关层实现访问控制 |

## 错误码

| code | 含义 |
|---|---|
| 0 | 成功 |
| -1 | 通用错误，message 携带具体原因 |

AST v3 规范仅约定 code 0 为成功，其余码段交由实现方定义。本服务统一使用 -1 表示可恢复的段处理错误，并在 header.message 给出描述。

## 会话结束语义

- 每个语音段的最终结果以 msgtype=sentence、header.status=1 返回。
- 中间结果以 msgtype=Progressive、header.status=1 返回。
- 整个会话结束时（客户端发送 status=2 或断开连接后），服务端补发一条 header.status=2 的终止帧，其 payload.result 不含 ws（ls=true），作为会话结束标志。
- 若整段会话没有任何有效语音，则只发送终止帧，不发送空的 sentence。

## Python 示例

完整可运行脚本见 [examples/ws_ast_v3.py](examples/ws_ast_v3.py)，集成测试客户端见 [tests/test_ast_v3_ws_client.py](../tests/test_ast_v3_ws_client.py)。

```bash
pip install websockets numpy

python docs/examples/ws_ast_v3.py sample.wav \
  --url ws://172.16.0.3:8080/tuling/ast/v3 \
  --hotwords "挚音科技,张硕"
```

目标说话人：先注册拿到 id，再用 `--enrollment-id` 传入（脚本会写进首帧 `header.resIdList[0]`）：

```bash
curl -X POST http://172.16.0.3:8080/api/asr/enrollment -F "audio=@speaker_enroll.wav"
# {"enrollment_id": "ule8QilVjZql30Q9oy9kiQ", "duration_sec": 3.0}

python docs/examples/ws_ast_v3.py sample.wav \
  --url ws://172.16.0.3:8080/tuling/ast/v3 \
  --enrollment-id "ule8QilVjZql30Q9oy9kiQ"
```

使用 `bash start.sh`（`wss://172.16.0.3:8443/...`）时可加 `--insecure` 跳过自签证书校验。

## Java SDK

可使用讯飞 `tuling-ast-sdk` 直接对接。响应字段名与本文档逐字对齐，可被 `ApiResponse` / `LatticeItem` / `ResultItem` / `ResultWordItem` 直接反序列化。客户端按 `header.status == 2` 判断会话结束、按 `header.code != 0` 判断错误。

## 相关文档

- [API 总览](api-reference.md)
- [通用流式 ASR WebSocket](transcribe-streaming-protocol.md)
- [分段情感识别 WebSocket](emotion-segmented-streaming-protocol.md)
