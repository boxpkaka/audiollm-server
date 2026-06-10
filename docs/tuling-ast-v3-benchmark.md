# /tuling/ast/v3 多并发性能与极限压测报告

日期：2026-06-10
测试脚本：`scripts/bench_tuling_ttft.py`（支持 lhotse manifests / 单 wav 数据源、RTF、尾部时延、并发梯度、错峰起跑）
原始数据：`bench_results/tuling_astv3_*.json`

## 1. 测试环境

| 项目 | 值 |
|---|---|
| 端点 | wss://127.0.0.1:8443(8444)/tuling/ast/v3，同机回环，网络开销可忽略 |
| 被测主模型 | AmphionASR-4.3B，vLLM，localhost:8009，独占 A800-SXM4-80GB（GPU0） |
| 端点策略 | primary-only（无副模型、无融合，协议恒定） |
| VAD | ten-vad，约 16 ms/帧 |
| 公共配置 | vad_start_frames=20、pseudo_stream_first_partial_ms=200、primary_asr_timeout=4.0s、http_max_connections=32 |
| 配置口径 A（默认） | vad_threshold=0.65、silence_duration_ms=350、pseudo_stream_interval_ms=500，用于第 3、4 节 aishell 测试 |
| 配置口径 B（调优） | vad_threshold=0.6、silence_duration_ms=1200、pseudo_stream_interval_ms=1000，用于第 5 节长对话测试 |
| 服务形态 | uvicorn 单进程单事件循环 |
| 数据集 1 | aishell test（lhotse manifests），短句 2.5-8.4 s，均长 5.0 s，shuffle seed=0 |
| 数据集 2 | manual_tests/audio/120报警电话16k.wav，81.2 s 真实电话对话，VAD 多段切分 |
| 发送参数 | chunk 128 ms（约 4096 B/帧，讯飞 SDK 默认），每路独立 WebSocket 连接 |

说明：压测期间同机 GPU2 有另一组 Qwen3-ASR 压测在跑（不同 GPU、不同 vLLM 实例），CPU 共享但核数充足；本报告全部数字均为 Amphion 链路。

## 2. 指标口径

| 指标 | 定义 |
|---|---|
| onset（首字-体感） | 用户开口（起音点）到首个非空文本，已剔除每条音频的前导静音，realtime 主指标 |
| ttft（首字-协议） | 第一帧发送到首个非空文本，含前导静音 |
| ttft_final（首段终稿） | 第一帧发送到首个 final（sentence）；受 VAD 切段语义影响，须整句结束 + 350 ms 静音才产出 |
| final_lag（尾字时延） | 末帧（status=2）发送完成到最后一个 final，用户停止说话后拿到完整结果的等待 |
| full（会话收尾） | 末帧发送完成到 status=2 终止帧 |
| rtf | 会话墙钟 / 音频时长，fast 模式下反映服务端纯处理速度 |
| agg_xRT（有效吞吐） | sum(音频时长) / 批次墙钟，跨并发可比 |

发送模式：realtime 按 128 ms 节奏推流（模拟真实麦克风），fast 不间断灌流（测服务端处理上限）。

## 3. 服务端处理上限（fast 模式，aishell 短句 32 条）

| 并发 | 单路 RTF p50 | 单路 RTF p90 | 有效吞吐 | 首字 p50/p90 (ms) | 尾字 p50/p90 (ms) |
|---|---|---|---|---|---|
| 1 | 0.037 | 0.046 | 25.8 倍实时 | 107/131 | 150/234 |
| 4 | 0.048 | 0.066 | 70.1 倍实时 | 134/233 | 197/345 |
| 8 | 0.076 | 0.130 | 76.3 倍实时 | 227/417 | 339/614 |
| 16 | 0.181 | 0.282 | 71.1 倍实时 | 685/957 | 849/1311 |

要点：

- 单路服务端处理 RTF 约 0.037（约 27 倍实时）。
- 总吞吐在并发 8 左右饱和，约 76 倍实时；继续加并发只抬高单路时延，不增吞吐。
- 76 倍实时即该 GPU 上 vLLM 的语音识别总容量上界（含 partial 推理开销时实际可用并发低于此值，见后文）。

## 4. 短句实时流（realtime，aishell，每路一句、高频建连）

| 并发 | 成功率 | 首字-体感 p50/p90 | 首字-协议 p50/p90 | 尾字 p50/p90 | 会话收尾 p50/p90 |
|---|---|---|---|---|---|
| 1 | 32/32 | 369/810 | 812/944 | 0/99 | 1/99 |
| 4 | 32/32 | 366/814 | 817/946 | 0/101 | 2/101 |
| 8 | 32/32 | 371/822 | 823/963 | 0/108 | 4/109 |
| 16 | 32/32 | 383/823 | 866/992 | 0/119 | 8/119 |
| 16（64 条） | 64/64 | 398/818 | 847/987 | 0/131 | 9/132 |
| 32（64 条） | 64/64 | 604/1026 | 1023/1159 | 37/178 | 39/178 |

要点：

- 1-16 路零退化：首字体感 p50 稳定在 370-400 ms（binding 项为 vad_start_frames=20 的约 320 ms 起音确认 + 推理约 60-100 ms），尾字 p50 恒为 0（边说边出字）。
- 32 路开始退化（首字 +约 220 ms）：与 http_max_connections=32 持平，且短句场景每约 5 s 全量重建连接，建连风暴放大了排队。
- 首字要进一步压低：首帧 asr_config 传 vad_start_frames=10，参考协议文档"首字延迟优化"一节（实测 onset p50 可到约 220 ms）。

## 5. 长对话实时流（realtime，120 报警电话 81.2 s，N 路全程同时在线）

本节全部使用配置口径 B（silence_duration_ms=1200 等），与生产调优形态一致；该口径下 120 电话被切成少而长的段（首段终稿约 32 s 处产出），段级推理时长更长，对 4 s 超时更敏感。

### 5.1 同相位（worst case：所有路同一时刻开始推同一条音频）

所有路的 VAD 在同一毫秒切段、final 推理请求同步爆发，是人为构造的最坏情况。

| 并发 | 成功率 | 首字-体感 p50 | 首段终稿 p50 | 尾字 p50 | 结论 |
|---|---|---|---|---|---|
| 8 | 8/8 | 3050 | 32149 | 0 | 稳定 |
| 16 | 16/16 | 3134 | 32161 | 0 | 稳定 |
| 32 | 32/32 | 3147 | 33115 | 0 | 稳定 |
| 48 | 48/48 | 3213 | 35233 | 0 | 临界：首段终稿 +3.1 s |
| 56 | 18/56 | - | - | - | 38 路会话失败 |
| 64 | 18/64 | - | - | - | 46 路会话失败（两轮复现） |

注：首字体感绝对值约 3.05 s 偏大是口径原因——音频开头约 3 s 为铃声/前奏，能量法起音点把铃声计入，ten-vad 等真实人声才出字；该列请看档间相对变化（8→48 路仅 +163 ms）。

### 5.2 错峰起跑（typical case：各路起点错开，模拟真实相位分布）

| 并发 | 错峰间隔 | 成功率 | 首字-体感 p50/p90 | 首段终稿 p50 | 尾字 p50 | 会话收尾 p50 |
|---|---|---|---|---|---|---|
| 64 | 1.5 s | 64/64 | 3049/3198 | 31994 | 0 | 14 |
| 96 | 1.2 s | 96/96 | 3059/4219 | 32325 | 0 | 42 |
| 128 | 1.0 s | 0/128 | - | - | - | - |

要点：

- 64 路错峰全部成功，首字与 8 路基线完全持平（3049 vs 3050 ms）——同相位档位的崩溃完全由同步 burst 造成，正常相位分布下 64 路毫无压力。
- 96 路错峰仍 100% 成功，p50 无退化，但首字 p90 出现 +约 1.0 s 的排队尾部（4219 vs 基线约 3200 ms），是接近容量上限的早期信号。
- 128 路错峰全部失败（error 帧风暴，与 6 节同一根因链）：128 路稳态 partial 请求约 128 QPS 加 final，超过推理容量，排队普遍越过 4 s 超时；且风暴后服务进入持续拒绝服务（见第 6 节），需重启恢复。
- 错峰口径的极限位于 96-128 路之间。

## 6. 故障定性（56/64 路同相位崩溃的根因链）

从服务端日志（`asyncio.exceptions.TimeoutError` traceback，65 处）确认：

1. 同步 burst 时刻，数十路 final 段同时进入 vLLM 排队，排队 + 推理超过 primary_asr_timeout=4.0 s。
2. `asyncio.wait_for` 抛 TimeoutError，`handle_segment` 未对超时做降级，异常冒泡为 WS error 帧；且 `str(TimeoutError())` 为空串，错误帧只有兜底文案 message="error"，几乎无法定位。
3. 按 AST v3 协议约定，客户端在 header.code != 0 时停止处理——一次瞬时超时杀死整条长连接会话（会话级雪崩，而非段级降级）。
4. 二次伤害（过载后不可自愈）：大规模风暴后服务进入持续拒绝服务状态（vLLM 队列已排空、新会话仍全部超时），重启 backend 进程才恢复。复现规律与 http_max_connections=32 吻合——单轮 56/64 路风暴后可自愈，连续两轮 64 路或单轮 128 路（取消的在途请求数明显超过 32）后持久故障，疑似 httpx 连接池在大规模请求取消下连接未归还、池被永久占满。
5. 触发面差异：同相位风暴在 56 路即触发（瞬时 burst）；错峰稳态在 128 路触发（稳态推理需求超容量）。两者殊途同归于同一条超时-错误帧链路。

瓶颈层次结论：vLLM 推理容量（76 倍实时）并不是先到的天花板，先到的是"4 s 超时 + 错误处理语义 + 连接池"这组工程上限。

## 7. 结论与建议

容量结论（Amphion-4.3B 单卡 A800、primary_asr_timeout=4 s、http_max_connections=32）：

| 场景 | 安全水位 | 极限/失效点 | 说明 |
|---|---|---|---|
| 短句高频建连（口径 A） | 16 路 | 32 路轻度退化 | 32 路起首字 +220 ms（连接池排队） |
| 长连接稳态-同相位（口径 B，worst case） | 48 路 | 56 路会话失败 | 全路同步切段的防御性下界 |
| 长连接稳态-错峰（口径 B，typical case） | 96 路（建议预留 buffer 按 80 路规划） | 128 路全军覆没且服务不可自愈 | 96 路 p90 已现 +1 s 排队尾部 |

改进建议（按优先级）：

1. final 超时降级而非断会话：`handle_segment` 捕获 TimeoutError 时返回空文本或以最近 partial 文本兜底，发 sentence 而非 error 帧；段级失败不应终止整条会话。
2. 错误信息可定位：error 帧 message 至少携带异常类型名（当前 `str(TimeoutError())` 为空导致 message="error"）。
3. 连接池扩容：http_max_connections 提升到不低于目标并发数（如 128），消除 32 路的池排队拐点。
4. 过载自愈问题跟进：复现路径为连续两轮 64 路同相位风暴；建议排查请求取消时 httpx response 未关闭导致的连接泄漏，或为连接池加健康检查/定期重建。
5. 容量规划按错峰口径执行，同相位口径仅作为防御性下界（真实业务天然错峰）。

## 8. 已知口径限制

- 同机回环测试，不含真实网络 RTT 与抖动。
- 长对话场景为同一条音频复制 N 路，文本内容相同可能略微提升 vLLM 前缀缓存命中（实测各路转写存在细微差异，影响有限）。
- 测试期间同机 GPU2 存在另一组独立压测（Qwen3-ASR），CPU 共享；GPU0（被测）独占。
- 首字体感（onset）依赖能量法起音点估计，对铃声/音乐前奏会提前计时，绝对值偏保守。

## 9. 复现命令

```bash
# fast 吞吐 sweep（aishell 短句）
uv run python scripts/bench_tuling_ttft.py \
  --url wss://127.0.0.1:8443/tuling/ast/v3 --insecure \
  --lhotse-recordings /ai_sds_wuzz/DATA_ASR/LHOTSE/data_aishell/data/manifests/aishell_recordings_test.jsonl.gz \
  --lhotse-supervisions /ai_sds_wuzz/DATA_ASR/LHOTSE/data_aishell/data/manifests/aishell_supervisions_test.jsonl.gz \
  --shuffle --seed 0 --limit 32 --warmup 2 \
  --send-mode fast --concurrency-sweep 1,4,8,16

# 长音频同相位 N 路（worst case）
uv run python scripts/bench_tuling_ttft.py \
  --url wss://127.0.0.1:8443/tuling/ast/v3 --insecure \
  --wav manual_tests/audio/120报警电话16k.wav \
  --limit 64 --warmup 0 --send-mode realtime --chunk-ms 128 --concurrency 64 \
  --final-timeout 90

# 长音频错峰 N 路（typical case）：加 --stagger-ms 1500
# 注意：>56 路的档位会触发会话失败风暴，风暴后服务可能需要重启才能恢复
```

## 10. 数据文件索引

| 文件（bench_results/） | 内容 |
|---|---|
| tuling_astv3_fast_sweep.json | aishell fast 1/4/8/16 路（口径 A） |
| tuling_astv3_realtime_sweep.json | aishell realtime 1/4/8/16 路（口径 A） |
| tuling_astv3_realtime_c32.json | aishell realtime 16/32 路 64 条（口径 A） |
| tuling_astv3_120call_c8/c16/c32/c48/c56.json | 120 电话同相位各档（口径 B） |
| tuling_astv3_120call_c64.json 与 c64_repro.json | 64 路同相位两轮复现（口径 B） |
| tuling_astv3_120call_c64_stagger.json | 64 路错峰（口径 B） |
| tuling_astv3_120call_c96_stagger.json | 96 路错峰（口径 B） |
| tuling_astv3_120call_c128_stagger.json | 128 路错峰，全失败（口径 B） |
