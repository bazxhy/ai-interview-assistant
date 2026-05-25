# AI 面试助手 — 实时语音监听 · AI 智能回答

> 面试中自动监听面试官提问 → 语音转文字 → AI 结合你的简历生成回答建议，助你从容应对。

## 功能特性

- **系统音频录制** — 通过 Windows WASAPI Loopback 录制电脑播放的声音（腾讯会议 / 微信通话 / 网页视频均可），支持蓝牙耳机
- **实时语音识别** — 接入讯飞语音听写 API，边说边出字，长句自动合并
- **AI 智能回答** — 支持 DeepSeek / OpenAI 兼容 LLM，流式逐字输出，回答围绕你的简历展开
- **简历自动读取** — 将 PDF/TXT 简历放在程序目录下，启动时自动解析并注入 AI 上下文
- **自我介绍秒返** — 检测到面试官要求自我介绍时，直接返回预写内容，零 AI 等待时间
- **关键词纠错** — 内置嵌入式领域专业术语表，自动修正语音识别中的技术名词误识别
- **可打包为 EXE** — PyInstaller 一键打包，发给其他人直接双击运行

## 工作环境

| 项目 | 要求 |
|------|------|
| 操作系统 | Windows 10 / 11 (64 位) |
| Python | 3.10+ (源码运行) 或 无需 Python (EXE 运行) |
| 网络 | 需要，用于调用讯飞 API 和 LLM API |
| 音频 | 系统音频设备正常（需启用立体声混音 / WASAPI） |

## 需要的 API 服务

| 服务 | 用途 | 获取方式 | 免费额度 |
|------|------|----------|----------|
| **讯飞语音听写** | 语音转文字 (STT) | [讯飞开放平台](https://console.xfyun.cn) 创建应用 | 每日 500 次 |
| **DeepSeek** (推荐) | AI 文本生成 (LLM) | [DeepSeek 开放平台](https://platform.deepseek.com) | 注册送额度 |
| OpenAI (可选) | AI 文本生成 / 语音识别 | [OpenAI](https://platform.openai.com) | 付费 |

> **推荐组合：** 讯飞 STT + DeepSeek LLM，均有免费额度，国内访问速度快。

## 快速开始（源码运行）

### 1. 克隆项目

```bash
git clone https://github.com/bazxhy/ai-interview-assistant.git
cd ai-interview-assistant
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置 .env 文件

```bash
copy .env.example .env
```

用文本编辑器打开 `.env`，填入你的 API 密钥：

```ini
# ========== API Key ==========
# 至少填一个 LLM 的 Key
DEEPSEEK_API_KEY=sk-你的DeepSeek密钥
# OPENAI_API_KEY=sk-你的OpenAI密钥  (如果用OpenAI)

# ========== AI 模型配置 ==========
LLM_PROVIDER=deepseek               # deepseek / openai
LLM_MODEL=deepseek-chat             # deepseek-chat / gpt-4o 等
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=1024

# ========== 语音识别配置 ==========
STT_PROVIDER=xunfei                 # xunfei (推荐) / local / openai

# 讯飞凭证 (前往 https://console.xfyun.cn 创建应用获取)
IFLYTEK_APP_ID=你的APP_ID
IFLYTEK_API_KEY=你的API_KEY
IFLYTEK_API_SECRET=你的API_SECRET

# ========== 音频输入配置 ==========
# loopback = 录制系统声音(面试官) / mic = 录制麦克风(你自己)
AUDIO_MODE=loopback

# VAD 能量阈值: loopback 模式 100-300, mic 模式 50-200
VAD_ENERGY_THRESHOLD=100
```

### 4. (可选) 放入简历

将你的简历 PDF 或 TXT 文件放到项目目录下（文件名包含"简历"、"resume"或"cv"即可），程序启动时自动识别。**不放简历也能运行**，但 AI 回答会缺乏针对性。

### 5. 运行

```bash
python main.py
```

## 打包为独立 EXE

```bash
pip install pyinstaller

# 一键打包
build.bat

# 产物在 dist/AI面试助手/ 目录
# 将 .env 和简历.pdf 也复制到该目录，整个文件夹可直接发给别人使用
```

## 使用方法

### 面试场景

1. 打开腾讯会议 / 微信通话 / 网页面试页面
2. 运行程序，观察 `系统声音: xxxx` 有没有跳动（确认能听到面试官声音）
3. 面试官说话时，控制台实时显示识别结果（紫色 `->` 行）
4. AI 思考中（绿色文字逐字输出）
5. 看完建议后向面试官回答

### 快捷键

| 按键 | 功能 |
|------|------|
| `Space` | 手动触发识别（处理最近 5 秒音频） |
| `C` | 清除对话历史 |
| `Q` | 退出程序 |
| `Ctrl+C` | 强制退出 |

### 自我介绍秒返

当面试官说出类似"你先自我介绍一下吧"的话时，程序自动匹配并直接返回 `SELF_INTRO` 中预写的内容，**不调用 AI，0 秒响应**。在 `.env` 中配置：

```ini
SELF_INTRO=面试官你好，我叫XXX，XX大学XX专业，应聘XX岗位...（你的自我介绍）
```

如果 `.env` 中未配置 `SELF_INTRO`，会自动走 AI 根据简历生成。

## 配置详解

### .env 全部配置项

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `DEEPSEEK_API_KEY` | - | DeepSeek API 密钥 |
| `OPENAI_API_KEY` | - | OpenAI API 密钥（优先级低于 DeepSeek） |
| `LLM_PROVIDER` | openai | LLM 提供商: `openai` / `deepseek` / `custom` |
| `LLM_MODEL` | gpt-4o | 模型名: `deepseek-chat` / `gpt-4o` / 自定义 |
| `LLM_TEMPERATURE` | 0.7 | 回答创造性 (0=保守, 1=创意, 面试建议 0.6-0.8) |
| `LLM_MAX_TOKENS` | 1024 | 回答最大长度 |
| `STT_PROVIDER` | xunfei | 语音识别: `xunfei` / `local` / `openai` |
| `IFLYTEK_APP_ID` | - | 讯飞应用 ID |
| `IFLYTEK_API_KEY` | - | 讯飞 API Key |
| `IFLYTEK_API_SECRET` | - | 讯飞 API Secret |
| `STT_LANGUAGE` | zh | 识别语言: `zh` / `en` / 留空自动 |
| `AUDIO_MODE` | mic | 音频来源: `loopback`(系统声音) / `mic`(麦克风) |
| `AUDIO_DEVICE` | -1 | 音频设备编号（仅 mic 模式），-1=系统默认 |
| `VAD_ENERGY_THRESHOLD` | 0 | VAD 触发阈值，0=自动校准 |
| `VAD_ADAPTIVE` | true | 是否自适应阈值 |
| `SILENCE_DURATION` | 1.5 | 静音多少秒认为说话结束 |
| `MIN_SPEECH_DURATION` | 0.5 | 最短有效语音时长(秒) |
| `SELF_INTRO` | - | 自我介绍模板（匹配到直接返回） |
| `SYSTEM_PROMPT` | - | 自定义 AI 系统提示词（留空=自动从简历生成） |
| `ENABLE_TTS` | false | 是否启用文字转语音 |
| `MAX_HISTORY` | 5 | 对话历史保留轮数 |
| `LOCAL_WHISPER_MODEL` | base | 本地模型: `tiny`/`base`/`small`/`medium`/`large` |
| `LOCAL_WHISPER_DEVICE` | cpu | 本地推理设备: `cpu` / `cuda` |
| `OPENAI_API_BASE` | - | 自定义 API 地址（用于代理或兼容 API） |

### 关键词纠错表

`keywords.txt` 包含嵌入式领域常用术语的误识别纠正规则。格式：

```
正确词,误识别1,误识别2,...
```

例如：`FreeRTOS,福瑞RTOS,free RTOS` — 当语音识别输出 "福瑞RTOS" 时自动替换为 "FreeRTOS"。

## 工作原理

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│ 系统音频     │────▶│ 语音活动检测  │────▶│ 讯飞语音听写  │────▶│ AI 生成回答   │
│ (WASAPI)    │     │ (Energy VAD) │     │ (WebSocket) │     │ (DeepSeek)   │
└─────────────┘     └──────────────┘     └─────────────┘     └──────────────┘
     ↑                     ↑                    ↑                   ↑
 面试官说话             检测说话/静音        实时转文字+合并       流式逐字输出
```

1. **音频采集** — PyAudioWPatch 通过 Windows WASAPI Loopback 抓取系统播放的所有声音（无论最终输出到音响还是蓝牙耳机）
2. **VAD 检测** — 基于能量的语音活动检测，自动判断说话开始和结束
3. **语音识别** — 实时 WebSocket 连接讯飞 IAT 服务，边说边出中间结果
4. **长句合并** — 面试官说话停顿不超过 1.5 秒时自动拼接为完整问题
5. **AI 回答** — 将简历 + 面试官问题 + 对话历史发给 LLM，流式输出回答建议

## 常见问题

### Q: 程序监听不到面试官声音？

**A:** 检查下面几点：
1. `.env` 中 `AUDIO_MODE=loopback`
2. 观察 `系统声音` 值：播放视频/音乐时数值应明显跳动（>500），如果始终为 0 则 WASAPI 未正常工作
3. 如 loopback 不可用，可改用 `AUDIO_MODE=mic` + 笔记本扬声器外放（麦克风会录到扬声器中的面试官声音）

### Q: 蓝牙耳机能用吗？

**A:** 可以。loopback 模式录制的是系统音频流本身，与输出设备（音响/有线耳机/蓝牙耳机）无关。

### Q: 识别结果不准确？

**A:** 
- 检查系统音量是否足够大
- 调整 `VAD_ENERGY_THRESHOLD`（增加或降低阈值）
- 编辑 `keywords.txt` 添加你的专业术语

### Q: AI 回答太慢？

**A:**
- 使用 `LLM_MODEL=deepseek-chat`（比 v4-pro 更快）
- 减少 `LLM_MAX_TOKENS`（1024 → 512，面试回答 300 字足够）
- 开启自我介绍秒返（免 AI 调用）

### Q: 如何支持英文面试？

**A:** `.env` 中设置 `STT_LANGUAGE=en`，并将 `SELF_INTRO` 和 `SYSTEM_PROMPT` 改为英文。

## 项目结构

```
├── main.py              # 主程序
├── requirements.txt     # Python 依赖
├── .env.example         # 配置文件模板
├── keywords.txt         # 专业术语纠错表
├── qa_bank.json         # Q&A 预设知识库 (备用)
├── build.bat            # 打包脚本
├── README.md            # 本文件
└── 简历.pdf             # 你的简历 (不上传 Git)
```

## License

MIT
