# Fly0 — 用公开数据集微调 Qwen3-VL-4B（视觉指代定位）

目标：让 `qwen3-vl:4b` 学会「**指令 + 图 → 目标位置**」的能力（视觉 grounding，输出 0–1000 归一化 `bbox_2d` JSON），最终缓解 Fly0 里「像素定位抖动 / 球方块分不清」两个问题。

> **重要预期**：公开 RefCOCO 数据（日常物体）能让模型在它自己的验证集上变准；但**能否迁移到你的 AirSim 红球场景 = 未经验证**。本 runbook 第 5 步先做冒烟测试，改 Fly0 接入是**可选最后一步**，等你在无人机上确认有效再做。

---

## 你需要准备

- AutoDL 账号（autodl.com，学生认证便宜；无需梯子）
- 本目录 4 个文件（README + 3 脚本）
- 费用预估：4090 约 ¥2/时，试跑 1 小时内 → **总共约 ¥10 以内**，关机不扣钱

---

## 本目录文件

| 文件 | 作用 |
|---|---|
| `README.md` | 本手册 |
| `train_qwen3vl.sh` | LoRA 训练命令（在云端跑） |
| `prepare_subset.py` | 兜底脚本：从内置 `swift/refcoco` 抽 2 万训练 + 2 千验证（仅当 ms-swift 不支持切片时用） |
| `infer_test.py` | 训练完冒烟验证：喂一张图，看模型是否输出可解析的 `bbox_2d` |

---

## 阶段 1 — AutoDL 开一台 4090

1. 注册/登录 autodl.com → 「算力市场」
2. 筛选：**GPU = RTX 4090（24G）**；镜像（框架）选 **PyTorch 2.x + CUDA 12.x**
3. **数据盘 ≥ 60GB**（模型 ~9GB + 数据集图片几 GB），地区随便
4. 租用开机。开机后在「容器实例」→ 你的机器 → 「**JupyterLab**」或「**终端**」进命令行
5. 把本 `finetune/` 目录上传：
   - JupyterLab 里直接拖文件；或
   - AutoDL「文件存储」挂载；或 scp（看你机器给的端口）
   - 上传后确认 `ls finetune/` 能看到 4 个文件

---

## 阶段 2 — 云端装环境

```bash
conda create -n swift python=3.11 -y
conda activate swift

pip install "ms-swift[all]>=4.0" \
            "transformers>=4.57" \
            "qwen_vl_utils>=0.0.14"

# 验证
swift sft --help > /dev/null && echo "swift OK"
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

> ms-swift 默认从 **ModelScope**（魔搭）取模型和数据集，国内直连，不需要梯子。

---

## 阶段 3 — 准备数据（2 万条子集）

**先试首选（ms-swift 是否支持内置数据集切片）：**

```bash
swift sft --help 2>&1 | grep -i -E "dataset|sample" | head -20
```

- 若你能看到类似 `dataset ... 支持 #N 取样` 的说明 → 直接用 `swift/refcoco#20000`、`swift/refcoco#2000`（见阶段 4 脚本注释，把两个变量改掉即可）。
- 看不到 / 不确定 → **用兜底脚本**：

```bash
cd finetune
python prepare_subset.py --train-n 20000 --val-n 2000
wc -l train.jsonl val.jsonl    # 应约为 20000 / 2000
head -1 train.jsonl            # 确认有 messages/images/objects
```

> **最省事的兜底（若连脚本都卡住）**：不抽子集，直接训全量 `swift/refcoco` 但加 `--max_steps 5000`（Trainer 原生参数，一定能跑），效果量级约等于一次小规模试跑。代价是验证时机略有不同，仅作冒烟够用。

---

## 阶段 4 — LoRA 训练（约 1 小时内）

先 `cd finetune`，再：

```bash
conda activate swift

# 情况 A：prepare_subset.py 生成了 jsonl → 直接跑（默认就是用这两个文件）
bash train_qwen3vl.sh

# 情况 B：ms-swift 支持切片 → 改脚本里两行再跑
#   TRAIN_DATASET="swift/refcoco#20000"
#   VAL_DATASET="swift/refcoco#2000"
```

跑前确认参数名（不同 ms-swift 版本可能略异）：

```bash
swift sft --help | grep -E "target_modules|freeze_vit|freeze_aligner|merge_lora|train_type|lora_rank"
```

训练期间重点看日志：
- `loss` 曲线逐步下降，无 `OOM` / `CUDA out of memory`
- 结束时有 `val` 指标输出
- 正常产物在 `output/qwen3vl-4b-refcoco-lora/vx-xxx-*/`，其中 `**merged**` 子目录是合并好权重的完整模型（供第 5 步用）

如果 OOM：把 `--gradient_accumulation_steps` 调大、`--max_length` 调小（如 2048），batch=1 保持不变。

---

## 阶段 5 — 冒烟验证（云端，只跑通、判断值不值得继续）

用**合并后**的模型目录测一张图，确认输出是 qwen 原生 grounding 格式、脚本能换算成像素中心：

```bash
conda activate swift
cd finetune

# 用 refcoco 验证集一张图（可选：把本地 images/test_*.png 无人机截图上传到这来喂它，更能贴近你的场景）
python infer_test.py \
  output/qwen3vl-4b-refcoco-lora/vx-xxx-xxx/merged \
  some_image.jpg \
  "red ball" \
  720 480
```

- 正常会打印：模型原始输出 + 解析出的 `bbox_2d` 像素中心。
- 把无人机截图（含红球/方块）传上去试几张：看它对「球 vs 方块」是否给得出位置。**这一步直接决定要不要进阶段 6。**

---

## 阶段 6（可选，等你在无人机上确认后再做）— 接回 Fly0

> 门槛：你在真机场景里确认微调模型确实能锁对红球后，再来找我改下面的代码。**默认不动 Fly0。**

改动点在 `src/airsim/core/detection/visual_target_detector.py`：
- 把运行时 prompt 换成 grounding 模板（图 + "找到&lt;目标&gt;"，要 qwen 原生 `bbox_2d` JSON 输出）
- 解析器新增分支：解析 `bbox_2d` JSON → 中心 = ((x1+x2)/2, (y1+y2)/2) → 像素 = 中心 × (宽/1000, 高/1000)
- 保留旧 `(X,Y)` 正则兜底

模型部署优先序：
1. **云端直连最快**：在 AutoDL 上 ollama/vLLM 起 merged 模型，Fly0 的 `config.json` 指向云端地址（`vision.base_url` 或 `API_TYPE=vllm`）。现有代码已支持 OpenAI 兼容 content-array 路径（`_detect_with_openai`）。
2. **搬回本机 Ollama**：下载 merged Safetensors → `ollama create`。⚠️ 已知坑：qwen3-vl 微调后转**分体 GGUF** 导入 Ollama 曾报 500（ollama issue #13101），Safetensors 导入是社区推荐路线；不行就退回方案 1。

---

## 常见问题

- **下载模型/数据慢**：确认是从 ModelScope 拉（国内快），别设 `USE_HF=1`。
- **OOM**：batch 已是 1，先降 `max_length` 或升 `gradient_accumulation_steps`。
- **val 指标没有 bbox 精度**：部分 ms-swift 版本 val 只看 loss，属正常，可忽略；格式对错看阶段 5 输出。
- **`prepare_subset.py` 报错**：多半是 ms-swift 版本改了内部 API。两种出路：① 改跑全量 + `--max_steps`（见阶段 3）；② 把报错贴回给我。

## 实测结论（2026-09-02，用户自家无人机截图验证）

> 这条路线**最终判定为不划算，Fly0 未接入**，保留本手册仅作训练管线参考。

- 全链路（COCO 图 → 本地 jsonl → `swift sft` LoRA 1200 步 → `swift merge-lora` → transformers 推理）**跑通**，输出为 qwen 原生 `bbox_2d` 0–1000 JSON，格式兼容。
- **但迁移失败**：用用户自己的 AirSim 红球截图对比，**未微调的基座模型本就锁得准，COCO 微调后反而偏**（框贴边、cube/block 认错目标）。公开数据域差太大 + 小步数过拟合，不解决真实场景问题。
- 已记录的环境配方（ms-swift 4.5.2 云端）：见项目记忆 `fly0-vlm-finetune-project`（torch≥2.6、tensorboard≥2.17、本地 jsonl 的 assistant-content 必须为 list、绕开 `swift/refcoco` 加载器直接用本地 parquet）。
- 若将来想真训好红球场景：需要**自家场景数据**（可用 OpenCV 检球做伪标签自动生成），公开 COCO 替代不了。当前 Fly0 维持 基座 qwen3-vl:4b + OpenCV 兜底 即可。

---

## 参考

- ms-swift Qwen3-VL 官方最佳实践（冻结 ViT/aligner 等参数出处）：
  `github.com/modelscope/ms-swift` → `docs/source/BestPractices/Qwen3-VL-Best-Practice.md`
- ms-swift grounding 数据格式：`docs/source/Customization/Custom-dataset.md`
- qwen3-vl 0–1000 坐标说明 + Ollama 部署坑：DeepWiki Qwen3-VL；ollama issue #13101 / #16095
