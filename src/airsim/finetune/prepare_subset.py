"""兜底脚本：从 ms-swift 内置公开数据集 'swift/refcoco' 抽取子集，生成 train.jsonl / val.jsonl。

仅在 ``swift sft --dataset 'swift/refcoco#N'`` 这种切片语法不被你当前 ms-swift 版本支持时使用。

用法（在装了 ms-swift 的云端 conda 环境 swift 里，于 finetune/ 目录下）::

    python prepare_subset.py --train-n 20000 --val-n 2000

原理：用 modelscope SDK 把数据集仓库 'swift/refcoco' 快照到本地缓存，
扫描其中的 *.jsonl（ms-swift 内置 grounding 数据的原始形态），
按顺序切出 train 与 val 两部分并原样写出，图片相对路径保持不变。

若结构与你装的 ms-swift 版本不一致（报错清晰提示），
README「阶段 3」给了一条更省事的路：全量 + --max_steps 5000。
"""
import argparse
import glob
import json
import os
import sys


def _abs(base_dir: str, p: str) -> str:
    """把图片引用转成绝对路径；已是绝对路径或 http(s) 链接则原样返回。"""
    p = p.strip()
    if p.startswith(("http://", "https://", "/", "\\")):
        return p
    joined = os.path.normpath(os.path.join(base_dir, p))
    return joined if os.path.exists(joined) else p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-n", type=int, default=20000)
    ap.add_argument("--val-n", type=int, default=2000)
    ap.add_argument("--dataset", default="swift/refcoco",
                    help="ModelScope 数据集仓库 id")
    args = ap.parse_args()

    try:
        from modelscope import snapshot_download
    except ImportError as e:
        print(f"[ERR] 需要 modelscope：pip install modelscope   ({e})")
        return 1

    print(f"==> 快照下载数据集 {args.dataset}（首次会下载图片，稍等）...")
    try:
        repo_dir = snapshot_download(args.dataset)
    except Exception as e:  # noqa: BLE001
        print(f"[ERR] 下载数据集失败: {e!r}")
        print("出路：改用全量训练 + --max_steps（见 README 阶段3），或把报错贴回给助手。")
        return 1
    print(f"    仓库目录: {repo_dir}")

    # 找出数据仓库根目录下的所有 jsonl（按名字排序，保证切分可复现）
    jsonls = sorted(glob.glob(os.path.join(repo_dir, "**", "*.jsonl"), recursive=True))
    if not jsonls:
        print(f"[ERR] 仓库 {repo_dir} 下没找到 *.jsonl，结构不符合预期。")
        print("出路：改用全量训练 + --max_steps（见 README 阶段3）。")
        return 1

    lines = []
    for jf in jsonls:
        base = os.path.dirname(jf)
        print(f"    读取 {jf}")
        with open(jf, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    rec = json.loads(ln)
                except json.JSONDecodeError as e:
                    print(f"    [WARN] 跳过一行坏 JSON: {e}")
                    continue
                # 图片相对路径 → 绝对路径（图片文件与 jsonl 同目录或子目录）
                imgs = rec.get("images")
                if isinstance(imgs, list):
                    fixed = []
                    for p in imgs:
                        fixed.append(_abs(base, p))
                    rec["images"] = fixed
                elif isinstance(imgs, str):
                    rec["images"] = _abs(base, imgs)
                lines.append(json.dumps(rec, ensure_ascii=False))

    total = len(lines)
    print(f"==> 共 {total} 条记录")
    if total == 0:
        print("[ERR] jsonl 内容为空。")
        return 1

    train_n = min(args.train_n, total)
    val_n = min(args.val_n, max(0, total - train_n))
    print(f"==> 切分：train={train_n}  val={val_n}")

    # 抽样校验一条结构
    sample = json.loads(lines[0])
    keys = sorted(sample.keys())
    print(f"==> 字段: {keys}")
    if not all(k in sample for k in ("messages", "images", "objects")):
        print("[WARN] 记录里没有 messages/images/objects 三件套，"
              "ms-swift grounding 可能无法识别；若训练报数据格式错，改用 README 阶段3 的 max_steps 路线。")

    with open("train.jsonl", "w", encoding="utf-8") as f:
        f.write("\n".join(lines[:train_n]) + ("\n" if train_n else ""))
    with open("val.jsonl", "w", encoding="utf-8") as f:
        f.write("\n".join(lines[train_n:train_n + val_n]) + ("\n" if val_n else ""))

    print("==> 完成。训练命令：bash train_qwen3vl.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
