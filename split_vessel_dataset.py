# -*- coding: utf-8 -*-
"""CF-CMoE 数据集划分工具。修改下方参数后，在 PyCharm 中右击运行即可。
仅使用 Python 标准库，不依赖 PyTorch。不会移动或修改原始图像。
"""
import csv
import math
import random
import re
from collections import defaultdict
from pathlib import Path

# ======================== 用户配置 ========================
DATASET_ROOT = r"D:\0_data\eye_datas\ROSSA"  # 数据集根目录，包含 images/ 和 labels/
OUTPUT_DIR = None  # None: 自动保存到 DATASET_ROOT/manifests；也可指定绝对路径
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 1
IMAGE_SUFFIX = ""  # 文件名后缀，不含扩展名，如 _image；同名时留空
LABEL_SUFFIX = ""  # 如标签是 001_mask.png、图像是 001.png，设为 "_mask"
# 同一患者/原图可能对应多个样本时，使用正则提取其共同 ID。
# 示例：patient001_left、patient001_right -> r"^(patient\d+)"
# None 表示一个图像/标签对作为独立样本划分。
GROUP_REGEX = None
OVERWRITE = False  # True: 允许覆盖已有 train.csv、val.csv、test.csv
# ==========================================================

EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".ppm", ".pgm"}
SPLITS = ("train", "val", "test")


def collect_files(folder, suffix):
    """根据文件名（去扩展名、指定后缀）匹配，而不是按文件排序强行配对。"""
    if not folder.is_dir():
        raise FileNotFoundError(f"文件夹不存在: {folder}")
    result = {}
    skipped = []
    for p in sorted(folder.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in EXTENSIONS:
            continue
        if any(part.startswith('.') for part in p.relative_to(folder).parts):
            continue
        name = p.stem
        if suffix:
            if not name.casefold().endswith(suffix.casefold()):
                skipped.append(p.name)
                continue
            name = name[:-len(suffix)]
        key = name.casefold()
        if not key:
            raise ValueError(f"空样本 ID: {p}")
        if key in result:
            raise ValueError(f"发现重复样本 ID {name!r}:\n{result[key][1]}\n{p}")
        result[key] = (name, p.resolve())
    if skipped:
        raise ValueError(f"{folder} 中有 {len(skipped)} 个图像文件不符合指定后缀 {suffix!r}，例如 {skipped[:5]}。请检查命名，避免遗漏样本。")
    if not result:
        raise ValueError(f"没有找到可读取的图像文件: {folder}")
    return result


def match_pairs(root, image_suffix, label_suffix):
    images = collect_files(root / "images", image_suffix)
    labels = collect_files(root / "labels", label_suffix)
    missing = sorted(images.keys() - labels.keys())
    extra = sorted(labels.keys() - images.keys())
    if missing or extra:
        raise ValueError(
            f"图像与标签不能一一配对。\n缺少标签: {len(missing)} 个，示例: {missing[:10]}"
            f"\n缺少图像: {len(extra)} 个，示例: {extra[:10]}"
            "\n请检查文件名或 IMAGE_SUFFIX / LABEL_SUFFIX。未生成任何 CSV。"
        )
    return [dict(id=images[k][0], image=str(images[k][1]),
                 mask=str(labels[k][1]), fov="") for k in sorted(images)]


def split_counts(n, ratios):
    """最大余数法分配数量；保证三个集合非空（至少三个独立单元）。"""
    if n < 3:
        raise ValueError("至少需要 3 个独立样本或分组，才能划分 train/val/test。")
    raw = [n * r / sum(ratios) for r in ratios]
    counts = [math.floor(x) for x in raw]
    for i in sorted(range(3), key=lambda j: (-(raw[j]-counts[j]), j))[:n-sum(counts)]:
        counts[i] += 1
    for i in range(3):
        if counts[i] == 0:
            donor = max((j for j in range(3) if counts[j] > 1), key=lambda j: counts[j]-raw[j])
            counts[donor] -= 1
            counts[i] += 1
    return counts


def split_rows(rows, ratios, seed, group_regex=None):
    if len(ratios) != 3 or not all(math.isfinite(r) and 0 < r < 1 for r in ratios) or not math.isclose(sum(ratios), 1.0, abs_tol=1e-8):
        raise ValueError("三个比例必须为正数，且总和必须为 1。")
    groups = defaultdict(list)
    pattern = re.compile(group_regex) if group_regex else None
    for row in rows:
        if pattern:
            match = pattern.search(row['id'])
            if not match:
                raise ValueError(f"样本 {row['id']!r} 不符合 GROUP_REGEX，请检查分组规则。")
            group = match.group(1) if pattern.groups else match.group(0)
            if not group:
                raise ValueError(f"样本 {row['id']!r} 的分组 ID 为空。")
        else:
            group = row['id']
        groups[group.casefold()].append(row)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    a, b, c = split_counts(len(keys), ratios)
    selected = (keys[:a], keys[a:a+b], keys[a+b:])
    splits = {name: sorted((row for k in ks for row in groups[k]), key=lambda r: r['id'].casefold())
              for name, ks in zip(SPLITS, selected)}
    # 检查集合间无重叠、无遗漏；分组模式下同组永不跨集合。
    sets = [set(r['id'].casefold() for r in splits[name]) for name in SPLITS]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i+1, 3)) or len(set.union(*sets)) != len(rows):
        raise RuntimeError("划分完整性检查失败。")
    return splits, dict(zip(SPLITS, (a, b, c)))


def main():
    root = Path(DATASET_ROOT).expanduser().resolve()
    output = Path(OUTPUT_DIR).expanduser().resolve() if OUTPUT_DIR else root / "manifests"
    rows = match_pairs(root, IMAGE_SUFFIX, LABEL_SUFFIX)
    splits, units = split_rows(rows, (TRAIN_RATIO, VAL_RATIO, TEST_RATIO), SEED, GROUP_REGEX)
    paths = {name: output / f"{name}.csv" for name in SPLITS}
    existing = [str(p) for p in paths.values() if p.exists()]
    if existing and not OVERWRITE:
        raise FileExistsError("CSV 已存在，为防止覆盖固定实验划分，程序已停止：\n" + "\n".join(existing) + "\n若确需重新生成，请设置 OVERWRITE = True。")
    output.mkdir(parents=True, exist_ok=True)
    for name, path in paths.items():
        with path.open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=['id', 'image', 'mask', 'fov'])
            writer.writeheader()
            writer.writerows(splits[name])
    print(f"\n数据集: {root.name}\n匹配成功: {len(rows)} 对\n随机种子: {SEED}")
    print("划分单位: " + ("分组（比例按独立组数计算）" if GROUP_REGEX else "独立样本"))
    for name in SPLITS:
        count = len(splits[name])
        print(f"{name:5s}: {count:5d} 张 ({count/len(rows):.2%})" + (f", {units[name]} 组" if GROUP_REGEX else "") + f" -> {paths[name]}")
    print("\n划分完成：无重复、无遗漏，原始数据未修改。")
    print("将以下内容配置到 CF-CMoE 的 config.yaml（或 config_gray.yaml）：")
    print("data:")
    for name in SPLITS:
        print(f"  {name}_csv: '{paths[name].as_posix().replace(chr(39), chr(39)*2)}'")


if __name__ == '__main__':
    main()
