"""Read-only Dataset Inspection script for FER-2013 (data/raw/train & data/raw/test).

This script inspects directory structure, category consistency, image counts,
dimensions, color modes, actual formats, corrupted/truncated files, constant-pixel
(blank) images, and exact duplicate/cross-split overlap without modifying or
moving any files in the original dataset.
"""

from __future__ import annotations

import hashlib
import json
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageFile

# Treat truncated images as errors rather than silently decoding them
ImageFile.LOAD_TRUNCATED_IMAGES = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
SPLITS = ("train", "test")

EXPECTED_FORMATS = {
    ".jpg": {"JPEG"},
    ".jpeg": {"JPEG"},
    ".png": {"PNG"},
    ".bmp": {"BMP"},
    ".gif": {"GIF"},
    ".tif": {"TIFF"},
    ".tiff": {"TIFF"},
    ".webp": {"WEBP"},
}


def analyze_duplicates(hash_map: dict[str, list[tuple[str, str, str]]]) -> dict[str, int]:
    dup_groups = {h: items for h, items in hash_map.items() if len(items) > 1}
    within_train_same_cat = 0
    within_train_diff_cat = 0
    within_test_same_cat = 0
    within_test_diff_cat = 0
    cross_split_groups = 0
    cross_split_same_cat = 0
    cross_split_diff_cat = 0
    extra_copies_in_train = 0
    extra_copies_in_test = 0
    test_files_also_in_train = 0

    for items in dup_groups.values():
        train_items = [x for x in items if x[0] == "train"]
        test_items = [x for x in items if x[0] == "test"]

        if len(train_items) > 1:
            extra_copies_in_train += len(train_items) - 1
            if len({x[1] for x in train_items}) == 1:
                within_train_same_cat += 1
            else:
                within_train_diff_cat += 1

        if len(test_items) > 1:
            extra_copies_in_test += len(test_items) - 1
            if len({x[1] for x in test_items}) == 1:
                within_test_same_cat += 1
            else:
                within_test_diff_cat += 1

        if train_items and test_items:
            cross_split_groups += 1
            test_files_also_in_train += len(test_items)
            if len({x[1] for x in items}) == 1:
                cross_split_same_cat += 1
            else:
                cross_split_diff_cat += 1

    return {
        "total_duplicate_groups": len(dup_groups),
        "extra_copies_in_train": extra_copies_in_train,
        "extra_copies_in_test": extra_copies_in_test,
        "within_train_same_cat_groups": within_train_same_cat,
        "within_train_diff_cat_groups": within_train_diff_cat,
        "within_test_same_cat_groups": within_test_same_cat,
        "within_test_diff_cat_groups": within_test_diff_cat,
        "cross_split_groups": cross_split_groups,
        "cross_split_same_cat_groups": cross_split_same_cat,
        "cross_split_diff_cat_groups": cross_split_diff_cat,
        "test_files_also_in_train": test_files_also_in_train,
    }


def inspect_dataset(data_raw: Path = DATA_RAW) -> dict:
    report: dict = {
        "root": str(data_raw),
        "top_level_entries": sorted(x.name for x in data_raw.iterdir()) if data_raw.exists() else [],
        "splits": {},
        "consistency": {},
        "cross_split_analysis": {},
    }

    all_file_hashes: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    all_pixel_hashes: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    all_filenames: dict[str, list[str]] = defaultdict(list)

    for split in SPLITS:
        split_dir = data_raw / split
        if not split_dir.exists():
            report["splits"][split] = {"exists": False}
            continue

        direct_files = sorted(x.name for x in split_dir.iterdir() if x.is_file())
        category_dirs = sorted(x.name for x in split_dir.iterdir() if x.is_dir())

        file_sizes: list[int] = []
        extensions_raw: Counter = Counter()
        extensions_lower: Counter = Counter()
        actual_formats: Counter = Counter()
        dimensions: Counter = Counter()
        modes: Counter = Counter()
        filename_prefixes: Counter = Counter()

        split_info: dict = {
            "exists": True,
            "direct_files": direct_files,
            "categories": category_dirs,
            "nested_subdirs": [],
            "total_files": 0,
            "total_valid_images": 0,
            "corrupted_files": [],
            "zero_byte_files": [],
            "constant_pixel_images": [],
            "category_counts": {},
            "category_valid_counts": {},
            "extension_format_mismatches": [],
            "pixel_min_global": None,
            "pixel_max_global": None,
        }

        for cat in category_dirs:
            cat_dir = split_dir / cat
            cat_file_count = 0
            cat_valid_count = 0

            for root_path, dirs, files in os.walk(cat_dir):
                for d in dirs:
                    split_info["nested_subdirs"].append(
                        str(Path(root_path, d).relative_to(data_raw))
                    )

                for fname in files:
                    fpath = Path(root_path) / fname
                    rel_path = str(fpath.relative_to(data_raw))
                    cat_file_count += 1
                    split_info["total_files"] += 1

                    prefix = fname.split("_")[0] if "_" in fname else "NO_PREFIX"
                    filename_prefixes[prefix] += 1
                    all_filenames[fname].append(rel_path)

                    ext_raw = fpath.suffix
                    ext_lower = ext_raw.lower()
                    extensions_raw[ext_raw] += 1
                    extensions_lower[ext_lower] += 1

                    fsize = fpath.stat().st_size
                    file_sizes.append(fsize)
                    if fsize == 0:
                        split_info["zero_byte_files"].append(rel_path)

                    file_hash = hashlib.sha256(fpath.read_bytes()).hexdigest()
                    all_file_hashes[file_hash].append((split, cat, rel_path))

                    try:
                        with Image.open(fpath) as img_v:
                            img_v.verify()

                        with Image.open(fpath) as img:
                            img.load()
                            fmt = img.format
                            size = img.size
                            mode = img.mode
                            extrema = img.getextrema()
                            px_hash = hashlib.md5(img.tobytes()).hexdigest()
                            all_pixel_hashes[px_hash].append((split, cat, rel_path))

                        cat_valid_count += 1
                        split_info["total_valid_images"] += 1
                        actual_formats[fmt] += 1
                        dimensions[f"{size[0]}x{size[1]}"] += 1
                        modes[mode] += 1

                        if ext_lower in EXPECTED_FORMATS and fmt not in EXPECTED_FORMATS[ext_lower]:
                            split_info["extension_format_mismatches"].append(
                                {"path": rel_path, "ext": ext_raw, "actual_format": fmt}
                            )

                        if isinstance(extrema[0], tuple):
                            pmin = min(ch[0] for ch in extrema)
                            pmax = max(ch[1] for ch in extrema)
                        else:
                            pmin, pmax = extrema[0], extrema[1]

                        if split_info["pixel_min_global"] is None or pmin < split_info["pixel_min_global"]:
                            split_info["pixel_min_global"] = pmin
                        if split_info["pixel_max_global"] is None or pmax > split_info["pixel_max_global"]:
                            split_info["pixel_max_global"] = pmax

                        if pmin == pmax:
                            split_info["constant_pixel_images"].append(
                                {"path": rel_path, "value": pmin, "category": cat}
                            )
                    except Exception as exc:
                        split_info["corrupted_files"].append(
                            {"path": rel_path, "error": f"{type(exc).__name__}: {exc}"}
                        )

            split_info["category_counts"][cat] = cat_file_count
            split_info["category_valid_counts"][cat] = cat_valid_count

        split_info["extensions_raw"] = dict(extensions_raw)
        split_info["extensions_lower"] = dict(extensions_lower)
        split_info["actual_formats"] = dict(actual_formats)
        split_info["dimensions"] = dict(dimensions)
        split_info["modes"] = dict(modes)
        split_info["filename_prefixes"] = dict(filename_prefixes)
        split_info["file_size_stats_bytes"] = (
            {
                "min": min(file_sizes),
                "max": max(file_sizes),
                "mean": round(statistics.mean(file_sizes), 2),
                "median": statistics.median(file_sizes),
                "total_mb": round(sum(file_sizes) / (1024 * 1024), 2),
            }
            if file_sizes
            else {}
        )
        report["splits"][split] = split_info

    train_cats = set(report["splits"].get("train", {}).get("categories", []))
    test_cats = set(report["splits"].get("test", {}).get("categories", []))
    report["consistency"] = {
        "train_categories": sorted(train_cats),
        "test_categories": sorted(test_cats),
        "exact_match": train_cats == test_cats,
        "only_in_train": sorted(train_cats - test_cats),
        "only_in_test": sorted(test_cats - train_cats),
    }

    report["cross_split_analysis"] = {
        "duplicate_filenames": {k: v for k, v in all_filenames.items() if len(v) > 1},
        "exact_file_bytes_duplicates": analyze_duplicates(all_file_hashes),
        "exact_pixel_content_duplicates": analyze_duplicates(all_pixel_hashes),
    }
    return report


def main() -> None:
    report = inspect_dataset()
    out_json = PROJECT_ROOT / "outputs" / "reports" / "dataset_inspection_summary.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved inspection summary to {out_json}")


if __name__ == "__main__":
    main()
