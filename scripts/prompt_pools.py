"""カテゴリ別の CSV からプロンプト生成用の候補を読み込む。"""

import csv
from itertools import cycle
from pathlib import Path


POOL_DIR = Path(__file__).resolve().parents[1] / "prompts" / "pools"

POOL_KEYS = {
    "bottoms": ("color_en", "style_en"),
    "footwear": ("color_en", "style_en"),
    "hairs": ("color_en", "style_en"),
    "palettes": ("en", "accessory_en"),
}

POOL_DESCRIPTIONS = {
    "bottoms": ("color_ja", "style_ja"),
    "footwear": ("color_ja", "style_ja"),
    "hairs": ("color_ja", "style_ja"),
    "palettes": ("color_ja", "accessory_ja"),
}


def read_pool(name, *columns, flag=None, english_only=False):
    path = POOL_DIR / f"{name}.csv"
    fields = (*columns, flag) if flag else columns
    key_columns = POOL_KEYS.get(name, ("en",))
    key_indices = tuple(fields.index(column) for column in key_columns)
    entries = []
    seen = {}
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.reader(source, strict=True)
        try:
            header = tuple(next(reader, ()))
            described_fields = fields + POOL_DESCRIPTIONS.get(name, ())
            if header not in (fields, described_fields):
                raise ValueError(f"{path}: ヘッダーは {', '.join(fields)} にしてください")
            for row in reader:
                if not row:
                    continue
                if len(row) != len(header):
                    raise ValueError(f"{path}:{reader.line_num}: 列数を確認してください")
                values = tuple(row[:len(columns)])
                if any(not value.strip() for value in values):
                    raise ValueError(f"{path}:{reader.line_num}: {', '.join(columns)} は必須です")
                key = tuple(row[index].strip().casefold() for index in key_indices)
                if key in seen:
                    candidate = " / ".join(row[index] for index in key_indices)
                    raise ValueError(f"{path}:{reader.line_num}: {candidate} は {seen[key]} 行目と重複しています")
                seen[key] = reader.line_num
                if flag:
                    if row[-1] not in ("yes", "no"):
                        raise ValueError(f"{path}:{reader.line_num}: {flag} は yes または no を指定してください")
                    entries.append((*values, row[-1] == "yes"))
                else:
                    entries.append(values[0] if english_only else values)
        except csv.Error as error:
            raise ValueError(f"{path}:{reader.line_num}: {error}") from error
    if not entries:
        raise ValueError(f"{path}: 候補を1行以上入力してください")
    return entries


def add_abstract_backgrounds(rows):
    """抽象背景を CSV の順番で巡回し、各行のプロンプトとメモに加える。"""
    backgrounds = read_pool("abstract_backgrounds", "en", "ja")
    for row, (background, description) in zip(rows, cycle(backgrounds)):
        row["positive"] += f", {background}"
        row["note"] += f"／背景 {description}"


def read_hair_prompts():
    return [f"{color} hair, {style}" for color, style, _ in
            read_pool("hairs", "color_en", "style_en", "ja")]


def read_colored_clothing(name):
    return [(f"{color} {style}", ja) for color, style, ja in
            read_pool(name, "color_en", "style_en", "ja")]


def read_outfit_prompts():
    path = POOL_DIR / "outfits.csv"
    fields = ["outfit_id", "item_no", "color_en", "style_en", "ja"]
    items_by_outfit = {}
    notes = {}
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.reader(source, strict=True)
        try:
            header = next(reader, None)
            fields_with_item_ja = [*fields, "item_ja"]
            if header not in (fields, fields_with_item_ja):
                raise ValueError(f"{path}: ヘッダーは {', '.join(fields_with_item_ja)} にしてください")
            for row in reader:
                if not row:
                    continue
                location = f"{path}:{reader.line_num}"
                if len(row) != len(header):
                    raise ValueError(f"{location}: 列数を確認してください")
                outfit_id, item_no, color, style, ja = (value.strip() for value in row[:len(fields)])
                if not outfit_id or not style:
                    raise ValueError(f"{location}: outfit_id、style_en は必須です")
                try:
                    number = int(item_no)
                except ValueError as error:
                    raise ValueError(f"{location}: item_no は正の整数にしてください") from error
                if number < 1:
                    raise ValueError(f"{location}: item_no は正の整数にしてください")
                items = items_by_outfit.setdefault(outfit_id, {})
                if number in items:
                    raise ValueError(f"{location}: {outfit_id} の item_no {number} が重複しています")
                if "{color_en}" in style:
                    if not color:
                        raise ValueError(f"{location}: {{color_en}} を使う場合は色を入力してください")
                    items[number] = style.replace("{color_en}", color)
                else:
                    items[number] = f"{color} {style}" if color else style
                if ja:
                    if outfit_id in notes:
                        raise ValueError(f"{location}: {outfit_id} の ja は1行に入力してください")
                    notes[outfit_id] = ja
        except csv.Error as error:
            raise ValueError(f"{path}:{reader.line_num}: {error}") from error
    if not items_by_outfit:
        raise ValueError(f"{path}: 候補を1行以上入力してください")
    outfits = []
    for outfit_id, items in items_by_outfit.items():
        if outfit_id not in notes:
            raise ValueError(f"{path}: {outfit_id} の ja を入力してください")
        prompt = ", ".join(items[number] for number in sorted(items))
        outfits.append((prompt, notes[outfit_id]))
    return outfits
