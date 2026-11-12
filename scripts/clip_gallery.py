"""Browse generated images ranked by their CLIP similarity to favorites."""

from __future__ import annotations

import errno
import json
import os
import shutil
import sqlite3
import sys
import tkinter as tk
from contextlib import ExitStack, closing
from itertools import count
from math import ceil
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread
from tkinter import filedialog, messagebox, ttk
from tkinter.font import Font

import numpy as np
from PIL import Image, ImageOps, ImageTk

ROOT = Path(__file__).resolve().parents[1]
COMFY = ROOT / "ComfyUI"
MODEL = COMFY / "models/clip_vision/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"
RAW = ROOT / "output/raw"
DEFAULT_FAVORITES = ROOT / "output/selected"
CACHE = ROOT / "logs/clip_gallery.sqlite3"
EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
IMAGE_TYPES = [("画像", " ".join(sorted("*" + ext for ext in EXTENSIONS)))]
IMAGE_ERRORS = (OSError, ValueError, Image.DecompressionBombError)
THUMBNAIL_COLUMN_WIDTH = 210
THUMBNAIL_SIZE = (190, 125)
MIN_THUMBNAIL_PERCENT = 60
MAX_THUMBNAIL_PERCENT = 160
NEGATIVE_SIMILARITY_THRESHOLD = 0.5
NEGATIVE_WEIGHT = 0.7
REGISTRATION_TABS = {
    "added_references": "追加した見本",
    "negative_references": "好みではない",
}


def image_files(folder: Path):
    return sorted(
        path.resolve()
        for path in folder.rglob("*")
        if path.suffix.lower() in EXTENSIONS and path.is_file()
    )


def move_image(source: Path, folder: Path):
    if source.parent == folder:
        raise ValueError("画像はすでに移動先フォルダーにあります。")
    for number in count(1):
        name = source.name if number == 1 else f"{source.stem} ({number}){source.suffix}"
        destination = folder / name
        try:
            try:
                # On Windows, rename fails if the destination already exists.
                source.rename(destination)
            except OSError as error:
                if error.errno != errno.EXDEV and getattr(error, "winerror", None) != 17:
                    raise
                # A different volume requires a copy before removing the source.
                with source.open("rb") as original:
                    stat = os.fstat(original.fileno())
                    output = destination.open("xb")
                    try:
                        with output:
                            shutil.copyfileobj(original, output)
                        os.utime(destination, ns=(stat.st_atime_ns, stat.st_mtime_ns))
                    except OSError:
                        destination.unlink()
                        raise
                try:
                    source.unlink()
                except OSError:
                    destination.unlink()
                    raise
            return destination
        except FileExistsError:
            continue


def reference_key(selected):
    paths = [selected] if isinstance(selected, Path) else selected
    kind = "folder" if isinstance(selected, Path) else "files"
    return json.dumps([kind, sorted({os.path.normcase(str(path.resolve())) for path in paths})])


def open_cache():
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(CACHE, timeout=15)


def prepare_cache(db):
    db.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            path TEXT PRIMARY KEY, size INTEGER, mtime INTEGER,
            model TEXT, vector BLOB
        )
    """)
    db.execute("""
        CREATE TABLE IF NOT EXISTS added_references (
            reference_set TEXT NOT NULL, path TEXT NOT NULL,
            PRIMARY KEY (reference_set, path)
        )
    """)
    db.execute("""
        CREATE TABLE IF NOT EXISTS negative_references (
            reference_set TEXT NOT NULL, path TEXT NOT NULL,
            PRIMARY KEY (reference_set, path)
        )
    """)
    db.execute("""
        CREATE TABLE IF NOT EXISTS excluded_references (
            reference_set TEXT NOT NULL, path TEXT NOT NULL,
            PRIMARY KEY (reference_set, path)
        )
    """)


def save_added_references(selected, paths, *, moved=None):
    key = reference_key(selected)
    with closing(open_cache()) as db, db:
        prepare_cache(db)
        rows = [(key, str(path)) for path in paths]
        db.executemany("INSERT OR IGNORE INTO added_references VALUES (?, ?)", rows)
        db.executemany("DELETE FROM excluded_references WHERE reference_set=? AND path=?", rows)
        db.executemany("DELETE FROM negative_references WHERE reference_set=? AND path=?", rows)
        for source, (destination, original) in (moved or {}).items():
            db.execute(
                "DELETE FROM negative_references WHERE reference_set=? AND path=?",
                (key, str(source)),
            )
            try:
                stat = destination.stat()
            except OSError:
                continue
            if (stat.st_size, stat.st_mtime_ns) == (original.st_size, original.st_mtime_ns):
                db.execute("""
                    INSERT OR REPLACE INTO embeddings
                    SELECT ?, size, mtime, model, vector FROM embeddings
                    WHERE path=? AND size=? AND mtime=?
                """, (str(destination), str(source), original.st_size, original.st_mtime_ns))


def update_negative_references(selected, paths, *, remove):
    key = reference_key(selected)
    with closing(open_cache()) as db, db:
        prepare_cache(db)
        sql = ("DELETE FROM negative_references WHERE reference_set=? AND path=?" if remove
               else "INSERT OR IGNORE INTO negative_references VALUES (?, ?)")
        db.executemany(sql, ((key, str(path)) for path in paths))


def registration_paths(db, table, key):
    if table not in REGISTRATION_TABS and table != "excluded_references":
        raise ValueError(f"Unknown registration table: {table}")
    rows = db.execute(f"SELECT path FROM {table} WHERE reference_set=? ORDER BY path", (key,))
    return [Path(value) for (value,) in rows]


def remove_registrations(key, table, paths):
    if table not in REGISTRATION_TABS:
        raise ValueError(f"Unknown registration table: {table}")
    with closing(open_cache()) as db, db:
        prepare_cache(db)
        rows = [(key, str(path)) for path in paths]
        db.executemany(f"DELETE FROM {table} WHERE reference_set=? AND path=?", rows)
        if table == "added_references":
            # The file may also be discovered by scanning the base favorites folder.
            db.executemany("INSERT OR IGNORE INTO excluded_references VALUES (?, ?)", rows)


def reset_registrations(key):
    with closing(open_cache()) as db, db:
        prepare_cache(db)
        removed = 0
        for table in (*REGISTRATION_TABS, "excluded_references"):
            removed += db.execute(f"DELETE FROM {table} WHERE reference_set=?", (key,)).rowcount
        return removed


class ClipEncoder:
    def __init__(self):
        sys.path.insert(0, str(COMFY))
        import torch
        from comfy.clip_vision import load
        from comfy.model_management import unload_all_models

        self.torch = torch
        self.close = unload_all_models
        self.model = load(str(MODEL))
        if self.model is None:
            raise RuntimeError(f"CLIPモデルを読み込めませんでした。\n{MODEL}")

    def encode(self, images):
        torch = self.torch
        pixels = torch.from_numpy(np.stack(images)).float().div_(255)
        with torch.inference_mode():
            features = self.model.encode_image(pixels).image_embeds.float()
            return torch.nn.functional.normalize(features, dim=1).cpu().numpy()


def rank_images(folder, selected, events, cancel, *, allow_empty=False):
    if not folder.is_dir():
        raise ValueError(f"検索先が見つかりません。\n{folder}")
    if isinstance(selected, Path) and not selected.is_dir():
        raise ValueError(f"見本フォルダーが見つかりません。\n{selected}")
    favorites = image_files(selected) if isinstance(selected, Path) else selected
    if cancel.is_set():
        return

    model_stat = MODEL.stat()
    model_key = f"{MODEL.name}:{model_stat.st_size}:{model_stat.st_mtime_ns}:center-crop-v1"
    with closing(open_cache()) as db, ExitStack() as resources:
        prepare_cache(db)
        db.commit()
        key = reference_key(selected)
        excluded = set(registration_paths(db, "excluded_references", key))
        base_paths = [path.resolve() for path in favorites]
        added_paths = [path for path in registration_paths(db, "added_references", key) if path.is_file()]
        favorites = list(dict.fromkeys([*(path for path in base_paths if path not in excluded), *added_paths]))
        if not favorites:
            raise ValueError("見本にする画像を1枚以上選んでください。")
        favorite_set = set(favorites)
        candidates = list(dict.fromkeys(path for path in image_files(folder) if path not in favorite_set))
        if not candidates and not allow_empty:
            raise ValueError("検索先に候補画像がありません。")
        negatives = [
            path for path in registration_paths(db, "negative_references", key)
            if path.is_file() and path not in favorite_set
        ]
        images = list(dict.fromkeys([*favorites, *candidates, *negatives]))
        vectors = {}
        pending = []
        skipped = 0
        reused = 0
        total = len(images)
        encoder = None

        def encode_pending():
            nonlocal encoder
            if encoder is None:
                events.put(("status", "CLIPモデルを読み込み中…"))
                encoder = resources.enter_context(closing(ClipEncoder()))
            features = encoder.encode([pixels for _, _, pixels in pending])
            rows = []
            for (path, stat, _), vector in zip(pending, features, strict=True):
                vectors[path] = vector
                rows.append((str(path), stat.st_size, stat.st_mtime_ns, model_key, vector.tobytes()))
            with db:
                db.executemany("INSERT OR REPLACE INTO embeddings VALUES (?, ?, ?, ?, ?)", rows)
            pending.clear()

        for index, path in enumerate(images, 1):
            if cancel.is_set():
                return
            try:
                stat = path.stat()
                row = db.execute(
                    "SELECT vector FROM embeddings WHERE path=? AND size=? AND mtime=? AND model=?",
                    (str(path), stat.st_size, stat.st_mtime_ns, model_key),
                ).fetchone()
                if row:
                    vectors[path] = np.frombuffer(row[0], dtype=np.float32)
                    reused += 1
                else:
                    with Image.open(path) as image:
                        rgb = ImageOps.exif_transpose(image).convert("RGB")
                        pixels = ImageOps.fit(rgb, (224, 224), Image.Resampling.BICUBIC)
                    pending.append((path, stat, np.asarray(pixels)))
            except IMAGE_ERRORS:
                skipped += 1
            if len(pending) >= 8 or index == total:
                if pending:
                    encode_pending()
            if index % 8 == 0 or index == total:
                events.put(("progress", (index - len(pending), total)))

    if cancel.is_set():
        return
    reference_vectors = [vectors[path] for path in favorites if path in vectors]
    ranked_paths = [path for path in candidates if path in vectors]
    if not reference_vectors:
        raise ValueError("読み込める見本画像がありません。")
    if not ranked_paths and not allow_empty:
        raise ValueError("読み込める候補画像がありません。")

    references = np.stack(reference_vectors)
    negative_paths = {path for path in negatives if path in vectors}
    negative_vectors = np.stack([vectors[path] for path in negative_paths]) if negative_paths else None
    results = []
    for start in range(0, len(ranked_paths), 256):
        if cancel.is_set():
            return
        paths = ranked_paths[start:start + 256]
        features = np.stack([vectors[path] for path in paths])
        scores = (features @ references.T).max(axis=1).clip(-1, 1)
        if negative_vectors is not None:
            similarity = (features @ negative_vectors.T).max(axis=1).clip(-1, 1)
            scores -= NEGATIVE_WEIGHT * np.maximum(0, similarity - NEGATIVE_SIMILARITY_THRESHOLD)
        results.extend(zip(paths, scores, strict=True))
    results.sort(key=lambda item: -item[1])
    events.put(("negative_references", negative_paths))
    events.put(("results", (results, len(reference_vectors), skipped, reused)))


def thumbnail_card(grid, column, path, caption, selected, toggle, *, size):
    card = ttk.Frame(grid, style="Card.TFrame", padding=8)
    card.grid(row=0, column=column, sticky="nsew", padx=4, pady=4)
    thumb = ttk.Label(card, style="Card.TLabel", cursor="hand2", anchor="center")
    thumb.pack(fill="both", expand=True)
    window = grid.winfo_toplevel()

    photo = None
    try:
        with Image.open(path) as image:
            preview = ImageOps.exif_transpose(image).convert("RGB")
            preview.thumbnail(size, Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(preview, master=window)
        thumb.configure(image=photo)
    except IMAGE_ERRORS:
        thumb.configure(text="画像を読み込めません", cursor="")

    name = path.name
    if len(name) > 25:
        name = f"{name[:12]}…{name[-12:]}"
    if caption:
        ttk.Label(card, style="Card.TLabel", text=caption).pack(anchor="w")
    checked = tk.BooleanVar(master=window, value=selected)
    checkbox = ttk.Checkbutton(
        card, style="Card.TCheckbutton", text=name, variable=checked,
        command=lambda: toggle(path, checked.get()),
    )
    checkbox.pack(anchor="w")

    clicked = False

    def select_thumbnail(_event):
        nonlocal clicked
        clicked = not checkbox.instate(["disabled"])
        if clicked:
            checkbox.invoke()

    def open_thumbnail(_event):
        nonlocal clicked
        # Tk reports the first press of a double-click as a single click.
        if clicked:
            checkbox.invoke()
            clicked = False
        if photo is not None:
            try:
                os.startfile(path)
            except OSError as error:
                messagebox.showerror("画像を開けません", str(error), parent=window)

    thumb.bind("<Button-1>", select_thumbnail)
    thumb.bind("<Double-Button-1>", open_thumbnail)
    return photo, checked, checkbox


class ThumbnailList(ttk.Frame):
    def __init__(self, parent, is_selected, toggle, *, empty_text=""):
        super().__init__(parent)
        self.is_selected = is_selected
        self.toggle = toggle
        self.items = []
        self.enabled = True
        self.columns = 1
        self.rows = {}
        self.redraw = None
        self.zoom = None
        self.percent = 100
        self.image_size = THUMBNAIL_SIZE
        self.font = Font(root=self, font=ttk.Style(self).lookup("Card.TLabel", "font"))
        self.card_chrome = 2 * self.font.metrics("linespace") + 32
        self.row_height = self.image_size[1] + self.card_chrome
        self.canvas = tk.Canvas(self, background="#f3f5f8", highlightthickness=0, yscrollincrement=1)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.configure(yscrollcommand=self.on_scroll)
        self.canvas.bind("<Configure>", self.resize)
        self.canvas.bind("<MouseWheel>", self.wheel)
        self.scrollbar.bind("<MouseWheel>", self.wheel)
        self.empty = self.canvas.create_text(
            12, 14, anchor="nw", text=empty_text, fill="#162638", font=self.font, state="hidden"
        )

    def set_items(self, items, *, reset=False):
        top = 0 if reset else self.canvas.canvasy(0)
        self.items = items
        self.clear_rows()
        self.update_scrollregion(top)

    def set_enabled(self, enabled):
        self.enabled = enabled
        for _, _, cards in self.rows.values():
            for _, _, checkbox in cards:
                checkbox.state(["!disabled" if enabled else "disabled"])

    def resize(self, event):
        self.layout(event.width, self.percent)

    def set_size(self, percent):
        if percent != self.percent:
            self.layout(self.canvas.winfo_width(), percent)

    def layout(self, width, percent):
        column_width = max(175, round(THUMBNAIL_COLUMN_WIDTH * percent / 100))
        columns = max(1, width // column_width)
        top = self.canvas.canvasy(0)
        if columns != self.columns or percent != self.percent:
            first = int(max(0, top) // self.row_height) * self.columns
            self.percent = percent
            self.image_size = tuple(round(side * percent / 100) for side in THUMBNAIL_SIZE)
            self.row_height = self.image_size[1] + self.card_chrome
            top = (first // columns) * self.row_height
            self.columns = columns
            self.clear_rows()
        for window_id, _, _ in self.rows.values():
            self.canvas.itemconfigure(window_id, width=width)
        self.update_scrollregion(top)

    def update_scrollregion(self, top):
        height = ceil(len(self.items) / self.columns) * self.row_height
        height = max(height, self.canvas.winfo_height(), 1)
        self.canvas.configure(scrollregion=(0, 0, self.canvas.winfo_width(), height))
        self.canvas.itemconfigure(self.empty, state="hidden" if self.items else "normal")
        self.canvas.yview_moveto(max(0, top) / height)
        self.schedule_redraw()

    def on_scroll(self, first, last):
        self.scrollbar.set(first, last)
        self.schedule_redraw()

    def wheel(self, event):
        if event.state & 0x0004 and self.zoom is not None:  # Tk Control modifier
            self.zoom(event.delta)
        else:
            pixels = round(-event.delta * self.row_height / 120)
            self.canvas.yview_scroll(pixels, "units")
        return "break"

    def schedule_redraw(self):
        if self.redraw is None:
            self.redraw = self.after_idle(self.render_visible_rows)

    def render_visible_rows(self):
        self.redraw = None
        top = max(0, self.canvas.canvasy(0))
        first = max(0, int(top // self.row_height) - 1)
        last = min(
            ceil(len(self.items) / self.columns),
            ceil((top + self.canvas.winfo_height()) / self.row_height) + 1,
        )
        for row in list(self.rows):
            if row < first or row >= last:
                self.remove_row(row)
        for row in range(first, last):
            if row not in self.rows:
                self.render_row(row)

    def render_row(self, row):
        frame = ttk.Frame(self.canvas)
        frame.rowconfigure(0, weight=1)
        for column in range(self.columns):
            frame.columnconfigure(column, weight=1, uniform="cards")
        start = row * self.columns
        cards = []
        for column, (path, caption) in enumerate(self.items[start:start + self.columns]):
            photo, checked, checkbox = thumbnail_card(
                frame, column, path, caption, self.is_selected(path), self.toggle, size=self.image_size
            )
            if not self.enabled:
                checkbox.state(["disabled"])
            widget = checkbox.master
            for child in (widget, *widget.winfo_children()):
                child.bind("<MouseWheel>", self.wheel)
            cards.append((photo, checked, checkbox))
        frame.bind("<MouseWheel>", self.wheel)
        window_id = self.canvas.create_window(
            0, row * self.row_height, anchor="nw", window=frame,
            width=self.canvas.winfo_width(), height=self.row_height,
        )
        self.rows[row] = (window_id, frame, cards)

    def remove_row(self, row):
        window_id, frame, _ = self.rows.pop(row)
        self.canvas.delete(window_id)
        frame.destroy()

    def clear_rows(self):
        for row in list(self.rows):
            self.remove_row(row)

    def destroy(self):
        if self.redraw is not None:
            self.after_cancel(self.redraw)
        super().destroy()


class ThumbnailSizeControl(ttk.Frame):
    def __init__(self, parent, *lists, percent=100):
        super().__init__(parent)
        self.lists = lists
        self.percent = percent
        self.pending = None
        self.value = tk.DoubleVar(master=self, value=percent)
        ttk.Label(self, text="サムネイルサイズ").pack(side="left", padx=(0, 8))
        self.scale = ttk.Scale(
            self, from_=MIN_THUMBNAIL_PERCENT, to=MAX_THUMBNAIL_PERCENT,
            length=130, variable=self.value, command=self.change,
        )
        self.scale.pack(side="left")
        self.label = ttk.Label(self, text=f"{percent}%", width=5, anchor="e")
        self.label.pack(side="left", padx=(6, 0))
        for thumbnails in self.lists:
            thumbnails.zoom = self.zoom
            thumbnails.set_size(percent)

    def change(self, value):
        percent = round(float(value) / 5) * 5
        self.value.set(percent)
        if percent == self.percent:
            return
        self.percent = percent
        self.label.configure(text=f"{percent}%")
        if self.pending is not None:
            self.after_cancel(self.pending)
        self.pending = self.after(70, self.apply_size)

    def apply_size(self):
        self.pending = None
        for thumbnails in self.lists:
            thumbnails.set_size(self.percent)

    def zoom(self, delta):
        if delta:
            step = 10 if delta > 0 else -10
            size = max(MIN_THUMBNAIL_PERCENT, min(MAX_THUMBNAIL_PERCENT, self.percent + step))
            self.scale.set(size)

    def destroy(self):
        if self.pending is not None:
            self.after_cancel(self.pending)
        super().destroy()


class RegistrationDialog:
    def __init__(self, parent, key, registrations, *, thumbnail_percent=100):
        self.key = key
        self.registrations = registrations
        self.selected = {table: set() for table in REGISTRATION_TABS}
        self.changed = False
        self.window = tk.Toplevel(parent)
        self.window.title("登録画像を管理")
        self.window.geometry("930x700")
        self.window.minsize(880, 680)
        self.window.transient(parent)
        self.window.bind("<Escape>", lambda _event: self.window.destroy())

        body = ttk.Frame(self.window, padding=14)
        body.pack(fill="both", expand=True)
        self.tabs = ttk.Notebook(body)
        self.tabs.pack(fill="both", expand=True)
        self.grids = {}
        for table, title in REGISTRATION_TABS.items():
            grid = ThumbnailList(
                self.tabs, self.selected[table].__contains__, self.toggle, empty_text="登録された画像はありません。"
            )
            self.tabs.add(grid, text=title)
            grid.set_items([(path, "") for path in self.registrations[table]])
            self.grids[table] = grid

        footer = ttk.Frame(body)
        footer.pack(fill="x", pady=(10, 0))
        self.remove_button = ttk.Button(footer, text="選択した登録を解除", command=self.remove)
        self.remove_button.pack(side="left")
        self.reset_button = ttk.Button(footer, text="登録を初期化…", command=self.reset)
        self.reset_button.pack(side="left", padx=(8, 0))
        self.selection_label = ttk.Label(footer)
        self.selection_label.pack(side="left", padx=10)
        ttk.Button(footer, text="閉じる", command=self.window.destroy).pack(side="right", padx=(12, 0))
        self.size_control = ThumbnailSizeControl(footer, *self.grids.values(), percent=thumbnail_percent)
        self.size_control.pack(side="right")
        self.tabs.bind("<<NotebookTabChanged>>", lambda _event: self.update_selection())
        self.update_selection()
        self.window.grab_set()

    def active_table(self):
        index = self.tabs.index("current")
        return tuple(REGISTRATION_TABS)[index]

    def update_selection(self):
        size = len(self.selected[self.active_table()])
        self.selection_label.configure(text=f"{size:,} 枚選択中")
        self.remove_button.configure(state="normal" if size else "disabled")

    def toggle(self, path, checked):
        selected = self.selected[self.active_table()]
        if checked:
            selected.add(path)
        else:
            selected.discard(path)
        self.update_selection()

    def remove(self):
        table = self.active_table()
        paths = self.selected[table]
        try:
            remove_registrations(self.key, table, paths)
        except (OSError, sqlite3.Error) as error:
            messagebox.showerror("登録を解除できませんでした", str(error), parent=self.window)
            return
        self.registrations[table] = [path for path in self.registrations[table] if path not in paths]
        paths.clear()
        self.changed = True
        self.grids[table].set_items([(path, "") for path in self.registrations[table]])
        self.update_selection()

    def reset(self):
        if not messagebox.askokcancel(
            "登録を初期化",
            "現在の見本設定で追加した見本と「好みではない」の登録、見本の解除履歴をすべてリセットします。"
            "\n\n画像ファイルは元の場所に残ります。",
            parent=self.window,
        ):
            return
        try:
            removed = reset_registrations(self.key)
        except (OSError, sqlite3.Error) as error:
            messagebox.showerror("登録を初期化できませんでした", str(error), parent=self.window)
            return
        self.changed = self.changed or removed > 0
        for table, grid in self.grids.items():
            self.registrations[table] = []
            self.selected[table].clear()
            grid.set_items([], reset=True)
        self.update_selection()
        messagebox.showinfo(
            "登録を初期化",
            "登録を初期化しました。" if removed else "初期化する登録はありません。",
            parent=self.window,
        )


class Gallery:
    def __init__(self, window):
        self.window = window
        window.title("CLIP 画像ギャラリー")
        window.geometry("1020x854")
        window.minsize(880, 780)
        window.configure(background="#f3f5f8")
        window.protocol("WM_DELETE_WINDOW", self.close)
        self.folder = RAW
        self.favorites = DEFAULT_FAVORITES if DEFAULT_FAVORITES.is_dir() else ()
        self.events = Queue()
        self.cancel = Event()
        self.worker = None
        self.results = []
        self.has_ranking = False
        self.selected = set()
        self.negative_references = set()
        self.moving = False

        style = ttk.Style(window)
        style.theme_use("clam")
        style.configure(".", font=("Yu Gothic UI", 10), background="#f3f5f8", foreground="#162638")
        style.configure("Title.TLabel", font=("Yu Gothic UI", 21, "bold"))
        style.configure("Card.TFrame", background="white")
        style.configure("Card.TLabel", background="white")
        style.configure("Card.TCheckbutton", background="white")
        style.configure("Primary.TButton", background="#225aaa", foreground="white", padding=(14, 7))
        style.map("Primary.TButton", background=[("active", "#184786"), ("disabled", "#8799b2")])
        style.configure("TButton", padding=(10, 5))

        body = ttk.Frame(window, padding=(24, 16))
        body.pack(fill="both", expand=True)
        header = ttk.Frame(body)
        header.pack(fill="x", pady=(0, 10))
        ttk.Label(header, text="CLIP 画像ギャラリー", style="Title.TLabel").pack(side="left")
        self.selection_label = ttk.Label(header)
        self.selection_label.pack(side="left", padx=16)
        self.manage_button = ttk.Button(header, text="登録画像を管理…", command=self.manage_registrations)
        self.manage_button.pack(side="right")
        form = ttk.Frame(body)
        form.pack(fill="x")
        form.columnconfigure(1, weight=1)
        self.sources = []
        for row, caption in enumerate(("検索先", "見本画像")):
            ttk.Label(form, text=caption, width=10).grid(row=row, column=0, sticky="w", pady=3)
            value = ttk.Entry(form, state="readonly")
            value.grid(row=row, column=1, sticky="ew", padx=(0, 8))
            self.sources.append(value)
        self.pick_buttons = [
            ttk.Button(form, text="フォルダー…", command=self.pick_folder),
            ttk.Button(form, text="フォルダー…", command=self.pick_favorite_folder),
            ttk.Button(form, text="画像を選ぶ…", command=self.pick_files),
        ]
        self.pick_buttons[0].grid(row=0, column=2, sticky="ew")
        self.pick_buttons[1].grid(row=1, column=2, sticky="ew")
        self.pick_buttons[2].grid(row=1, column=3, padx=(6, 0))
        self.update_sources()

        actions = ttk.Frame(body)
        actions.pack(fill="x", pady=(10, 9))
        self.start_button = ttk.Button(actions, text="類似度順に並べる", style="Primary.TButton", command=self.start)
        self.start_button.pack(side="left")
        self.stop_button = ttk.Button(actions, text="中止", command=self.stop, state="disabled")
        self.stop_button.pack(side="left", padx=8)
        self.progress = ttk.Progressbar(actions, length=135)
        self.progress.pack(side="left", padx=(6, 10))
        self.status = ttk.Label(actions, text="見本と検索先を選んで開始")
        self.status.pack(side="left")

        navigation = ttk.Frame(body)
        navigation.pack(side="bottom", fill="x", pady=(8, 0))
        self.move_button = ttk.Button(navigation, text="移動…", command=self.move_selected)
        self.move_button.pack(side="left")
        self.reference_button = ttk.Button(navigation, text="見本に追加", command=self.add_selected_references)
        self.reference_button.pack(side="left", padx=(8, 0))
        self.add_button = ttk.Button(
            navigation, text="移動して見本に追加…", command=lambda: self.move_selected(add_to_favorites=True)
        )
        self.add_button.pack(side="left", padx=(8, 0))
        self.negative_button = ttk.Button(navigation, command=self.change_negative_selection)
        self.negative_button.pack(side="left", padx=(8, 0))
        self.update_selection()
        self.grid = ThumbnailList(body, self.selected.__contains__, self.toggle_selection)
        self.grid.pack(fill="both", expand=True)
        self.size_control = ThumbnailSizeControl(header, self.grid)
        self.size_control.pack(side="right", padx=(0, 14))

    def update_sources(self):
        if isinstance(self.favorites, Path):
            label = str(self.favorites)
        elif self.favorites:
            label = f"{len(self.favorites)} 枚  ·  " + ", ".join(path.name for path in self.favorites[:4])
        else:
            label = "未選択"
        for field, value in zip(self.sources, (str(self.folder), label), strict=True):
            field.configure(state="normal")
            field.delete(0, "end")
            field.insert(0, value)
            field.configure(state="readonly")

    def reset_results(self):
        self.update_sources()
        self.results = []
        self.has_ranking = False
        self.selected.clear()
        self.negative_references.clear()
        self.update_selection()
        self.grid.set_items([], reset=True)
        self.progress.configure(value=0)
        self.status.configure(text="「類似度順に並べる」を押して検索")

    def pick_folder(self):
        path = filedialog.askdirectory(parent=self.window, initialdir=self.folder)
        if path:
            self.folder = Path(path)
            self.reset_results()

    def pick_favorite_folder(self):
        initial = self.favorites if isinstance(self.favorites, Path) else self.folder
        path = filedialog.askdirectory(parent=self.window, initialdir=initial)
        if path:
            self.favorites = Path(path)
            self.reset_results()

    def pick_files(self):
        initial = self.favorites if isinstance(self.favorites, Path) else self.folder
        paths = filedialog.askopenfilenames(parent=self.window, initialdir=initial, filetypes=IMAGE_TYPES)
        if paths:
            self.favorites = tuple(map(Path, paths))
            self.reset_results()

    def manage_registrations(self):
        if self.worker is not None:
            return
        try:
            key = reference_key(self.favorites)
            with closing(open_cache()) as db, db:
                prepare_cache(db)
                registrations = {table: registration_paths(db, table, key) for table in REGISTRATION_TABS}
        except (OSError, sqlite3.Error) as error:
            messagebox.showerror("登録画像を読み込めませんでした", str(error), parent=self.window)
            return
        dialog = RegistrationDialog(self.window, key, registrations, thumbnail_percent=self.size_control.percent)
        self.window.wait_window(dialog.window)
        if dialog.changed and self.has_ranking:
            self.start(allow_empty=True)

    def start(self, *, allow_empty=False):
        if not self.favorites:
            messagebox.showinfo("見本画像", "見本にする画像を1枚以上選んでください。", parent=self.window)
            return
        self.cancel.clear()
        self.selected.clear()
        self.update_selection()
        self.results = []
        self.has_ranking = False
        self.grid.set_items([], reset=True)
        self.progress.configure(value=0)
        self.status.configure(text="画像を探しています…")
        self.stop_button.configure(state="normal")
        self.run_task(lambda: rank_images(self.folder, self.favorites, self.events, self.cancel, allow_empty=allow_empty))

    def update_selection(self):
        self.selection_label.configure(text=f"{len(self.selected):,} 枚選択中")
        state = "normal" if self.selected and self.worker is None else "disabled"
        for button in (self.move_button, self.reference_button, self.add_button, self.negative_button):
            button.configure(state=state)
        remove = self.selected and self.selected <= self.negative_references
        self.negative_button.configure(text="好みではない登録を解除" if remove else "好みではない")

    def toggle_selection(self, path, selected):
        if selected:
            self.selected.add(path)
        else:
            self.selected.discard(path)
        self.update_selection()

    def add_selected_references(self):
        if not self.selected or self.worker is not None:
            return
        paths = self.selected.copy()
        self.cancel.clear()
        self.status.configure(text="見本に追加しています…")
        self.progress.configure(value=0)

        def rerank():
            # A file may have been moved or removed after the gallery was loaded.
            existing = {path for path in paths if path.is_file()}
            if not existing:
                raise ValueError("選択した画像が見つかりません。")
            save_added_references(self.favorites, existing)
            self.events.put(("negative_references", self.negative_references - existing))
            rank_images(self.folder, self.favorites, self.events, self.cancel, allow_empty=True)
            if missing := paths - existing:
                self.events.put(("reference_missing", len(missing)))

        self.run_task(rerank)

    def change_negative_selection(self):
        if not self.selected or self.worker is not None:
            return
        paths = self.selected.copy()
        remove = paths <= self.negative_references
        negatives = self.negative_references - paths if remove else self.negative_references | paths
        self.cancel.clear()
        self.status.configure(text="好みではない画像の登録を更新中…")
        self.progress.configure(value=0)

        def rerank():
            update_negative_references(self.favorites, paths, remove=remove)
            self.events.put(("negative_references", negatives))
            rank_images(self.folder, self.favorites, self.events, self.cancel, allow_empty=True)

        self.run_task(rerank)

    def move_selected(self, *, add_to_favorites=False):
        if not self.selected or self.worker is not None:
            return
        folder = filedialog.askdirectory(
            parent=self.window, title="画像の移動先フォルダーを選ぶ",
            initialdir=self.folder.parent, mustexist=True,
        )
        if not folder:
            return
        try:
            destination = Path(folder).resolve(strict=True)
        except OSError as error:
            messagebox.showerror("移動先を開けません", str(error), parent=self.window)
            return
        paths = [path for path, _ in self.results if path in self.selected]
        confirmation = f"{len(paths):,} 枚の画像を次のフォルダーに移動します。\n\n{destination}"
        if add_to_favorites:
            confirmation += "\n\n移動した画像を見本に追加し、残りを並べ直します。"
        if not messagebox.askokcancel(
            "画像を移動", confirmation, parent=self.window,
        ):
            return
        self.cancel.clear()
        self.moving = True
        self.progress.configure(value=0, maximum=len(paths))
        self.status.configure(text="画像を移動中…")

        def move():
            moved = {}
            failures = []
            for index, path in enumerate(paths, 1):
                try:
                    stat = path.stat()
                    moved[path] = (move_image(path, destination), stat)
                except (OSError, ValueError) as error:
                    failures.append((path, error))
                if index % 8 == 0 or index == len(paths):
                    self.events.put(("move_progress", (index, len(paths))))
            self.events.put(("moved", (moved, failures)))
            if moved and add_to_favorites:
                try:
                    save_added_references(
                        self.favorites, (destination for destination, _ in moved.values()), moved=moved
                    )
                except (OSError, sqlite3.Error) as error:
                    raise RuntimeError(f"画像は移動しましたが、見本への追加を保存できませんでした。\n{error}") from error
                self.events.put(("status", "追加した見本で並べ直しています…"))
                rank_images(self.folder, self.favorites, self.events, self.cancel, allow_empty=True)

        self.run_task(move)

    def run_task(self, action):
        for button in [*self.pick_buttons, self.start_button, self.manage_button]:
            button.configure(state="disabled")

        def work():
            try:
                action()
            except Exception as error:
                self.events.put(("error", str(error)))
            finally:
                self.events.put(("finished", None))

        self.worker = Thread(target=work, daemon=True)
        self.worker.start()
        self.grid.set_enabled(False)
        self.update_selection()
        self.window.after(80, self.poll)

    def stop(self):
        self.cancel.set()
        self.stop_button.configure(state="disabled")
        self.status.configure(text="中止しています…")

    def poll(self):
        try:
            while True:
                kind, data = self.events.get_nowait()
                if kind == "finished":
                    for button in [*self.pick_buttons, self.start_button, self.manage_button]:
                        button.configure(state="normal")
                    self.stop_button.configure(state="disabled")
                    self.moving = False
                    self.worker = None
                    self.grid.set_enabled(True)
                    self.update_selection()
                    if self.cancel.is_set():
                        self.status.configure(text="中止しました。処理済みの画像は再利用できます。")
                    return
                if self.cancel.is_set():
                    continue
                if kind in ("progress", "move_progress"):
                    done, total = data
                    self.progress.configure(value=done, maximum=total)
                    operation = "移動中" if kind == "move_progress" else "照合中"
                    self.status.configure(text=f"{operation}  {done:,} / {total:,} 枚")
                elif kind == "status":
                    self.status.configure(text=data)
                elif kind == "negative_references":
                    self.negative_references = data
                elif kind == "reference_missing":
                    messagebox.showwarning(
                        "一部の画像を見本に追加できませんでした",
                        f"選択した画像のうち {data:,} 枚が見つかりませんでした。",
                        parent=self.window,
                    )
                elif kind == "results":
                    self.results, favorites, skipped, reused = data
                    self.has_ranking = True
                    self.selected.intersection_update(path for path, _ in self.results)
                    negative_summary = (
                        f"  ·  好みではない {len(self.negative_references):,} 枚" if self.negative_references else ""
                    )
                    self.status.configure(
                        text=f"{len(self.results):,} 件  ·  見本 {favorites:,} 枚{negative_summary}"
                             f"  ·  読込不可 {skipped:,} 枚  ·  再利用 {reused:,} 枚"
                    )
                    self.show_results(reset=True)
                elif kind == "moved":
                    moved, failures = data
                    self.selected.difference_update(moved)
                    self.results = [(path, score) for path, score in self.results if path not in moved]
                    self.show_results()
                    self.status.configure(text=f"{len(moved):,} 枚移動  ·  移動できなかった画像 {len(failures):,} 枚")
                    if failures:
                        details = "\n".join(f"{path.name}: {error}" for path, error in failures[:5])
                        if len(failures) > 5:
                            details += f"\nほか {len(failures) - 5:,} 枚"
                        messagebox.showwarning("一部の画像を移動できませんでした", details, parent=self.window)
                elif kind == "error":
                    operation = "処理" if self.moving else "照合"
                    self.status.configure(text=f"{operation}を完了できませんでした")
                    messagebox.showerror("CLIP 画像ギャラリー", data, parent=self.window)
        except Empty:
            self.window.after(80, self.poll)

    def show_results(self, *, reset=False):
        items = []
        for index, (path, score) in enumerate(self.results, 1):
            rank = f"{index:,}位  ·  スコア {score:.3f}"
            if path in self.negative_references:
                rank += "  ·  登録済み"
            items.append((path, rank))
        self.grid.set_items(items, reset=reset)

    def close(self):
        if self.moving:
            messagebox.showinfo("画像を移動中", "移動が完了してからウィンドウを閉じてください。", parent=self.window)
        else:
            self.window.destroy()


if __name__ == "__main__":
    window = tk.Tk()
    Gallery(window)
    window.mainloop()
