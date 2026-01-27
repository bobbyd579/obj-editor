"""
OBJ Suffix Checker
Scans a directory (and subfolders) for OBJ files, lists base OBJs (no _s1/_s2/_s3),
and indicates whether _s1, _s2, _s3 variants exist in the same folder.
"""

# Set DPI awareness BEFORE importing tkinter to prevent scaling issues
# Use level 1 (Per Monitor DPI Aware) which works better with tkinter
try:
    import ctypes
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)  # PROCESS_PER_MONITOR_DPI_AWARE
    except:
        ctypes.windll.user32.SetProcessDPIAware()
except (ImportError, AttributeError, OSError):
    pass

import csv
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


def scan_folder(root_dir):
    """Recursively scan folder and subfolders for base OBJs and their _s1/_s2/_s3 variants.
    Returns [(rel_path, has_s1, has_s2, has_s3), ...] sorted by rel_path.
    """
    rows = []
    try:
        for dirpath, _, filenames in os.walk(root_dir):
            stems = {
                os.path.splitext(f)[0]
                for f in filenames
                if f.lower().endswith(".obj")
            }
            base_stems = [
                s
                for s in stems
                if not (s.endswith("_s1") or s.endswith("_s2") or s.endswith("_s3"))
            ]
            for base in sorted(base_stems):
                has_s1 = (base + "_s1") in stems
                has_s2 = (base + "_s2") in stems
                has_s3 = (base + "_s3") in stems
                rel_dir = os.path.relpath(dirpath, root_dir)
                if rel_dir == ".":
                    rel_path = base + ".obj"
                else:
                    rel_path = os.path.join(rel_dir, base + ".obj")
                rows.append((rel_path, has_s1, has_s2, has_s3))
    except Exception as e:
        raise e
    rows.sort(key=lambda r: r[0])
    return rows


class SuffixCheckerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("OBJ Suffix Checker")
        self.root.geometry("700x500")
        self.folder_path = None
        self.rows = []
        self.setup_ui()

    def setup_ui(self):
        main_frame = ttk.LabelFrame(self.root, text="Suffix Check", padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Folder selection
        folder_frame = ttk.Frame(main_frame)
        folder_frame.pack(fill=tk.X, pady=5)
        ttk.Button(folder_frame, text="Select Folder", command=self.select_folder).pack(
            side=tk.LEFT, padx=5
        )
        self.folder_label = ttk.Label(
            folder_frame, text="No folder selected", foreground="gray"
        )
        self.folder_label.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # Table: canvas with scrollable frame of labels (for row height and per-cell colors)
        table_frame = ttk.Frame(main_frame)
        table_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.table_canvas = tk.Canvas(table_frame, highlightthickness=0)
        v_scroll = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.table_canvas.yview)
        h_scroll = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL, command=self.table_canvas.xview)
        self.table_canvas.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)

        self.table_inner = tk.Frame(self.table_canvas)
        self.table_canvas_window = self.table_canvas.create_window(
            (0, 0), window=self.table_inner, anchor="nw"
        )

        def _on_frame_configure(event):
            self.table_canvas.configure(scrollregion=self.table_canvas.bbox("all"))

        self.table_inner.bind("<Configure>", _on_frame_configure)

        def _on_mousewheel(event):
            self.table_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        self.table_canvas.bind("<MouseWheel>", _on_mousewheel)
        self.table_inner.bind("<MouseWheel>", _on_mousewheel)

        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.table_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Count label
        self.count_label = ttk.Label(main_frame, text="0 base OBJ(s) found")
        self.count_label.pack(pady=2)

        # Export CSV button
        ttk.Button(main_frame, text="Export CSV", command=self.export_csv).pack(pady=5)

    def _populate_table(self):
        """Clear and rebuild the table from self.rows. Uses Labels for row height and green/red."""
        for w in self.table_inner.winfo_children():
            w.destroy()
        font = ("TkDefaultFont", 10)
        header_font = ("TkDefaultFont", 10, "bold")
        cell_pady = 5
        cell_padx = 8
        # Header
        for c, txt in enumerate(("OBJ", "_s1", "_s2", "_s3")):
            lbl = tk.Label(
                self.table_inner,
                text=txt,
                font=header_font,
                anchor="w" if c == 0 else "center",
                pady=cell_pady,
                padx=cell_padx,
            )
            lbl.grid(row=0, column=c, sticky="ew")
        self.table_inner.grid_columnconfigure(0, minsize=220)
        for c in (1, 2, 3):
            self.table_inner.grid_columnconfigure(c, minsize=70)
        # Data rows
        for i, (rel, h1, h2, h3) in enumerate(self.rows):
            r = i + 1
            tk.Label(
                self.table_inner,
                text=rel,
                font=font,
                anchor="w",
                pady=cell_pady,
                padx=cell_padx,
            ).grid(row=r, column=0, sticky="w")
            for col, val in enumerate((h1, h2, h3), start=1):
                txt = "True" if val else "False"
                fg = "green" if val else "red"
                tk.Label(
                    self.table_inner,
                    text=txt,
                    font=font,
                    fg=fg,
                    anchor="center",
                    pady=cell_pady,
                    padx=cell_padx,
                ).grid(row=r, column=col, sticky="")

    def select_folder(self):
        folder = filedialog.askdirectory(title="Select Folder with OBJ Files")
        if not folder:
            return
        self.folder_path = folder
        self.folder_label.config(text=os.path.basename(folder), foreground="black")
        try:
            self.rows = scan_folder(folder)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to scan folder: {str(e)}")
            return
        self._populate_table()
        n = len(self.rows)
        self.count_label.config(text=f"{n} base OBJ(s) found")
        if not self.rows:
            messagebox.showinfo("Info", "No base OBJ files found (no .obj without _s1/_s2/_s3).")

    def export_csv(self):
        if not self.rows:
            messagebox.showwarning("Warning", "No data to export. Select a folder first.")
            return
        path = filedialog.asksaveasfilename(
            title="Export CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            defaultextension=".csv",
        )
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(("OBJ", "_s1", "_s2", "_s3"))
                for rel, h1, h2, h3 in self.rows:
                    w.writerow((rel, h1, h2, h3))
            messagebox.showinfo("Success", f"Exported to {path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = SuffixCheckerGUI(root)
    root.mainloop()
