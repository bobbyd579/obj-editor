"""
OBJ Combiner
Combines two OBJ files (each with 1 MTL and 1 JPG in the same folder) into
one OBJ, one MTL, and two JPG textures.
"""

# Set DPI awareness BEFORE importing tkinter
try:
    import ctypes
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        ctypes.windll.user32.SetProcessDPIAware()
except (ImportError, AttributeError, OSError):
    pass

import os
import re
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


def resolve_mtl_for_obj(obj_path):
    """Resolve MTL path for an OBJ. Uses mtllib if present (exactly one), else same-name .mtl.
    Returns (mtl_path, None) or (None, error_message)."""
    obj_dir = os.path.dirname(obj_path)
    mtllibs = []
    try:
        with open(obj_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if parts and parts[0].lower() == "mtllib" and len(parts) >= 2:
                    # mtllib can have the filename with spaces; join rest
                    name = " ".join(parts[1:]).strip()
                    if name:
                        mtllibs.append(name)
    except Exception as e:
        return None, f"Could not read OBJ: {e}"
    if len(mtllibs) > 1:
        return None, "More than one mtllib in OBJ"
    if mtllibs:
        mtl_path = os.path.join(obj_dir, mtllibs[0])
    else:
        base = os.path.splitext(os.path.basename(obj_path))[0]
        mtl_path = os.path.join(obj_dir, base + ".mtl")
    if not os.path.isfile(mtl_path):
        return None, "No MTL found in folder"
    return mtl_path, None


def parse_mtl_textures(mtl_path):
    """Return list of (map_Kd) texture paths found in MTL. Paths are as in file (relative or abs)."""
    mtl_dir = os.path.dirname(mtl_path)
    paths = []
    try:
        with open(mtl_path, "r", encoding="utf-8") as f:
            for line in f:
                m = re.search(r"map_Kd\s+([^\s#]+)", line, re.IGNORECASE)
                if m:
                    p = m.group(1).strip()
                    paths.append(p)
    except Exception:
        pass
    return paths


def validate_obj_mtl_texture(obj_path, label="OBJ"):
    """Validate: 1 MTL, 1 texture, JPG only. Returns (mtl_path, tex_full_path) or raises ValueError."""
    mtl_path, err = resolve_mtl_for_obj(obj_path)
    if err:
        raise ValueError(f"{label}: {err}")
    tex_paths = parse_mtl_textures(mtl_path)
    if len(tex_paths) == 0:
        raise ValueError(f"{label}: MTL has no texture (map_Kd)")
    if len(tex_paths) > 1:
        raise ValueError(f"{label}: MTL has more than one texture")
    raw = tex_paths[0]
    mtl_dir = os.path.dirname(mtl_path)
    if os.path.isabs(raw):
        full = raw
    else:
        full = os.path.normpath(os.path.join(mtl_dir, raw))
    if not os.path.isfile(full):
        raise ValueError(f"{label}: Texture file not found: {raw}")
    ext = os.path.splitext(full)[1].lower()
    if ext not in (".jpg", ".jpeg"):
        raise ValueError(f"{label}: Texture must be JPG (got {ext})")
    return mtl_path, full


def parse_obj_full(obj_path):
    """Parse OBJ: v, vt, vn, faces (with v, vt, vn indices, format, usemtl)."""
    v, vt, vn = [], [], []
    faces = []  # {vertices, textures, normals, format, usemtl}
    current_material = None
    try:
        with open(obj_path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                parts = s.split()
                if not parts:
                    continue
                if parts[0] == "v" and len(parts) >= 4:
                    v.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif parts[0] == "vt" and len(parts) >= 3:
                    u, vv = float(parts[1]), float(parts[2])
                    w = float(parts[3]) if len(parts) >= 4 else 0.0
                    vt.append([u, vv, w])
                elif parts[0] == "vn" and len(parts) >= 4:
                    vn.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif parts[0] == "usemtl" and len(parts) >= 2:
                    current_material = parts[1]
                elif parts[0] == "f":
                    fv, ft, fn, fmt = [], [], [], []
                    for part in parts[1:]:
                        idx = part.split("/")
                        a = int(idx[0]) - 1 if idx[0] else None
                        b = (int(idx[1]) - 1) if (len(idx) > 1 and idx[1]) else None
                        c = (int(idx[2]) - 1) if (len(idx) > 2 and idx[2]) else None
                        if a is None:
                            continue
                        fv.append(a)
                        ft.append(b)
                        fn.append(c)
                        fmt.append(part)
                    if fv:
                        faces.append({
                            "vertices": fv, "textures": ft, "normals": fn,
                            "format": fmt, "usemtl": current_material,
                        })
    except Exception as e:
        raise RuntimeError(f"Parse OBJ: {e}") from e
    return {"v": v, "vt": vt, "vn": vn, "faces": faces}


def parse_mtl_materials(mtl_path):
    """Parse MTL into list of {name, map_Kd or None, and raw lines for the block}."""
    mtl_dir = os.path.dirname(mtl_path)
    blocks = []
    current = None
    try:
        with open(mtl_path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    if current is not None:
                        current["lines"].append(line)
                    continue
                parts = s.split()
                if parts and parts[0].lower() == "newmtl" and len(parts) >= 2:
                    name = " ".join(parts[1:]).strip()
                    current = {"name": name, "map_Kd": None, "lines": [line]}
                    blocks.append(current)
                elif current is not None:
                    m = re.search(r"map_Kd\s+([^\s#]+)", line, re.IGNORECASE)
                    if m:
                        p = m.group(1).strip()
                        if os.path.isabs(p):
                            current["map_Kd"] = p
                        else:
                            current["map_Kd"] = os.path.normpath(os.path.join(mtl_dir, p))
                    current["lines"].append(line)
    except Exception as e:
        raise RuntimeError(f"Parse MTL: {e}") from e
    return blocks


def merge_objs(data1, data2, blocks1, blocks2, prefix1, prefix2, out_tex1, out_tex2):
    """Merge two parsed OBJs and MTL blocks. Returns (obj_lines, mtl_lines)."""
    v1, vt1, vn1 = data1["v"], data1["vt"], data1["vn"]
    v2, vt2, vn2 = data2["v"], data2["vt"], data2["vn"]
    nv1, nvt1, nvn1 = len(v1), len(vt1), len(vn1)

    # Material name mapping: old -> new (prefixed)
    mat1_map = {b["name"]: f"{prefix1}_{b['name']}" for b in blocks1}
    mat2_map = {b["name"]: f"{prefix2}_{b['name']}" for b in blocks2}

    lines = []
    lines.append("# Merged OBJ\n")
    # mtllib will be set by caller to match output .mtl name
    # lines.append("mtllib <stem>.mtl\n")

    # v
    for a in v1:
        lines.append(f"v {a[0]:.6f} {a[1]:.6f} {a[2]:.6f}\n")
    for a in v2:
        lines.append(f"v {a[0]:.6f} {a[1]:.6f} {a[2]:.6f}\n")
    if vt1 or vt2:
        lines.append("\n")
        for a in vt1:
            lines.append(f"vt {a[0]:.6f} {a[1]:.6f}\n")
        for a in vt2:
            lines.append(f"vt {a[0]:.6f} {a[1]:.6f}\n")
    if vn1 or vn2:
        lines.append("\n")
        for a in vn1:
            lines.append(f"vn {a[0]:.6f} {a[1]:.6f} {a[2]:.6f}\n")
        for a in vn2:
            lines.append(f"vn {a[0]:.6f} {a[1]:.6f} {a[2]:.6f}\n")

    default1 = f"{prefix1}_{blocks1[0]['name']}" if blocks1 else f"{prefix1}_default"
    default2 = f"{prefix2}_{blocks2[0]['name']}" if blocks2 else f"{prefix2}_default"

    last_mat = None
    for face in data1["faces"]:
        mat = mat1_map.get(face["usemtl"], default1)
        if mat != last_mat:
            lines.append(f"usemtl {mat}\n")
            last_mat = mat
        p = "f"
        for fmt in face["format"]:
            p += " " + fmt
        lines.append(p + "\n")

    for face in data2["faces"]:
        mat = mat2_map.get(face["usemtl"], default2)
        if mat != last_mat:
            lines.append(f"usemtl {mat}\n")
            last_mat = mat
        p = "f"
        for i in range(len(face["vertices"])):
            vv = face["vertices"][i] + nv1 + 1
            tt = face["textures"][i]
            nn = face["normals"][i]
            if tt is not None and nn is not None:
                p += f" {vv}/{tt + nvt1 + 1}/{nn + nvn1 + 1}"
            elif tt is not None:
                p += f" {vv}/{tt + nvt1 + 1}"
            elif nn is not None:
                p += f" {vv}//{nn + nvn1 + 1}"
            else:
                p += f" {vv}"
        lines.append(p + "\n")

    # Merged MTL: rewrite blocks with new names and map_Kd -> out_tex1/out_tex2
    mtl_lines = []
    for b in blocks1:
        mat = f"{prefix1}_{b['name']}"
        mtl_lines.append(f"newmtl {mat}\n")
        for ln in b["lines"][1:]:  # skip newmtl
            ln = re.sub(r"map_Kd\s+[^\s#]+", f"map_Kd {out_tex1}", ln, flags=re.IGNORECASE)
            mtl_lines.append(ln)
    for b in blocks2:
        mat = f"{prefix2}_{b['name']}"
        mtl_lines.append(f"newmtl {mat}\n")
        for ln in b["lines"][1:]:
            ln = re.sub(r"map_Kd\s+[^\s#]+", f"map_Kd {out_tex2}", ln, flags=re.IGNORECASE)
            mtl_lines.append(ln)

    return lines, mtl_lines


class CombinerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("OBJ Combiner")
        self.root.geometry("600x280")
        self.obj1_path = None
        self.obj2_path = None
        self.setup_ui()

    def setup_ui(self):
        main = ttk.LabelFrame(self.root, text="Combine two OBJs (each: 1 MTL, 1 JPG in same folder)", padding="10")
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # OBJ 1
        f1 = ttk.Frame(main)
        f1.pack(fill=tk.X, pady=5)
        ttk.Button(f1, text="OBJ 1", command=self.select_obj1).pack(side=tk.LEFT, padx=5)
        self.l1 = ttk.Label(f1, text="No file selected", foreground="gray")
        self.l1.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # OBJ 2
        f2 = ttk.Frame(main)
        f2.pack(fill=tk.X, pady=5)
        ttk.Button(f2, text="OBJ 2", command=self.select_obj2).pack(side=tk.LEFT, padx=5)
        self.l2 = ttk.Label(f2, text="No file selected", foreground="gray")
        self.l2.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # Buttons
        bf = ttk.Frame(main)
        bf.pack(fill=tk.X, pady=10)
        ttk.Button(bf, text="Combine and Save", command=self.combine_and_save).pack(side=tk.LEFT, padx=5)
        ttk.Button(bf, text="Clear", command=self.clear).pack(side=tk.LEFT, padx=5)

    def select_obj1(self):
        p = filedialog.askopenfilename(title="Select OBJ 1", filetypes=[("OBJ files", "*.obj"), ("All", "*.*")])
        if p:
            self.obj1_path = p
            self.l1.config(text=os.path.basename(p), foreground="black")

    def select_obj2(self):
        p = filedialog.askopenfilename(title="Select OBJ 2", filetypes=[("OBJ files", "*.obj"), ("All", "*.*")])
        if p:
            self.obj2_path = p
            self.l2.config(text=os.path.basename(p), foreground="black")

    def clear(self):
        self.obj1_path = self.obj2_path = None
        self.l1.config(text="No file selected", foreground="gray")
        self.l2.config(text="No file selected", foreground="gray")

    def combine_and_save(self):
        if not self.obj1_path:
            messagebox.showerror("Error", "Please select OBJ 1.")
            return
        if not self.obj2_path:
            messagebox.showerror("Error", "Please select OBJ 2.")
            return
        try:
            mtl1, tex1 = validate_obj_mtl_texture(self.obj1_path, "OBJ 1")
            mtl2, tex2 = validate_obj_mtl_texture(self.obj2_path, "OBJ 2")
        except ValueError as e:
            messagebox.showerror("Validation Error", str(e))
            return

        try:
            data1 = parse_obj_full(self.obj1_path)
            data2 = parse_obj_full(self.obj2_path)
            blocks1 = parse_mtl_materials(mtl1)
            blocks2 = parse_mtl_materials(mtl2)
        except RuntimeError as e:
            messagebox.showerror("Error", str(e))
            return

        out_path = filedialog.asksaveasfilename(
            title="Save merged OBJ",
            filetypes=[("OBJ files", "*.obj"), ("All", "*.*")],
            defaultextension=".obj",
        )
        if not out_path:
            return

        out_dir = os.path.dirname(out_path)
        stem = os.path.splitext(os.path.basename(out_path))[0]
        out_mtl = os.path.join(out_dir, stem + ".mtl")
        out_tex1_name = stem + "_tex1.jpg"
        out_tex2_name = stem + "_tex2.jpg"
        out_tex1_path = os.path.join(out_dir, out_tex1_name)
        out_tex2_path = os.path.join(out_dir, out_tex2_name)

        obj_lines, mtl_lines = merge_objs(
            data1, data2, blocks1, blocks2,
            "obj1", "obj2", out_tex1_name, out_tex2_name,
        )

        try:
            mtllib_line = f"mtllib {stem}.mtl\n"
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(mtllib_line)
                f.writelines(obj_lines)
            with open(out_mtl, "w", encoding="utf-8") as f:
                f.writelines(mtl_lines)
            shutil.copy2(tex1, out_tex1_path)
            shutil.copy2(tex2, out_tex2_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to write: {e}")
            return

        messagebox.showinfo(
            "Success",
            f"Merged files saved to:\n{out_dir}\n\n"
            f"OBJ: {os.path.basename(out_path)}\n"
            f"MTL: {stem}.mtl\n"
            f"Textures: {out_tex1_name}, {out_tex2_name}",
        )


if __name__ == "__main__":
    root = tk.Tk()
    CombinerGUI(root)
    root.mainloop()
