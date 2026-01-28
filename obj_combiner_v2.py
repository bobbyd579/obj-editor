"""
OBJ Combiner v2
Copy of obj_combiner with room for additional features.
Combines two OBJ files (each with 1 MTL and 1 JPG in the same folder) into
one OBJ, one MTL, and two JPG textures.
Includes 3D viewers for OBJ_1, OBJ_2, and combined; vertex picking; alignment; Save _s2.
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

import ctypes
import os
import re
import shutil
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np

try:
    import glfw
    from OpenGL.GL import *
    from OpenGL.GLU import gluPerspective, gluLookAt, gluProject
    GLFW_AVAILABLE = True
except ImportError:
    GLFW_AVAILABLE = False

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


def _make_number_texture(ch):
    """Create OpenGL texture for character '1','2','3'. Returns texture id or None. Call in GL context."""
    if not PIL_AVAILABLE:
        return None
    try:
        img = Image.new("RGBA", (24, 24), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 18)
        except Exception:
            try:
                font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 18)
            except Exception:
                font = ImageFont.load_default()
        try:
            bbox = draw.textbbox((0, 0), ch, font=font)
        except AttributeError:
            bbox = (0, 0, 8, 12)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = max(0, (24 - w) / 2)
        y = max(0, (24 - h) / 2)
        draw.rectangle((0, 0, 24, 24), fill=(0, 0, 0, 220))
        draw.text((x, y), ch, font=font, fill=(255, 255, 0))
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        data = img.tobytes()
        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 24, 24, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        return tex
    except Exception:
        return None


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

    # Merged MTL
    mtl_lines = []
    for b in blocks1:
        mat = f"{prefix1}_{b['name']}"
        mtl_lines.append(f"newmtl {mat}\n")
        for ln in b["lines"][1:]:
            ln = re.sub(r"map_Kd\s+[^\s#]+", f"map_Kd {out_tex1}", ln, flags=re.IGNORECASE)
            mtl_lines.append(ln)
    for b in blocks2:
        mat = f"{prefix2}_{b['name']}"
        mtl_lines.append(f"newmtl {mat}\n")
        for ln in b["lines"][1:]:
            ln = re.sub(r"map_Kd\s+[^\s#]+", f"map_Kd {out_tex2}", ln, flags=re.IGNORECASE)
            mtl_lines.append(ln)

    return lines, mtl_lines


def _axa4(axis, angle_rad):
    """4x4 rotation matrix (numpy) from axis (3,) and angle in radians."""
    a = np.asarray(axis, dtype=float)
    n = np.linalg.norm(a)
    if n < 1e-9:
        return np.eye(4)
    a = a / n
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    R = np.eye(3) + s * K + (1 - c) * (K @ K)
    out = np.eye(4)
    out[:3, :3] = R
    return out


def _rot_vec_to_vec(v_from, v_to):
    """4x4 rotation that rotates unit vector v_from to v_to."""
    v = np.asarray(v_from, dtype=float) / (np.linalg.norm(v_from) + 1e-9)
    u = np.asarray(v_to, dtype=float) / (np.linalg.norm(v_to) + 1e-9)
    d = np.dot(v, u)
    if d > 1 - 1e-6:
        return np.eye(4)
    if d < -1 + 1e-6:
        # 180 about any perpendicular
        perp = np.array([1, 0, 0]) if abs(v[0]) < 0.9 else np.array([0, 1, 0])
        ax = np.cross(v, perp)
        ax = ax / (np.linalg.norm(ax) + 1e-9)
        return _axa4(ax, np.pi)
    ax = np.cross(v, u)
    th = np.arccos(np.clip(d, -1, 1))
    return _axa4(ax, th)


def compute_align_transform_2point(p1, p2, q1, q2, extra_angle_rad=0.0):
    """2-point: map Q1->P1, Q2->P2 up to rotation about P1-P2. Returns 4x4.
    p1,p2: obj1 points; q1,q2: obj2 points. extra_angle_rad: rotation about P1-P2 axis."""
    p1, p2, q1, q2 = np.asarray(p1), np.asarray(p2), np.asarray(q1), np.asarray(q2)
    T_neg_q1 = np.eye(4)
    T_neg_q1[:3, 3] = -q1
    dq = np.linalg.norm(q2 - q1)
    dp = np.linalg.norm(p2 - p1)
    s = dp / (dq + 1e-9)
    S = np.eye(4)
    S[0, 0] = S[1, 1] = S[2, 2] = s
    v = (q2 - q1) / (dq + 1e-9)
    u = (p2 - p1) / (dp + 1e-9)
    R_align = _rot_vec_to_vec(v, u)
    axis = (p2 - p1) / (np.linalg.norm(p2 - p1) + 1e-9)
    R_axis = _axa4(axis, extra_angle_rad)
    T_p1 = np.eye(4)
    T_p1[:3, 3] = p1
    return T_p1 @ R_axis @ R_align @ S @ T_neg_q1


def compute_align_transform_3point(points1, points2):
    """3-point Procrustes: points1 and points2 are (3,3) or list of 3 [x,y,z]. Returns 4x4."""
    p = np.asarray(points1, dtype=float)
    q = np.asarray(points2, dtype=float)
    if p.shape[0] != 3 or q.shape[0] != 3:
        return np.eye(4)
    c1 = p.mean(axis=0)
    c2 = q.mean(axis=0)
    P = p - c1
    Q = q - c2
    sp = np.sqrt((P * P).sum() / 3)
    sq = np.sqrt((Q * Q).sum() / 3)
    scale = sp / (sq + 1e-9)
    H = P.T @ Q
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    R = R * scale
    out = np.eye(4)
    out[:3, :3] = R
    out[:3, 3] = c1 - R @ c2
    return out


def _build_triangle_indices(faces):
    """Build flat triangle index array from faces. Each face is triangulated as fan (0,i,i+1). Returns np.uint32."""
    indices = []
    for face in faces:
        fv = face.get("vertices", [])
        for i in range(1, len(fv) - 1):
            indices.append(fv[0])
            indices.append(fv[i])
            indices.append(fv[i + 1])
    return np.array(indices, dtype=np.uint32) if indices else np.zeros(0, dtype=np.uint32)


def _transform_vertices_numpy(vertices, M, n):
    """Batch transform: (N,3) = (M @ [v,1])[:3] + n. M: 4x4 or None (identity). n: (3,) or None (no add)."""
    v = np.asarray(vertices, dtype=np.float64)
    if v.size == 0:
        return v.reshape(0, 3) if v.ndim == 2 else np.zeros((0, 3))
    if v.ndim == 1:
        v = v.reshape(1, -1)
    if v.shape[1] < 3:
        return np.zeros((v.shape[0], 3))
    v = v[:, :3].copy()
    if M is not None:
        M = np.asarray(M, dtype=np.float64)
        one = np.ones((v.shape[0], 1))
        v_h = np.hstack([v, one])
        v = (M @ v_h.T).T[:, :3]
    if n is not None:
        v = v + np.asarray(n, dtype=np.float64)
    return v


def bbox_from_vertices(v):
    """Given list of [x,y,z] or (N,3) array, return (center, extents). extents = (sx,sy,sz)."""
    if v is None:
        return [0, 0, 0], [1, 1, 1]
    arr = np.array(v, dtype=float)
    if arr.size == 0:
        return [0, 0, 0], [1, 1, 1]
    mn = arr.min(axis=0)
    mx = arr.max(axis=0)
    center = ((mn + mx) / 2).tolist()
    extents = (mx - mn).tolist()
    for i in range(3):
        if extents[i] < 1e-6:
            extents[i] = 1.0
    return center, extents


def _build_vertex_to_faces(faces):
    out = {}
    for fi, face in enumerate(faces):
        for vi in face.get("vertices", []):
            out.setdefault(vi, []).append(fi)
    return out


def find_nearest_vertex_screen_space(mouse_x, mouse_y, width, height, vertices, faces, camera_pos, transform=None):
    """Find nearest front-facing vertex to the mouse. vertices/faces in model space; transform 4x4 or None.
    camera_pos in world. Uses gluProject and back-face filter (only vertices on a front-facing face)."""
    if not vertices or not faces:
        return None
    # Drawn coords for projection and front-facing
    M = np.asarray(transform, dtype=float) if transform is not None else None
    if M is not None:
        v_drawn = [(M @ [x, y, z, 1.0])[:3] for x, y, z in vertices]
    else:
        v_drawn = [[float(x), float(y), float(z)] for x, y, z in vertices]
    v2f = _build_vertex_to_faces(faces)
    cam = np.array(camera_pos, dtype=float)
    radius = 30.0
    best_dist = float("inf")
    best_vi = None

    modelview = (ctypes.c_double * 16)()
    projection = (ctypes.c_double * 16)()
    viewport = (ctypes.c_int * 4)()
    glGetDoublev(GL_MODELVIEW_MATRIX, modelview)
    glGetDoublev(GL_PROJECTION_MATRIX, projection)
    glGetIntegerv(GL_VIEWPORT, viewport)

    for vi, vw in enumerate(v_drawn):
        try:
            res = gluProject(vw[0], vw[1], vw[2], modelview, projection, viewport)
            if res is None:
                continue
            win_x, win_y, win_z = res
            sy = height - win_y
            if not (0.0 <= win_z <= 1.0 and 0 <= win_x <= width and 0 <= sy <= height):
                continue
            dx, dy = win_x - mouse_x, sy - mouse_y
            d2 = dx * dx + dy * dy
            if d2 >= radius * radius:
                continue
            # Front-facing: at least one adjacent face points toward camera
            view_dir = cam - np.array(vw, dtype=float)
            ln = np.linalg.norm(view_dir)
            if ln < 1e-9:
                continue
            view_dir = view_dir / ln
            pickable = False
            for fi in v2f.get(vi, []):
                f = faces[fi]
                fv = f.get("vertices", [])
                if len(fv) < 3:
                    continue
                a = np.array(v_drawn[fv[0]], dtype=float)
                b = np.array(v_drawn[fv[1]], dtype=float)
                c = np.array(v_drawn[fv[2]], dtype=float)
                n = np.cross(b - a, c - a)
                ln = np.linalg.norm(n)
                if ln < 1e-9:
                    continue
                n = n / ln
                if np.dot(n, view_dir) > 0:
                    pickable = True
                    break
            if not pickable:
                continue
            dist = np.sqrt(d2)
            if dist < best_dist:
                best_dist = dist
                best_vi = vi
        except Exception:
            continue
    return best_vi


def draw_mesh_filled(vertices, faces, transform=None, color=None, nudge=None, world_verts=None):
    """Draw mesh as wireframe (edges). vertices: list of [x,y,z]; faces: list of {vertices: [...]}.
    If world_verts (N,3) is provided, use it and ignore transform/nudge. Else transform/nudge as before. color: (r,g,b) or None."""
    if not faces:
        return
    if world_verts is not None:
        vs = world_verts
        if hasattr(vs, '__len__') and len(vs) == 0:
            return
    else:
        if not vertices:
            return
        vs = vertices
        if transform is not None:
            M = np.asarray(transform, dtype=float)
            n = np.asarray(nudge, dtype=float) if nudge is not None else None
            vs = _transform_vertices_numpy(vertices, M, n)
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
    r, g, b = color if color else (0.7, 0.7, 0.7)
    glColor3f(r, g, b)
    vs = np.asarray(vs, dtype=np.float64)
    if vs.ndim == 1:
        vs = vs.reshape(-1, 3)
    glBegin(GL_TRIANGLES)
    for face in faces:
        fv = face["vertices"]
        for i in range(1, len(fv) - 1):
            for j in (0, i, i + 1):
                idx = fv[j]
                if 0 <= idx < len(vs):
                    p = vs[idx]
                    glVertex3f(float(p[0]), float(p[1]), float(p[2]))
    glEnd()


def draw_vertices_on_surface(vertices, transform=None, nudge=None, point_size=4, color=(0.25, 0.5, 0.9), world_verts=None):
    """Draw each vertex as a small point. If world_verts (N,3) is provided, use it and ignore transform/nudge."""
    if world_verts is not None:
        vs = np.asarray(world_verts, dtype=np.float64)
        if vs.size == 0:
            return
        if vs.ndim == 1:
            vs = vs.reshape(-1, 3)
    else:
        if not vertices:
            return
        if transform is not None:
            M = np.asarray(transform, dtype=float)
            n = np.asarray(nudge, dtype=float) if nudge is not None else None
            vs = _transform_vertices_numpy(vertices, M, n)
        else:
            vs = np.asarray(vertices, dtype=np.float64)
            if vs.ndim == 1:
                vs = vs.reshape(-1, 3)
            vs = vs[:, :3] if vs.shape[1] >= 3 else vs
    glPointSize(point_size)
    glColor3f(color[0], color[1], color[2])
    glBegin(GL_POINTS)
    for p in vs:
        glVertex3f(float(p[0]), float(p[1]), float(p[2]))
    glEnd()


def draw_mesh_and_points_vbo(world_verts, faces, color, point_size, point_color, vbo_ibo_cache, cache_key, vbo_static, vbo_dirty):
    """Draw wireframe mesh and points using VBO/IBO. world_verts: (N,3) float32. vbo_ibo_cache: dict; cache_key: 'obj1'|'obj2'.
    vbo_static: if True, upload VBO only on first use. vbo_dirty: if True, re-upload VBO (for obj2 when transform changed)."""
    if world_verts is None or (isinstance(world_verts, np.ndarray) and world_verts.size == 0) or not faces:
        return
    arr = np.ascontiguousarray(world_verts, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 3:
        arr = arr.reshape(-1, 3) if arr.size >= 3 else np.zeros((0, 3), dtype=np.float32)
    else:
        arr = arr[:, :3].copy()
    nv = arr.shape[0]
    indices = _build_triangle_indices(faces)
    ni = len(indices)
    if cache_key not in vbo_ibo_cache:
        vbo = glGenBuffers(1)
        ibo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
        vbo_ibo_cache[cache_key] = {"vbo": vbo, "ibo": ibo, "num_indices": ni, "num_vertices": nv, "uploaded": False}
    ent = vbo_ibo_cache[cache_key]
    if "vbo" not in ent:
        ent["vbo"] = glGenBuffers(1)
        ent.setdefault("uploaded", False)
    vbo, ibo = ent["vbo"], ent["ibo"]
    do_upload = (vbo_static and not ent["uploaded"]) and nv > 0
    do_update = (not vbo_static and vbo_dirty) and nv > 0
    if do_upload:
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, arr.nbytes, arr, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        ent["uploaded"] = True
    elif do_update:
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        if not ent["uploaded"]:
            glBufferData(GL_ARRAY_BUFFER, arr.nbytes, arr, GL_DYNAMIC_DRAW)
            ent["uploaded"] = True
        else:
            glBufferSubData(GL_ARRAY_BUFFER, 0, arr.nbytes, arr)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
    ent["num_vertices"] = nv
    ent["num_indices"] = ni
    num_vertices = ent["num_vertices"]
    num_indices = ent["num_indices"]
    if num_indices == 0 and num_vertices == 0:
        return
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
    glColor3f(color[0], color[1], color[2])
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glVertexPointer(3, GL_FLOAT, 0, None)
    glEnableClientState(GL_VERTEX_ARRAY)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo)
    if num_indices > 0:
        glDrawElements(GL_TRIANGLES, num_indices, GL_UNSIGNED_INT, None)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
    _LOD_POINTS_MAX = 50000
    if num_vertices > 0 and num_vertices <= _LOD_POINTS_MAX:
        glPointSize(point_size)
        glColor3f(point_color[0], point_color[1], point_color[2])
        glDrawArrays(GL_POINTS, 0, num_vertices)
    glDisableClientState(GL_VERTEX_ARRAY)
    glBindBuffer(GL_ARRAY_BUFFER, 0)


def _world_verts(vertices, transform=None, nudge=None):
    """Return list of [x,y,z] in world space (transform and nudge applied if given)."""
    if transform is not None:
        M = np.asarray(transform, dtype=float)
        n = np.asarray(nudge, dtype=float) if nudge is not None else None
        if n is not None:
            return [((M @ [x, y, z, 1.0])[:3] + n).tolist() for x, y, z in vertices]
        return [(M @ [x, y, z, 1.0])[:3].tolist() for x, y, z in vertices]
    return [[float(x), float(y), float(z)] for x, y, z in vertices]


def draw_ground_plane(cx, cz, y_ground, size, color=(0.16, 0.16, 0.2)):
    """Draw a quad in the XZ plane at y=y_ground, centered at (cx, cz), half-width/half-depth = size."""
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
    glColor3f(color[0], color[1], color[2])
    glBegin(GL_QUADS)
    glVertex3f(cx - size, y_ground, cz - size)
    glVertex3f(cx + size, y_ground, cz - size)
    glVertex3f(cx + size, y_ground, cz + size)
    glVertex3f(cx - size, y_ground, cz + size)
    glEnd()


def draw_ground_shadow(vertices, faces, transform, nudge, y_ground, shadow_color=(0.06, 0.06, 0.1), world_verts=None, vbo_ibo_cache=None, cache_key=None):
    """Project mesh onto the XZ plane at y=y_ground (light from above) and draw as a dark shadow.
    If world_verts (N,3) is provided, use it instead of vertices/transform/nudge.
    If vbo_ibo_cache and cache_key are provided, use VBO+IBO (reuses mesh IBO)."""
    if not faces:
        return
    if world_verts is not None:
        proj = np.asarray(world_verts, dtype=np.float64)
        if proj.size == 0:
            return
        if proj.ndim == 1:
            proj = proj.reshape(-1, 3)
        proj = proj[:, :3].copy() if proj.shape[1] >= 3 else proj.copy()
    else:
        if not vertices:
            return
        vs = _world_verts(vertices, transform, nudge)
        proj = np.asarray(vs, dtype=np.float64)
        if proj.ndim == 1:
            proj = proj.reshape(-1, 3)
        proj = proj[:, :3].copy() if proj.shape[1] >= 3 else proj.copy()
    eps = 0.002
    proj[:, 1] = y_ground + eps
    nv, ni = proj.shape[0], len(_build_triangle_indices(faces))
    if nv == 0 or ni == 0:
        return
    _SHADOW_MAX_VERTS = 50000
    if nv > _SHADOW_MAX_VERTS:
        return
    if vbo_ibo_cache is not None and cache_key is not None:
        if cache_key not in vbo_ibo_cache:
            indices = _build_triangle_indices(faces)
            ibo = glGenBuffers(1)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo)
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
            vbo_ibo_cache[cache_key] = {"ibo": ibo, "num_indices": len(indices), "shadow_vbo": None, "shadow_uploaded": False}
        ent = vbo_ibo_cache[cache_key]
        ibo, num_indices = ent.get("ibo"), ent.get("num_indices", ni)
        if "shadow_vbo" not in ent or ent["shadow_vbo"] is None:
            ent["shadow_vbo"] = glGenBuffers(1)
        shadow_vbo = ent["shadow_vbo"]
        arr = np.ascontiguousarray(proj, dtype=np.float32)
        glBindBuffer(GL_ARRAY_BUFFER, shadow_vbo)
        if not ent.get("shadow_uploaded", False):
            glBufferData(GL_ARRAY_BUFFER, arr.nbytes, arr, GL_DYNAMIC_DRAW)
            ent["shadow_uploaded"] = True
        else:
            glBufferSubData(GL_ARRAY_BUFFER, 0, arr.nbytes, arr)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glColor3f(shadow_color[0], shadow_color[1], shadow_color[2])
        glBindBuffer(GL_ARRAY_BUFFER, shadow_vbo)
        glVertexPointer(3, GL_FLOAT, 0, None)
        glEnableClientState(GL_VERTEX_ARRAY)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo)
        glDrawElements(GL_TRIANGLES, num_indices, GL_UNSIGNED_INT, None)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
        glDisableClientState(GL_VERTEX_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        return
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
    glColor3f(shadow_color[0], shadow_color[1], shadow_color[2])
    glBegin(GL_TRIANGLES)
    for face in faces:
        fv = face["vertices"]
        for i in range(1, len(fv) - 1):
            for j in (0, i, i + 1):
                idx = fv[j]
                if 0 <= idx < len(proj):
                    q = proj[idx]
                    glVertex3f(float(q[0]), float(q[1]), float(q[2]))
    glEnd()


def run_glfw_viewer(gui, mode):
    """Run GLFW 3D viewer in a thread. mode in ('obj1','obj2','combined').
    One at a time: closes existing viewer_window before opening. Draws filled mesh, depth test, camera."""
    if not GLFW_AVAILABLE:
        messagebox.showerror("Error", "PyOpenGL/glfw not available. pip install PyOpenGL glfw")
        return

    # One-at-a-time: close existing viewer
    if gui.viewer_window is not None:
        try:
            glfw.set_window_should_close(gui.viewer_window, True)
        except Exception:
            pass
        gui.viewer_window = None
    if gui.viewer_thread is not None and gui.viewer_thread.is_alive():
        gui.viewer_thread.join(timeout=2.0)
    gui.viewer_thread = None

    def worker():
        if not glfw.init():
            gui.root.after(0, lambda: messagebox.showerror("Error", "Failed to initialize GLFW"))
            return
        try:
            win_w, win_h = 800, 600
            title = {"obj1": "OBJ_1", "obj2": "OBJ_2", "combined": "Combined"}.get(mode, "Viewer")
            window = glfw.create_window(win_w, win_h, f"OBJ Combiner v2 — {title}  [1=3D 2=Top 3=Front 4=Side]", None, None)
            if not window:
                gui.root.after(0, lambda: messagebox.showerror("Error", "Failed to create GLFW window"))
                return
            glfw.make_context_current(window)
            gui.viewer_window = window

            glEnable(GL_DEPTH_TEST)
            glClearColor(0.2, 0.2, 0.25, 1.0)

            # Choose data and transform for this mode
            if mode == "obj1":
                data = gui.obj1_data
                transform = None
            elif mode == "obj2":
                data = gui.obj2_data
                transform = gui.obj2_transform
            else:
                data = None
                transform = None

            if data is None or not data.get("v"):
                # placeholder: draw nothing meaningful; still need bbox for camera
                center, extents = [0, 0, 0], [1, 1, 1]
                verts, faces = [], []
            else:
                verts, faces = data["v"], data["faces"]
                center, extents = bbox_from_vertices(verts)

            max_ext = max(extents) if extents else 1.0
            camera_distance = max_ext * 2.5
            cam_x = center[0]
            cam_y = center[1]
            cam_z = center[2] + camera_distance
            rotation_x, rotation_y = 0.0, 0.0
            zoom, pan_x, pan_y = 1.0, 0.0, 0.0
            pan_world_y, pan_world_z = 0.0, 0.0
            last_x, last_y = 0.0, 0.0
            mouse_down = False
            mouse_button = None
            click_start_x, click_start_y = 0.0, 0.0
            mouse_has_moved = False
            pending_click_ref = [None]  # [ (x_fb, y_fb, w, h) ] or [None]
            ortho_view_ref = [None]  # None | 'xy' (top) | 'yz' (front) | 'xz' (side)

            def on_mouse(w, b, a, mods):
                nonlocal mouse_down, mouse_button, click_start_x, click_start_y, mouse_has_moved, last_x, last_y
                if a == glfw.PRESS:
                    mouse_down = True
                    mouse_button = b
                    mouse_has_moved = False
                    click_start_x, click_start_y = glfw.get_cursor_pos(w)
                    last_x, last_y = click_start_x, click_start_y
                    if b == glfw.MOUSE_BUTTON_LEFT and mode in ("obj1", "obj2"):
                        xpos, ypos = glfw.get_cursor_pos(w)
                        fw, fh = glfw.get_framebuffer_size(w)
                        ww, wh = glfw.get_window_size(w)
                        if ww and wh:
                            x_fb = xpos * (fw / ww)
                            y_fb = ypos * (fh / wh)
                        else:
                            x_fb, y_fb = xpos, ypos
                        pending_click_ref[0] = (x_fb, y_fb, fw, fh)
                elif a == glfw.RELEASE:
                    mouse_down = False
                    mouse_button = None

            def on_cursor(w, x, y):
                nonlocal last_x, last_y, rotation_x, rotation_y, pan_x, pan_y, mouse_has_moved, click_start_x, click_start_y
                if mouse_down:
                    if not mouse_has_moved:
                        d = np.sqrt((x - click_start_x) ** 2 + (y - click_start_y) ** 2)
                        if d > 5.0:
                            mouse_has_moved = True
                            if mouse_button == glfw.MOUSE_BUTTON_LEFT:
                                pending_click_ref[0] = None
                    if mouse_has_moved:
                        dx, dy = x - last_x, y - last_y
                        if abs(dx) > 0.1 or abs(dy) > 0.1:
                            if mouse_button == glfw.MOUSE_BUTTON_RIGHT:
                                s = max_ext * 0.001 * zoom
                                pan_x += dx * s
                                pan_y -= dy * s
                            elif mouse_button == glfw.MOUSE_BUTTON_LEFT:
                                rotation_y += dx * 0.5
                                rotation_x += dy * 0.5
                last_x, last_y = x, y

            def on_scroll(w, xo, yo):
                nonlocal zoom
                if yo != 0:
                    f = 1.15 if yo > 0 else 1.0 / 1.15
                    zoom *= f
                    zoom = max(0.05, min(20.0, zoom))

            def on_key(w, key, sc, a, mods):
                nonlocal pan_world_y, pan_world_z
                if a != glfw.PRESS:
                    return
                if key == glfw.KEY_ESCAPE:
                    glfw.set_window_should_close(w, True)
                    return
                # Arrows move view (Up/Down=Y, Left/Right=Z) when arrow_keys_move_view
                if getattr(gui, "arrow_keys_move_view", False):
                    step = max_ext / 10.0
                    if key == glfw.KEY_UP:
                        pan_world_y += step
                        return
                    if key == glfw.KEY_DOWN:
                        pan_world_y -= step
                        return
                    if key == glfw.KEY_LEFT:
                        pan_world_z -= step
                        return
                    if key == glfw.KEY_RIGHT:
                        pan_world_z += step
                        return
                # View: 1=Perspective, 2=Top (XY), 3=Front (YZ), 4=Side (XZ)
                if key == glfw.KEY_1:
                    ortho_view_ref[0] = None
                    return
                if key == glfw.KEY_2:
                    ortho_view_ref[0] = "xy"
                    return
                if key == glfw.KEY_3:
                    ortho_view_ref[0] = "yz"
                    return
                if key == glfw.KEY_4:
                    ortho_view_ref[0] = "xz"
                    return
                if mode not in ("obj2", "combined"):
                    return
                # Up/Down: 2-point rotation about P1-P2 axis
                if len(gui.picked1) == 2 and len(gui.picked2) == 2 and gui.obj1_data and gui.obj2_data:
                    v1, v2 = gui.obj1_data["v"], gui.obj2_data["v"]
                    p1, p2 = v1[gui.picked1[0]], v1[gui.picked1[1]]
                    q1, q2 = v2[gui.picked2[0]], v2[gui.picked2[1]]
                    if key == glfw.KEY_UP:
                        gui.obj2_extra_angle += 5.0 * np.pi / 180.0
                        gui.obj2_transform = compute_align_transform_2point(p1, p2, q1, q2, gui.obj2_extra_angle)
                    elif key == glfw.KEY_DOWN:
                        gui.obj2_extra_angle -= 5.0 * np.pi / 180.0
                        gui.obj2_transform = compute_align_transform_2point(p1, p2, q1, q2, gui.obj2_extra_angle)
                # X/Y/Z + Left/Right nudge
                if len(gui.picked1) >= 2 and gui.obj1_data:
                    v1 = gui.obj1_data["v"]
                    pa = np.array(v1[gui.picked1[0]])
                    pb = np.array(v1[gui.picked1[1]])
                    step = float(np.linalg.norm(pb - pa)) / 10.0
                    sign = 1.0 if key == glfw.KEY_RIGHT else -1.0 if key == glfw.KEY_LEFT else None
                    if sign is not None:
                        if glfw.get_key(w, glfw.KEY_X) == glfw.PRESS:
                            gui.obj2_nudge[0] += sign * step
                        elif glfw.get_key(w, glfw.KEY_Y) == glfw.PRESS:
                            gui.obj2_nudge[1] += sign * step
                        elif glfw.get_key(w, glfw.KEY_Z) == glfw.PRESS:
                            gui.obj2_nudge[2] += sign * step

            glfw.set_mouse_button_callback(window, on_mouse)
            glfw.set_cursor_pos_callback(window, on_cursor)
            glfw.set_scroll_callback(window, on_scroll)
            glfw.set_key_callback(window, on_key)
            glfw.swap_interval(1)
            num_tex_cache = {}
            # Cache for bbox and obj2 world verts; recompute only when transform/nudge/data change
            cached_obj1_valid = False
            cached_obj1_center, cached_obj1_extents, cached_obj1_max_ext = None, None, None
            prev_obj2_transform = None
            prev_obj2_nudge = None
            cached_center, cached_extents, cached_max_ext = None, None, None
            cached_vs2 = None
            vbo_ibo_cache = {}
            obj2_vbo_dirty = False

            while not glfw.window_should_close(window) and not getattr(gui, "close_viewer_flag", False):
                glfw.poll_events()
                if getattr(gui, "close_viewer_flag", False):
                    glfw.set_window_should_close(window, True)
                    break

                # Zoom to fit OBJ_2 when Align was run (obj2 or combined viewer)
                skip_bbox_update = False
                if getattr(gui, "viewer_zoom_to_fit_obj2", False) and mode in ("obj2", "combined") and gui.obj2_data:
                    tr_z = gui.obj2_transform
                    M_z = np.eye(4) if tr_z is None else np.asarray(tr_z, dtype=float)
                    n_z = np.asarray(gui.obj2_nudge, dtype=float)
                    vs2 = _transform_vertices_numpy(gui.obj2_data["v"], M_z, n_z)
                    if mode == "combined" and gui.obj1_data and gui.obj1_data.get("v"):
                        vs1 = np.asarray(gui.obj1_data["v"], dtype=np.float64)
                        if vs1.ndim == 1:
                            vs1 = vs1.reshape(1, -1)
                        vs1 = vs1[:, :3] if vs1.shape[1] >= 3 else np.zeros((vs1.shape[0], 3))
                        center, extents = bbox_from_vertices(np.vstack([vs1, vs2]))
                    else:
                        center, extents = bbox_from_vertices(vs2)
                    max_ext = max(extents) if extents else 1.0
                    camera_distance = max_ext * 2.5
                    zoom = 1.0
                    pan_x, pan_y = 0.0, 0.0
                    pan_world_y, pan_world_z = 0.0, 0.0
                    rotation_x, rotation_y = 0.0, 0.0
                    cam_x, cam_y, cam_z = center[0], center[1], center[2] + camera_distance
                    ortho_view_ref[0] = None
                    gui.viewer_zoom_to_fit_obj2 = False
                    skip_bbox_update = True
                    cached_vs2 = vs2
                    prev_obj2_transform = np.array(tr_z, copy=True) if tr_z is not None else None
                    prev_obj2_nudge = np.array(gui.obj2_nudge, copy=True)
                    cached_center, cached_extents, cached_max_ext = center, extents, max_ext
                    obj2_vbo_dirty = True

                # Refresh data/transform and bbox (unless zoom-to-fit just ran)
                if mode == "obj1":
                    d = gui.obj1_data
                    tr = None
                elif mode == "obj2":
                    d = gui.obj2_data
                    tr = gui.obj2_transform
                else:
                    d = None
                    tr = gui.obj2_transform
                if not skip_bbox_update:
                    if mode == "obj1" and d and d.get("v"):
                        verts, faces = d["v"], d["faces"]
                        if not cached_obj1_valid:
                            center, extents = bbox_from_vertices(verts)
                            max_ext = max(extents)
                            cached_obj1_valid = True
                            cached_obj1_center, cached_obj1_extents, cached_obj1_max_ext = center, extents, max_ext
                        else:
                            center, extents, max_ext = cached_obj1_center, cached_obj1_extents, cached_obj1_max_ext
                    elif mode == "obj2" and d and d.get("v"):
                        verts, faces = d["v"], d["faces"]
                        nudge = np.asarray(gui.obj2_nudge, dtype=float)
                        tr_eq = (tr is None and prev_obj2_transform is None) or (tr is not None and prev_obj2_transform is not None and np.array_equal(np.asarray(tr), prev_obj2_transform))
                        nudge_eq = prev_obj2_nudge is not None and np.allclose(nudge, prev_obj2_nudge)
                        if not (tr_eq and nudge_eq):
                            M = np.eye(4) if tr is None else np.asarray(tr, dtype=float)
                            cached_vs2 = _transform_vertices_numpy(verts, M, nudge)
                            center, extents = bbox_from_vertices(cached_vs2)
                            max_ext = max(extents)
                            prev_obj2_transform = np.array(tr, copy=True) if tr is not None else None
                            prev_obj2_nudge = np.array(nudge, copy=True)
                            cached_center, cached_extents, cached_max_ext = center, extents, max_ext
                            obj2_vbo_dirty = True
                        else:
                            center, extents, max_ext = cached_center, cached_extents, cached_max_ext
                    elif mode == "combined" and gui.obj1_data and gui.obj2_data:
                        nudge = np.asarray(gui.obj2_nudge, dtype=float)
                        tr_eq = (tr is None and prev_obj2_transform is None) or (tr is not None and prev_obj2_transform is not None and np.array_equal(np.asarray(tr), prev_obj2_transform))
                        nudge_eq = prev_obj2_nudge is not None and np.allclose(nudge, prev_obj2_nudge)
                        if not (tr_eq and nudge_eq):
                            M = np.eye(4) if tr is None else np.asarray(tr, dtype=float)
                            cached_vs2 = _transform_vertices_numpy(gui.obj2_data["v"], M, nudge)
                            vs1 = np.asarray(gui.obj1_data["v"], dtype=np.float64)
                            if vs1.ndim == 1:
                                vs1 = vs1.reshape(1, -1)
                            vs1 = vs1[:, :3] if vs1.shape[1] >= 3 else np.zeros((vs1.shape[0], 3))
                            center, extents = bbox_from_vertices(np.vstack([vs1, cached_vs2]))
                            max_ext = max(extents)
                            prev_obj2_transform = np.array(tr, copy=True) if tr is not None else None
                            prev_obj2_nudge = np.array(nudge, copy=True)
                            cached_center, cached_extents, cached_max_ext = center, extents, max_ext
                            obj2_vbo_dirty = True
                        else:
                            center, extents, max_ext = cached_center, cached_extents, cached_max_ext

                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
                width, height = glfw.get_framebuffer_size(window)
                glViewport(0, 0, width, height)
                asp = width / height if height > 0 else 1.0
                ortho = ortho_view_ref[0]

                glMatrixMode(GL_PROJECTION)
                glLoadIdentity()
                if ortho is not None:
                    ortho_size = max_ext * 1.2 / zoom
                    if asp >= 1.0:
                        left, right = -ortho_size * asp, ortho_size * asp
                        bottom, top = -ortho_size, ortho_size
                    else:
                        left, right = -ortho_size, ortho_size
                        bottom, top = -ortho_size / asp, ortho_size / asp
                    glOrtho(left, right, bottom, top, -max_ext * 10, max_ext * 10)
                else:
                    gluPerspective(45.0, asp, 0.1, camera_distance * 10)

                glMatrixMode(GL_MODELVIEW)
                glLoadIdentity()
                if ortho == "xy":
                    eye = [center[0] + pan_x, center[1] + pan_y + pan_world_y, center[2] + camera_distance + pan_world_z]
                    gluLookAt(eye[0], eye[1], eye[2], center[0] + pan_x, center[1] + pan_y + pan_world_y, center[2] + pan_world_z, 0, 1, 0)
                elif ortho == "yz":
                    eye = [center[0] + camera_distance, center[1] + pan_y + pan_world_y, center[2] + pan_x + pan_world_z]
                    gluLookAt(eye[0], eye[1], eye[2], center[0], center[1] + pan_y + pan_world_y, center[2] + pan_x + pan_world_z, 0, 1, 0)
                elif ortho == "xz":
                    eye = [center[0] + pan_x, center[1] + camera_distance + pan_world_y, center[2] + pan_y + pan_world_z]
                    gluLookAt(eye[0], eye[1], eye[2], center[0] + pan_x, center[1] + pan_world_y, center[2] + pan_y + pan_world_z, 0, 0, -1)
                else:
                    cd = camera_distance / zoom
                    dx = cam_x - center[0]
                    dy = cam_y - center[1]
                    dz = cam_z - center[2]
                    dlen = np.sqrt(dx * dx + dy * dy + dz * dz) or 0.001
                    dx, dy, dz = dx / dlen * cd, dy / dlen * cd, dz / dlen * cd
                    eye = [center[0] + dx + pan_x, center[1] + dy + pan_y + pan_world_y, center[2] + dz + pan_world_z]
                    gluLookAt(eye[0], eye[1], eye[2], center[0] + pan_x, center[1] + pan_y + pan_world_y, center[2] + pan_world_z, 0, 1, 0)
                    glTranslatef(center[0], center[1], center[2])
                    glRotatef(rotation_x, 1, 0, 0)
                    glRotatef(rotation_y, 0, 1, 0)
                    glTranslatef(-center[0], -center[1], -center[2])

                # Ground plane and projected shadows (all views)
                y_ground = center[1] - max_ext
                size = max(1.5 * max_ext, 0.1)
                draw_ground_plane(center[0], center[2], y_ground, size)
                if mode == "combined":
                    if gui.obj1_data and gui.obj1_data.get("v"):
                        wv1 = np.asarray(gui.obj1_data["v"], dtype=np.float64).reshape(-1, 3)[:, :3]
                        draw_ground_shadow(gui.obj1_data["v"], gui.obj1_data["faces"], None, None, y_ground, shadow_color=(0.08, 0.06, 0.06), world_verts=wv1, vbo_ibo_cache=vbo_ibo_cache, cache_key="obj1")
                    if gui.obj2_data and gui.obj2_data.get("v") and cached_vs2 is not None:
                        draw_ground_shadow(gui.obj2_data["v"], gui.obj2_data["faces"], gui.obj2_transform, gui.obj2_nudge, y_ground, shadow_color=(0.06, 0.06, 0.1), world_verts=cached_vs2, vbo_ibo_cache=vbo_ibo_cache, cache_key="obj2")
                else:
                    if verts and faces:
                        wv = np.asarray(verts, dtype=np.float64).reshape(-1, 3) if tr is None else cached_vs2
                        if wv is not None and len(wv) > 0:
                            draw_ground_shadow(verts, faces, tr, gui.obj2_nudge if (tr is not None and mode == "obj2") else None, y_ground, world_verts=wv, vbo_ibo_cache=vbo_ibo_cache, cache_key="obj1" if tr is None else "obj2")

                # Process pending vertex pick (obj1/obj2 only; excludes back-facing)
                pc = pending_click_ref[0]
                if pc is not None and mode in ("obj1", "obj2") and verts and faces:
                    pending_click_ref[0] = None
                    cx, cy, cw, ch = pc
                    nearest = find_nearest_vertex_screen_space(cx, cy, cw, ch, verts, faces, eye, tr)
                    if nearest is not None:
                        lst = gui.picked1 if mode == "obj1" else gui.picked2
                        if len(lst) >= 3:
                            if mode == "obj1":
                                gui.picked1 = [nearest]
                            else:
                                gui.picked2 = [nearest]
                        elif nearest not in lst:
                            if mode == "obj1":
                                gui.picked1 = list(gui.picked1) + [nearest]
                            else:
                                gui.picked2 = list(gui.picked2) + [nearest]

                if mode == "combined":
                    if gui.obj1_data and gui.obj1_data.get("v"):
                        wv1 = np.asarray(gui.obj1_data["v"], dtype=np.float64).reshape(-1, 3)[:, :3]
                        draw_mesh_and_points_vbo(wv1, gui.obj1_data["faces"], (0.75, 0.6, 0.5), 4, (0.25, 0.5, 0.9), vbo_ibo_cache, "obj1", True, False)
                    if gui.obj2_data and gui.obj2_data.get("v") and cached_vs2 is not None:
                        draw_mesh_and_points_vbo(cached_vs2, gui.obj2_data["faces"], (0.5, 0.6, 0.75), 4, (0.25, 0.5, 0.9), vbo_ibo_cache, "obj2", False, obj2_vbo_dirty)
                        obj2_vbo_dirty = False
                else:
                    wv = np.asarray(verts, dtype=np.float64).reshape(-1, 3)[:, :3] if tr is None else cached_vs2
                    if wv is not None and len(wv) > 0:
                        if tr is None:
                            draw_mesh_and_points_vbo(wv, faces, (0.7, 0.7, 0.7), 4, (0.25, 0.5, 0.9), vbo_ibo_cache, "obj1", True, False)
                        else:
                            draw_mesh_and_points_vbo(wv, faces, (0.7, 0.7, 0.7), 4, (0.25, 0.5, 0.9), vbo_ibo_cache, "obj2", False, obj2_vbo_dirty)
                            obj2_vbo_dirty = False

                # Numbered selection markers: large points + 1,2,3 labels (obj1/obj2 only)
                picked = (gui.picked1 if mode == "obj1" else gui.picked2) if mode in ("obj1", "obj2") else []
                if picked and verts and mode in ("obj1", "obj2"):
                    tr_arr = np.asarray(tr, dtype=float) if tr is not None else None
                    nudge_arr = np.asarray(gui.obj2_nudge, dtype=float) if (mode == "obj2" and tr is not None) else None
                    positions = []
                    for idx in picked:
                        if 0 <= idx < len(verts):
                            v = verts[idx]
                            if tr_arr is not None:
                                p = (tr_arr @ [v[0], v[1], v[2], 1.0])[:3]
                                if nudge_arr is not None:
                                    p = (p + nudge_arr).tolist()
                                else:
                                    p = p.tolist()
                            else:
                                p = [float(v[0]), float(v[1]), float(v[2])]
                            positions.append(p)
                    # Large points
                    glPointSize(12.0)
                    glColor3f(1.0, 1.0, 0.0)
                    glBegin(GL_POINTS)
                    for p in positions:
                        glVertex3f(p[0], p[1], p[2])
                    glEnd()
                    # Project to screen for labels
                    modelview = (ctypes.c_double * 16)()
                    projection = (ctypes.c_double * 16)()
                    viewport = (ctypes.c_int * 4)()
                    glGetDoublev(GL_MODELVIEW_MATRIX, modelview)
                    glGetDoublev(GL_PROJECTION_MATRIX, projection)
                    glGetIntegerv(GL_VIEWPORT, viewport)
                    labels_xy = []
                    for i, pos in enumerate(positions):
                        if i >= 3:
                            break
                        res = gluProject(pos[0], pos[1], pos[2], modelview, projection, viewport)
                        if res is not None:
                            sx, wy, _ = res
                            labels_xy.append((i + 1, sx, height - wy))
                    # 2D overlay for number labels
                    if labels_xy and PIL_AVAILABLE:
                        glDisable(GL_DEPTH_TEST)
                        glEnable(GL_BLEND)
                        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                        glEnable(GL_TEXTURE_2D)
                        glMatrixMode(GL_PROJECTION)
                        glPushMatrix()
                        glLoadIdentity()
                        glOrtho(0, width, height, 0, -1, 1)
                        glMatrixMode(GL_MODELVIEW)
                        glPushMatrix()
                        glLoadIdentity()
                        for rank, sx, sy in labels_xy:
                            key = str(rank)
                            if key not in num_tex_cache:
                                num_tex_cache[key] = _make_number_texture(key)
                            tid = num_tex_cache.get(key)
                            if tid is not None:
                                glBindTexture(GL_TEXTURE_2D, tid)
                                glColor4f(1, 1, 1, 1)
                                glBegin(GL_QUADS)
                                for (tx, ty, vx, vy) in [(0, 0, sx - 12, sy - 12), (1, 0, sx + 12, sy - 12), (1, 1, sx + 12, sy + 12), (0, 1, sx - 12, sy + 12)]:
                                    glTexCoord2f(tx, ty)
                                    glVertex2f(vx, vy)
                                glEnd()
                        glPopMatrix()
                        glMatrixMode(GL_PROJECTION)
                        glPopMatrix()
                        glMatrixMode(GL_MODELVIEW)
                        glDisable(GL_TEXTURE_2D)
                        glEnable(GL_DEPTH_TEST)

                glfw.swap_buffers(window)

            glfw.destroy_window(window)
        finally:
            gui.viewer_window = None
            gui.close_viewer_flag = False
            try:
                glfw.terminate()
            except Exception:
                pass

    gui.viewer_thread = threading.Thread(target=worker, daemon=True)
    gui.viewer_thread.start()


class CombinerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("OBJ Combiner v2")
        self.root.geometry("600x400")
        self.obj1_path = None
        self.obj2_path = None
        self.obj1_data = None
        self.obj2_data = None
        self.blocks1 = None
        self.blocks2 = None
        self.tex1_path = None
        self.tex2_path = None
        self.picked1 = []
        self.picked2 = []
        self.obj2_transform = None
        self.obj2_extra_angle = 0.0
        self.obj2_nudge = np.array([0.0, 0.0, 0.0], dtype=float)
        self.viewer_window = None
        self.viewer_thread = None
        self.close_viewer_flag = False
        self.viewer_zoom_to_fit_obj2 = False
        self.arrow_keys_move_view = False  # True=arrows move view (Y/Z), False=arrows move OBJ_2
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

        # Viewers (one at a time)
        vf = ttk.Frame(main)
        vf.pack(fill=tk.X, pady=5)
        ttk.Label(vf, text="View:").pack(side=tk.LEFT, padx=(0, 5))
        self.btn_view1 = ttk.Button(vf, text="View OBJ_1", command=self.view_obj1)
        self.btn_view1.pack(side=tk.LEFT, padx=2)
        self.btn_view2 = ttk.Button(vf, text="View OBJ_2", command=self.view_obj2)
        self.btn_view2.pack(side=tk.LEFT, padx=2)
        self.btn_view_combined = ttk.Button(vf, text="View Combined", command=self.view_combined)
        self.btn_view_combined.pack(side=tk.LEFT, padx=2)

        # Align and Arrows mode
        af = ttk.Frame(main)
        af.pack(fill=tk.X, pady=2)
        self.btn_align = ttk.Button(af, text="Align", command=self.run_align)
        self.btn_align.pack(side=tk.LEFT, padx=2)
        self.btn_arrows = ttk.Button(af, text="Arrows: OBJ_2", command=self.toggle_arrow_keys_mode)
        self.btn_arrows.pack(side=tk.LEFT, padx=2)
        ttk.Button(af, text="Reset points", command=self.reset_points).pack(side=tk.LEFT, padx=2)

        # Buttons
        bf = ttk.Frame(main)
        bf.pack(fill=tk.X, pady=10)
        ttk.Button(bf, text="Combine and Save", command=self.combine_and_save).pack(side=tk.LEFT, padx=5)
        ttk.Button(bf, text="Clear", command=self.clear).pack(side=tk.LEFT, padx=5)

    def select_obj1(self):
        p = filedialog.askopenfilename(title="Select OBJ 1", filetypes=[("OBJ files", "*.obj"), ("All", "*.*")])
        if p:
            self.obj1_path = p
            self.obj1_data = None
            self.blocks1 = None
            self.tex1_path = None
            self.picked1 = []
            self.l1.config(text=os.path.basename(p), foreground="black")

    def select_obj2(self):
        p = filedialog.askopenfilename(title="Select OBJ 2", filetypes=[("OBJ files", "*.obj"), ("All", "*.*")])
        if p:
            self.obj2_path = p
            self.obj2_data = None
            self.blocks2 = None
            self.tex2_path = None
            self.picked2 = []
            self.obj2_transform = None
            self.obj2_extra_angle = 0.0
            self.obj2_nudge = np.array([0.0, 0.0, 0.0], dtype=float)
            self.l2.config(text=os.path.basename(p), foreground="black")

    def clear(self):
        self.obj1_path = self.obj2_path = None
        self.obj1_data = self.obj2_data = None
        self.blocks1 = self.blocks2 = None
        self.tex1_path = self.tex2_path = None
        self.picked1 = []
        self.picked2 = []
        self.obj2_transform = None
        self.obj2_extra_angle = 0.0
        self.obj2_nudge = np.array([0.0, 0.0, 0.0], dtype=float)
        self.viewer_zoom_to_fit_obj2 = False
        self.arrow_keys_move_view = False
        self._update_arrows_button()
        self.l1.config(text="No file selected", foreground="gray")
        self.l2.config(text="No file selected", foreground="gray")

    def _ensure_obj1_loaded(self):
        if not self.obj1_path:
            return False, "Select OBJ 1 first."
        if self.obj1_data is not None:
            return True, None
        try:
            mtl1, tex1 = validate_obj_mtl_texture(self.obj1_path, "OBJ 1")
            self.obj1_data = parse_obj_full(self.obj1_path)
            self.blocks1 = parse_mtl_materials(mtl1)
            self.tex1_path = tex1
            return True, None
        except (ValueError, RuntimeError) as e:
            return False, str(e)

    def _ensure_obj2_loaded(self):
        if not self.obj2_path:
            return False, "Select OBJ 2 first."
        if self.obj2_data is not None:
            return True, None
        try:
            mtl2, tex2 = validate_obj_mtl_texture(self.obj2_path, "OBJ 2")
            self.obj2_data = parse_obj_full(self.obj2_path)
            self.blocks2 = parse_mtl_materials(mtl2)
            self.tex2_path = tex2
            return True, None
        except (ValueError, RuntimeError) as e:
            return False, str(e)

    def view_obj1(self):
        ok, err = self._ensure_obj1_loaded()
        if not ok:
            messagebox.showerror("Error", err)
            return
        run_glfw_viewer(self, "obj1")

    def view_obj2(self):
        ok, err = self._ensure_obj2_loaded()
        if not ok:
            messagebox.showerror("Error", err)
            return
        run_glfw_viewer(self, "obj2")

    def view_combined(self):
        ok1, e1 = self._ensure_obj1_loaded()
        ok2, e2 = self._ensure_obj2_loaded()
        if not ok1:
            messagebox.showerror("Error", e1 or "Select OBJ 1 first.")
            return
        if not ok2:
            messagebox.showerror("Error", e2 or "Select OBJ 2 first.")
            return
        run_glfw_viewer(self, "combined")

    def toggle_arrow_keys_mode(self):
        """Switch between: arrows move view (Up/Down=Y, Left/Right=Z) vs arrows move OBJ_2."""
        self.arrow_keys_move_view = not self.arrow_keys_move_view
        self._update_arrows_button()

    def _update_arrows_button(self):
        if hasattr(self, "btn_arrows") and self.btn_arrows.winfo_exists():
            self.btn_arrows.config(text="Arrows: View (Y/Z)" if self.arrow_keys_move_view else "Arrows: OBJ_2")

    def reset_points(self):
        """Clear all picked vertices on OBJ_1 and OBJ_2."""
        self.picked1 = []
        self.picked2 = []

    def run_align(self):
        """Compute obj2_transform from picked1 and picked2 (2- or 3-point)."""
        if len(self.picked1) < 2 or len(self.picked2) < 2:
            messagebox.showwarning("Align", "Pick at least 2 vertices on each model (View OBJ_1, View OBJ_2).")
            return
        if len(self.picked1) != len(self.picked2):
            messagebox.showwarning("Align", "Pick the same number of points on each (2 or 3).")
            return
        ok1, _ = self._ensure_obj1_loaded()
        ok2, _ = self._ensure_obj2_loaded()
        if not ok1 or not ok2:
            return
        v1, v2 = self.obj1_data["v"], self.obj2_data["v"]
        n = len(self.picked1)
        if n == 2:
            p1 = v1[self.picked1[0]]
            p2 = v1[self.picked1[1]]
            q1 = v2[self.picked2[0]]
            q2 = v2[self.picked2[1]]
            self.obj2_transform = compute_align_transform_2point(p1, p2, q1, q2, self.obj2_extra_angle)
        else:
            p = [v1[self.picked1[i]] for i in range(3)]
            q = [v2[self.picked2[i]] for i in range(3)]
            self.obj2_transform = compute_align_transform_3point(p, q)
        self.obj2_nudge = np.array([0.0, 0.0, 0.0], dtype=float)
        self.viewer_zoom_to_fit_obj2 = True
        messagebox.showinfo("Align", "Alignment applied. Use View Combined to fine-tune (Up/Down, X/Y/Z+arrows). Views: 1=3D, 2=Top, 3=Front, 4=Side.")

    def combine_and_save(self):
        """Save merged OBJ with _s2 naming: {base}_s2.obj, .mtl, _tex1.jpg, _tex2.jpg in OBJ_1's directory.
        Uses obj2_transform and obj2_nudge; if Align was not run, OBJ_2 is merged untransformed."""
        ok1, e1 = self._ensure_obj1_loaded()
        ok2, e2 = self._ensure_obj2_loaded()
        if not ok1:
            messagebox.showerror("Error", e1 or "Please select OBJ 1.")
            return
        if not ok2:
            messagebox.showerror("Error", e2 or "Please select OBJ 2.")
            return

        base = os.path.splitext(os.path.basename(self.obj1_path))[0]
        if base.endswith("_s1"):
            base = base[:-3]
        out_stem = base + "_s2"
        out_dir = os.path.dirname(self.obj1_path)
        out_tex1_name = out_stem + "_tex1.jpg"
        out_tex2_name = out_stem + "_tex2.jpg"

        data1 = self.obj1_data
        data2 = self.obj2_data
        M = np.asarray(self.obj2_transform, dtype=float) if self.obj2_transform is not None else np.eye(4)
        nudge = np.asarray(self.obj2_nudge, dtype=float)

        data2_mod = {
            **data2,
            "v": [((M @ [x, y, z, 1.0])[:3] + nudge).tolist() for x, y, z in data2["v"]],
        }

        obj_lines, mtl_lines = merge_objs(
            data1, data2_mod, self.blocks1, self.blocks2,
            "obj1", "obj2", out_tex1_name, out_tex2_name,
        )

        out_obj = os.path.join(out_dir, out_stem + ".obj")
        out_mtl = os.path.join(out_dir, out_stem + ".mtl")
        try:
            with open(out_obj, "w", encoding="utf-8") as f:
                f.write(f"mtllib {out_stem}.mtl\n")
                f.writelines(obj_lines)
            with open(out_mtl, "w", encoding="utf-8") as f:
                f.writelines(mtl_lines)
            shutil.copy2(self.tex1_path, os.path.join(out_dir, out_tex1_name))
            shutil.copy2(self.tex2_path, os.path.join(out_dir, out_tex2_name))
        except Exception as e:
            messagebox.showerror("Error", f"Failed to write: {e}")
            return

        messagebox.showinfo(
            "Success",
            f"Merged files saved to:\n{out_dir}\n\n"
            f"OBJ: {out_stem}.obj\n"
            f"MTL: {out_stem}.mtl\n"
            f"Textures: {out_tex1_name}, {out_tex2_name}",
        )


if __name__ == "__main__":
    root = tk.Tk()
    CombinerGUI(root)
    root.mainloop()
