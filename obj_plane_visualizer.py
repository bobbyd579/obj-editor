"""
OBJ Plane Distance Visualizer
Loads an OBJ file, displays it in 3D, allows selecting 3 vertices to define a plane,
and highlights vertices within a specified distance from the plane.
"""

# Set DPI awareness BEFORE importing tkinter to prevent scaling issues
# Use level 1 (Per Monitor DPI Aware) which works better with tkinter
try:
    import ctypes
    # Set to Per Monitor DPI Aware (level 1) for better tkinter compatibility
    # Level 2 can cause issues with tkinter on high-DPI displays
    # This must be done before any GUI libraries are imported
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)  # PROCESS_PER_MONITOR_DPI_AWARE
    except:
        # Fallback for older Windows
        ctypes.windll.user32.SetProcessDPIAware()
except (ImportError, AttributeError, OSError):
    pass

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import os
import threading
import time

try:
    import glfw
    from OpenGL.GL import *
    from OpenGL.GLU import *
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class OBJPlaneVisualizer:
    def __init__(self):
        self.obj_path = None
        self.vertices = []
        self.faces = []  # List of face data: [vertex_indices, texture_indices, normal_indices, format_string]
        self.obj_lines = []
        self.texture_coords = []  # Texture coordinates (vt)
        self.normals = []  # Vertex normals (vn)
        self.material_refs = []  # Material references (usemtl, mtllib) with line numbers
        self.bbox_min = None
        self.bbox_max = None
        self.bbox_center = None
        self.extents = None
        
        # Selection state
        self.selected_vertices = []  # List of 3 vertex indices
        self.plane_equation = None  # [a, b, c, d] for ax + by + cz + d = 0
        self.distance_threshold = 1.0  # Distance in OBJ units
        self.vertex_distances = None  # Distance from each vertex to plane
        self.vertices_within_threshold = set()  # Set of vertex indices within threshold
        
        # Visualization state
        self.opengl_window = None
        self.opengl_thread = None
        self.needs_redraw = True
        self.gui = None  # Reference to GUI for status updates
        self.reset_view_flag = False  # Flag to reset camera view
        self.view_xy_flag = False  # Flag to set XY plane view (top view)
        self.view_yz_flag = False  # Flag to set YZ plane view (front view)
        self.view_xz_flag = False  # Flag to set XZ plane view (side view)
        self.orthographic_view = None  # Current orthographic view: 'xy', 'yz', 'xz', or None for perspective
        self.pending_click = None  # (x, y, width, height) for screen-space selection
        
    def select_obj_file(self):
        """Use tkinter to select OBJ file"""
        root = tk.Tk()
        root.withdraw()
        
        file_path = filedialog.askopenfilename(
            title="Select OBJ file",
            filetypes=[("OBJ files", "*.obj"), ("All files", "*.*")]
        )
        
        root.destroy()
        
        if file_path:
            self.obj_path = file_path
            return True
        return False
    
    def parse_obj_file(self):
        """Parse OBJ file to extract vertices, faces, texture coordinates, and normals"""
        if not self.obj_path:
            return False
        
        self.vertices = []
        self.faces = []  # Will store full face data: (vertex_indices, texture_indices, normal_indices, original_format)
        self.texture_coords = []
        self.normals = []
        self.material_refs = []  # Store (line_number, type, value)
        self.obj_lines = []
        
        try:
            with open(self.obj_path, 'r', encoding='utf-8') as f:
                line_num = 0
                for line in f:
                    self.obj_lines.append(line)
                    line_stripped = line.strip()
                    line_num += 1
                    
                    if not line_stripped or line_stripped.startswith('#'):
                        continue
                    
                    parts = line_stripped.split()
                    if not parts:
                        continue
                    
                    if parts[0] == 'v':  # Vertex
                        if len(parts) >= 4:
                            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                            self.vertices.append([x, y, z])
                    
                    elif parts[0] == 'vt':  # Texture coordinate
                        if len(parts) >= 3:
                            u, v = float(parts[1]), float(parts[2])
                            w = float(parts[3]) if len(parts) >= 4 else 0.0
                            self.texture_coords.append([u, v, w])
                    
                    elif parts[0] == 'vn':  # Normal
                        if len(parts) >= 4:
                            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                            self.normals.append([x, y, z])
                    
                    elif parts[0] == 'f':  # Face - handle full format: v/vt/vn, v//vn, v/vt, or v
                        face_vertices = []
                        face_textures = []
                        face_normals = []
                        original_format = []
                        
                        for part in parts[1:]:
                            indices = part.split('/')
                            # Store original format for writing back
                            original_format.append(part)
                            
                            # Vertex index (required)
                            if indices[0]:
                                face_vertices.append(int(indices[0]) - 1)  # OBJ is 1-indexed
                            else:
                                continue  # Skip invalid face entries
                            
                            # Texture coordinate index (optional)
                            if len(indices) > 1 and indices[1]:
                                face_textures.append(int(indices[1]) - 1)
                            else:
                                face_textures.append(None)
                            
                            # Normal index (optional)
                            if len(indices) > 2 and indices[2]:
                                face_normals.append(int(indices[2]) - 1)
                            else:
                                face_normals.append(None)
                        
                        if face_vertices:
                            # Store face data: (vertex_indices, texture_indices, normal_indices, original_format_string)
                            self.faces.append({
                                'vertices': face_vertices,
                                'textures': face_textures,
                                'normals': face_normals,
                                'format': original_format,
                                'original_line': line_stripped
                            })
                    
                    elif parts[0] == 'usemtl':  # Material reference
                        if len(parts) >= 2:
                            self.material_refs.append((line_num - 1, 'usemtl', parts[1]))
                    
                    elif parts[0] == 'mtllib':  # MTL library reference
                        if len(parts) >= 2:
                            self.material_refs.append((line_num - 1, 'mtllib', ' '.join(parts[1:])))
            
            if not self.vertices:
                return False
            
            self.vertices = np.array(self.vertices)
            self.calculate_bounding_box()
            return True
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to parse OBJ file: {str(e)}")
            return False
    
    def calculate_bounding_box(self):
        """Calculate bounding box from vertices"""
        if len(self.vertices) == 0:
            return
        
        self.bbox_min = np.min(self.vertices, axis=0)
        self.bbox_max = np.max(self.vertices, axis=0)
        self.bbox_center = (self.bbox_min + self.bbox_max) / 2.0
        self.extents = self.bbox_max - self.bbox_min
    
    def delete_selected_vertices(self):
        """Delete all vertices within the plane threshold and faces that reference them, return deletion stats"""
        # Use vertices within threshold (the red ones) instead of just the 3 selected vertices
        if not self.vertices_within_threshold:
            return None, None, None
        
        # Convert to set for faster lookup
        vertices_to_delete = set(self.vertices_within_threshold)
        
        # Find faces that reference any deleted vertex
        faces_to_remove = []
        for i, face in enumerate(self.faces):
            face_vertex_indices = face['vertices']
            if any(v_idx in vertices_to_delete for v_idx in face_vertex_indices):
                faces_to_remove.append(i)
        
        # Remove faces (in reverse order to maintain indices)
        for i in reversed(faces_to_remove):
            self.faces.pop(i)
        
        # Create mapping from old vertex index to new index
        # Only keep vertices that are NOT being deleted
        old_to_new_vertex = {}
        new_vertices = []
        new_index = 0
        
        for old_index in range(len(self.vertices)):
            if old_index not in vertices_to_delete:
                old_to_new_vertex[old_index] = new_index
                new_vertices.append(self.vertices[old_index])
                new_index += 1
        
        # Update vertices
        self.vertices = np.array(new_vertices) if new_vertices else np.array([])
        
        # Update face vertex indices using the mapping
        # Remove faces that have no valid vertices after deletion
        faces_to_remove = []
        for face_idx, face in enumerate(self.faces):
            new_vertex_indices = []
            new_texture_indices = []
            new_normal_indices = []
            new_format = []
            
            for i, v_idx in enumerate(face['vertices']):
                if v_idx in old_to_new_vertex:
                    new_vertex_indices.append(old_to_new_vertex[v_idx])
                    new_texture_indices.append(face['textures'][i])
                    new_normal_indices.append(face['normals'][i])
                    # Reconstruct format string with new vertex index
                    orig_format = face['format'][i]
                    parts = orig_format.split('/')
                    if len(parts) > 0:
                        parts[0] = str(old_to_new_vertex[v_idx] + 1)  # OBJ is 1-indexed
                        new_format.append('/'.join(parts))
                    else:
                        # Fallback: just the vertex index
                        new_format.append(str(old_to_new_vertex[v_idx] + 1))
            
            # If face has less than 3 vertices after deletion, mark for removal
            if len(new_vertex_indices) < 3:
                faces_to_remove.append(face_idx)
            else:
                # Update face data
                face['vertices'] = new_vertex_indices
                face['textures'] = new_texture_indices
                face['normals'] = new_normal_indices
                face['format'] = new_format
        
        # Remove invalid faces (in reverse order)
        for face_idx in reversed(faces_to_remove):
            self.faces.pop(face_idx)
        
        # Recalculate bounding box
        if len(self.vertices) > 0:
            self.calculate_bounding_box()
        
        # Clear selection after deletion
        deleted_vertex_count = len(vertices_to_delete)
        deleted_face_count = len(faces_to_remove)
        remaining_vertex_count = len(self.vertices)
        
        self.selected_vertices = []
        self.plane_equation = None
        self.vertex_distances = None
        self.vertices_within_threshold = set()
        
        return deleted_vertex_count, deleted_face_count, remaining_vertex_count
    
    def write_obj_file(self, output_path):
        """Write OBJ file with current vertices, faces, texture coords, and normals"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                # Write header comment
                f.write("# OBJ file exported from OBJ Plane Visualizer\n")
                f.write(f"# Original file: {os.path.basename(self.obj_path)}\n\n")
                
                # Write MTL library references (if any) - write at the top
                for line_num, ref_type, value in self.material_refs:
                    if ref_type == 'mtllib':
                        f.write(f"mtllib {value}\n")
                
                # Write vertices
                for vertex in self.vertices:
                    f.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
                
                # Write texture coordinates (preserve all, even if unused)
                if self.texture_coords:
                    f.write("\n")
                    for uv in self.texture_coords:
                        if len(uv) >= 3 and uv[2] != 0.0:
                            f.write(f"vt {uv[0]:.6f} {uv[1]:.6f} {uv[2]:.6f}\n")
                        else:
                            f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
                
                # Write normals (preserve all, even if unused)
                if self.normals:
                    f.write("\n")
                    for normal in self.normals:
                        f.write(f"vn {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}\n")
                
                # Write faces with material references
                if self.faces:
                    f.write("\n")
                    current_material = None
                    
                    # Get usemtl references sorted by original line number
                    usemtl_refs = [(line_num, value) for line_num, ref_type, value in self.material_refs if ref_type == 'usemtl']
                    usemtl_refs.sort()
                    
                    # Estimate which face index corresponds to which material
                    # This is approximate since we don't track exact face-to-material mapping
                    # We'll write materials at the beginning and when they change
                    if usemtl_refs:
                        # Write first material at the start
                        _, first_mat = usemtl_refs[0]
                        f.write(f"usemtl {first_mat}\n")
                        current_material = first_mat
                    
                    # Write faces - reconstruct from components
                    for face_idx, face in enumerate(self.faces):
                        # Write face - use stored format if available, otherwise reconstruct
                        if 'format' in face and len(face['format']) == len(face['vertices']):
                            # Use stored format strings (already updated with new vertex indices)
                            face_str = "f"
                            for fmt_str in face['format']:
                                face_str += f" {fmt_str}"
                            f.write(face_str + "\n")
                        else:
                            # Fallback: reconstruct from components
                            face_str = "f"
                            for i in range(len(face['vertices'])):
                                v_idx = face['vertices'][i] + 1  # OBJ is 1-indexed
                                vt_idx = face['textures'][i]
                                vn_idx = face['normals'][i]
                                
                                if vt_idx is not None and vn_idx is not None:
                                    # Format: v/vt/vn
                                    face_str += f" {v_idx}/{vt_idx + 1}/{vn_idx + 1}"
                                elif vt_idx is not None:
                                    # Format: v/vt
                                    face_str += f" {v_idx}/{vt_idx + 1}"
                                elif vn_idx is not None:
                                    # Format: v//vn
                                    face_str += f" {v_idx}//{vn_idx + 1}"
                                else:
                                    # Format: v
                                    face_str += f" {v_idx}"
                            
                            f.write(face_str + "\n")
            
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Failed to write OBJ file: {str(e)}")
            return False
    
    def calculate_plane_from_points(self, p1, p2, p3):
        """Calculate plane equation from 3 points"""
        # Convert to numpy arrays
        p1 = np.array(p1)
        p2 = np.array(p2)
        p3 = np.array(p3)
        
        # Calculate two vectors in the plane
        v1 = p2 - p1
        v2 = p3 - p1
        
        # Calculate normal vector (cross product)
        normal = np.cross(v1, v2)
        
        # Check if points are collinear
        norm_length = np.linalg.norm(normal)
        if norm_length < 1e-10:
            return None  # Points are collinear
        
        # Normalize normal vector
        normal = normal / norm_length
        
        # Calculate d: ax + by + cz + d = 0, so d = -(ax + by + cz)
        d = -np.dot(normal, p1)
        
        # Return plane equation [a, b, c, d]
        return np.array([normal[0], normal[1], normal[2], d])
    
    def calculate_vertex_distances(self):
        """Calculate distance from each vertex to the plane"""
        if self.plane_equation is None:
            self.vertex_distances = None
            return
        
        a, b, c, d = self.plane_equation
        
        # Point-to-plane distance: |ax + by + cz + d| / sqrt(a² + b² + c²)
        # Since normal is normalized, denominator is 1.0
        distances = np.abs(
            a * self.vertices[:, 0] + 
            b * self.vertices[:, 1] + 
            c * self.vertices[:, 2] + d
        )
        
        self.vertex_distances = distances
        
        # Find vertices within threshold
        self.vertices_within_threshold = set(
            np.where(distances <= self.distance_threshold)[0]
        )
    
    def screen_to_world_ray(self, mouse_x, mouse_y, width, height, view_matrix, proj_matrix):
        """Convert screen coordinates to 3D ray"""
        # Normalize mouse coordinates to [-1, 1]
        x = (2.0 * mouse_x / width) - 1.0
        y = 1.0 - (2.0 * mouse_y / height)  # Flip Y
        
        # Create ray in clip space
        ray_clip = np.array([x, y, -1.0, 1.0])
        
        # Transform to eye space
        inv_proj = np.linalg.inv(proj_matrix)
        ray_eye = inv_proj @ ray_clip
        ray_eye = np.array([ray_eye[0], ray_eye[1], -1.0, 0.0])
        
        # Transform to world space
        inv_view = np.linalg.inv(view_matrix)
        ray_world = inv_view @ ray_eye
        ray_dir = ray_world[:3]
        ray_dir = ray_dir / np.linalg.norm(ray_dir)
        
        return ray_dir
    
    def find_nearest_vertex_screen_space(self, mouse_x, mouse_y, width, height):
        """Find nearest vertex to the mouse click using screen-space coordinates"""
        if len(self.vertices) == 0:
            return None
        
        # Selection radius in pixels - larger for easier selection
        selection_radius_pixels = 30.0  # 30 pixel radius
        
        min_screen_distance = float('inf')
        nearest_vertex = None
        
        # Get current OpenGL matrices and viewport
        from OpenGL.GL import glGetDoublev, GL_MODELVIEW_MATRIX, GL_PROJECTION_MATRIX, GL_VIEWPORT
        from OpenGL.GLU import gluProject
        import ctypes
        
        # Get matrices and viewport
        modelview = (ctypes.c_double * 16)()
        projection = (ctypes.c_double * 16)()
        viewport = (ctypes.c_int * 4)()
        
        glGetDoublev(GL_MODELVIEW_MATRIX, modelview)
        glGetDoublev(GL_PROJECTION_MATRIX, projection)
        glGetIntegerv(GL_VIEWPORT, viewport)
        
        # Convert viewport to tuple for gluProject
        vp_tuple = tuple(viewport)
        vp_x, vp_y, vp_width, vp_height = vp_tuple
        
        for i, vertex in enumerate(self.vertices):
            # Use gluProject for accurate projection
            # PyOpenGL's gluProject returns a tuple (winX, winY, winZ) or None on failure
            try:
                result = gluProject(
                    vertex[0],
                    vertex[1],
                    vertex[2],
                    modelview,
                    projection,
                    viewport
                )
                
                if result is not None:  # Projection successful
                    win_x, win_y, win_z = result
                    
                    # gluProject returns coordinates in viewport space
                    # win_x, win_y are in viewport coordinates (Y from bottom)
                    # mouse_x, mouse_y are in window coordinates (Y from top)
                    screen_x = win_x
                    # Convert gluProject Y (bottom origin) to window Y (top origin)
                    screen_y = height - win_y
                    
                    # Check if vertex is in front of camera and in valid depth range
                    # win_z is in [0, 1] range, where 0 is near plane, 1 is far plane
                    if 0.0 <= win_z <= 1.0:
                        # Check if vertex is within window bounds (not just viewport)
                        if 0 <= screen_x <= width and 0 <= screen_y <= height:
                            # Calculate distance in screen space (pixels)
                            dx = screen_x - mouse_x
                            dy = screen_y - mouse_y
                            screen_distance = np.sqrt(dx * dx + dy * dy)
                            
                            # If within selection radius and closer than previous best
                            if screen_distance < selection_radius_pixels and screen_distance < min_screen_distance:
                                min_screen_distance = screen_distance
                                nearest_vertex = i
            except:
                # If gluProject fails, skip this vertex
                continue
        
        return nearest_vertex
    
    def run_opengl_window(self):
        """Run the OpenGL visualization window"""
        if not glfw.init():
            messagebox.showerror("Error", "Failed to initialize GLFW")
            return
        
        window = glfw.create_window(1024, 768, "OBJ Plane Visualizer - Click 3 vertices to define plane", None, None)
        if not window:
            glfw.terminate()
            messagebox.showerror("Error", "Failed to create GLFW window")
            return
        
        glfw.make_context_current(window)
        self.opengl_window = window
        
        # OpenGL settings
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(1.0, 1.0, 1.0, 1.0)  # White background
        
        # Camera settings - store initial values for reset
        initial_camera_distance = np.max(self.extents) * 2.5
        initial_camera_x = self.bbox_center[0]
        initial_camera_y = self.bbox_center[1]
        initial_camera_z = self.bbox_center[2] + initial_camera_distance
        
        camera_distance = initial_camera_distance
        camera_x = initial_camera_x
        camera_y = initial_camera_y
        camera_z = initial_camera_z
        
        rotation_x = 0.0
        rotation_y = 0.0
        zoom = 1.0
        pan_x = 0.0
        pan_y = 0.0
        
        # Mouse state
        last_x, last_y = 0, 0
        mouse_down = False
        mouse_button = None
        
        def mouse_button_callback(window, button, action, mods):
            nonlocal mouse_down, mouse_button
            if action == glfw.PRESS:
                mouse_down = True
                mouse_button = button
                
                if button == glfw.MOUSE_BUTTON_LEFT:
                    # Get mouse position (window coordinates, top-left origin)
                    xpos, ypos = glfw.get_cursor_pos(window)
                    width, height = glfw.get_framebuffer_size(window)
                    
                    # Store click for processing in render loop (after matrices are set up)
                    # Note: glfw.get_cursor_pos gives window coordinates (top-left origin)
                    # We'll convert to viewport coordinates in the selection function
                    self.pending_click = (xpos, ypos, width, height)
            
            elif action == glfw.RELEASE:
                mouse_down = False
                mouse_button = None
        
        def cursor_pos_callback(window, xpos, ypos):
            nonlocal last_x, last_y, rotation_x, rotation_y, pan_x, pan_y
            if mouse_down:
                dx = xpos - last_x
                dy = ypos - last_y
                
                # Right mouse button or Shift+Left = Pan
                if mouse_button == glfw.MOUSE_BUTTON_RIGHT or \
                   (mouse_button == glfw.MOUSE_BUTTON_LEFT and glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS):
                    # Pan - calculate pan speed based on zoom level and model size
                    model_size = np.max(self.extents) if self.extents is not None else 1.0
                    pan_speed = (model_size * 0.001) * zoom  # Pan more when zoomed in, scale with model size
                    pan_x += dx * pan_speed
                    pan_y -= dy * pan_speed
                elif mouse_button == glfw.MOUSE_BUTTON_LEFT:
                    # Rotate
                    rotation_y += dx * 0.5
                    rotation_x += dy * 0.5
            
            last_x, last_y = xpos, ypos
        
        def scroll_callback(window, xoffset, yoffset):
            nonlocal zoom
            # Zoom with mouse wheel - exponential zoom for better control
            # Use yoffset directly (positive = zoom in, negative = zoom out)
            if yoffset != 0:
                zoom_factor = 1.15 if yoffset > 0 else 1.0 / 1.15
                zoom *= zoom_factor
                zoom = max(0.05, min(20.0, zoom))  # Wider zoom range
                self.needs_redraw = True
        
        glfw.set_mouse_button_callback(window, mouse_button_callback)
        glfw.set_cursor_pos_callback(window, cursor_pos_callback)
        glfw.set_scroll_callback(window, scroll_callback)
        
        # Enable vsync for better performance
        glfw.swap_interval(1)
        
        # Main render loop - simplified for better performance
        while not glfw.window_should_close(window):
            glfw.poll_events()
            
            # Check for reset view flag
            if self.reset_view_flag:
                rotation_x = 0.0
                rotation_y = 0.0
                zoom = 1.0
                pan_x = 0.0
                pan_y = 0.0
                camera_distance = initial_camera_distance
                camera_x = initial_camera_x
                camera_y = initial_camera_y
                camera_z = initial_camera_z
                self.orthographic_view = None  # Reset to perspective
                self.reset_view_flag = False
            
            # Check for plane view flags
            if self.view_xy_flag:
                # XY plane view (top view) - look down Z-axis, orthographic
                rotation_x = 0.0
                rotation_y = 0.0
                zoom = 1.0
                pan_x = 0.0
                pan_y = 0.0
                camera_distance = initial_camera_distance
                self.orthographic_view = 'xy'
                self.view_xy_flag = False
            
            if self.view_yz_flag:
                # YZ plane view (front view) - look down X-axis, orthographic
                rotation_x = 0.0
                rotation_y = 0.0
                zoom = 1.0
                pan_x = 0.0
                pan_y = 0.0
                camera_distance = initial_camera_distance
                self.orthographic_view = 'yz'
                self.view_yz_flag = False
            
            if self.view_xz_flag:
                # XZ plane view (side view) - look down Y-axis, orthographic
                rotation_x = 0.0
                rotation_y = 0.0
                zoom = 1.0
                pan_x = 0.0
                pan_y = 0.0
                camera_distance = initial_camera_distance
                self.orthographic_view = 'xz'
                self.view_xz_flag = False
            
            # Clear
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            
            # Set up projection
            width, height = glfw.get_framebuffer_size(window)
            glViewport(0, 0, width, height)
            
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            aspect = width / height if height > 0 else 1.0
            
            # Use orthographic projection for standard plane views
            if self.orthographic_view is not None:
                # Calculate orthographic bounds based on model extents
                max_extent = np.max(self.extents) if self.extents is not None else 1.0
                ortho_size = max_extent * 1.2 / zoom  # Scale with zoom
                
                if aspect >= 1.0:
                    # Wider than tall
                    left = -ortho_size * aspect
                    right = ortho_size * aspect
                    bottom = -ortho_size
                    top = ortho_size
                else:
                    # Taller than wide
                    left = -ortho_size
                    right = ortho_size
                    bottom = -ortho_size / aspect
                    top = ortho_size / aspect
                
                glOrtho(left, right, bottom, top, -max_extent * 10, max_extent * 10)
            else:
                # Perspective projection for normal view
                gluPerspective(45.0, aspect, 0.1, camera_distance * 10)
            
            # Set up modelview
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            
            # Camera setup - different for orthographic vs perspective
            if self.orthographic_view is not None:
                # For orthographic views, use simpler camera setup
                # Position camera directly along the viewing axis, perpendicular to the plane
                if self.orthographic_view == 'xy':
                    # Top view - look down Z-axis (positive Z to negative Z)
                    # Camera above, looking straight down
                    gluLookAt(
                        self.bbox_center[0] + pan_x, self.bbox_center[1] + pan_y, self.bbox_center[2] + camera_distance,
                        self.bbox_center[0] + pan_x, self.bbox_center[1] + pan_y, self.bbox_center[2],
                        0, 1, 0  # Up vector is +Y
                    )
                    # No rotations needed - camera is already aligned
                elif self.orthographic_view == 'yz':
                    # Front view - look down X-axis (positive X to negative X)
                    # Camera to the right, looking left along -X
                    gluLookAt(
                        self.bbox_center[0] + camera_distance, self.bbox_center[1] + pan_y, self.bbox_center[2] + pan_x,
                        self.bbox_center[0], self.bbox_center[1] + pan_y, self.bbox_center[2] + pan_x,
                        0, 1, 0  # Up vector is +Y
                    )
                    # No rotations needed - camera is already aligned
                elif self.orthographic_view == 'xz':
                    # Side view - look down Y-axis (positive Y to negative Y)
                    # Camera above, looking down along -Y
                    gluLookAt(
                        self.bbox_center[0] + pan_x, self.bbox_center[1] + camera_distance, self.bbox_center[2] + pan_y,
                        self.bbox_center[0] + pan_x, self.bbox_center[1], self.bbox_center[2] + pan_y,
                        0, 0, -1  # Up vector is -Z (so X is right, Z is up in screen space)
                    )
                    # No rotations needed - camera is already aligned
            else:
                # Perspective view - use original camera setup
                current_distance = camera_distance / zoom
                # Calculate direction from center to original camera position
                dir_x = camera_x - self.bbox_center[0]
                dir_y = camera_y - self.bbox_center[1]
                dir_z = camera_z - self.bbox_center[2]
                dir_length = np.sqrt(dir_x*dir_x + dir_y*dir_y + dir_z*dir_z)
                
                if dir_length > 0.001:  # Avoid division by zero
                    # Normalize and scale by zoomed distance
                    dir_x = dir_x / dir_length * current_distance
                    dir_y = dir_y / dir_length * current_distance
                    dir_z = dir_z / dir_length * current_distance
                    
                    zoomed_camera_x = self.bbox_center[0] + dir_x
                    zoomed_camera_y = self.bbox_center[1] + dir_y
                    zoomed_camera_z = self.bbox_center[2] + dir_z
                else:
                    # Fallback to original position
                    zoomed_camera_x = camera_x
                    zoomed_camera_y = camera_y
                    zoomed_camera_z = camera_z
                
                gluLookAt(
                    zoomed_camera_x + pan_x, zoomed_camera_y + pan_y, zoomed_camera_z,
                    self.bbox_center[0] + pan_x, self.bbox_center[1] + pan_y, self.bbox_center[2],
                    0, 1, 0
                )
                
                # Rotate around center
                glTranslatef(self.bbox_center[0], self.bbox_center[1], self.bbox_center[2])
                glRotatef(rotation_x, 1, 0, 0)
                glRotatef(rotation_y, 0, 1, 0)
                glTranslatef(-self.bbox_center[0], -self.bbox_center[1], -self.bbox_center[2])
            
            # Process pending click for vertex selection (after matrices are set up)
            if self.pending_click is not None:
                click_x, click_y, click_width, click_height = self.pending_click
                self.pending_click = None  # Clear the pending click
                
                # Now that matrices are set up, do screen-space selection
                nearest = self.find_nearest_vertex_screen_space(click_x, click_y, click_width, click_height)
                
                if nearest is not None:
                    if len(self.selected_vertices) < 3:
                        if nearest not in self.selected_vertices:
                            self.selected_vertices.append(nearest)
                            
                            # If we have 3 vertices, calculate plane
                            if len(self.selected_vertices) == 3:
                                p1 = self.vertices[self.selected_vertices[0]]
                                p2 = self.vertices[self.selected_vertices[1]]
                                p3 = self.vertices[self.selected_vertices[2]]
                                
                                plane = self.calculate_plane_from_points(p1, p2, p3)
                                if plane is not None:
                                    self.plane_equation = plane
                                    self.calculate_vertex_distances()
                                    # Update GUI status (use threading-safe method)
                                    if self.gui:
                                        try:
                                            self.gui.root.after(0, self.gui.update_status)
                                        except:
                                            pass
                                else:
                                    # Show warning in a thread-safe way
                                    if self.gui:
                                        self.gui.root.after(0, lambda: messagebox.showwarning("Warning", "Selected points are collinear. Please select 3 non-collinear points."))
                                    self.selected_vertices.pop()
                            else:
                                # Update status when vertex is selected
                                if self.gui:
                                    try:
                                        self.gui.root.after(0, self.gui.update_status)
                                    except:
                                        pass
                    else:
                        # Reset selection
                        self.selected_vertices = [nearest]
                        self.plane_equation = None
                        self.vertex_distances = None
                        self.vertices_within_threshold = set()
                        if self.gui:
                            try:
                                self.gui.root.after(0, self.gui.update_status)
                            except:
                                pass
            
            # Draw vertices - optimized single pass
            if len(self.vertices) > 0:
                glPointSize(3.0)
                glBegin(GL_POINTS)
                
                selected_set = set(self.selected_vertices)
                threshold_set = self.vertices_within_threshold
                
                for i, vertex in enumerate(self.vertices):
                    # Color based on selection and distance
                    if i in selected_set:
                        # Selected vertices - yellow (will be redrawn larger below)
                        glColor3f(1.0, 1.0, 0.0)
                    elif i in threshold_set:
                        # Vertices within threshold - red
                        glColor3f(1.0, 0.0, 0.0)
                    else:
                        # Default color - blue
                        glColor3f(0.0, 0.2, 0.6)
                    
                    glVertex3f(vertex[0], vertex[1], vertex[2])
                
                glEnd()
            
            # Draw selected vertices with larger markers on top
            if len(self.selected_vertices) > 0:
                glPointSize(8.0)
                glColor3f(1.0, 1.0, 0.0)  # Yellow
                glBegin(GL_POINTS)
                for idx in self.selected_vertices:
                    v = self.vertices[idx]
                    glVertex3f(v[0], v[1], v[2])
                glEnd()
            
            # Draw plane if defined
            if self.plane_equation is not None:
                self._draw_plane()
            
            # Draw coordinate system
            self._draw_coordinate_system()
            
            glfw.swap_buffers(window)
        
        glfw.terminate()
        self.opengl_window = None
    
    def _draw_plane(self):
        """Draw the plane as a semi-transparent surface"""
        if self.plane_equation is None or len(self.selected_vertices) < 3:
            return
        
        a, b, c, d = self.plane_equation
        
        # Use the first selected vertex as plane center
        plane_center = self.vertices[self.selected_vertices[0]]
        
        # Calculate plane bounds (extend to bounding box size)
        plane_size = np.max(self.extents) * 1.5
        
        # Create two vectors in the plane
        # Use arbitrary perpendicular vectors
        if abs(c) > 0.1:
            v1 = np.array([1, 0, -a/c])
        else:
            v1 = np.array([0, 1, -b/c])
        
        v1 = v1 / np.linalg.norm(v1)
        v2 = np.cross(np.array([a, b, c]), v1)
        v2 = v2 / np.linalg.norm(v2)
        
        # Draw plane as grid
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glColor4f(0.5, 0.5, 0.5, 0.3)  # Semi-transparent gray
        
        grid_size = 20
        step = plane_size / grid_size
        
        glBegin(GL_QUADS)
        for i in range(grid_size):
            for j in range(grid_size):
                p1 = plane_center + v1 * (i * step - plane_size/2) + v2 * (j * step - plane_size/2)
                p2 = plane_center + v1 * ((i+1) * step - plane_size/2) + v2 * (j * step - plane_size/2)
                p3 = plane_center + v1 * ((i+1) * step - plane_size/2) + v2 * ((j+1) * step - plane_size/2)
                p4 = plane_center + v1 * (i * step - plane_size/2) + v2 * ((j+1) * step - plane_size/2)
                
                glVertex3f(p1[0], p1[1], p1[2])
                glVertex3f(p2[0], p2[1], p2[2])
                glVertex3f(p3[0], p3[1], p3[2])
                glVertex3f(p4[0], p4[1], p4[2])
        glEnd()
        
        # Draw plane edges
        glLineWidth(2.0)
        glColor3f(0.3, 0.3, 0.3)
        glBegin(GL_LINE_LOOP)
        corners = [
            plane_center + v1 * (-plane_size/2) + v2 * (-plane_size/2),
            plane_center + v1 * (plane_size/2) + v2 * (-plane_size/2),
            plane_center + v1 * (plane_size/2) + v2 * (plane_size/2),
            plane_center + v1 * (-plane_size/2) + v2 * (plane_size/2)
        ]
        for corner in corners:
            glVertex3f(corner[0], corner[1], corner[2])
        glEnd()
    
    def _draw_coordinate_system(self):
        """Draw coordinate system at lower front corner"""
        axis_origin = np.array([
            self.bbox_min[0],
            self.bbox_min[1],
            self.bbox_min[2]
        ])
        axis_length = np.max(self.extents) * 0.15
        glLineWidth(6.0)
        
        glBegin(GL_LINES)
        # X axis - Red
        glColor3f(0.8, 0.0, 0.0)
        glVertex3f(axis_origin[0], axis_origin[1], axis_origin[2])
        glVertex3f(axis_origin[0] + axis_length, axis_origin[1], axis_origin[2])
        # Y axis - Green
        glColor3f(0.0, 0.6, 0.0)
        glVertex3f(axis_origin[0], axis_origin[1], axis_origin[2])
        glVertex3f(axis_origin[0], axis_origin[1] + axis_length, axis_origin[2])
        # Z axis - Blue
        glColor3f(0.0, 0.0, 0.8)
        glVertex3f(axis_origin[0], axis_origin[1], axis_origin[2])
        glVertex3f(axis_origin[0], axis_origin[1], axis_origin[2] + axis_length)
        glEnd()


class PlaneVisualizerGUI:
    instance = None
    
    def __init__(self):
        PlaneVisualizerGUI.instance = self
        self.visualizer = OBJPlaneVisualizer()
        self.visualizer.gui = self  # Reference to GUI for updates
        self.root = tk.Tk()
        self.root.title("OBJ Plane Distance Visualizer")
        self.root.geometry("450x500")
        
        # Store initial size for restoration if needed
        self.initial_geometry = "400x350"
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the GUI"""
        # File selection frame
        file_frame = ttk.Frame(self.root, padding="10")
        file_frame.pack(fill=tk.X)
        
        ttk.Label(file_frame, text="OBJ File:").pack(side=tk.LEFT, padx=5)
        self.file_label = ttk.Label(file_frame, text="No file loaded", foreground="gray")
        self.file_label.pack(side=tk.LEFT, padx=5, expand=True)
        
        ttk.Button(file_frame, text="Load OBJ", command=self.load_obj).pack(side=tk.RIGHT, padx=5)
        
        # Status frame
        status_frame = ttk.LabelFrame(self.root, text="Status", padding="10")
        status_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.status_text = tk.Text(status_frame, height=4, wrap=tk.WORD, state=tk.DISABLED)
        self.status_text.pack(fill=tk.BOTH, expand=True)
        
        # Distance threshold frame
        threshold_frame = ttk.LabelFrame(self.root, text="Distance Threshold (OBJ units)", padding="10")
        threshold_frame.pack(fill=tk.X, padx=10, pady=5)
        
        input_frame = ttk.Frame(threshold_frame)
        input_frame.pack(fill=tk.X)
        
        ttk.Label(input_frame, text="Distance:").pack(side=tk.LEFT, padx=5)
        self.distance_var = tk.StringVar(value="1.0")
        self.distance_entry = ttk.Entry(input_frame, textvariable=self.distance_var, width=15)
        self.distance_entry.pack(side=tk.LEFT, padx=5)
        self.distance_entry.bind('<KeyRelease>', self.on_distance_changed)
        
        ttk.Label(input_frame, text="units").pack(side=tk.LEFT, padx=5)
        
        # Buttons frame
        button_frame = ttk.Frame(self.root, padding="10")
        button_frame.pack(fill=tk.X)
        
        ttk.Button(button_frame, text="Open 3D View", command=self.open_3d_view).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Reset Selection", command=self.reset_selection).pack(side=tk.RIGHT, padx=5)
        
        # View controls frame
        view_frame = ttk.Frame(self.root, padding="10")
        view_frame.pack(fill=tk.X)
        
        ttk.Button(view_frame, text="Reset View / Fit Object", command=self.reset_view).pack(side=tk.LEFT, padx=5)
        ttk.Button(view_frame, text="Delete Vertices in Plane", command=self.delete_vertices).pack(side=tk.LEFT, padx=5)
        
        # Plane view buttons frame
        plane_view_frame = ttk.Frame(self.root, padding="10")
        plane_view_frame.pack(fill=tk.X)
        
        ttk.Label(plane_view_frame, text="Standard Views:").pack(side=tk.LEFT, padx=5)
        ttk.Button(plane_view_frame, text="XY (Top)", command=self.set_xy_view).pack(side=tk.LEFT, padx=2)
        ttk.Button(plane_view_frame, text="YZ (Front)", command=self.set_yz_view).pack(side=tk.LEFT, padx=2)
        ttk.Button(plane_view_frame, text="XZ (Side)", command=self.set_xz_view).pack(side=tk.LEFT, padx=2)
        
        # Instructions
        instructions = ttk.Label(
            self.root,
            text="Instructions:\n1. Load OBJ file\n2. Open 3D view\n3. Click 3 vertices to define plane\n4. Adjust distance threshold\n5. Click 'Delete Vertices in Plane' to delete all red vertices\n\n3D View Controls:\n• Left-click + drag: Rotate\n• Right-click + drag: Pan\n• Shift + Left-click + drag: Pan\n• Mouse wheel: Zoom\n• Reset View button: Fit object to center",
            justify=tk.LEFT
        )
        instructions.pack(fill=tk.X, padx=10, pady=5)
    
    def load_obj(self):
        """Load OBJ file"""
        if not self.visualizer.select_obj_file():
            return
        
        self.file_label.config(text=os.path.basename(self.visualizer.obj_path), foreground="black")
        
        if not self.visualizer.parse_obj_file():
            messagebox.showerror("Error", "Failed to parse OBJ file")
            return
        
        self.update_status()
    
    def update_status(self):
        """Update status display"""
        self.status_text.config(state=tk.NORMAL)
        self.status_text.delete(1.0, tk.END)
        
        if self.visualizer.extents is not None:
            status = f"Vertices: {len(self.visualizer.vertices)}\n"
            status += f"Faces: {len(self.visualizer.faces)}\n"
            status += f"Selected vertices: {len(self.visualizer.selected_vertices)}/3\n"
            
            if len(self.visualizer.selected_vertices) == 3:
                status += f"Vertices within threshold: {len(self.visualizer.vertices_within_threshold)}"
            else:
                status += "Click 3 vertices in 3D view to define plane"
        
        self.status_text.insert(1.0, status)
        self.status_text.config(state=tk.DISABLED)
    
    def on_distance_changed(self, event=None):
        """Handle distance threshold change"""
        try:
            new_threshold = float(self.distance_var.get())
            if new_threshold >= 0:
                self.visualizer.distance_threshold = new_threshold
                if self.visualizer.plane_equation is not None:
                    self.visualizer.calculate_vertex_distances()
                    self.visualizer.needs_redraw = True
                self.update_status()
        except ValueError:
            pass
    
    def reset_selection(self):
        """Reset vertex selection"""
        self.visualizer.selected_vertices = []
        self.visualizer.plane_equation = None
        self.visualizer.vertex_distances = None
        self.visualizer.vertices_within_threshold = set()
        self.visualizer.needs_redraw = True
        self.update_status()
    
    def reset_view(self):
        """Reset camera view to fit object"""
        if self.visualizer.opengl_window is None:
            messagebox.showinfo("Info", "Please open the 3D view first")
            return
        
        # Set flag to reset view in OpenGL thread
        self.visualizer.reset_view_flag = True
        self.visualizer.needs_redraw = True
    
    def set_xy_view(self):
        """Set view to XY plane (top view)"""
        if self.visualizer.opengl_window is None:
            messagebox.showinfo("Info", "Please open the 3D view first")
            return
        
        self.visualizer.view_xy_flag = True
        self.visualizer.needs_redraw = True
    
    def set_yz_view(self):
        """Set view to YZ plane (front view)"""
        if self.visualizer.opengl_window is None:
            messagebox.showinfo("Info", "Please open the 3D view first")
            return
        
        self.visualizer.view_yz_flag = True
        self.visualizer.needs_redraw = True
    
    def set_xz_view(self):
        """Set view to XZ plane (side view)"""
        if self.visualizer.opengl_window is None:
            messagebox.showinfo("Info", "Please open the 3D view first")
            return
        
        self.visualizer.view_xz_flag = True
        self.visualizer.needs_redraw = True
    
    def delete_vertices(self):
        """Delete all vertices within the plane threshold and save to new file"""
        if not self.visualizer.vertices_within_threshold:
            messagebox.showwarning("Warning", "Please define a plane first by selecting 3 vertices.\nThen adjust the distance threshold to select vertices to delete.")
            return
        
        if len(self.visualizer.vertices) == 0:
            messagebox.showwarning("Warning", "No OBJ file loaded")
            return
        
        # Count faces that will be deleted
        vertices_to_delete = set(self.visualizer.vertices_within_threshold)
        faces_to_delete = sum(1 for face in self.visualizer.faces 
                             if any(v_idx in vertices_to_delete for v_idx in face['vertices']))
        
        # Confirmation dialog
        confirm_msg = f"Delete {len(self.visualizer.vertices_within_threshold)} vertices within plane threshold?\n"
        confirm_msg += f"This will also delete {faces_to_delete} faces that reference them.\n"
        confirm_msg += f"Remaining: {len(self.visualizer.vertices) - len(self.visualizer.vertices_within_threshold)} vertices"
        
        if not messagebox.askyesno("Confirm Deletion", confirm_msg):
            return
        
        # Perform deletion
        deleted_verts, deleted_faces, remaining_verts = self.visualizer.delete_selected_vertices()
        
        if deleted_verts is None:
            messagebox.showerror("Error", "Failed to delete vertices")
            return
        
        # Ask for output file location
        root = tk.Tk()
        root.withdraw()
        
        # Suggest output filename
        if self.visualizer.obj_path:
            base_name = os.path.splitext(os.path.basename(self.visualizer.obj_path))[0]
            dir_name = os.path.dirname(self.visualizer.obj_path)
            default_name = os.path.join(dir_name, f"{base_name}_edited.obj")
        else:
            default_name = "edited.obj"
        
        output_path = filedialog.asksaveasfilename(
            title="Save edited OBJ file",
            defaultextension=".obj",
            filetypes=[("OBJ files", "*.obj"), ("All files", "*.*")],
            initialfile=os.path.basename(default_name),
            initialdir=os.path.dirname(default_name) if os.path.dirname(default_name) else "."
        )
        
        root.destroy()
        
        if not output_path:
            # User cancelled - restore vertices (we already deleted them, so reload file)
            if self.visualizer.obj_path:
                self.visualizer.parse_obj_file()
                self.update_status()
            return
        
        # Write the edited OBJ file
        if self.visualizer.write_obj_file(output_path):
            success_msg = f"Successfully saved edited OBJ file!\n\n"
            success_msg += f"Deleted: {deleted_verts} vertices, {deleted_faces} faces\n"
            success_msg += f"Remaining: {remaining_verts} vertices, {len(self.visualizer.faces)} faces\n"
            success_msg += f"Saved to: {os.path.basename(output_path)}"
            messagebox.showinfo("Success", success_msg)
            
            # Update status and reload visualization
            self.update_status()
            self.visualizer.needs_redraw = True
        else:
            messagebox.showerror("Error", "Failed to save edited OBJ file")
    
    def open_3d_view(self):
        """Open 3D visualization window"""
        if len(self.visualizer.vertices) == 0:
            messagebox.showwarning("Warning", "Please load an OBJ file first")
            return
        
        if not OPENGL_AVAILABLE:
            messagebox.showerror("Error", "PyOpenGL not available. Please install: pip install PyOpenGL glfw")
            return
        
        # Get current window size and position in actual pixels before GLFW initializes
        try:
            self.root.update_idletasks()
            # Get actual pixel dimensions (accounts for DPI scaling)
            w = self.root.winfo_width()
            h = self.root.winfo_height()
            x_pos = self.root.winfo_x()
            y_pos = self.root.winfo_y()
            
            # Fallback to geometry string if winfo fails
            if w <= 1 or h <= 1:
                current_geometry = self.root.geometry()
                parts = current_geometry.split('+')
                size_part = parts[0]
                if 'x' in size_part:
                    w, h = map(int, size_part.split('x'))
                else:
                    w, h = 400, 350
                x_pos = int(parts[1]) if len(parts) > 1 else x_pos
                y_pos = int(parts[2]) if len(parts) > 2 else y_pos
        except:
            w, h = 400, 350
            x_pos = y_pos = None
        
        # Run OpenGL window in separate thread
        if self.visualizer.opengl_thread is None or not self.visualizer.opengl_thread.is_alive():
            self.visualizer.opengl_thread = threading.Thread(target=self.visualizer.run_opengl_window, daemon=True)
            self.visualizer.opengl_thread.start()
            
            # Restore window size after GLFW initializes (multiple attempts)
            def restore_window_size(attempt=0):
                try:
                    self.root.update_idletasks()
                    
                    # Force the window to maintain its exact pixel size
                    if x_pos is not None and y_pos is not None:
                        restore_geometry = f"{w}x{h}+{x_pos}+{y_pos}"
                    else:
                        restore_geometry = f"{w}x{h}"
                    
                    # Set geometry multiple times to force it
                    self.root.geometry(restore_geometry)
                    self.root.update_idletasks()
                    self.root.geometry(restore_geometry)  # Set again to force
                    self.root.update_idletasks()
                    
                    # Verify it worked
                    current_w = self.root.winfo_width()
                    current_h = self.root.winfo_height()
                    
                    # If size changed, try again
                    if (abs(current_w - w) > 5 or abs(current_h - h) > 5) and attempt < 5:
                        self.root.after(50, lambda: restore_window_size(attempt + 1))
                except Exception:
                    # If restoration fails, try again
                    if attempt < 5:
                        self.root.after(50, lambda: restore_window_size(attempt + 1))
            
            # Start restoration attempts - more frequent at first
            self.root.after(10, lambda: restore_window_size(0))
            self.root.after(50, lambda: restore_window_size(1))
            self.root.after(100, lambda: restore_window_size(2))
            self.root.after(200, lambda: restore_window_size(3))
            self.root.after(500, lambda: restore_window_size(4))
    
    def run(self):
        """Run the GUI"""
        self.root.mainloop()


def main():
    app = PlaneVisualizerGUI()
    app.run()


if __name__ == "__main__":
    main()

