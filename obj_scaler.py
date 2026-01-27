"""
OBJ Scaler with Bounding Box Visualization
Loads an OBJ file, calculates bounding box, displays it visually,
and allows scaling based on target dimensions.
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
import json
import numpy as np
import os
import re
import shutil
from pathlib import Path
import threading

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


class OBJScaler:
    def __init__(self):
        self.obj_path = None
        self.mtl_path = None
        self.texture_paths = []
        self.vertices = []
        self.faces = []
        self.normals = []
        self.uvs = []
        self.materials = {}
        self.obj_lines = []
        self.bbox_min = None
        self.bbox_max = None
        self.bbox_center = None
        self.extents = None
        
        # Batch processing state
        self.batch_file_list = []  # List of full paths to OBJ files found
        self.current_file_index = -1  # Index of currently selected file in batch list
        
        # Window position/size memory (persists for session)
        self.saved_window_x = None  # Saved window X position
        self.saved_window_y = None  # Saved window Y position
        self.saved_window_width = 1024  # Default window width
        self.saved_window_height = 768  # Default window height
        
        # View and orthographic state (set from GUI, read by OpenGL loop)
        self.view_xy_flag = False
        self.view_yz_flag = False
        self.view_xz_flag = False
        self.reset_view_flag = False
        self.orthographic_view = None  # None | 'xy' | 'yz' | 'xz'
        
        # Distance measurement state (ortho views only)
        self.measure_mode = False
        self.measure_point1 = None  # [x,y,z] or None
        self.measure_point2 = None  # [x,y,z] or None; during drag = current mouse world
        self.measure_dragging = False
        self.pending_measure_click = None  # (x_fb, y_fb, width, height) or None
        self.pending_measure_track = None  # (x_fb, y_fb, width, height) or None
        
        # Global scale adjustment (applies before any scale; 1.0 = no change)
        self.adjustment = 1.0
        
    def select_obj_file(self):
        """Use tkinter to select OBJ file"""
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        
        file_path = filedialog.askopenfilename(
            title="Select OBJ file",
            filetypes=[("OBJ files", "*.obj"), ("All files", "*.*")]
        )
        
        root.destroy()
        
        if file_path:
            self.obj_path = file_path
            return True
        return False
    
    def find_mtl_file(self):
        """Find associated MTL file"""
        if not self.obj_path:
            return False
        
        obj_dir = os.path.dirname(self.obj_path)
        obj_name = os.path.splitext(os.path.basename(self.obj_path))[0]
        mtl_path = os.path.join(obj_dir, obj_name + ".mtl")
        
        if os.path.exists(mtl_path):
            self.mtl_path = mtl_path
            return True
        return False
    
    def parse_obj_file(self, suppress_dialogs=False):
        """Parse OBJ file to extract vertices, faces, and other data"""
        if not self.obj_path:
            return False
        
        self.vertices = []
        self.faces = []
        self.normals = []
        self.uvs = []
        self.obj_lines = []
        current_material = None
        
        try:
            with open(self.obj_path, 'r', encoding='utf-8') as f:
                for line in f:
                    self.obj_lines.append(line)
                    line = line.strip()
                    
                    if not line or line.startswith('#'):
                        continue
                    
                    parts = line.split()
                    if not parts:
                        continue
                    
                    if parts[0] == 'v':  # Vertex
                        if len(parts) >= 4:
                            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                            self.vertices.append([x, y, z])
                    
                    elif parts[0] == 'vn':  # Normal
                        if len(parts) >= 4:
                            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                            self.normals.append([x, y, z])
                    
                    elif parts[0] == 'vt':  # Texture coordinate
                        if len(parts) >= 3:
                            u, v = float(parts[1]), float(parts[2])
                            self.uvs.append([u, v])
                    
                    elif parts[0] == 'f':  # Face
                        face_vertices = []
                        for part in parts[1:]:
                            # Handle format: v/vt/vn or v//vn or v
                            indices = part.split('/')
                            if indices[0]:
                                face_vertices.append(int(indices[0]) - 1)  # OBJ is 1-indexed
                        if face_vertices:
                            self.faces.append(face_vertices)
                    
                    elif parts[0] == 'usemtl':  # Material
                        if len(parts) >= 2:
                            current_material = parts[1]
                    
                    elif parts[0] == 'mtllib':  # MTL library reference
                        if len(parts) >= 2:
                            # Store MTL reference
                            pass
            
            if not self.vertices:
                return False
            
            self.vertices = np.array(self.vertices)
            self.calculate_bounding_box()
            return True
            
        except Exception as e:
            if not suppress_dialogs:
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
    
    def get_bounding_box_corners(self):
        """Get the 8 corners of the bounding box"""
        min_x, min_y, min_z = self.bbox_min
        max_x, max_y, max_z = self.bbox_max
        
        corners = np.array([
            [min_x, min_y, min_z],
            [max_x, min_y, min_z],
            [max_x, max_y, min_z],
            [min_x, max_y, min_z],
            [min_x, min_y, max_z],
            [max_x, min_y, max_z],
            [max_x, max_y, max_z],
            [min_x, max_y, max_z],
        ])
        return corners
    
    def get_adjusted_vertices(self):
        """Vertices scaled by adjustment around bbox_center. If adjustment==1.0, return originals."""
        if self.extents is None or len(self.vertices) == 0:
            return self.vertices
        if self.adjustment == 1.0:
            return self.vertices
        adj = (self.vertices - self.bbox_center) * self.adjustment + self.bbox_center
        return adj
    
    def get_adjusted_extents(self):
        """Extents scaled by adjustment."""
        if self.extents is None:
            return None
        return self.extents * self.adjustment
    
    def get_adjusted_bbox_center(self):
        """Bbox center (unchanged by adjustment)."""
        return self.bbox_center
    
    def get_adjusted_bbox_min(self):
        """Bbox min for adjusted geometry."""
        if self.extents is None or self.bbox_center is None:
            return self.bbox_min
        return self.bbox_center - (self.extents * 0.5) * self.adjustment
    
    def get_adjusted_bbox_max(self):
        """Bbox max for adjusted geometry."""
        if self.extents is None or self.bbox_center is None:
            return self.bbox_max
        return self.bbox_center + (self.extents * 0.5) * self.adjustment
    
    def get_adjusted_corners(self):
        """Eight corners of adjusted bounding box."""
        if self.extents is None:
            return self.get_bounding_box_corners()
        mn = self.get_adjusted_bbox_min()
        mx = self.get_adjusted_bbox_max()
        min_x, min_y, min_z = mn
        max_x, max_y, max_z = mx
        corners = np.array([
            [min_x, min_y, min_z],
            [max_x, min_y, min_z],
            [max_x, max_y, min_z],
            [min_x, max_y, min_z],
            [min_x, min_y, max_z],
            [max_x, min_y, max_z],
            [max_x, max_y, max_z],
            [min_x, max_y, max_z],
        ])
        return corners
    
    def screen_to_world_on_view_plane(self, mouse_x, mouse_y, width, height, view_type):
        """Convert screen coordinates to world coordinates on the view plane.
        
        Uses gluUnProject. For XY: projects to z=0; YZ: x=0; XZ: y=0.
        Call from render loop after projection/modelview are set.
        """
        import ctypes
        from OpenGL.GL import glGetDoublev, glGetIntegerv, GL_MODELVIEW_MATRIX, GL_PROJECTION_MATRIX, GL_VIEWPORT
        from OpenGL.GLU import gluUnProject
        
        modelview = (ctypes.c_double * 16)()
        projection = (ctypes.c_double * 16)()
        viewport = (ctypes.c_int * 4)()
        glGetDoublev(GL_MODELVIEW_MATRIX, modelview)
        glGetDoublev(GL_PROJECTION_MATRIX, projection)
        glGetIntegerv(GL_VIEWPORT, viewport)
        win_y = height - mouse_y
        win_z = 0.5
        try:
            result = gluUnProject(mouse_x, win_y, win_z, modelview, projection, viewport)
            if result is not None:
                obj_x, obj_y, obj_z = result
                world_point = np.array([float(obj_x), float(obj_y), float(obj_z)])
                if view_type == 'xy':
                    world_point[2] = 0.0
                elif view_type == 'yz':
                    world_point[0] = 0.0
                elif view_type == 'xz':
                    world_point[1] = 0.0
                return world_point
        except Exception:
            pass
        return None
    
    def visualize_bounding_box(self):
        """Display 3D visualization with OBJ and bounding box using PyOpenGL"""
        if not OPENGL_AVAILABLE:
            messagebox.showerror("Error", "PyOpenGL not available. Please install: pip install PyOpenGL PyOpenGL-accelerate glfw")
            return
        
        # Run OpenGL window in a separate thread to avoid blocking
        thread = threading.Thread(target=self._run_opengl_window, daemon=True)
        thread.start()
    
    def _run_opengl_window(self):
        """Run the OpenGL visualization window"""
        # Initialize GLFW
        if not glfw.init():
            messagebox.showerror("Error", "Failed to initialize GLFW")
            return
        
        # Use saved window size and position if available
        window_width = self.saved_window_width if self.saved_window_width else 1024
        window_height = self.saved_window_height if self.saved_window_height else 768
        
        # Create window
        window = glfw.create_window(window_width, window_height, "OBJ Model with Bounding Box", None, None)
        if not window:
            glfw.terminate()
            messagebox.showerror("Error", "Failed to create GLFW window")
            return
        
        # Set window position if we have saved position
        if self.saved_window_x is not None and self.saved_window_y is not None:
            glfw.set_window_pos(window, self.saved_window_x, self.saved_window_y)
        
        glfw.make_context_current(window)
        
        # Set up callbacks to track window position and size changes
        def window_pos_callback(window, x, y):
            """Track window position changes"""
            self.saved_window_x = x
            self.saved_window_y = y
        
        def window_size_callback(window, width, height):
            """Track window size changes"""
            self.saved_window_width = width
            self.saved_window_height = height
        
        glfw.set_window_pos_callback(window, window_pos_callback)
        glfw.set_window_size_callback(window, window_size_callback)
        
        # Update window title with dimensions (use adjusted extents)
        adj_ext = self.get_adjusted_extents()
        dim_text = f"OBJ Model - BBox: X={adj_ext[0]:.2f} Y={adj_ext[1]:.2f} Z={adj_ext[2]:.2f}" if adj_ext is not None else "OBJ Model"
        glfw.set_window_title(window, dim_text)
        
        # OpenGL settings
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(1.0, 1.0, 1.0, 1.0)  # White background
        
        # Initial camera (for reset); use adjusted extents
        _ext = adj_ext if adj_ext is not None else self.extents
        _max = np.max(_ext) * 2.5 if _ext is not None else 1.0
        initial_camera_distance = _max
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
        click_start_x, click_start_y = 0, 0
        mouse_has_moved = False
        
        def window_to_framebuffer_coords(window, xpos, ypos):
            width, height = glfw.get_framebuffer_size(window)
            ww, wh = glfw.get_window_size(window)
            if ww > 0 and wh > 0:
                return xpos * (width / ww), ypos * (height / wh), width, height
            return xpos, ypos, width, height
        
        def mouse_button_callback(win, button, action, mods):
            nonlocal mouse_down, mouse_button, click_start_x, click_start_y, mouse_has_moved, last_x, last_y
            if action == glfw.PRESS:
                mouse_down = True
                mouse_button = button
                mouse_has_moved = False
                click_start_x, click_start_y = glfw.get_cursor_pos(win)
                last_x, last_y = click_start_x, click_start_y
                if button == glfw.MOUSE_BUTTON_LEFT and self.measure_mode and self.orthographic_view is not None:
                    xpos_fb, ypos_fb, w, h = window_to_framebuffer_coords(win, click_start_x, click_start_y)
                    if self.measure_point1 is not None and self.measure_point2 is not None:
                        self.measure_point1 = None
                        self.measure_point2 = None
                    self.pending_measure_click = (xpos_fb, ypos_fb, w, h)
            elif action == glfw.RELEASE:
                mouse_down = False
                mouse_button = None
                mouse_has_moved = False
                if button == glfw.MOUSE_BUTTON_LEFT:
                    self.measure_dragging = False
        
        def cursor_pos_callback(win, xpos, ypos):
            nonlocal last_x, last_y, rotation_x, rotation_y, pan_x, pan_y, mouse_has_moved, click_start_x, click_start_y
            measuring = (self.measure_mode and self.orthographic_view is not None and
                        self.measure_point1 is not None and mouse_down and mouse_button == glfw.MOUSE_BUTTON_LEFT)
            if measuring:
                xpos_fb, ypos_fb, w, h = window_to_framebuffer_coords(win, xpos, ypos)
                self.pending_measure_track = (xpos_fb, ypos_fb, w, h)
            elif mouse_down:
                if not mouse_has_moved:
                    dist = np.sqrt((xpos - click_start_x)**2 + (ypos - click_start_y)**2)
                    if dist > 5.0:
                        mouse_has_moved = True
                        if mouse_button == glfw.MOUSE_BUTTON_LEFT and not (self.measure_mode and self.orthographic_view is not None):
                            self.pending_measure_click = None
                if mouse_has_moved:
                    dx = xpos - last_x
                    dy = ypos - last_y
                    if abs(dx) > 0.1 or abs(dy) > 0.1:
                        adj_e = self.get_adjusted_extents()
                        model_size = np.max(adj_e) if adj_e is not None else 1.0
                        pan_speed = (model_size * 0.001) * zoom
                        if mouse_button == glfw.MOUSE_BUTTON_RIGHT or \
                           (mouse_button == glfw.MOUSE_BUTTON_LEFT and glfw.get_key(win, glfw.KEY_LEFT_SHIFT) == glfw.PRESS):
                            pan_x += dx * pan_speed
                            pan_y -= dy * pan_speed
                        elif mouse_button == glfw.MOUSE_BUTTON_LEFT and self.orthographic_view is None:
                            rotation_y += dx * 0.5
                            rotation_x += dy * 0.5
            last_x, last_y = xpos, ypos
        
        def scroll_callback(win, xoffset, yoffset):
            nonlocal zoom
            if yoffset != 0:
                zoom_factor = 1.15 if yoffset > 0 else 1.0 / 1.15
                zoom *= zoom_factor
                zoom = max(0.05, min(20.0, zoom))
        
        def key_callback(win, key, scancode, action, mods):
            if action == glfw.PRESS and key == glfw.KEY_ESCAPE:
                if self.measure_mode and (self.measure_point1 is not None or self.measure_point2 is not None):
                    self.measure_point1 = None
                    self.measure_point2 = None
                    self.measure_dragging = False
                    self.pending_measure_click = None
                    self.pending_measure_track = None
        
        glfw.set_mouse_button_callback(window, mouse_button_callback)
        glfw.set_cursor_pos_callback(window, cursor_pos_callback)
        glfw.set_scroll_callback(window, scroll_callback)
        glfw.set_key_callback(window, key_callback)
        
        # Bbox edges (indices into corners)
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
            [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
            [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
        ]
        
        # Main render loop
        while not glfw.window_should_close(window):
            glfw.poll_events()
            
            # Reset view
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
                self.orthographic_view = None
                self.measure_point1 = None
                self.measure_point2 = None
                self.measure_dragging = False
                self.pending_measure_click = None
                self.pending_measure_track = None
                self.reset_view_flag = False
            
            # View XY
            if self.view_xy_flag:
                rotation_x = 0.0
                rotation_y = 0.0
                zoom = 1.0
                pan_x = 0.0
                pan_y = 0.0
                camera_distance = initial_camera_distance
                self.orthographic_view = 'xy'
                self.measure_point1 = None
                self.measure_point2 = None
                self.measure_dragging = False
                self.pending_measure_click = None
                self.pending_measure_track = None
                self.view_xy_flag = False
            
            # View YZ
            if self.view_yz_flag:
                rotation_x = 0.0
                rotation_y = 0.0
                zoom = 1.0
                pan_x = 0.0
                pan_y = 0.0
                camera_distance = initial_camera_distance
                self.orthographic_view = 'yz'
                self.measure_point1 = None
                self.measure_point2 = None
                self.measure_dragging = False
                self.pending_measure_click = None
                self.pending_measure_track = None
                self.view_yz_flag = False
            
            # View XZ
            if self.view_xz_flag:
                rotation_x = 0.0
                rotation_y = 0.0
                zoom = 1.0
                pan_x = 0.0
                pan_y = 0.0
                camera_distance = initial_camera_distance
                self.orthographic_view = 'xz'
                self.measure_point1 = None
                self.measure_point2 = None
                self.measure_dragging = False
                self.pending_measure_click = None
                self.pending_measure_track = None
                self.view_xz_flag = False
            
            # Clear
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            
            width, height = glfw.get_framebuffer_size(window)
            glViewport(0, 0, width, height)
            aspect = width / height if height > 0 else 1.0
            
            # Projection: ortho vs perspective
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            if self.orthographic_view is not None:
                adj_e = self.get_adjusted_extents()
                max_extent = np.max(adj_e) if adj_e is not None else 1.0
                ortho_size = max_extent * 1.2 / zoom
                if aspect >= 1.0:
                    left, right = -ortho_size * aspect, ortho_size * aspect
                    bottom, top = -ortho_size, ortho_size
                else:
                    left, right = -ortho_size, ortho_size
                    bottom, top = -ortho_size / aspect, ortho_size / aspect
                glOrtho(left, right, bottom, top, -max_extent * 10, max_extent * 10)
            else:
                gluPerspective(45.0, aspect, 0.1, camera_distance * 10)
            
            # Modelview / camera
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            if self.orthographic_view is not None:
                if self.orthographic_view == 'xy':
                    gluLookAt(
                        self.bbox_center[0] + pan_x, self.bbox_center[1] + pan_y, self.bbox_center[2] + camera_distance,
                        self.bbox_center[0] + pan_x, self.bbox_center[1] + pan_y, self.bbox_center[2],
                        0, 1, 0
                    )
                elif self.orthographic_view == 'yz':
                    gluLookAt(
                        self.bbox_center[0] + camera_distance, self.bbox_center[1] + pan_y, self.bbox_center[2] + pan_x,
                        self.bbox_center[0], self.bbox_center[1] + pan_y, self.bbox_center[2] + pan_x,
                        0, 1, 0
                    )
                elif self.orthographic_view == 'xz':
                    gluLookAt(
                        self.bbox_center[0] + pan_x, self.bbox_center[1] + camera_distance, self.bbox_center[2] + pan_y,
                        self.bbox_center[0] + pan_x, self.bbox_center[1], self.bbox_center[2] + pan_y,
                        0, 0, -1
                    )
            else:
                current_distance = camera_distance / zoom
                dir_x = camera_x - self.bbox_center[0]
                dir_y = camera_y - self.bbox_center[1]
                dir_z = camera_z - self.bbox_center[2]
                dlen = np.sqrt(dir_x*dir_x + dir_y*dir_y + dir_z*dir_z)
                if dlen > 0.001:
                    f = current_distance / dlen
                    cx = self.bbox_center[0] + dir_x * f
                    cy = self.bbox_center[1] + dir_y * f
                    cz = self.bbox_center[2] + dir_z * f
                else:
                    cx, cy, cz = camera_x, camera_y, camera_z
                gluLookAt(
                    cx + pan_x, cy + pan_y, cz,
                    self.bbox_center[0] + pan_x, self.bbox_center[1] + pan_y, self.bbox_center[2],
                    0, 1, 0
                )
                glTranslatef(self.bbox_center[0], self.bbox_center[1], self.bbox_center[2])
                glRotatef(rotation_x, 1, 0, 0)
                glRotatef(rotation_y, 0, 1, 0)
                glTranslatef(-self.bbox_center[0], -self.bbox_center[1], -self.bbox_center[2])
            
            # Process pending measure (after matrices are set)
            if self.orthographic_view is not None:
                if self.pending_measure_click is not None:
                    x_fb, y_fb, w, h = self.pending_measure_click
                    self.pending_measure_click = None
                    pt = self.screen_to_world_on_view_plane(x_fb, y_fb, w, h, self.orthographic_view)
                    if pt is not None:
                        self.measure_point1 = pt.copy()
                        self.measure_dragging = True
                if self.pending_measure_track is not None:
                    x_fb, y_fb, w, h = self.pending_measure_track
                    self.pending_measure_track = None
                    pt = self.screen_to_world_on_view_plane(x_fb, y_fb, w, h, self.orthographic_view)
                    if pt is not None:
                        self.measure_point2 = pt.copy()
            
            # Adjusted geometry for display
            corners = self.get_adjusted_corners()
            adj_verts = self.get_adjusted_vertices()
            adj_ext = self.get_adjusted_extents()
            axis_origin = self.get_adjusted_bbox_min()
            if adj_ext is not None:
                glfw.set_window_title(window, f"OBJ Model - BBox: X={adj_ext[0]:.2f} Y={adj_ext[1]:.2f} Z={adj_ext[2]:.2f}")
            
            # Draw vertices as points (adjusted)
            if len(adj_verts) > 0:
                glPointSize(2.0)
                glColor3f(0.0, 0.2, 0.6)  # Darker blue for white background
                glBegin(GL_POINTS)
                for vertex in adj_verts:
                    glVertex3f(vertex[0], vertex[1], vertex[2])
                glEnd()
            
            # Draw bounding box wireframe (adjusted corners)
            glLineWidth(2.0)
            glColor3f(0.8, 0.0, 0.0)  # Darker red for white background
            for edge in edges:
                glBegin(GL_LINES)
                p1 = corners[edge[0]]
                p2 = corners[edge[1]]
                glVertex3f(p1[0], p1[1], p1[2])
                glVertex3f(p2[0], p2[1], p2[2])
                glEnd()
            
            # Draw coordinate axes at lower front corner (adjusted)
            axis_length = np.max(adj_ext) * 0.15 if adj_ext is not None else 0.15
            glLineWidth(6.0)  # 2x thicker for better visibility
            glBegin(GL_LINES)
            # X axis - Red (pointing right/positive X)
            glColor3f(0.8, 0.0, 0.0)
            glVertex3f(axis_origin[0], axis_origin[1], axis_origin[2])
            glVertex3f(axis_origin[0] + axis_length, axis_origin[1], axis_origin[2])
            # Y axis - Green (pointing up/positive Y)
            glColor3f(0.0, 0.6, 0.0)
            glVertex3f(axis_origin[0], axis_origin[1], axis_origin[2])
            glVertex3f(axis_origin[0], axis_origin[1] + axis_length, axis_origin[2])
            # Z axis - Blue (pointing forward/positive Z)
            glColor3f(0.0, 0.0, 0.8)
            glVertex3f(axis_origin[0], axis_origin[1], axis_origin[2])
            glVertex3f(axis_origin[0], axis_origin[1], axis_origin[2] + axis_length)
            glEnd()
            
            # Draw arrow heads for better visibility
            arrow_size = axis_length * 0.15
            glBegin(GL_TRIANGLES)
            # X arrow head
            glColor3f(0.8, 0.0, 0.0)
            x_end = axis_origin[0] + axis_length
            glVertex3f(x_end, axis_origin[1], axis_origin[2])
            glVertex3f(x_end - arrow_size, axis_origin[1] - arrow_size * 0.5, axis_origin[2])
            glVertex3f(x_end - arrow_size, axis_origin[1] + arrow_size * 0.5, axis_origin[2])
            # Y arrow head
            glColor3f(0.0, 0.6, 0.0)
            y_end = axis_origin[1] + axis_length
            glVertex3f(axis_origin[0], y_end, axis_origin[2])
            glVertex3f(axis_origin[0] - arrow_size * 0.5, y_end - arrow_size, axis_origin[2])
            glVertex3f(axis_origin[0] + arrow_size * 0.5, y_end - arrow_size, axis_origin[2])
            # Z arrow head
            glColor3f(0.0, 0.0, 0.8)
            z_end = axis_origin[2] + axis_length
            glVertex3f(axis_origin[0], axis_origin[1], z_end)
            glVertex3f(axis_origin[0] - arrow_size * 0.5, axis_origin[1], z_end - arrow_size)
            glVertex3f(axis_origin[0] + arrow_size * 0.5, axis_origin[1], z_end - arrow_size)
            glEnd()
            
            # Draw measurement line (ortho + measure mode, both points set)
            if (self.orthographic_view is not None and self.measure_mode and
                    self.measure_point1 is not None and self.measure_point2 is not None):
                glLineWidth(2.0)
                glColor3f(0.0, 0.6, 0.0)
                glBegin(GL_LINES)
                glVertex3f(self.measure_point1[0], self.measure_point1[1], self.measure_point1[2])
                glVertex3f(self.measure_point2[0], self.measure_point2[1], self.measure_point2[2])
                glEnd()
                glPointSize(6.0)
                glColor3f(0.0, 0.8, 0.0)
                glBegin(GL_POINTS)
                glVertex3f(self.measure_point1[0], self.measure_point1[1], self.measure_point1[2])
                glVertex3f(self.measure_point2[0], self.measure_point2[1], self.measure_point2[2])
                glEnd()
            
            # Draw text overlay with dimensions
            self._draw_text_overlay(width, height)
            
            # Draw measurement overlay (length box) when measuring
            self._draw_measurement_overlay(width, height)
            
            # Draw coordinate system indicator
            self._draw_coordinate_indicator(width, height)
            
            glfw.swap_buffers(window)
        
        glfw.terminate()
    
    def _draw_text_overlay(self, width, height):
        """Draw text overlay showing bounding box dimensions"""
        # Switch to 2D orthographic projection for text
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, width, height, 0, -1, 1)
        
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        # Disable depth test for overlay
        glDisable(GL_DEPTH_TEST)
        
        # Draw semi-transparent background for text
        text_x = 10
        text_y = 10
        text_width = 300
        text_height = 100
        
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glColor4f(0.0, 0.0, 0.0, 0.7)  # Semi-transparent black background
        glBegin(GL_QUADS)
        glVertex2f(text_x, text_y)
        glVertex2f(text_x + text_width, text_y)
        glVertex2f(text_x + text_width, text_y + text_height)
        glVertex2f(text_x, text_y + text_height)
        glEnd()
        
        # Render text using PIL if available, otherwise use simple method
        if PIL_AVAILABLE:
            try:
                # Create text image
                font_size = 16
                img = Image.new('RGBA', (text_width, text_height), (0, 0, 0, 0))
                draw = ImageDraw.Draw(img)
                
                try:
                    font = ImageFont.truetype("arial.ttf", font_size)
                except:
                    try:
                        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", font_size)
                    except:
                        font = ImageFont.load_default()
                
                adj_e = self.get_adjusted_extents()
                text_lines = [
                    f"Bounding Box Dimensions:",
                    f"X: {adj_e[0]:.6f} units",
                    f"Y: {adj_e[1]:.6f} units",
                    f"Z: {adj_e[2]:.6f} units"
                ]
                if self.adjustment != 1.0:
                    text_lines.append(f"Global adjustment: {self.adjustment:.4f}")
                
                y_offset = 5
                for line in text_lines:
                    draw.text((5, y_offset), line, fill=(255, 255, 255, 255), font=font)
                    y_offset += font_size + 2
                
                # Convert to texture (flip vertically for OpenGL)
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
                img_data = img.tobytes("raw", "RGBA", 0, -1)
                
                # Create and bind texture
                texture = glGenTextures(1)
                glBindTexture(GL_TEXTURE_2D, texture)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, text_width, text_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
                
                # Draw texture as quad
                glEnable(GL_TEXTURE_2D)
                glColor4f(1.0, 1.0, 1.0, 1.0)
                glBegin(GL_QUADS)
                glTexCoord2f(0, 0)
                glVertex2f(text_x, text_y)
                glTexCoord2f(1, 0)
                glVertex2f(text_x + text_width, text_y)
                glTexCoord2f(1, 1)
                glVertex2f(text_x + text_width, text_y + text_height)
                glTexCoord2f(0, 1)
                glVertex2f(text_x, text_y + text_height)
                glEnd()
                
                glDisable(GL_TEXTURE_2D)
                glDeleteTextures([texture])
                
            except Exception as e:
                # Fallback: draw simple text using lines (very basic)
                pass
        
        # Re-enable depth test
        glEnable(GL_DEPTH_TEST)
        
        # Restore matrices
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
    
    def _draw_measurement_overlay(self, width, height):
        """Draw measurement distance box when measuring in ortho view."""
        if not (self.measure_mode and self.orthographic_view is not None):
            return
        if self.measure_point1 is None:
            return
        if not (self.measure_point2 is not None or self.measure_dragging):
            return
        p2 = self.measure_point2 if self.measure_point2 is not None else self.measure_point1
        length = float(np.linalg.norm(np.array(p2) - np.array(self.measure_point1)))
        
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, width, height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        box_x, box_y = 10, 120  # Below dimension overlay
        box_w, box_h = 220, 36
        glColor4f(0.0, 0.0, 0.0, 0.7)
        glBegin(GL_QUADS)
        glVertex2f(box_x, box_y)
        glVertex2f(box_x + box_w, box_y)
        glVertex2f(box_x + box_w, box_y + box_h)
        glVertex2f(box_x, box_y + box_h)
        glEnd()
        
        if PIL_AVAILABLE:
            try:
                font_size = 16
                img = Image.new('RGBA', (box_w, box_h), (0, 0, 0, 0))
                draw = ImageDraw.Draw(img)
                try:
                    font = ImageFont.truetype("arial.ttf", font_size)
                except Exception:
                    try:
                        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", font_size)
                    except Exception:
                        font = ImageFont.load_default()
                draw.text((5, 4), f"Distance: {length:.4f} units", fill=(255, 255, 255, 255), font=font)
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
                img_data = img.tobytes("raw", "RGBA", 0, -1)
                texture = glGenTextures(1)
                glBindTexture(GL_TEXTURE_2D, texture)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, box_w, box_h, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
                glEnable(GL_TEXTURE_2D)
                glColor4f(1.0, 1.0, 1.0, 1.0)
                glBegin(GL_QUADS)
                glTexCoord2f(0, 0)
                glVertex2f(box_x, box_y)
                glTexCoord2f(1, 0)
                glVertex2f(box_x + box_w, box_y)
                glTexCoord2f(1, 1)
                glVertex2f(box_x + box_w, box_y + box_h)
                glTexCoord2f(0, 1)
                glVertex2f(box_x, box_y + box_h)
                glEnd()
                glDisable(GL_TEXTURE_2D)
                glDeleteTextures([texture])
            except Exception:
                pass
        glEnable(GL_DEPTH_TEST)
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
    
    def _draw_coordinate_indicator(self, width, height):
        """Draw a coordinate system indicator in the corner showing X, Y, Z axes"""
        # Switch to 2D orthographic projection for indicator
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, width, height, 0, -1, 1)
        
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        # Disable depth test for overlay
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Position in bottom-right corner
        indicator_size = 120
        indicator_x = width - indicator_size - 20
        indicator_y = height - indicator_size - 20
        center_x = indicator_x + indicator_size // 2
        center_y = indicator_y + indicator_size // 2
        axis_length = 40
        
        # Draw semi-transparent background
        glColor4f(0.0, 0.0, 0.0, 0.5)
        glBegin(GL_QUADS)
        glVertex2f(indicator_x - 10, indicator_y - 10)
        glVertex2f(indicator_x + indicator_size + 10, indicator_y - 10)
        glVertex2f(indicator_x + indicator_size + 10, indicator_y + indicator_size + 10)
        glVertex2f(indicator_x - 10, indicator_y + indicator_size + 10)
        glEnd()
        
        # Draw axes with arrows
        glLineWidth(3.0)
        
        # X axis - Red (pointing right)
        glColor3f(1.0, 0.0, 0.0)
        glBegin(GL_LINES)
        glVertex2f(center_x, center_y)
        glVertex2f(center_x + axis_length, center_y)
        glEnd()
        # X arrow head
        glBegin(GL_TRIANGLES)
        glVertex2f(center_x + axis_length, center_y)
        glVertex2f(center_x + axis_length - 8, center_y - 5)
        glVertex2f(center_x + axis_length - 8, center_y + 5)
        glEnd()
        
        # Y axis - Green (pointing up, but screen Y is inverted)
        glColor3f(0.0, 0.8, 0.0)
        glBegin(GL_LINES)
        glVertex2f(center_x, center_y)
        glVertex2f(center_x, center_y - axis_length)  # Negative because screen Y is inverted
        glEnd()
        # Y arrow head
        glBegin(GL_TRIANGLES)
        glVertex2f(center_x, center_y - axis_length)
        glVertex2f(center_x - 5, center_y - axis_length + 8)
        glVertex2f(center_x + 5, center_y - axis_length + 8)
        glEnd()
        
        # Z axis - Blue (pointing diagonally to represent depth)
        glColor3f(0.0, 0.0, 1.0)
        z_offset_x = axis_length * 0.707  # cos(45)
        z_offset_y = axis_length * 0.707  # sin(45)
        glBegin(GL_LINES)
        glVertex2f(center_x, center_y)
        glVertex2f(center_x + z_offset_x, center_y + z_offset_y)
        glEnd()
        # Z arrow head
        arrow_base_x = center_x + z_offset_x
        arrow_base_y = center_y + z_offset_y
        glBegin(GL_TRIANGLES)
        glVertex2f(arrow_base_x, arrow_base_y)
        glVertex2f(arrow_base_x - 6, arrow_base_y - 6)
        glVertex2f(arrow_base_x - 6, arrow_base_y + 6)
        glEnd()
        
        # Draw axis labels using PIL if available
        if PIL_AVAILABLE:
            try:
                label_size = indicator_size
                label_img = Image.new('RGBA', (label_size, label_size), (0, 0, 0, 0))
                label_draw = ImageDraw.Draw(label_img)
                
                try:
                    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 16)
                except:
                    try:
                        font = ImageFont.truetype("arial.ttf", 16)
                    except:
                        font = ImageFont.load_default()
                
                # Calculate label positions relative to center (which is at label_size/2, label_size/2)
                label_center = label_size // 2
                
                # X label (red) - at end of X axis (right side)
                x_label_x = label_center + axis_length + 5
                x_label_y = label_center - 8
                label_draw.text((x_label_x, x_label_y), "X", fill=(255, 0, 0, 255), font=font)
                
                # Y label (green) - at end of Y axis (top)
                y_label_x = label_center - 8
                y_label_y = label_center - axis_length - 20
                label_draw.text((y_label_x, y_label_y), "Y", fill=(0, 200, 0, 255), font=font)
                
                # Z label (blue) - at end of Z axis (diagonal)
                z_label_x = label_center + int(z_offset_x) + 5
                z_label_y = label_center + int(z_offset_y) + 5
                label_draw.text((z_label_x, z_label_y), "Z", fill=(0, 0, 255, 255), font=font)
                
                # Convert to texture
                label_img = label_img.transpose(Image.FLIP_TOP_BOTTOM)
                label_data = label_img.tobytes("raw", "RGBA", 0, -1)
                
                # Create and bind texture
                label_texture = glGenTextures(1)
                glBindTexture(GL_TEXTURE_2D, label_texture)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, label_size, label_size, 0, GL_RGBA, GL_UNSIGNED_BYTE, label_data)
                
                # Draw label texture
                glEnable(GL_TEXTURE_2D)
                glColor4f(1.0, 1.0, 1.0, 1.0)
                glBegin(GL_QUADS)
                glTexCoord2f(0, 0)
                glVertex2f(indicator_x, indicator_y)
                glTexCoord2f(1, 0)
                glVertex2f(indicator_x + label_size, indicator_y)
                glTexCoord2f(1, 1)
                glVertex2f(indicator_x + label_size, indicator_y + label_size)
                glTexCoord2f(0, 1)
                glVertex2f(indicator_x, indicator_y + label_size)
                glEnd()
                
                glDisable(GL_TEXTURE_2D)
                glDeleteTextures([label_texture])
                
            except Exception:
                pass
        
        # Re-enable depth test
        glEnable(GL_DEPTH_TEST)
        
        # Restore matrices
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
    
    def scale_vertices(self, target_x=None, target_y=None, target_z=None):
        """Scale vertices based on target dimensions. Uses adjusted geometry (global adjustment)."""
        if self.extents is None:
            return False
        
        adj_extents = self.get_adjusted_extents()
        adj_vertices = self.get_adjusted_vertices()
        center = self.bbox_center
        
        scale_factors = np.ones(3)
        if target_x is not None and adj_extents[0] > 0:
            scale_factors[0] = target_x / adj_extents[0]
        if target_y is not None and adj_extents[1] > 0:
            scale_factors[1] = target_y / adj_extents[1]
        if target_z is not None and adj_extents[2] > 0:
            scale_factors[2] = target_z / adj_extents[2]
        
        if target_x is not None and target_y is None and target_z is None:
            uniform_scale = scale_factors[0]
        elif target_y is not None and target_x is None and target_z is None:
            uniform_scale = scale_factors[1]
        elif target_z is not None and target_x is None and target_y is None:
            uniform_scale = scale_factors[2]
        else:
            active_scales = [s for i, s in enumerate(scale_factors) 
                           if (i == 0 and target_x is not None) or
                              (i == 1 and target_y is not None) or
                              (i == 2 and target_z is not None)]
            uniform_scale = min(active_scales) if active_scales else 1.0
        
        scaled_vertices = (adj_vertices - center) * uniform_scale + center
        return scaled_vertices, uniform_scale
    
    def rotate_vertices_neg90_deg_x(self, vertices):
        """Rotate vertices -90 degrees about the X axis.
        Y becomes -Z, Z becomes Y.
        Transformation: (x, y, z) -> (x, -z, y)
        
        Args:
            vertices: numpy array of vertices (N x 3)
            
        Returns:
            Rotated vertices (N x 3)
        """
        if len(vertices) == 0:
            return vertices
        
        rotated = vertices.copy()
        # Store original Y and Z values
        y_orig = rotated[:, 1].copy()
        z_orig = rotated[:, 2].copy()
        
        # Apply rotation: (x, y, z) -> (x, -z, y)
        rotated[:, 1] = -z_orig  # Y becomes -Z
        rotated[:, 2] = y_orig  # Z becomes Y
        
        return rotated
    
    def write_scaled_obj(self, scaled_vertices, output_path):
        """Write scaled OBJ file preserving all original data"""
        try:
            vertex_index = 0
            normal_index = 0
            uv_index = 0
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for line in self.obj_lines:
                    original_line = line
                    line = line.strip()
                    
                    if not line or line.startswith('#'):
                        f.write(original_line)
                        continue
                    
                    parts = line.split()
                    if not parts:
                        f.write(original_line)
                        continue
                    
                    if parts[0] == 'v':  # Vertex - replace with scaled version
                        if vertex_index < len(scaled_vertices):
                            v = scaled_vertices[vertex_index]
                            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                            vertex_index += 1
                        else:
                            f.write(original_line)
                    else:
                        f.write(original_line)
            
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Failed to write scaled OBJ: {str(e)}")
            return False
    
    def copy_mtl_and_textures(self, output_obj_path):
        """Copy MTL file and texture files to output location"""
        if not self.mtl_path or not os.path.exists(self.mtl_path):
            return True  # No MTL file, nothing to copy
        
        obj_dir = os.path.dirname(output_obj_path)
        output_mtl_name = os.path.splitext(os.path.basename(output_obj_path))[0] + ".mtl"
        output_mtl_path = os.path.join(obj_dir, output_mtl_name)
        
        # Copy MTL file
        try:
            shutil.copy2(self.mtl_path, output_mtl_path)
            
            # Parse MTL file to find texture references
            mtl_dir = os.path.dirname(self.mtl_path)
            texture_files = []
            
            with open(self.mtl_path, 'r', encoding='utf-8') as f:
                mtl_content = f.read()
                # Look for texture map references (map_Kd, map_Ks, map_Ka, map_bump, etc.)
                import re
                texture_patterns = [
                    r'map_Kd\s+(.+)',
                    r'map_Ks\s+(.+)',
                    r'map_Ka\s+(.+)',
                    r'map_bump\s+(.+)',
                    r'map_d\s+(.+)',
                    r'map_Ns\s+(.+)',
                ]
                
                for pattern in texture_patterns:
                    matches = re.findall(pattern, mtl_content, re.IGNORECASE)
                    for match in matches:
                        texture_path = match.strip()
                        # Handle both absolute and relative paths
                        if os.path.isabs(texture_path):
                            full_path = texture_path
                        else:
                            full_path = os.path.join(mtl_dir, texture_path)
                        
                        if os.path.exists(full_path):
                            texture_files.append((texture_path, full_path))
            
            # Copy texture files
            for rel_path, full_path in texture_files:
                texture_name = os.path.basename(rel_path)
                dest_path = os.path.join(obj_dir, texture_name)
                if not os.path.exists(dest_path):
                    shutil.copy2(full_path, dest_path)
            
            return True
            
        except Exception as e:
            messagebox.showwarning("Warning", f"Could not copy MTL/texture files: {str(e)}")
            return False
    
    def get_texture_file_from_mtl(self):
        """Extract texture file path from MTL file.
        Returns the first texture file found, or None if no texture."""
        if not self.mtl_path or not os.path.exists(self.mtl_path):
            return None
        
        try:
            mtl_dir = os.path.dirname(self.mtl_path)
            with open(self.mtl_path, 'r', encoding='utf-8') as f:
                mtl_content = f.read()
                import re
                texture_patterns = [
                    r'map_Kd\s+(.+)',
                    r'map_Ks\s+(.+)',
                    r'map_Ka\s+(.+)',
                    r'map_bump\s+(.+)',
                    r'map_d\s+(.+)',
                    r'map_Ns\s+(.+)',
                ]
                
                for pattern in texture_patterns:
                    matches = re.findall(pattern, mtl_content, re.IGNORECASE)
                    for match in matches:
                        texture_path = match.strip()
                        # Handle both absolute and relative paths
                        if os.path.isabs(texture_path):
                            full_path = texture_path
                        else:
                            full_path = os.path.join(mtl_dir, texture_path)
                        
                        if os.path.exists(full_path):
                            return full_path
        except Exception:
            pass
        
        return None
    
    def copy_mtl_and_update_textures(self, output_mtl_path, output_dir, suppress_dialogs=False):
        """Copy MTL file and update texture paths to point to textures in output directory.
        Also copies all texture files to output directory.
        
        Args:
            output_mtl_path: Path where the new MTL file should be saved
            output_dir: Directory where textures should be copied
            suppress_dialogs: If True, suppress error message dialogs (for batch processing)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.mtl_path or not os.path.exists(self.mtl_path):
            return True  # No MTL file, nothing to copy
        
        try:
            mtl_dir = os.path.dirname(self.mtl_path)
            
            # Read original MTL file
            with open(self.mtl_path, 'r', encoding='utf-8') as f:
                mtl_lines = f.readlines()
            
            # Find all texture references and copy textures
            import re
            # Pattern to match texture map lines: prefix, whitespace, texture path (may have trailing comment)
            # Capture the texture path (non-whitespace characters, may include path separators)
            texture_patterns = [
                (r'(map_Kd)\s+([^\s#]+)', 'map_Kd'),
                (r'(map_Ks)\s+([^\s#]+)', 'map_Ks'),
                (r'(map_Ka)\s+([^\s#]+)', 'map_Ka'),
                (r'(map_bump)\s+([^\s#]+)', 'map_bump'),
                (r'(map_d)\s+([^\s#]+)', 'map_d'),
                (r'(map_Ns)\s+([^\s#]+)', 'map_Ns'),
            ]
            
            # Process each line and update texture paths
            updated_lines = []
            for line in mtl_lines:
                updated_line = line
                
                # Check each texture pattern
                for pattern, prefix in texture_patterns:
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        texture_path = match.group(2).strip()  # Group 2 is the texture path
                        
                        # Handle both absolute and relative paths
                        if os.path.isabs(texture_path):
                            full_path = texture_path
                        else:
                            full_path = os.path.join(mtl_dir, texture_path)
                        
                        # Copy texture if it exists
                        if os.path.exists(full_path):
                            texture_name = os.path.basename(texture_path)
                            dest_texture_path = os.path.join(output_dir, texture_name)
                            
                            # Copy texture if not already copied
                            if not os.path.exists(dest_texture_path):
                                shutil.copy2(full_path, dest_texture_path)
                            
                            # Replace the texture path with just the filename
                            # This preserves the prefix, whitespace, and any trailing content (comments, etc.)
                            updated_line = re.sub(
                                pattern,
                                lambda m: f"{m.group(1)} {texture_name}",
                                line,
                                flags=re.IGNORECASE
                            )
                            break  # Only replace first match per line
                
                updated_lines.append(updated_line)
            
            # Write updated MTL file
            with open(output_mtl_path, 'w', encoding='utf-8') as f:
                f.writelines(updated_lines)
            
            return True
            
        except Exception as e:
            if not suppress_dialogs:
                messagebox.showerror("Error", f"Failed to copy MTL file: {str(e)}")
            return False
    
    def create_scaled_file_with_mtl(self, target_y, scale_name, output_dir, rotate=False, suppress_dialogs=False):
        """Create a scaled OBJ file with MTL file (copied from original) pointing to textures.
        
        Args:
            target_y: Target Y dimension in mm
            scale_name: Name for the scale (e.g., "G", "O", "S")
            output_dir: Directory to save the scaled files
            rotate: If True, rotate -90 degrees about X axis before saving
            suppress_dialogs: If True, suppress error message dialogs (for batch processing)
            
        Returns:
            (success, output_path) tuple
        """
        if self.extents is None:
            return False, None
        
        # Scale vertices to target Y dimension
        scaled_vertices, scale_factor = self.scale_vertices(target_y=target_y)
        
        # Rotate -90 degrees about X axis if requested
        if rotate:
            scaled_vertices = self.rotate_vertices_neg90_deg_x(scaled_vertices)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output filename
        obj_name = os.path.splitext(os.path.basename(self.obj_path))[0]
        output_obj_name = f"{obj_name}_{scale_name}.obj"
        output_obj_path = os.path.join(output_dir, output_obj_name)
        
        # Write scaled (and optionally rotated) OBJ
        if not self.write_scaled_obj(scaled_vertices, output_obj_path):
            return False, None
        
        # Copy MTL file and update texture paths
        output_mtl_name = f"{obj_name}_{scale_name}.mtl"
        output_mtl_path = os.path.join(output_dir, output_mtl_name)
        
        if not self.copy_mtl_and_update_textures(output_mtl_path, output_dir, suppress_dialogs=suppress_dialogs):
            return False, None
        
        # Update OBJ file to reference the MTL file
        try:
            # Read the OBJ file we just wrote
            with open(output_obj_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Check if mtllib already exists and update it, or add it
            mtllib_found = False
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.lower().startswith('mtllib'):
                    # Update existing mtllib line
                    lines[i] = f"mtllib {output_mtl_name}\n"
                    mtllib_found = True
                    break
            
            if not mtllib_found:
                # Add mtllib reference at the beginning (after comments)
                insert_pos = 0
                for i, line in enumerate(lines):
                    stripped = line.strip()
                    if stripped and not stripped.startswith('#'):
                        insert_pos = i
                        break
                
                lines.insert(insert_pos, f"mtllib {output_mtl_name}\n")
            
            # Write updated OBJ with mtllib reference
            with open(output_obj_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            
            return True, output_obj_path
            
        except Exception as e:
            if not suppress_dialogs:
                messagebox.showerror("Error", f"Failed to update OBJ file: {str(e)}")
            return False, None


class ScalingGUI:
    def __init__(self):
        self.scaler = OBJScaler()
        self.root = tk.Tk()
        self.root.title("OBJ Scaler")
        self.root.geometry("900x1000")  # Larger initial size to show all buttons, panes, and log window
        
        # Batch processing UI elements
        self.batch_folder_path = None
        self.file_listbox = None
        self.file_list_scrollbar = None
        self._s2_s3_filter_active = False
        
        # Standard scales in mm (Y dimension). None = 1:1 true scale (use model Y)
        self.standard_scales = {
            'G': 81,
            'O': 40,
            'S': 29,
            'HO': 20,
            'N': 12,
            'DD': 32,
            'true': None,
            '6in': 150,
            '4in': 100,
            '3in': 75
        }
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the GUI"""
        # File selection frame
        file_frame = ttk.Frame(self.root, padding="10")
        file_frame.pack(fill=tk.X)
        
        ttk.Label(file_frame, text="OBJ File:").pack(side=tk.LEFT, padx=5)
        self.file_label = ttk.Label(file_frame, text="No file selected", foreground="gray")
        self.file_label.pack(side=tk.LEFT, padx=5, expand=True)
        
        ttk.Button(file_frame, text="Select OBJ", command=self.load_obj).pack(side=tk.RIGHT, padx=5)
        
        # Batch processing frame
        batch_frame = ttk.LabelFrame(self.root, text="Batch Processing", padding="10")
        batch_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Folder selection
        folder_frame = ttk.Frame(batch_frame)
        folder_frame.pack(fill=tk.X, pady=5)
        ttk.Button(folder_frame, text="Select Folder", command=self.select_batch_folder).pack(side=tk.LEFT, padx=5)
        self.batch_folder_label = ttk.Label(folder_frame, text="No folder selected", foreground="gray")
        self.batch_folder_label.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(folder_frame, text="Show only _s2/_s3", command=self.show_only_s2_s3).pack(side=tk.LEFT, padx=5)
        
        # File list with scrollbar
        list_frame = ttk.Frame(batch_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.file_list_scrollbar = ttk.Scrollbar(list_frame)
        self.file_list_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.file_listbox = tk.Listbox(list_frame, yscrollcommand=self.file_list_scrollbar.set, height=6)
        self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.file_list_scrollbar.config(command=self.file_listbox.yview)
        
        # Bind selection events
        self.file_listbox.bind('<<ListboxSelect>>', self.on_file_selected)
        self.file_listbox.bind('<Return>', self.on_file_enter)
        self.file_listbox.bind('<Button-1>', lambda e: self.file_listbox.focus_set())
        
        # File count label
        self.file_count_label = ttk.Label(batch_frame, text="0 files found")
        self.file_count_label.pack(pady=2)
        
        # Bounding box info frame
        info_frame = ttk.LabelFrame(self.root, text="Bounding Box Information", padding="10")
        info_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.info_text = tk.Text(info_frame, height=6, wrap=tk.WORD, state=tk.DISABLED)
        self.info_text.pack(fill=tk.BOTH, expand=True)
        
        # Log window frame
        log_frame = ttk.LabelFrame(self.root, text="Processing Log", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Log text area with scrollbar
        log_text_frame = ttk.Frame(log_frame)
        log_text_frame.pack(fill=tk.BOTH, expand=True)
        
        self.log_scrollbar = ttk.Scrollbar(log_text_frame)
        self.log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.log_text = tk.Text(log_text_frame, height=8, wrap=tk.WORD, state=tk.DISABLED, 
                               yscrollcommand=self.log_scrollbar.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.log_scrollbar.config(command=self.log_text.yview)
        
        # Clear log button
        ttk.Button(log_frame, text="Clear Log", command=self.clear_log).pack(pady=5)
        
        # Global scale adjustment (applies before any scale)
        adj_frame = ttk.Frame(self.root, padding="10")
        adj_frame.pack(fill=tk.X)
        ttk.Label(adj_frame, text="Global scale adjustment:").pack(side=tk.LEFT, padx=5)
        self.adjustment_entry = ttk.Entry(adj_frame, width=10)
        self.adjustment_entry.insert(0, "1.0")
        self.adjustment_entry.pack(side=tk.LEFT, padx=2)
        ttk.Button(adj_frame, text="Apply", command=self.apply_adjustment).pack(side=tk.LEFT, padx=2)
        ttk.Label(adj_frame, text="(1.0 = no change, 1.5 = 1.5x, 0.5 = half)", foreground="gray").pack(side=tk.LEFT, padx=5)
        
        # Standard scales frame
        scales_frame = ttk.LabelFrame(self.root, text="Standard Scales (Y dimension in mm)", padding="10")
        scales_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # First row of scale buttons
        scale_row1 = ttk.Frame(scales_frame)
        scale_row1.pack(fill=tk.X, pady=2)
        ttk.Button(scale_row1, text="G (81mm)", command=lambda: self.create_scale('G')).pack(side=tk.LEFT, padx=2)
        ttk.Button(scale_row1, text="O (40mm)", command=lambda: self.create_scale('O')).pack(side=tk.LEFT, padx=2)
        ttk.Button(scale_row1, text="S (29mm)", command=lambda: self.create_scale('S')).pack(side=tk.LEFT, padx=2)
        ttk.Button(scale_row1, text="HO (20mm)", command=lambda: self.create_scale('HO')).pack(side=tk.LEFT, padx=2)
        
        # Second row of scale buttons
        scale_row2 = ttk.Frame(scales_frame)
        scale_row2.pack(fill=tk.X, pady=2)
        ttk.Button(scale_row2, text="N (12mm)", command=lambda: self.create_scale('N')).pack(side=tk.LEFT, padx=2)
        ttk.Button(scale_row2, text="6in (150mm)", command=lambda: self.create_scale('6in')).pack(side=tk.LEFT, padx=2)
        ttk.Button(scale_row2, text="4in (100mm)", command=lambda: self.create_scale('4in')).pack(side=tk.LEFT, padx=2)
        ttk.Button(scale_row2, text="3in (75mm)", command=lambda: self.create_scale('3in')).pack(side=tk.LEFT, padx=2)
        ttk.Button(scale_row2, text="DD 1:56 (32mm)", command=lambda: self.create_scale('DD')).pack(side=tk.LEFT, padx=2)
        ttk.Button(scale_row2, text="True 1:1 (m)", command=lambda: self.create_scale('true')).pack(side=tk.LEFT, padx=2)
        
        # Rotation checkbox
        rotation_frame = ttk.Frame(scales_frame)
        rotation_frame.pack(fill=tk.X, pady=5)
        self.rotate_checkbox_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(rotation_frame, text="Rotate -90° about X axis (Y→-Z, Z→Y)", 
                       variable=self.rotate_checkbox_var).pack(side=tk.LEFT, padx=5)
        
        # Run All checkbox
        run_all_frame = ttk.Frame(scales_frame)
        run_all_frame.pack(fill=tk.X, pady=5)
        self.run_all_checkbox_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(run_all_frame, text="Run All: Process all files in folder", 
                       variable=self.run_all_checkbox_var).pack(side=tk.LEFT, padx=5)
        
        # Create All button
        ttk.Button(scales_frame, text="Create All Scales", command=self.create_all_scales).pack(pady=5)
        
        # Manual scaling input frame (keep for custom scales)
        scale_frame = ttk.LabelFrame(self.root, text="Custom Scale (leave empty to keep original)", padding="10")
        scale_frame.pack(fill=tk.X, padx=10, pady=5)
        
        input_frame = ttk.Frame(scale_frame)
        input_frame.pack(fill=tk.X)
        
        ttk.Label(input_frame, text="X:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.x_entry = ttk.Entry(input_frame, width=15)
        self.x_entry.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(input_frame, text="Y:").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        self.y_entry = ttk.Entry(input_frame, width=15)
        self.y_entry.grid(row=0, column=3, padx=5, pady=5)
        
        ttk.Label(input_frame, text="Z:").grid(row=0, column=4, padx=5, pady=5, sticky=tk.W)
        self.z_entry = ttk.Entry(input_frame, width=15)
        self.z_entry.grid(row=0, column=5, padx=5, pady=5)
        
        # View frame (XY/YZ/XZ, Reset, Measure)
        view_frame = ttk.Frame(self.root, padding="10")
        view_frame.pack(fill=tk.X)
        ttk.Label(view_frame, text="Standard Views:").pack(side=tk.LEFT, padx=5)
        ttk.Button(view_frame, text="XY (Top)", command=self.set_xy_view).pack(side=tk.LEFT, padx=2)
        ttk.Button(view_frame, text="YZ (Front)", command=self.set_yz_view).pack(side=tk.LEFT, padx=2)
        ttk.Button(view_frame, text="XZ (Side)", command=self.set_xz_view).pack(side=tk.LEFT, padx=2)
        ttk.Button(view_frame, text="Reset View / Fit Object", command=self.reset_view).pack(side=tk.LEFT, padx=5)
        self.measure_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(view_frame, text="Measure Distance", variable=self.measure_var, command=self.toggle_measure).pack(side=tk.LEFT, padx=5)
        
        # Buttons frame
        button_frame = ttk.Frame(self.root, padding="10")
        button_frame.pack(fill=tk.X)
        
        ttk.Button(button_frame, text="Show Visualization", command=self.show_visualization).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Custom Scale and Save", command=self.scale_and_save).pack(side=tk.RIGHT, padx=5)
    
    def scan_folder_for_obj_files(self, folder_path):
        """Recursively scan folder and subfolders for OBJ files"""
        obj_files = []
        try:
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if file.lower().endswith('.obj'):
                        full_path = os.path.join(root, file)
                        obj_files.append(full_path)
            obj_files.sort()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to scan folder: {str(e)}")
        return obj_files
    
    def filter_s2_s3(self, paths):
        """Keep only paths whose basename (no .obj) ends with _s2 or _s3.
        Group by (directory, base). If both _s2 and _s3 exist for same base in same dir, keep only _s3.
        Returns filtered list of full paths, sorted.
        """
        kept = []
        for p in paths:
            base = os.path.splitext(os.path.basename(p))[0]
            if not (base.endswith('_s2') or base.endswith('_s3')):
                continue
            kept.append(p)
        groups = {}  # (dir, base) -> list of (path, suffix)
        for p in kept:
            d = os.path.dirname(p)
            stem = os.path.splitext(os.path.basename(p))[0]
            if stem.endswith('_s3'):
                base = stem[:-3]
                suf = '_s3'
            else:
                base = stem[:-3]
                suf = '_s2'
            key = (d, base)
            if key not in groups:
                groups[key] = []
            groups[key].append((p, suf))
        out = []
        for key, lst in groups.items():
            has_s3 = any(s == '_s3' for _, s in lst)
            if has_s3:
                for p, s in lst:
                    if s == '_s3':
                        out.append(p)
            else:
                for p, _ in lst:
                    out.append(p)
        out.sort()
        return out
    
    def select_batch_folder(self):
        """Open folder dialog to select folder for batch processing"""
        folder_path = filedialog.askdirectory(title="Select Folder with OBJ Files")
        if not folder_path:
            return
        
        self.batch_folder_path = folder_path
        self.batch_folder_label.config(text=os.path.basename(folder_path), foreground="black")
        self._s2_s3_filter_active = False
        
        # Scan for OBJ files
        obj_files = self.scan_folder_for_obj_files(folder_path)
        self.scaler.batch_file_list = obj_files
        
        # Update file listbox
        self.file_listbox.delete(0, tk.END)
        if len(obj_files) > 0:
            # Show relative paths from selected folder
            for file_path in obj_files:
                rel_path = os.path.relpath(file_path, folder_path)
                self.file_listbox.insert(tk.END, rel_path)
            
            self.file_count_label.config(text=f"{len(obj_files)} files found")
            self.file_listbox.focus_set()
        else:
            self.file_count_label.config(text="0 files found")
            messagebox.showinfo("Info", "No OBJ files found in selected folder")
    
    def show_only_s2_s3(self):
        """Filter file list to _s2/_s3 only; prefer _s3 when both exist in same folder."""
        if not self.scaler.batch_file_list:
            messagebox.showwarning("Warning", "Select a folder first.")
            return
        filtered = self.filter_s2_s3(self.scaler.batch_file_list)
        self.scaler.batch_file_list = filtered
        self._s2_s3_filter_active = True
        self.file_listbox.delete(0, tk.END)
        if filtered:
            for file_path in filtered:
                rel_path = os.path.relpath(file_path, self.batch_folder_path)
                self.file_listbox.insert(tk.END, rel_path)
            self.file_count_label.config(text=f"{len(filtered)} files (filtered _s2/_s3)")
            self.file_listbox.focus_set()
        else:
            self.file_count_label.config(text="0 files found")
            messagebox.showinfo("Info", "No _s2 or _s3 files found.")
    
    def on_file_selected(self, event=None):
        """Handle file selection from listbox"""
        if not self.file_listbox.curselection():
            return
        
        selected_indices = self.file_listbox.curselection()
        if not selected_indices:
            return
        
        file_index = selected_indices[0]
        self.navigate_to_file(file_index)
    
    def on_file_enter(self, event=None):
        """Handle Enter key to load selected file"""
        self.on_file_selected(event)
        return "break"
    
    def navigate_to_file(self, file_index):
        """Load and display the selected file"""
        if file_index < 0 or file_index >= len(self.scaler.batch_file_list):
            return
        
        file_path = self.scaler.batch_file_list[file_index]
        
        # Load new file
        self.scaler.obj_path = file_path
        self.scaler.current_file_index = file_index
        
        # Update file label
        self.file_label.config(text=os.path.basename(file_path), foreground="black")
        
        # Find MTL file
        self.scaler.find_mtl_file()
        
        # Parse the file
        if not self.scaler.parse_obj_file():
            messagebox.showerror("Error", f"Failed to parse OBJ file:\n{file_path}")
            return
        
        # Update status
        self.update_info_display()
        
        # Highlight current file in list
        self.file_listbox.selection_clear(0, tk.END)
        self.file_listbox.selection_set(file_index)
        self.file_listbox.see(file_index)
        self.file_listbox.activate(file_index)
        self.file_listbox.focus_set()
        
        # Auto-open visualizer
        self.show_visualization()
    
    def load_obj(self):
        """Load OBJ file"""
        if not self.scaler.select_obj_file():
            return
        
        self.file_label.config(text=os.path.basename(self.scaler.obj_path), foreground="black")
        
        # Find MTL file
        self.scaler.find_mtl_file()
        
        # Parse OBJ
        if not self.scaler.parse_obj_file():
            messagebox.showerror("Error", "Failed to parse OBJ file")
            return
        
        # Update info display
        self.update_info_display()
    
    def update_info_display(self):
        """Update bounding box information display"""
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)
        
        if self.scaler.extents is not None:
            adj_e = self.scaler.get_adjusted_extents()
            adj_min = self.scaler.get_adjusted_bbox_min()
            adj_max = self.scaler.get_adjusted_bbox_max()
            adj_c = self.scaler.get_adjusted_bbox_center()
            info = f"Current Extents:\n"
            info += f"  X: {adj_e[0]:.6f} units\n"
            info += f"  Y: {adj_e[1]:.6f} units\n"
            info += f"  Z: {adj_e[2]:.6f} units\n\n"
            info += f"Bounding Box:\n"
            info += f"  Min: ({adj_min[0]:.6f}, {adj_min[1]:.6f}, {adj_min[2]:.6f})\n"
            info += f"  Max: ({adj_max[0]:.6f}, {adj_max[1]:.6f}, {adj_max[2]:.6f})\n"
            info += f"  Center: ({adj_c[0]:.6f}, {adj_c[1]:.6f}, {adj_c[2]:.6f})\n\n"
            if self.scaler.adjustment != 1.0:
                info += f"Global adjustment: {self.scaler.adjustment:.4f}\n\n"
            info += f"Vertices: {len(self.scaler.vertices)}\n"
            info += f"Faces: {len(self.scaler.faces)}"
            
            if self.scaler.mtl_path:
                info += f"\nMTL file: {os.path.basename(self.scaler.mtl_path)}"
        
        self.info_text.config(state=tk.DISABLED)
    
    def log_message(self, message, newline=True):
        """Append a message to the log window
        
        Args:
            message: Message to append
            newline: If True, add newline at end (default True)
        """
        self.log_text.config(state=tk.NORMAL)
        if newline:
            self.log_text.insert(tk.END, message + "\n")
        else:
            self.log_text.insert(tk.END, message)
        self.log_text.see(tk.END)  # Auto-scroll to bottom
        self.log_text.config(state=tk.DISABLED)
        # Force GUI update
        self.root.update_idletasks()
    
    def clear_log(self):
        """Clear the log window"""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)
    
    def apply_adjustment(self):
        """Apply global scale adjustment from Entry. Must be > 0."""
        s = self.adjustment_entry.get().strip()
        if not s:
            s = "1.0"
        try:
            val = float(s)
        except ValueError:
            messagebox.showerror("Error", "Invalid value. Enter a positive number (e.g. 1.0, 1.5, 0.5).")
            return
        if val <= 0:
            messagebox.showerror("Error", "Global scale adjustment must be > 0.")
            return
        self.scaler.adjustment = val
        self.update_info_display()
    
    def show_visualization(self):
        """Show 3D visualization"""
        if self.scaler.extents is None:
            messagebox.showwarning("Warning", "Please load an OBJ file first")
            return
        
        self.scaler.visualize_bounding_box()
    
    def set_xy_view(self):
        """Set view to XY plane (top)"""
        if self.scaler.extents is None:
            messagebox.showwarning("Warning", "Please load an OBJ file first")
            return
        self.scaler.view_xy_flag = True
    
    def set_yz_view(self):
        """Set view to YZ plane (front)"""
        if self.scaler.extents is None:
            messagebox.showwarning("Warning", "Please load an OBJ file first")
            return
        self.scaler.view_yz_flag = True
    
    def set_xz_view(self):
        """Set view to XZ plane (side)"""
        if self.scaler.extents is None:
            messagebox.showwarning("Warning", "Please load an OBJ file first")
            return
        self.scaler.view_xz_flag = True
    
    def reset_view(self):
        """Reset camera view to fit object; return to perspective"""
        if self.scaler.extents is None:
            messagebox.showwarning("Warning", "Please load an OBJ file first")
            return
        self.scaler.reset_view_flag = True
    
    def toggle_measure(self):
        """Toggle measure distance mode"""
        self.scaler.measure_mode = self.measure_var.get()
    
    def scale_and_save(self):
        """Scale and save the OBJ file"""
        if self.scaler.extents is None:
            messagebox.showwarning("Warning", "Please load an OBJ file first")
            return
        
        # Get target dimensions
        target_x = None
        target_y = None
        target_z = None
        
        try:
            x_val = self.x_entry.get().strip()
            if x_val:
                target_x = float(x_val)
            
            y_val = self.y_entry.get().strip()
            if y_val:
                target_y = float(y_val)
            
            z_val = self.z_entry.get().strip()
            if z_val:
                target_z = float(z_val)
        except ValueError:
            messagebox.showerror("Error", "Invalid input. Please enter numeric values.")
            return
        
        if target_x is None and target_y is None and target_z is None:
            messagebox.showwarning("Warning", "Please enter at least one target dimension")
            return
        
        # Scale vertices
        scaled_vertices, scale_factor = self.scaler.scale_vertices(target_x, target_y, target_z)
        
        # Generate output path
        obj_dir = os.path.dirname(self.scaler.obj_path)
        obj_name = os.path.splitext(os.path.basename(self.scaler.obj_path))[0]
        output_path = os.path.join(obj_dir, obj_name + "_scaled.obj")
        
        # Write scaled OBJ
        if self.scaler.write_scaled_obj(scaled_vertices, output_path):
            # Copy MTL and textures
            self.scaler.copy_mtl_and_textures(output_path)
            
            messagebox.showinfo("Success", 
                              f"Scaled OBJ saved to:\n{output_path}\n\n"
                              f"Scale factor: {scale_factor:.6f}")
            
            # Update display with new extents; reset adjustment so display matches new base
            self.scaler.vertices = scaled_vertices
            self.scaler.calculate_bounding_box()
            self.scaler.adjustment = 1.0
            self.adjustment_entry.delete(0, tk.END)
            self.adjustment_entry.insert(0, "1.0")
            self.update_info_display()
    
    def is_scaled_file(self, file_path):
        """Check if a file is already a scaled file (in scaled folder or has scale suffix)"""
        # Check if file is in a "scaled" subdirectory
        normalized_path = os.path.normpath(file_path)
        if 'scaled' in normalized_path.split(os.sep):
            return True
        
        # Check if filename has a scale suffix (e.g., _G, _O, _S, etc.)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        for scale_name in self.standard_scales.keys():
            if base_name.endswith(f'_{scale_name}'):
                return True
        
        return False
    
    def get_original_files(self):
        """Get list of original files (excluding scaled files) from batch_file_list"""
        if not self.scaler.batch_file_list:
            return []
        
        original_files = []
        for file_path in self.scaler.batch_file_list:
            if not self.is_scaled_file(file_path):
                original_files.append(file_path)
        
        return original_files
    
    def get_files_to_process(self):
        """Files to use for Run All / Process all in folder.
        When _s2/_s3 filter is active, use exactly what's shown (batch_file_list).
        Otherwise use get_original_files() (exclude scaled)."""
        if not self.scaler.batch_file_list:
            return []
        if self._s2_s3_filter_active:
            return list(self.scaler.batch_file_list)
        return self.get_original_files()
    
    def find_session_metadata_path(self, obj_dir):
        """Look for session_metadata.json in the parent of obj_dir. Return path or None."""
        parent = os.path.normpath(os.path.join(obj_dir, ".."))
        path = os.path.join(parent, "session_metadata.json")
        return path if os.path.isfile(path) else None
    
    def parse_scale_from_notes(self, notes_str):
        """Parse notes for scale=xx (ratio) and scale=yy mm / scale=yymm. Returns list of
        {"kind": "ratio"|"mm", "value": float}. Validates > 0, deduplicates by (kind, value)."""
        if not notes_str or not isinstance(notes_str, str):
            return []
        seen = set()
        out = []
        # MM: scale=N mm or scale=Nmm
        for m in re.finditer(r'scale\s*=\s*(\d+(?:\.\d+)?)\s*mm\b', notes_str, re.IGNORECASE):
            v = float(m.group(1))
            if v > 0 and ("mm", v) not in seen:
                seen.add(("mm", v))
                out.append({"kind": "mm", "value": v})
        # Ratio: scale=N where N is not followed by optional space and "mm"
        # Use a positive constraint: number must be followed by \s*[\s,;.$]|$ to avoid
        # matching "5" in "scale=50 mm" (after 5 we have "0", not space/end)
        for m in re.finditer(r'scale\s*=\s*(\d+(?:\.\d+)?)(?=\s*[\s,;.]|\s*$)', notes_str, re.IGNORECASE):
            after = notes_str[m.end():m.end() + 10]
            if re.match(r'\s*mm\b', after, re.IGNORECASE):
                continue  # skip: this is the mm case
            v = float(m.group(1))
            if v > 0 and ("ratio", v) not in seen:
                seen.add(("ratio", v))
                out.append({"kind": "ratio", "value": v})
        return out
    
    def get_metadata_scale_specs(self, file_path):
        """Find session_metadata.json in parent of OBJ dir, read notes, parse scale=.
        Returns list of {"kind":"ratio"|"mm", "value":float} or []."""
        obj_dir = os.path.dirname(file_path)
        meta_path = self.find_session_metadata_path(obj_dir)
        if not meta_path:
            return []
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []
        notes = data.get("notes") if isinstance(data, dict) else None
        return self.parse_scale_from_notes(notes)
    
    def _value_to_scale_suffix(self, value, mm=False):
        """Convert numeric value to scale_name part: decimals become _. E.g. 13.7 -> '13_7', 10 -> '10'."""
        v = float(value)
        s = str(int(v)) if v == int(v) else str(v).replace(".", "_")
        return f"c{s}mm" if mm else f"c{s}"
    
    def get_metadata_scales_to_create(self, specs):
        """Given parse specs from get_metadata_scale_specs, and with scaler loaded (extents available),
        return [(target_y, scale_name), ...]. Ratio: target_y = current_Y / value; mm: target_y = value."""
        if not specs or self.scaler.extents is None:
            return []
        current_y = self.scaler.get_adjusted_extents()[1]
        result = []
        seen = set()
        for s in specs:
            if s["kind"] == "ratio":
                v = s["value"]
                target_y = current_y / v
                name = self._value_to_scale_suffix(v, mm=False)
            else:
                target_y = s["value"]
                name = self._value_to_scale_suffix(s["value"], mm=True)
            key = (target_y, name)
            if key not in seen:
                seen.add(key)
                result.append(key)
        return result
    
    def create_scale_for_file(self, file_path, scale_name, suppress_dialogs=False):
        """Create a scaled file for a specific file and scale name"""
        # Load the file
        self.scaler.obj_path = file_path
        self.scaler.find_mtl_file()
        
        if not self.scaler.parse_obj_file(suppress_dialogs=suppress_dialogs):
            return False
        
        if scale_name not in self.standard_scales:
            if not suppress_dialogs:
                messagebox.showerror("Error", f"Unknown scale: {scale_name}")
            return False
        
        target_y = self.standard_scales[scale_name]
        if target_y is None:
            target_y = self.scaler.get_adjusted_extents()[1]
        
        # Get output directory (scaled subdirectory of original file's directory)
        obj_dir = os.path.dirname(file_path)
        output_dir = os.path.join(obj_dir, "scaled")
        
        # Get rotation setting from checkbox
        rotate = self.rotate_checkbox_var.get()
        
        # Create scaled file
        success, output_path = self.scaler.create_scaled_file_with_mtl(target_y, scale_name, output_dir, rotate=rotate, suppress_dialogs=suppress_dialogs)
        
        # Metadata-driven scales from session_metadata.json notes (scale=xx, scale=yy mm)
        self.create_metadata_scales_for_file(file_path, suppress_dialogs=suppress_dialogs,
            log_message_fn=self.log_message if suppress_dialogs else None)
        
        return success
    
    def create_metadata_scales_for_file(self, file_path, suppress_dialogs=False, log_message_fn=None):
        """If session_metadata.json in parent of OBJ dir has scale= in notes, create those scales.
        Scaler must already be loaded with this file. output_dir=obj_dir/scaled, rotate from checkbox."""
        specs = self.get_metadata_scale_specs(file_path)
        if not specs:
            return
        scales = self.get_metadata_scales_to_create(specs)
        if not scales:
            return
        obj_dir = os.path.dirname(file_path)
        output_dir = os.path.join(obj_dir, "scaled")
        rotate = self.rotate_checkbox_var.get()
        for ty, name in scales:
            ok, _ = self.scaler.create_scaled_file_with_mtl(ty, name, output_dir, rotate=rotate, suppress_dialogs=suppress_dialogs)
            if log_message_fn and ok:
                log_message_fn(f"  ✓ Created {name} from metadata")
    
    def create_scale(self, scale_name):
        """Create a scaled file for the given scale name"""
        if scale_name not in self.standard_scales:
            messagebox.showerror("Error", f"Unknown scale: {scale_name}")
            return
        
        # Check if "Run All" is enabled
        run_all = self.run_all_checkbox_var.get()
        
        if run_all:
            # Process all files (filtered _s2/_s3 list when that filter is on, else originals only)
            files_to_process = self.get_files_to_process()
            
            if not files_to_process:
                messagebox.showwarning("Warning", "No OBJ files to process in folder")
                return
            
            # Clear log and start batch processing
            self.clear_log()
            desc = "1:1 (m)" if self.standard_scales[scale_name] is None else f"{self.standard_scales[scale_name]}mm"
            self.log_message(f"Starting batch processing: Scale {scale_name} ({desc})")
            self.log_message(f"Found {len(files_to_process)} files to process")
            self.log_message("-" * 60)
            
            # Batch process all files
            processed = 0
            failed = 0
            
            for i, file_path in enumerate(files_to_process, 1):
                file_name = os.path.basename(file_path)
                self.log_message(f"[{i}/{len(files_to_process)}] Processing: {file_name}")
                self.root.update_idletasks()
                
                if self.create_scale_for_file(file_path, scale_name, suppress_dialogs=True):
                    processed += 1
                    self.log_message(f"  ✓ Success: Created {scale_name} scale")
                else:
                    failed += 1
                    self.log_message(f"  ✗ Failed: Could not process file")
            
            # Show summary
            self.log_message("-" * 60)
            self.log_message(f"Batch processing complete!")
            self.log_message(f"  Successfully processed: {processed} files")
            self.log_message(f"  Failed: {failed} files")
            
            # Also show summary dialog
            desc = "1:1 (m)" if self.standard_scales[scale_name] is None else f"{self.standard_scales[scale_name]}mm"
            messagebox.showinfo("Batch Processing Complete", 
                              f"Processed {processed} files successfully\n"
                              f"Failed: {failed} files\n\n"
                              f"Scale: {scale_name} ({desc})")
        else:
            # Process current file only
            if self.scaler.extents is None:
                messagebox.showwarning("Warning", "Please load an OBJ file first")
                return
            
            target_y = self.standard_scales[scale_name]
            if target_y is None:
                target_y = self.scaler.get_adjusted_extents()[1]
            
            # Get output directory (scaled subdirectory of original file's directory)
            obj_dir = os.path.dirname(self.scaler.obj_path)
            output_dir = os.path.join(obj_dir, "scaled")
            
            # Get rotation setting from checkbox
            rotate = self.rotate_checkbox_var.get()
            
            # Create scaled file
            success, output_path = self.scaler.create_scaled_file_with_mtl(target_y, scale_name, output_dir, rotate=rotate)
            
            if success:
                self.create_metadata_scales_for_file(self.scaler.obj_path, suppress_dialogs=False, log_message_fn=None)
                rotate_text = " (rotated -90° about X)" if rotate else ""
                desc = "1:1 (m)" if self.standard_scales[scale_name] is None else f"{target_y}mm Y dimension"
                messagebox.showinfo("Success", 
                                  f"Scaled file created:\n{os.path.basename(output_path)}\n\n"
                                  f"Scale: {scale_name} ({desc}){rotate_text}\n"
                                  f"Saved to: {output_dir}")
            else:
                messagebox.showerror("Error", f"Failed to create scaled file for {scale_name}")
    
    def create_all_scales(self):
        """Create all standard scales at once"""
        # Check if "Run All" is enabled
        run_all = self.run_all_checkbox_var.get()
        
        if run_all:
            # Process all files (filtered _s2/_s3 list when that filter is on, else originals only)
            files_to_process = self.get_files_to_process()
            
            if not files_to_process:
                messagebox.showwarning("Warning", "No OBJ files to process in folder")
                return
            
            # Get rotation setting from checkbox
            rotate = self.rotate_checkbox_var.get()
            rotate_text = " (with -90° X rotation)" if rotate else ""
            
            # Clear log and start batch processing
            self.clear_log()
            self.log_message(f"Starting batch processing: All scales{rotate_text}")
            self.log_message(f"Found {len(files_to_process)} files to process")
            self.log_message("-" * 60)
            
            # Batch process all files with all scales
            total_processed = 0
            total_failed = 0
            files_processed = 0
            files_failed = 0
            
            for file_idx, file_path in enumerate(files_to_process, 1):
                file_name = os.path.basename(file_path)
                self.log_message(f"\n[{file_idx}/{len(files_to_process)}] Processing: {file_name}")
                self.root.update_idletasks()
                
                file_success = True
                file_created = []
                file_failed = []
                
                # Load the file
                self.scaler.obj_path = file_path
                self.scaler.find_mtl_file()
                
                if not self.scaler.parse_obj_file(suppress_dialogs=True):
                    self.log_message(f"  ✗ Failed to parse file")
                    files_failed += 1
                    total_failed += len(self.standard_scales)
                    continue
                
                # Get output directory
                obj_dir = os.path.dirname(file_path)
                output_dir = os.path.join(obj_dir, "scaled")
                
                # Create all scales for this file
                for scale_name, target_y in self.standard_scales.items():
                    ty = self.scaler.get_adjusted_extents()[1] if target_y is None else target_y
                    self.log_message(f"  Creating {scale_name} scale...", newline=False)
                    self.root.update_idletasks()
                    
                    success, output_path = self.scaler.create_scaled_file_with_mtl(
                        ty, scale_name, output_dir, rotate=rotate, suppress_dialogs=True
                    )
                    if success:
                        file_created.append(scale_name)
                        total_processed += 1
                        self.log_message(f" ✓")
                    else:
                        file_failed.append(scale_name)
                        total_failed += 1
                        file_success = False
                        self.log_message(f" ✗")
                
                self.create_metadata_scales_for_file(file_path, suppress_dialogs=True, log_message_fn=self.log_message)
                
                if file_success:
                    files_processed += 1
                    self.log_message(f"  All scales created successfully")
                else:
                    files_failed += 1
                    if file_created:
                        self.log_message(f"  Created: {', '.join(file_created)}")
                    if file_failed:
                        self.log_message(f"  Failed: {', '.join(file_failed)}")
            
            # Show summary
            self.log_message("\n" + "-" * 60)
            self.log_message(f"Batch processing complete{rotate_text}!")
            self.log_message(f"  Files processed successfully: {files_processed}")
            self.log_message(f"  Files with errors: {files_failed}")
            self.log_message(f"  Total scales created: {total_processed}")
            self.log_message(f"  Total scales failed: {total_failed}")
            
            # Also show summary dialog
            message = f"Batch Processing Complete{rotate_text}:\n\n"
            message += f"Files processed: {files_processed}\n"
            message += f"Files with errors: {files_failed}\n"
            message += f"Total scales created: {total_processed}\n"
            message += f"Total scales failed: {total_failed}"
            
            messagebox.showinfo("Batch Scale Complete", message)
        else:
            # Process current file only
            if self.scaler.extents is None:
                messagebox.showwarning("Warning", "Please load an OBJ file first")
                return
            
            # Get output directory
            obj_dir = os.path.dirname(self.scaler.obj_path)
            output_dir = os.path.join(obj_dir, "scaled")
            
            # Get rotation setting from checkbox
            rotate = self.rotate_checkbox_var.get()
            
            # Create all scales
            created = []
            failed = []
            
            for scale_name, target_y in self.standard_scales.items():
                ty = self.scaler.get_adjusted_extents()[1] if target_y is None else target_y
                success, output_path = self.scaler.create_scaled_file_with_mtl(ty, scale_name, output_dir, rotate=rotate)
                if success:
                    created.append(scale_name)
                else:
                    failed.append(scale_name)
            
            self.create_metadata_scales_for_file(self.scaler.obj_path, suppress_dialogs=False, log_message_fn=None)
            
            # Show results
            rotate_text = " (rotated -90° about X)" if rotate else ""
            message = f"Created {len(created)} scaled files{rotate_text}:\n"
            message += ", ".join(created) + "\n\n"
            message += f"Saved to: {output_dir}"
            
            if failed:
                message += f"\n\nFailed to create: {', '.join(failed)}"
            
            messagebox.showinfo("Batch Scale Complete", message)
    
    def run(self):
        """Run the GUI"""
        self.root.mainloop()


def main():
    app = ScalingGUI()
    app.run()


if __name__ == "__main__":
    main()

