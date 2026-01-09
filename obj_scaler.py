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
import numpy as np
import os
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
    
    def parse_obj_file(self):
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
        
        # Create window
        window = glfw.create_window(1024, 768, "OBJ Model with Bounding Box", None, None)
        if not window:
            glfw.terminate()
            messagebox.showerror("Error", "Failed to create GLFW window")
            return
        
        glfw.make_context_current(window)
        
        # Update window title with dimensions
        dim_text = f"OBJ Model - BBox: X={self.extents[0]:.2f} Y={self.extents[1]:.2f} Z={self.extents[2]:.2f}"
        glfw.set_window_title(window, dim_text)
        
        # OpenGL settings
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(1.0, 1.0, 1.0, 1.0)  # White background
        
        # Camera settings
        camera_distance = np.max(self.extents) * 2.5
        camera_x = self.bbox_center[0]
        camera_y = self.bbox_center[1]
        camera_z = self.bbox_center[2] + camera_distance
        
        rotation_x = 0.0
        rotation_y = 0.0
        zoom = 1.0
        
        # Mouse state
        last_x, last_y = 0, 0
        mouse_down = False
        
        def mouse_button_callback(window, button, action, mods):
            nonlocal mouse_down
            if button == glfw.MOUSE_BUTTON_LEFT:
                mouse_down = (action == glfw.PRESS)
        
        def cursor_pos_callback(window, xpos, ypos):
            nonlocal last_x, last_y, rotation_x, rotation_y
            if mouse_down:
                dx = xpos - last_x
                dy = ypos - last_y
                rotation_y += dx * 0.5
                rotation_x += dy * 0.5
            last_x, last_y = xpos, ypos
        
        def scroll_callback(window, xoffset, yoffset):
            nonlocal zoom
            zoom += yoffset * 0.1
            zoom = max(0.1, min(5.0, zoom))
        
        glfw.set_mouse_button_callback(window, mouse_button_callback)
        glfw.set_cursor_pos_callback(window, cursor_pos_callback)
        glfw.set_scroll_callback(window, scroll_callback)
        
        # Get bounding box corners
        corners = self.get_bounding_box_corners()
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
            [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
            [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
        ]
        
        # Main render loop
        while not glfw.window_should_close(window):
            glfw.poll_events()
            
            # Clear
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            
            # Set up projection
            width, height = glfw.get_framebuffer_size(window)
            glViewport(0, 0, width, height)
            
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            aspect = width / height if height > 0 else 1.0
            gluPerspective(45.0, aspect, 0.1, camera_distance * 10)
            
            # Set up modelview
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            
            # Camera position
            current_distance = camera_distance / zoom
            gluLookAt(
                camera_x, camera_y, camera_z,
                self.bbox_center[0], self.bbox_center[1], self.bbox_center[2],
                0, 1, 0
            )
            
            # Rotate around center
            glTranslatef(self.bbox_center[0], self.bbox_center[1], self.bbox_center[2])
            glRotatef(rotation_x, 1, 0, 0)
            glRotatef(rotation_y, 0, 1, 0)
            glTranslatef(-self.bbox_center[0], -self.bbox_center[1], -self.bbox_center[2])
            
            # Draw vertices as points
            if len(self.vertices) > 0:
                glPointSize(2.0)
                glColor3f(0.0, 0.2, 0.6)  # Darker blue for white background
                glBegin(GL_POINTS)
                for vertex in self.vertices:
                    glVertex3f(vertex[0], vertex[1], vertex[2])
                glEnd()
            
            # Draw bounding box wireframe
            glLineWidth(2.0)
            glColor3f(0.8, 0.0, 0.0)  # Darker red for white background
            for edge in edges:
                glBegin(GL_LINES)
                p1 = corners[edge[0]]
                p2 = corners[edge[1]]
                glVertex3f(p1[0], p1[1], p1[2])
                glVertex3f(p2[0], p2[1], p2[2])
                glEnd()
            
            # Draw coordinate axes at lower front corner
            # Lower front corner: min X, min Y, min Z (assuming Z increases going back)
            axis_origin = np.array([
                self.bbox_min[0],  # Lower (min X)
                self.bbox_min[1],  # Lower (min Y)
                self.bbox_min[2]   # Front (min Z)
            ])
            axis_length = np.max(self.extents) * 0.15  # Slightly smaller to fit better
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
            
            # Draw text overlay with dimensions
            self._draw_text_overlay(width, height)
            
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
                
                text_lines = [
                    f"Bounding Box Dimensions:",
                    f"X: {self.extents[0]:.6f} units",
                    f"Y: {self.extents[1]:.6f} units",
                    f"Z: {self.extents[2]:.6f} units"
                ]
                
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
        """Scale vertices based on target dimensions"""
        if self.extents is None:
            return False
        
        scale_factors = np.ones(3)
        
        if target_x is not None and self.extents[0] > 0:
            scale_factors[0] = target_x / self.extents[0]
        if target_y is not None and self.extents[1] > 0:
            scale_factors[1] = target_y / self.extents[1]
        if target_z is not None and self.extents[2] > 0:
            scale_factors[2] = target_z / self.extents[2]
        
        # Use uniform scaling (minimum scale factor to maintain proportions)
        # Or use the specified scale factor if only one dimension is provided
        if target_x is not None and target_y is None and target_z is None:
            uniform_scale = scale_factors[0]
        elif target_y is not None and target_x is None and target_z is None:
            uniform_scale = scale_factors[1]
        elif target_z is not None and target_x is None and target_y is None:
            uniform_scale = scale_factors[2]
        else:
            # If multiple dimensions specified, use minimum to maintain proportions
            active_scales = [s for i, s in enumerate(scale_factors) 
                           if (i == 0 and target_x is not None) or
                              (i == 1 and target_y is not None) or
                              (i == 2 and target_z is not None)]
            uniform_scale = min(active_scales) if active_scales else 1.0
        
        # Scale vertices relative to center
        scaled_vertices = (self.vertices - self.bbox_center) * uniform_scale + self.bbox_center
        
        return scaled_vertices, uniform_scale
    
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
    
    def copy_mtl_and_update_textures(self, output_mtl_path, output_dir):
        """Copy MTL file and update texture paths to point to textures in output directory.
        Also copies all texture files to output directory.
        
        Args:
            output_mtl_path: Path where the new MTL file should be saved
            output_dir: Directory where textures should be copied
            
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
            messagebox.showerror("Error", f"Failed to copy MTL file: {str(e)}")
            return False
    
    def create_scaled_file_with_mtl(self, target_y, scale_name, output_dir):
        """Create a scaled OBJ file with MTL file (copied from original) pointing to textures.
        
        Args:
            target_y: Target Y dimension in mm
            scale_name: Name for the scale (e.g., "G", "O", "S")
            output_dir: Directory to save the scaled files
            
        Returns:
            (success, output_path) tuple
        """
        if self.extents is None:
            return False, None
        
        # Scale vertices to target Y dimension
        scaled_vertices, scale_factor = self.scale_vertices(target_y=target_y)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output filename
        obj_name = os.path.splitext(os.path.basename(self.obj_path))[0]
        output_obj_name = f"{obj_name}_{scale_name}.obj"
        output_obj_path = os.path.join(output_dir, output_obj_name)
        
        # Write scaled OBJ
        if not self.write_scaled_obj(scaled_vertices, output_obj_path):
            return False, None
        
        # Copy MTL file and update texture paths
        output_mtl_name = f"{obj_name}_{scale_name}.mtl"
        output_mtl_path = os.path.join(output_dir, output_mtl_name)
        
        if not self.copy_mtl_and_update_textures(output_mtl_path, output_dir):
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
            messagebox.showerror("Error", f"Failed to update OBJ file: {str(e)}")
            return False, None


class ScalingGUI:
    def __init__(self):
        self.scaler = OBJScaler()
        self.root = tk.Tk()
        self.root.title("OBJ Scaler")
        self.root.geometry("600x700")
        
        # Batch processing UI elements
        self.batch_folder_path = None
        self.file_listbox = None
        self.file_list_scrollbar = None
        
        # Standard scales in mm (Y dimension)
        self.standard_scales = {
            'G': 81,
            'O': 40,
            'S': 29,
            'HO': 20,
            'N': 12,
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
    
    def select_batch_folder(self):
        """Open folder dialog to select folder for batch processing"""
        folder_path = filedialog.askdirectory(title="Select Folder with OBJ Files")
        if not folder_path:
            return
        
        self.batch_folder_path = folder_path
        self.batch_folder_label.config(text=os.path.basename(folder_path), foreground="black")
        
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
            info = f"Current Extents:\n"
            info += f"  X: {self.scaler.extents[0]:.6f} units\n"
            info += f"  Y: {self.scaler.extents[1]:.6f} units\n"
            info += f"  Z: {self.scaler.extents[2]:.6f} units\n\n"
            info += f"Bounding Box:\n"
            info += f"  Min: ({self.scaler.bbox_min[0]:.6f}, {self.scaler.bbox_min[1]:.6f}, {self.scaler.bbox_min[2]:.6f})\n"
            info += f"  Max: ({self.scaler.bbox_max[0]:.6f}, {self.scaler.bbox_max[1]:.6f}, {self.scaler.bbox_max[2]:.6f})\n"
            info += f"  Center: ({self.scaler.bbox_center[0]:.6f}, {self.scaler.bbox_center[1]:.6f}, {self.scaler.bbox_center[2]:.6f})\n\n"
            info += f"Vertices: {len(self.scaler.vertices)}\n"
            info += f"Faces: {len(self.scaler.faces)}"
            
            if self.scaler.mtl_path:
                info += f"\nMTL file: {os.path.basename(self.scaler.mtl_path)}"
        
        self.info_text.config(state=tk.DISABLED)
    
    def show_visualization(self):
        """Show 3D visualization"""
        if self.scaler.extents is None:
            messagebox.showwarning("Warning", "Please load an OBJ file first")
            return
        
        self.scaler.visualize_bounding_box()
    
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
            
            # Update display with new extents
            self.scaler.vertices = scaled_vertices
            self.scaler.calculate_bounding_box()
            self.update_info_display()
    
    def create_scale(self, scale_name):
        """Create a scaled file for the given scale name"""
        if self.scaler.extents is None:
            messagebox.showwarning("Warning", "Please load an OBJ file first")
            return
        
        if scale_name not in self.standard_scales:
            messagebox.showerror("Error", f"Unknown scale: {scale_name}")
            return
        
        target_y = self.standard_scales[scale_name]
        
        # Get output directory (scaled subdirectory of original file's directory)
        obj_dir = os.path.dirname(self.scaler.obj_path)
        output_dir = os.path.join(obj_dir, "scaled")
        
        # Create scaled file
        success, output_path = self.scaler.create_scaled_file_with_mtl(target_y, scale_name, output_dir)
        
        if success:
            messagebox.showinfo("Success", 
                              f"Scaled file created:\n{os.path.basename(output_path)}\n\n"
                              f"Scale: {scale_name} ({target_y}mm Y dimension)\n"
                              f"Saved to: {output_dir}")
        else:
            messagebox.showerror("Error", f"Failed to create scaled file for {scale_name}")
    
    def create_all_scales(self):
        """Create all standard scales at once"""
        if self.scaler.extents is None:
            messagebox.showwarning("Warning", "Please load an OBJ file first")
            return
        
        # Get output directory
        obj_dir = os.path.dirname(self.scaler.obj_path)
        output_dir = os.path.join(obj_dir, "scaled")
        
        # Create all scales
        created = []
        failed = []
        
        for scale_name, target_y in self.standard_scales.items():
            success, output_path = self.scaler.create_scaled_file_with_mtl(target_y, scale_name, output_dir)
            if success:
                created.append(scale_name)
            else:
                failed.append(scale_name)
        
        # Show results
        message = f"Created {len(created)} scaled files:\n"
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

