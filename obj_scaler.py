"""
OBJ Scaler with Bounding Box Visualization
Loads an OBJ file, calculates bounding box, displays it visually,
and allows scaling based on target dimensions.
"""

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


class ScalingGUI:
    def __init__(self):
        self.scaler = OBJScaler()
        self.root = tk.Tk()
        self.root.title("OBJ Scaler")
        self.root.geometry("500x400")
        
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
        
        # Bounding box info frame
        info_frame = ttk.LabelFrame(self.root, text="Bounding Box Information", padding="10")
        info_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.info_text = tk.Text(info_frame, height=8, wrap=tk.WORD, state=tk.DISABLED)
        self.info_text.pack(fill=tk.BOTH, expand=True)
        
        # Scaling input frame
        scale_frame = ttk.LabelFrame(self.root, text="Target Dimensions (leave empty to keep original)", padding="10")
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
        ttk.Button(button_frame, text="Scale and Save", command=self.scale_and_save).pack(side=tk.RIGHT, padx=5)
    
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
    
    def run(self):
        """Run the GUI"""
        self.root.mainloop()


def main():
    app = ScalingGUI()
    app.run()


if __name__ == "__main__":
    main()

