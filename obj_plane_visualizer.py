"""
OBJ Plane Distance Visualizer
Loads an OBJ file, displays it in 3D, allows selecting 3 vertices to define a plane,
and highlights vertices within a specified distance from the plane.
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import os
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


class OBJPlaneVisualizer:
    def __init__(self):
        self.obj_path = None
        self.vertices = []
        self.faces = []
        self.obj_lines = []
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
        """Parse OBJ file to extract vertices and faces"""
        if not self.obj_path:
            return False
        
        self.vertices = []
        self.faces = []
        self.obj_lines = []
        
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
                    
                    elif parts[0] == 'f':  # Face
                        face_vertices = []
                        for part in parts[1:]:
                            indices = part.split('/')
                            if indices[0]:
                                face_vertices.append(int(indices[0]) - 1)  # OBJ is 1-indexed
                        if face_vertices:
                            self.faces.append(face_vertices)
            
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
    
    def find_nearest_vertex(self, mouse_x, mouse_y, camera_pos, ray_dir):
        """Find nearest vertex to the mouse click"""
        if len(self.vertices) == 0:
            return None
        
        min_distance = float('inf')
        nearest_vertex = None
        selection_tolerance = 0.1  # Fraction of bounding box size
        
        # Calculate selection radius based on model size
        model_size = np.max(self.extents)
        selection_radius = model_size * selection_tolerance
        
        for i, vertex in enumerate(self.vertices):
            # Vector from camera to vertex
            to_vertex = vertex - camera_pos
            
            # Project to_vertex onto ray direction
            projection_length = np.dot(to_vertex, ray_dir)
            
            # Closest point on ray to vertex
            closest_point = camera_pos + ray_dir * projection_length
            
            # Distance from vertex to ray
            distance = np.linalg.norm(vertex - closest_point)
            
            # Also check if vertex is in front of camera
            if projection_length > 0 and distance < selection_radius:
                if distance < min_distance:
                    min_distance = distance
                    nearest_vertex = i
        
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
        
        # Camera settings
        camera_distance = np.max(self.extents) * 2.5
        camera_x = self.bbox_center[0]
        camera_y = self.bbox_center[1]
        camera_z = self.bbox_center[2] + camera_distance
        
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
                    # Get mouse position
                    xpos, ypos = glfw.get_cursor_pos(window)
                    width, height = glfw.get_framebuffer_size(window)
                    
                    # Calculate camera position
                    camera_pos = np.array([camera_x + pan_x, camera_y + pan_y, camera_z])
                    
                    # Calculate view direction (toward center)
                    view_dir = np.array([
                        self.bbox_center[0] + pan_x - camera_pos[0],
                        self.bbox_center[1] + pan_y - camera_pos[1],
                        self.bbox_center[2] - camera_pos[2]
                    ])
                    view_dir = view_dir / np.linalg.norm(view_dir)
                    
                    # Calculate right and up vectors
                    right = np.cross(view_dir, np.array([0, 1, 0]))
                    if np.linalg.norm(right) < 0.1:
                        right = np.cross(view_dir, np.array([1, 0, 0]))
                    right = right / np.linalg.norm(right)
                    up = np.cross(right, view_dir)
                    up = up / np.linalg.norm(up)
                    
                    # Calculate field of view
                    fov_rad = np.radians(45.0)
                    aspect = width / height if height > 0 else 1.0
                    tan_fov = np.tan(fov_rad / 2.0)
                    
                    # Convert screen coordinates to normalized device coordinates
                    x_ndc = (2.0 * xpos / width) - 1.0
                    y_ndc = 1.0 - (2.0 * ypos / height)
                    
                    # Calculate ray direction in camera space
                    ray_camera_x = x_ndc * tan_fov * aspect
                    ray_camera_y = y_ndc * tan_fov
                    
                    # Transform to world space
                    ray_dir = view_dir + right * ray_camera_x + up * ray_camera_y
                    ray_dir = ray_dir / np.linalg.norm(ray_dir)
                    
                    nearest = self.find_nearest_vertex(xpos, ypos, camera_pos, ray_dir)
                    
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
            
            elif action == glfw.RELEASE:
                mouse_down = False
                mouse_button = None
        
        def cursor_pos_callback(window, xpos, ypos):
            nonlocal last_x, last_y, rotation_x, rotation_y, pan_x, pan_y
            if mouse_down:
                dx = xpos - last_x
                dy = ypos - last_y
                
                if mouse_button == glfw.MOUSE_BUTTON_LEFT and glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS:
                    # Pan
                    pan_x += dx * 0.01
                    pan_y -= dy * 0.01
                else:
                    # Rotate
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
                camera_x + pan_x, camera_y + pan_y, camera_z,
                self.bbox_center[0] + pan_x, self.bbox_center[1] + pan_y, self.bbox_center[2],
                0, 1, 0
            )
            
            # Rotate around center
            glTranslatef(self.bbox_center[0], self.bbox_center[1], self.bbox_center[2])
            glRotatef(rotation_x, 1, 0, 0)
            glRotatef(rotation_y, 0, 1, 0)
            glTranslatef(-self.bbox_center[0], -self.bbox_center[1], -self.bbox_center[2])
            
            # Draw vertices
            if len(self.vertices) > 0:
                glPointSize(3.0)
                glBegin(GL_POINTS)
                
                for i, vertex in enumerate(self.vertices):
                    # Color based on selection and distance
                    if i in self.selected_vertices:
                        # Selected vertices - yellow
                        glColor3f(1.0, 1.0, 0.0)
                    elif i in self.vertices_within_threshold:
                        # Vertices within threshold - red
                        glColor3f(1.0, 0.0, 0.0)
                    else:
                        # Default color - blue
                        glColor3f(0.0, 0.2, 0.6)
                    
                    glVertex3f(vertex[0], vertex[1], vertex[2])
                
                glEnd()
            
            # Draw selected vertices with larger markers
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
        self.root.geometry("400x350")
        
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
        
        # Instructions
        instructions = ttk.Label(
            self.root,
            text="Instructions:\n1. Load OBJ file\n2. Open 3D view\n3. Click 3 vertices to define plane\n4. Adjust distance threshold",
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
    
    def open_3d_view(self):
        """Open 3D visualization window"""
        if len(self.visualizer.vertices) == 0:
            messagebox.showwarning("Warning", "Please load an OBJ file first")
            return
        
        if not OPENGL_AVAILABLE:
            messagebox.showerror("Error", "PyOpenGL not available. Please install: pip install PyOpenGL glfw")
            return
        
        # Run OpenGL window in separate thread
        if self.visualizer.opengl_thread is None or not self.visualizer.opengl_thread.is_alive():
            self.visualizer.opengl_thread = threading.Thread(target=self.visualizer.run_opengl_window, daemon=True)
            self.visualizer.opengl_thread.start()
    
    def run(self):
        """Run the GUI"""
        self.root.mainloop()


def main():
    app = PlaneVisualizerGUI()
    app.run()


if __name__ == "__main__":
    main()

