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
        self.open_edges = []  # List of open edges: [(v1, v2), ...]
        self.show_open_edges = True  # Flag to show/hide open edges
        self.edge_loops = []  # List of edge loops: [[(v1, v2), (v2, v3), ...], ...]
        self.selected_loops = set()  # Set of loop indices that are selected
        self.pending_edge_click = None  # (x, y, width, height) for edge selection
        self.edge_loop_selection_mode = False  # True = select edge loops, False = select vertices
        self.open_edges = []  # List of open edges: [(v1, v2), ...]
        self.show_open_edges = True  # Flag to show/hide open edges
        
        # Batch processing state
        self.has_unsaved_changes = False  # Flag to track if current file has been modified
        self.original_vertices = None  # Store original vertices for comparison
        self.batch_mode = False  # Whether batch mode is active
        self.current_file_index = -1  # Index of currently selected file in batch list
        self.batch_file_list = []  # List of full paths to OBJ files found
        self.close_window_flag = False  # Flag to signal OpenGL window to close
        
        # Window position/size memory (persists for session)
        self.saved_window_x = None  # Saved window X position
        self.saved_window_y = None  # Saved window Y position
        self.saved_window_width = 1024  # Default window width
        self.saved_window_height = 768  # Default window height
        
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
    
    def mark_as_changed(self):
        """Mark the current file as having unsaved changes"""
        self.has_unsaved_changes = True
    
    def reset_change_tracking(self):
        """Reset change tracking and store original vertices"""
        self.has_unsaved_changes = False
        if len(self.vertices) > 0:
            self.original_vertices = self.vertices.copy()
        else:
            self.original_vertices = None
    
    def close_opengl_window(self):
        """Signal the OpenGL window to close"""
        self.close_window_flag = True
        if self.opengl_window is not None:
            try:
                import glfw
                glfw.set_window_should_close(self.opengl_window, True)
            except:
                pass
    
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
            # Calculate open edges and loops after loading
            self.open_edges = self.detect_open_edges()
            self.edge_loops = self.find_edge_loops()
            self.selected_loops = set()
            # Reset change tracking after successful parse
            self.reset_change_tracking()
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
    
    def detect_open_edges(self):
        """Detect open/boundary edges in the mesh.
        An open edge is an edge that belongs to only one face.
        Returns a list of tuples (v1, v2) representing open edges, where v1 < v2."""
        if len(self.faces) == 0:
            return []
        
        # Dictionary to count how many times each edge appears
        # Edge is stored as (min(v1, v2), max(v1, v2)) to handle both directions
        edge_count = {}
        
        # Extract edges from all faces
        for face in self.faces:
            face_vertices = face['vertices']
            num_vertices = len(face_vertices)
            
            # Each face has edges between consecutive vertices
            for i in range(num_vertices):
                v1 = face_vertices[i]
                v2 = face_vertices[(i + 1) % num_vertices]  # Wrap around for last edge
                
                # Store edge with smaller index first for consistency
                edge = (min(v1, v2), max(v1, v2))
                edge_count[edge] = edge_count.get(edge, 0) + 1
        
        # Find edges that appear only once (open edges)
        open_edges = [edge for edge, count in edge_count.items() if count == 1]
        
        return open_edges
    
    def find_edge_loops(self):
        """Find connected edge loops from open edges.
        Returns a list of loops, where each loop is a list of edges in order."""
        if len(self.open_edges) == 0:
            return []
        
        # Build adjacency map: for each vertex, which edges connect to it
        # Store edges normalized (min, max) for consistency
        vertex_to_edges = {}
        
        # Create a set of all edges (normalized) for quick lookup
        all_edges_set = set()
        for edge in self.open_edges:
            v1, v2 = edge
            all_edges_set.add(edge)  # Already normalized (min, max)
            
            if v1 not in vertex_to_edges:
                vertex_to_edges[v1] = []
            if v2 not in vertex_to_edges:
                vertex_to_edges[v2] = []
            
            # Store both directions for traversal
            vertex_to_edges[v1].append(edge)
            vertex_to_edges[v2].append(edge)
        
        loops = []
        used_edges = set()
        
        # Find all loops - try each unused edge as a potential start
        for start_edge in self.open_edges:
            # Skip if this edge is already in a loop
            if start_edge in used_edges:
                continue
            
            # Start a new loop
            loop = []
            current_edge = start_edge
            v1, v2 = current_edge
            
            # Traverse forward from start_edge
            while current_edge not in used_edges:
                used_edges.add(current_edge)
                loop.append(current_edge)
                
                # Find next edge connected to v2 (the end vertex of current edge)
                # Look for edges that connect to v2 but aren't the current edge
                next_edge = None
                for candidate_edge in vertex_to_edges.get(v2, []):
                    # candidate_edge is normalized (min, max)
                    if candidate_edge == current_edge:
                        continue  # Skip the edge we're coming from
                    
                    # Check if this edge connects to v2
                    cand_v1, cand_v2 = candidate_edge
                    if cand_v1 == v2 or cand_v2 == v2:
                        # This edge connects to v2
                        if candidate_edge not in used_edges:
                            next_edge = candidate_edge
                            break
                
                if next_edge is not None:
                    # Move to next edge
                    current_edge = next_edge
                    # Update v1, v2 for next iteration
                    nv1, nv2 = current_edge
                    # v2 should be the vertex we're coming from, v1 should be the new vertex
                    if nv1 == v2:
                        v1, v2 = nv1, nv2
                    else:
                        v1, v2 = nv2, nv1
                else:
                    # No more connected edges, this loop is complete
                    break
            
            # Also try traversing backward from start_edge to complete the loop
            # Find the starting vertex (v1 of start_edge) and see if we can continue backward
            if len(loop) > 0:
                # Get the first vertex of the start edge
                start_v1, start_v2 = start_edge
                # We traversed forward from v2, so try backward from v1
                current_vertex = start_v1
                
                while True:
                    # Find edge connected to current_vertex (going backward)
                    prev_edge = None
                    for candidate_edge in vertex_to_edges.get(current_vertex, []):
                        if candidate_edge in used_edges:
                            continue
                        
                        cand_v1, cand_v2 = candidate_edge
                        if cand_v1 == current_vertex or cand_v2 == current_vertex:
                            prev_edge = candidate_edge
                            break
                    
                    if prev_edge is not None:
                        used_edges.add(prev_edge)
                        loop.insert(0, prev_edge)  # Add to beginning
                        # Update current_vertex to the other end of prev_edge
                        pv1, pv2 = prev_edge
                        if pv1 == current_vertex:
                            current_vertex = pv2
                        else:
                            current_vertex = pv1
                    else:
                        break
            
            if len(loop) > 0:
                loops.append(loop)
        
        print(f"DEBUG: find_edge_loops found {len(loops)} loops from {len(self.open_edges)} open edges")
        total_edges_in_loops = sum(len(loop) for loop in loops)
        print(f"DEBUG: Total edges in loops: {total_edges_in_loops}, unused edges: {len(self.open_edges) - total_edges_in_loops}")
        if len(loops) > 0:
            print(f"DEBUG: Loop sizes: {[len(loop) for loop in loops]}")
        
        return loops
    
    def find_nearest_edge(self, mouse_x, mouse_y, width, height):
        """Find the nearest open edge to the mouse position in screen space."""
        if len(self.open_edges) == 0:
            return None
        
        from OpenGL.GL import glGetDoublev, GL_MODELVIEW_MATRIX, GL_PROJECTION_MATRIX, GL_VIEWPORT
        from OpenGL.GLU import gluProject
        import ctypes
        
        # Get matrices and viewport (same way as vertex selection)
        modelview = (ctypes.c_double * 16)()
        projection = (ctypes.c_double * 16)()
        viewport = (ctypes.c_int * 4)()
        
        glGetDoublev(GL_MODELVIEW_MATRIX, modelview)
        glGetDoublev(GL_PROJECTION_MATRIX, projection)
        glGetIntegerv(GL_VIEWPORT, viewport)
        
        # Convert viewport to tuple for gluProject (same as vertex selection)
        vp_tuple = tuple(viewport)
        vp_x, vp_y, vp_width, vp_height = vp_tuple
        
        selection_radius_pixels = 30.0  # Increased from 20 to match vertex selection
        min_distance = float('inf')
        nearest_edge = None
        
        for edge in self.open_edges:
            v1_idx, v2_idx = edge
            if v1_idx >= len(self.vertices) or v2_idx >= len(self.vertices):
                continue
            
            v1 = self.vertices[v1_idx]
            v2 = self.vertices[v2_idx]
            
            # Project both vertices to screen space
            try:
                result1 = gluProject(v1[0], v1[1], v1[2], modelview, projection, viewport)
                result2 = gluProject(v2[0], v2[1], v2[2], modelview, projection, viewport)
                
                if result1 is None or result2 is None:
                    continue
                
                win_x1, win_y1, win_z1 = result1
                win_x2, win_y2, win_z2 = result2
                
                # Convert to screen coordinates (Y-axis flip) - same as vertex selection
                screen_x1 = win_x1
                screen_y1 = height - win_y1
                screen_x2 = win_x2
                screen_y2 = height - win_y2
                
                # Check if both vertices are in valid depth range
                if not (0.0 <= win_z1 <= 1.0 and 0.0 <= win_z2 <= 1.0):
                    continue
                
                # Calculate distance from mouse to edge (line segment)
                # Vector from v1 to v2
                dx = screen_x2 - screen_x1
                dy = screen_y2 - screen_y1
                edge_length_sq = dx * dx + dy * dy
                
                if edge_length_sq < 1e-6:  # Edge too short
                    continue
                
                # Vector from v1 to mouse
                mx = mouse_x - screen_x1
                my = mouse_y - screen_y1
                
                # Project mouse onto edge
                t = max(0.0, min(1.0, (mx * dx + my * dy) / edge_length_sq))
                
                # Closest point on edge
                closest_x = screen_x1 + t * dx
                closest_y = screen_y1 + t * dy
                
                # Distance from mouse to closest point
                dist_x = mouse_x - closest_x
                dist_y = mouse_y - closest_y
                distance = np.sqrt(dist_x * dist_x + dist_y * dist_y)
                
                if distance < selection_radius_pixels and distance < min_distance:
                    min_distance = distance
                    nearest_edge = edge
            except Exception as e:
                # Debug: print error if needed
                # print(f"Error in find_nearest_edge: {e}")
                continue
        
        return nearest_edge
    
    def get_loop_index_for_edge(self, edge):
        """Find which loop contains the given edge.
        Edge can be in any format (min,max) or (v1,v2), we need to check both."""
        # Normalize edge to (min, max) format for comparison
        edge_normalized = (min(edge[0], edge[1]), max(edge[0], edge[1]))
        
        for i, loop in enumerate(self.edge_loops):
            # Check all possible formats
            for loop_edge in loop:
                loop_edge_normalized = (min(loop_edge[0], loop_edge[1]), max(loop_edge[0], loop_edge[1]))
                # Check if edges match (in normalized form)
                if edge_normalized == loop_edge_normalized:
                    return i
        
        return None
    
    def _find_loop_from_edge(self, start_edge):
        """Find the loop containing the given edge by traversing connected edges.
        Returns a list of edges in the loop."""
        if start_edge not in self.open_edges:
            return []
        
        # Build adjacency map
        vertex_to_edges = {}
        for edge in self.open_edges:
            v1, v2 = edge
            if v1 not in vertex_to_edges:
                vertex_to_edges[v1] = []
            if v2 not in vertex_to_edges:
                vertex_to_edges[v2] = []
            vertex_to_edges[v1].append(edge)
            vertex_to_edges[v2].append(edge)
        
        loop = []
        used_edges = set()
        current_edge = start_edge
        v1, v2 = start_edge
        current_vertex = v2  # Start traversing from v2
        
        # Traverse forward
        while current_edge not in used_edges:
            used_edges.add(current_edge)
            loop.append(current_edge)
            
            # Find next edge
            next_edge = None
            for candidate_edge in vertex_to_edges.get(current_vertex, []):
                if candidate_edge == current_edge or candidate_edge in used_edges:
                    continue
                cand_v1, cand_v2 = candidate_edge
                if cand_v1 == current_vertex or cand_v2 == current_vertex:
                    next_edge = candidate_edge
                    break
            
            if next_edge is not None:
                current_edge = next_edge
                nv1, nv2 = next_edge
                if nv1 == current_vertex:
                    current_vertex = nv2
                else:
                    current_vertex = nv1
            else:
                break
        
        # Traverse backward from start
        if len(loop) > 0:
            start_v1, start_v2 = start_edge
            current_vertex = start_v1
            
            while True:
                prev_edge = None
                for candidate_edge in vertex_to_edges.get(current_vertex, []):
                    if candidate_edge in used_edges:
                        continue
                    cand_v1, cand_v2 = candidate_edge
                    if cand_v1 == current_vertex or cand_v2 == current_vertex:
                        prev_edge = candidate_edge
                        break
                
                if prev_edge is not None:
                    used_edges.add(prev_edge)
                    loop.insert(0, prev_edge)
                    pv1, pv2 = prev_edge
                    if pv1 == current_vertex:
                        current_vertex = pv2
                    else:
                        current_vertex = pv1
                else:
                    break
        
        return loop
    
    def get_open_edge_info(self):
        """Get information about open edges in the mesh.
        Returns a dictionary with statistics about open edges."""
        open_edges = self.detect_open_edges()
        
        # Count unique vertices that are part of open edges
        vertices_on_boundary = set()
        for v1, v2 in open_edges:
            vertices_on_boundary.add(v1)
            vertices_on_boundary.add(v2)
        
        return {
            'num_open_edges': len(open_edges),
            'num_boundary_vertices': len(vertices_on_boundary),
            'open_edges': open_edges,
            'boundary_vertices': vertices_on_boundary
        }
    
    def flatten_selected_loops(self, axis):
        """Flatten selected loops along the specified axis (0=X, 1=Y, 2=Z).
        Sets all vertices in selected loops to the minimum value for that axis.
        Returns (success, message) tuple."""
        if len(self.selected_loops) == 0:
            return False, "No loops selected"
        
        if axis not in [0, 1, 2]:
            return False, "Invalid axis"
        
        axis_names = ['X', 'Y', 'Z']
        
        # Collect all vertices in selected loops
        vertices_to_flatten = set()
        for loop_idx in self.selected_loops:
            if loop_idx < len(self.edge_loops):
                loop = self.edge_loops[loop_idx]
                for v1_idx, v2_idx in loop:
                    vertices_to_flatten.add(v1_idx)
                    vertices_to_flatten.add(v2_idx)
        
        if len(vertices_to_flatten) == 0:
            return False, "No vertices found in selected loops"
        
        # Find minimum value for the axis across all selected vertices
        min_value = float('inf')
        for v_idx in vertices_to_flatten:
            if v_idx < len(self.vertices):
                value = self.vertices[v_idx][axis]
                if value < min_value:
                    min_value = value
        
        if min_value == float('inf'):
            return False, "Could not determine minimum value"
        
        # Validate geometry BEFORE modifying (to avoid issues)
        # We'll do a quick check - if flattening would create overlapping vertices, warn
        # But we'll proceed anyway since the user wants to flatten
        
        # Flatten vertices
        vertices_modified = 0
        for v_idx in vertices_to_flatten:
            if v_idx < len(self.vertices):
                self.vertices[v_idx][axis] = min_value
                vertices_modified += 1
        
        # Recalculate bounding box and open edges
        # Do this in a try-except to prevent crashes
        try:
            self.calculate_bounding_box()
            self.open_edges = self.detect_open_edges()
            # Only recalculate loops if we have open edges and not too many (to avoid hanging)
            if len(self.open_edges) > 0 and len(self.open_edges) < 10000:
                self.edge_loops = self.find_edge_loops()
            else:
                # For large meshes, skip loop recalculation to prevent hanging
                if len(self.open_edges) >= 10000:
                    print(f"Warning: Too many open edges ({len(self.open_edges)}), skipping loop recalculation")
                self.edge_loops = []
        except Exception as e:
            # If recalculation fails, log but don't crash
            print(f"Warning: Error recalculating geometry after flattening: {e}")
            # Still mark for redraw
            pass
        
        self.needs_redraw = True
        
        # Mark as changed
        self.mark_as_changed()
        
        return True, f"Flattened {vertices_modified} vertices along {axis_names[axis]} axis to {min_value:.6f}"
    
    def validate_geometry(self):
        """Validate geometry for overlapping vertices and bad geometry.
        Returns (is_valid, error_message) tuple.
        Note: This is a quick check - for large meshes, it may skip some checks."""
        if len(self.vertices) == 0:
            return True, ""
        
        tolerance = 1e-6
        
        # For large meshes, skip expensive validation to prevent hanging
        # Only check if mesh is reasonably sized
        if len(self.vertices) > 10000:
            # Just check for degenerate faces (faster)
            degenerate_faces = []
            for face_idx, face in enumerate(self.faces):
                face_vertices = face['vertices']
                if len(face_vertices) < 3:
                    degenerate_faces.append(face_idx)
                    continue
                
                # Check for duplicate vertices in face
                if len(face_vertices) != len(set(face_vertices)):
                    degenerate_faces.append(face_idx)
            
            if len(degenerate_faces) > 0:
                return False, f"Found {len(degenerate_faces)} degenerate faces"
            return True, ""
        
        # For smaller meshes, do full validation
        # Check for overlapping vertices (within a small tolerance)
        overlapping_pairs = []
        
        # Limit the number of checks to prevent hanging
        max_checks = 1000000  # Limit to prevent O(n²) from hanging
        check_count = 0
        
        for i in range(len(self.vertices)):
            for j in range(i + 1, len(self.vertices)):
                check_count += 1
                if check_count > max_checks:
                    # Too many vertices, skip detailed overlap check
                    break
                v1 = self.vertices[i]
                v2 = self.vertices[j]
                distance = np.linalg.norm(v1 - v2)
                if distance < tolerance:
                    overlapping_pairs.append((i, j))
            if check_count > max_checks:
                break
        
        if len(overlapping_pairs) > 0:
            return False, f"Found {len(overlapping_pairs)} overlapping vertex pairs"
        
        # Check for degenerate faces (faces with zero area or duplicate vertices)
        degenerate_faces = []
        for face_idx, face in enumerate(self.faces):
            face_vertices = face['vertices']
            if len(face_vertices) < 3:
                degenerate_faces.append(face_idx)
                continue
            
            # Check for duplicate vertices in face
            if len(face_vertices) != len(set(face_vertices)):
                degenerate_faces.append(face_idx)
                continue
            
            # Check for zero area (if we have at least 3 vertices)
            if len(face_vertices) >= 3:
                v1 = self.vertices[face_vertices[0]]
                v2 = self.vertices[face_vertices[1]]
                v3 = self.vertices[face_vertices[2]]
                
                edge1 = v2 - v1
                edge2 = v3 - v1
                cross_product = np.cross(edge1, edge2)
                area = 0.5 * np.linalg.norm(cross_product)
                
                if area < tolerance:
                    degenerate_faces.append(face_idx)
        
        if len(degenerate_faces) > 0:
            return False, f"Found {len(degenerate_faces)} degenerate faces"
        
        return True, ""
    
    def delete_selected_vertices(self):
        """Delete vertices within the plane threshold that are only connected to other vertices in the selection.
        Only deletes faces that are completely within the selection window."""
        # Use vertices within threshold (the red ones) instead of just the 3 selected vertices
        if not self.vertices_within_threshold:
            return None, None, None
        
        # Convert to set for faster lookup
        selection_vertices = set(self.vertices_within_threshold)
        all_vertices = set(range(len(self.vertices)))
        outside_vertices = all_vertices - selection_vertices
        
        # Build connectivity graph: for each vertex, find all vertices it's connected to via faces
        vertex_connections = {}
        for vertex_idx in range(len(self.vertices)):
            vertex_connections[vertex_idx] = set()
        
        for face in self.faces:
            face_vertex_indices = face['vertices']
            # Each vertex in the face is connected to all other vertices in the face
            for i, v1 in enumerate(face_vertex_indices):
                for j, v2 in enumerate(face_vertex_indices):
                    if i != j:
                        vertex_connections[v1].add(v2)
        
        # Check each vertex in selection: only delete if it has NO connections to outside vertices
        vertices_to_delete = set()
        for vertex_idx in selection_vertices:
            # Check if this vertex is connected to any vertex outside the selection
            connected_to_outside = False
            for connected_vertex in vertex_connections[vertex_idx]:
                if connected_vertex in outside_vertices:
                    connected_to_outside = True
                    break
            
            # Only delete if not connected to any outside vertex
            if not connected_to_outside:
                vertices_to_delete.add(vertex_idx)
        
        # Find faces that are COMPLETELY within the deletable set (all vertices must be deletable)
        faces_to_remove = []
        for i, face in enumerate(self.faces):
            face_vertex_indices = face['vertices']
            # Check if ALL vertices in this face are in the deletable set
            if len(face_vertex_indices) > 0 and all(v_idx in vertices_to_delete for v_idx in face_vertex_indices):
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
        
        # Recalculate open edges after deletion
        self.open_edges = self.detect_open_edges()
        
        # Clear selection after deletion
        deleted_vertex_count = len(vertices_to_delete)
        deleted_face_count = len(faces_to_remove)
        remaining_vertex_count = len(self.vertices)
        
        self.selected_vertices = []
        self.plane_equation = None
        self.vertex_distances = None
        self.vertices_within_threshold = set()
        
        # Mark as changed
        self.mark_as_changed()
        
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
        
        # Use saved window size and position if available
        window_width = self.saved_window_width if self.saved_window_width else 1024
        window_height = self.saved_window_height if self.saved_window_height else 768
        
        window = glfw.create_window(window_width, window_height, "OBJ Plane Visualizer - Click 3 vertices to define plane", None, None)
        if not window:
            glfw.terminate()
            messagebox.showerror("Error", "Failed to create GLFW window")
            return
        
        # Set window position if we have saved position
        if self.saved_window_x is not None and self.saved_window_y is not None:
            glfw.set_window_pos(window, self.saved_window_x, self.saved_window_y)
        
        glfw.make_context_current(window)
        self.opengl_window = window
        
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
        click_start_x, click_start_y = 0, 0
        mouse_has_moved = False  # Track if mouse moved after click
        
        def mouse_button_callback(window, button, action, mods):
            nonlocal mouse_down, mouse_button, click_start_x, click_start_y, mouse_has_moved, last_x, last_y
            if action == glfw.PRESS:
                mouse_down = True
                mouse_button = button
                mouse_has_moved = False  # Reset movement flag
                
                # Store click start position and update last position to prevent immediate panning
                click_start_x, click_start_y = glfw.get_cursor_pos(window)
                last_x, last_y = click_start_x, click_start_y  # Initialize to prevent false movement
                
                if button == glfw.MOUSE_BUTTON_LEFT:
                    # Get mouse position (window coordinates, top-left origin)
                    xpos, ypos = glfw.get_cursor_pos(window)
                    width, height = glfw.get_framebuffer_size(window)
                    
                    # Convert window coordinates to framebuffer coordinates (for high-DPI)
                    # glfw.get_cursor_pos returns window coords, but we need framebuffer coords
                    window_width, window_height = glfw.get_window_size(window)
                    if window_width > 0 and window_height > 0:
                        scale_x = width / window_width
                        scale_y = height / window_height
                        xpos_fb = xpos * scale_x
                        ypos_fb = ypos * scale_y
                    else:
                        xpos_fb = xpos
                        ypos_fb = ypos
                    
                    # Check current selection mode
                    print(f"DEBUG: Mouse button pressed. Mode: {'Edge Loops' if self.edge_loop_selection_mode else 'Vertices'}")
                    if self.edge_loop_selection_mode:
                        # Edge loop selection mode
                        self.pending_edge_click = (xpos_fb, ypos_fb, width, height)
                        print(f"DEBUG: Set pending_edge_click = ({xpos_fb}, {ypos_fb}, {width}, {height})")
                    else:
                        # Vertex selection mode
                        self.pending_click = (xpos_fb, ypos_fb, width, height)
            
            elif action == glfw.RELEASE:
                # On release, if it was a click (no significant movement) and Shift was held, edge selection will be processed
                mouse_down = False
                mouse_button = None
                mouse_has_moved = False
        
        def cursor_pos_callback(window, xpos, ypos):
            nonlocal last_x, last_y, rotation_x, rotation_y, pan_x, pan_y, mouse_has_moved, click_start_x, click_start_y
            if mouse_down:
                # Check if mouse has moved significantly (more than 5 pixels from click start)
                if not mouse_has_moved:
                    move_distance = np.sqrt((xpos - click_start_x)**2 + (ypos - click_start_y)**2)
                    if move_distance > 5.0:  # Threshold for considering it a drag
                        mouse_has_moved = True
                        # Clear pending clicks if we're dragging - this prevents edge/vertex selection
                        if mouse_button == glfw.MOUSE_BUTTON_LEFT:
                            if self.edge_loop_selection_mode:
                                self.pending_edge_click = None
                            else:
                                self.pending_click = None
                
                    # Only pan/rotate if mouse has moved significantly (it's a drag, not a click)
                    # Also check that dx/dy are non-zero to avoid panning on first callback
                    if mouse_has_moved:
                        dx = xpos - last_x
                        dy = ypos - last_y
                        
                        # Only apply pan/rotate if there's actual movement (avoid first callback issue)
                        if abs(dx) > 0.1 or abs(dy) > 0.1:
                            # Right mouse button = Pan
                            # Shift+Left (when dragging) = Pan (for convenience)
                            if mouse_button == glfw.MOUSE_BUTTON_RIGHT or \
                               (mouse_button == glfw.MOUSE_BUTTON_LEFT and glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS):
                                # Pan - calculate pan speed based on zoom level and model size
                                model_size = np.max(self.extents) if self.extents is not None else 1.0
                                pan_speed = (model_size * 0.001) * zoom  # Pan more when zoomed in, scale with model size
                                pan_x += dx * pan_speed
                                pan_y -= dy * pan_speed
                            elif mouse_button == glfw.MOUSE_BUTTON_LEFT:
                                # Rotate (only if not in edge loop selection mode, or if dragging)
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
        while not glfw.window_should_close(window) and not self.close_window_flag:
            glfw.poll_events()
            
            # Check for close window flag
            if self.close_window_flag:
                glfw.set_window_should_close(window, True)
                break
            
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
            
            # Process pending edge click for loop selection (after matrices are set up)
            # Only process if it's still pending (wasn't cleared by drag)
            if self.pending_edge_click is not None:
                # Process the edge click - the mouse button check in cursor_pos_callback already
                # clears pending clicks if it's a drag, so if we get here it's a real click
                click_x, click_y, click_width, click_height = self.pending_edge_click
                self.pending_edge_click = None  # Clear the pending click immediately
                
                print(f"DEBUG: Edge click received at ({click_x}, {click_y}), window size: {click_width}x{click_height}")
                print(f"DEBUG: Open edges count: {len(self.open_edges)}, Edge loops count: {len(self.edge_loops)}")
                print(f"DEBUG: Currently selected loops: {self.selected_loops}")
                
                if len(self.open_edges) == 0:
                    # No open edges - silently ignore the click
                    print("DEBUG: No open edges detected!")
                else:
                    # Find nearest edge and select its loop
                    nearest_edge = self.find_nearest_edge(click_x, click_y, click_width, click_height)
                    
                    if nearest_edge is not None:
                        print(f"DEBUG: Found nearest edge: {nearest_edge}")
                        loop_idx = self.get_loop_index_for_edge(nearest_edge)
                        print(f"DEBUG: Loop index for edge: {loop_idx}")
                        
                        if loop_idx is not None:
                            # Toggle loop selection
                            if loop_idx in self.selected_loops:
                                print(f"DEBUG: Deselecting loop {loop_idx}")
                                self.selected_loops.remove(loop_idx)
                            else:
                                print(f"DEBUG: Selecting loop {loop_idx}")
                                self.selected_loops.add(loop_idx)
                            print(f"DEBUG: Selected loops after toggle: {self.selected_loops}")
                            self.needs_redraw = True
                            # Update GUI status
                            if self.gui:
                                try:
                                    self.gui.root.after(0, self.gui.update_status)
                                except:
                                    pass
                        else:
                            # Edge not in any loop - find the loop containing this edge by traversing from it
                            print(f"DEBUG: Edge {nearest_edge} not in any detected loop. Finding loop containing this edge...")
                            
                            # Build a loop starting from this edge
                            loop_edges = self._find_loop_from_edge(nearest_edge)
                            
                            if len(loop_edges) > 0:
                                # Check if this loop already exists (might be a duplicate)
                                existing_loop_idx = None
                                for i, existing_loop in enumerate(self.edge_loops):
                                    if set(loop_edges) == set(existing_loop):
                                        existing_loop_idx = i
                                        break
                                
                                if existing_loop_idx is not None:
                                    # Loop already exists, use it
                                    loop_idx = existing_loop_idx
                                    print(f"DEBUG: Found existing loop {loop_idx}")
                                else:
                                    # New loop, add it
                                    loop_idx = len(self.edge_loops)
                                    self.edge_loops.append(loop_edges)
                                    print(f"DEBUG: Created new loop {loop_idx} with {len(loop_edges)} edges")
                                
                                # Toggle selection
                                if loop_idx in self.selected_loops:
                                    print(f"DEBUG: Deselecting loop {loop_idx}")
                                    self.selected_loops.remove(loop_idx)
                                else:
                                    print(f"DEBUG: Selecting loop {loop_idx}")
                                    self.selected_loops.add(loop_idx)
                                print(f"DEBUG: Selected loops after toggle: {self.selected_loops}")
                                self.needs_redraw = True
                                if self.gui:
                                    try:
                                        self.gui.root.after(0, self.gui.update_status)
                                    except:
                                        pass
                            else:
                                # Couldn't find a loop - create a single-edge loop
                                print(f"DEBUG: Could not find loop, creating single-edge loop for edge {nearest_edge}")
                                new_loop_idx = len(self.edge_loops)
                                self.edge_loops.append([nearest_edge])
                                self.selected_loops.add(new_loop_idx)
                                print(f"DEBUG: Selected loops after adding single-edge loop: {self.selected_loops}")
                                self.needs_redraw = True
                                if self.gui:
                                    try:
                                        self.gui.root.after(0, self.gui.update_status)
                                    except:
                                        pass
                    else:
                        # No edge found near click - silently ignore (user might be clicking to rotate/pan)
                        print(f"DEBUG: No edge found near click position")
                        pass
            
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
            
            # Draw open edges if enabled
            if self.show_open_edges and len(self.open_edges) > 0:
                self._draw_open_edges()
            
            # Draw selected loops with different color
            if len(self.selected_loops) > 0:
                self._draw_selected_loops()
            
            # Draw plane if defined
            if self.plane_equation is not None:
                self._draw_plane()
            
            # Draw coordinate system
            self._draw_coordinate_system()
            
            glfw.swap_buffers(window)
        
        glfw.terminate()
        self.opengl_window = None
        self.close_window_flag = False  # Reset flag after window closes
    
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
    
    def _draw_selected_loops(self):
        """Draw selected edge loops with highlighted color"""
        if len(self.selected_loops) == 0 or len(self.vertices) == 0:
            return
        
        glLineWidth(5.0)
        glColor3f(0.0, 1.0, 1.0)  # Cyan color for selected loops
        
        glBegin(GL_LINES)
        for loop_idx in self.selected_loops:
            if loop_idx < len(self.edge_loops):
                loop = self.edge_loops[loop_idx]
                for v1_idx, v2_idx in loop:
                    if v1_idx < len(self.vertices) and v2_idx < len(self.vertices):
                        v1 = self.vertices[v1_idx]
                        v2 = self.vertices[v2_idx]
                        glVertex3f(v1[0], v1[1], v1[2])
                        glVertex3f(v2[0], v2[1], v2[2])
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
    
    def _draw_open_edges(self):
        """Draw open/boundary edges in the mesh"""
        if len(self.open_edges) == 0 or len(self.vertices) == 0:
            return
        
        glLineWidth(3.0)
        glColor3f(1.0, 0.0, 1.0)  # Magenta color for open edges
        
        glBegin(GL_LINES)
        for v1_idx, v2_idx in self.open_edges:
            if v1_idx < len(self.vertices) and v2_idx < len(self.vertices):
                v1 = self.vertices[v1_idx]
                v2 = self.vertices[v2_idx]
                glVertex3f(v1[0], v1[1], v1[2])
                glVertex3f(v2[0], v2[1], v2[2])
        glEnd()
    
    def _draw_selected_loops(self):
        """Draw selected edge loops with highlighted color"""
        if len(self.selected_loops) == 0 or len(self.vertices) == 0:
            return
        
        glLineWidth(5.0)
        glColor3f(0.0, 1.0, 1.0)  # Cyan color for selected loops
        
        glBegin(GL_LINES)
        for loop_idx in self.selected_loops:
            if loop_idx < len(self.edge_loops):
                loop = self.edge_loops[loop_idx]
                for v1_idx, v2_idx in loop:
                    if v1_idx < len(self.vertices) and v2_idx < len(self.vertices):
                        v1 = self.vertices[v1_idx]
                        v2 = self.vertices[v2_idx]
                        glVertex3f(v1[0], v1[1], v1[2])
                        glVertex3f(v2[0], v2[1], v2[2])
        glEnd()


class PlaneVisualizerGUI:
    instance = None
    
    def __init__(self):
        PlaneVisualizerGUI.instance = self
        self.visualizer = OBJPlaneVisualizer()
        self.visualizer.gui = self  # Reference to GUI for updates
        self.root = tk.Tk()
        self.root.title("OBJ Plane Distance Visualizer")
        self.root.geometry("800x600")
        
        # Store initial size for restoration if needed
        self.initial_geometry = "400x350"
        
        # Batch processing UI elements (will be created in setup_ui)
        self.batch_folder_path = None
        self.file_listbox = None
        self.file_list_scrollbar = None
        
        self.setup_ui()
    
    def scan_folder_for_obj_files(self, folder_path):
        """Recursively scan folder and subfolders for OBJ files
        
        Args:
            folder_path: Root folder path to scan
            
        Returns:
            List of full paths to OBJ files found, sorted
        """
        obj_files = []
        try:
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if file.lower().endswith('.obj'):
                        full_path = os.path.join(root, file)
                        obj_files.append(full_path)
            # Sort for consistent ordering
            obj_files.sort()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to scan folder: {str(e)}")
        return obj_files
    
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
        ttk.Button(view_frame, text="Save OBJ", command=self.save_obj_wrapper).pack(side=tk.LEFT, padx=5)
        
        # Selection mode frame
        selection_mode_frame = ttk.Frame(self.root, padding="10")
        selection_mode_frame.pack(fill=tk.X)
        
        self.selection_mode_var = tk.StringVar(value="Vertices")
        ttk.Label(selection_mode_frame, text="Selection Mode:").pack(side=tk.LEFT, padx=5)
        self.selection_mode_button = ttk.Button(
            selection_mode_frame, 
            text="Select Vertices", 
            command=self.toggle_selection_mode
        )
        self.selection_mode_button.pack(side=tk.LEFT, padx=5)
        
        # Edge loop flattening frame
        flatten_frame = ttk.LabelFrame(self.root, text="Flatten Selected Loops", padding="10")
        flatten_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(flatten_frame, text="Click 'Select Edge Loops' mode, then click on open edges to select loops:").pack(side=tk.LEFT, padx=5)
        ttk.Button(flatten_frame, text="Flatten X", command=lambda: self.flatten_loops(0)).pack(side=tk.LEFT, padx=2)
        ttk.Button(flatten_frame, text="Flatten Y", command=lambda: self.flatten_loops(1)).pack(side=tk.LEFT, padx=2)
        ttk.Button(flatten_frame, text="Flatten Z", command=lambda: self.flatten_loops(2)).pack(side=tk.LEFT, padx=2)
        
        # Plane view buttons frame
        plane_view_frame = ttk.Frame(self.root, padding="10")
        plane_view_frame.pack(fill=tk.X)
        
        ttk.Label(plane_view_frame, text="Standard Views:").pack(side=tk.LEFT, padx=5)
        ttk.Button(plane_view_frame, text="XY (Top)", command=self.set_xy_view).pack(side=tk.LEFT, padx=2)
        ttk.Button(plane_view_frame, text="YZ (Front)", command=self.set_yz_view).pack(side=tk.LEFT, padx=2)
        ttk.Button(plane_view_frame, text="XZ (Side)", command=self.set_xz_view).pack(side=tk.LEFT, padx=2)
        
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
        
        self.file_listbox = tk.Listbox(list_frame, yscrollcommand=self.file_list_scrollbar.set, height=8)
        self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.file_list_scrollbar.config(command=self.file_listbox.yview)
        
        # Bind selection events
        # Arrow keys work automatically with listbox - <<ListboxSelect>> will fire
        self.file_listbox.bind('<<ListboxSelect>>', self.on_file_selected)
        self.file_listbox.bind('<Return>', self.on_file_enter)
        # Bind focus events to ensure listbox can receive keyboard input
        self.file_listbox.bind('<Button-1>', lambda e: self.file_listbox.focus_set())
        
        # File count label
        self.file_count_label = ttk.Label(batch_frame, text="0 files found")
        self.file_count_label.pack(pady=2)
        
        # Instructions
        instructions = ttk.Label(
            self.root,
            text="Instructions:\n1. Load OBJ file\n2. Open 3D view\n3. Click 'Select Vertices' or 'Select Edge Loops' to choose mode\n4. Click 3 vertices OR click on open edges (magenta) to select loops\n5. Adjust distance threshold\n6. Click 'Delete Vertices in Plane' to delete all red vertices\n7. Use Flatten X/Y/Z to align selected loops\n8. Click 'Save OBJ' when done\n\n3D View Controls:\n• Left-click + drag: Rotate\n• Right-click + drag: Pan\n• Shift + Left-click + drag: Pan\n• Mouse wheel: Zoom\n• Reset View button: Fit object to center",
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
            status += f"Open edge loops: {len(self.visualizer.edge_loops)}\n"
            status += f"Selected loops: {len(self.visualizer.selected_loops)}\n"
            status += f"Selection mode: {'Edge Loops' if self.visualizer.edge_loop_selection_mode else 'Vertices'}\n"
            status += f"Selected vertices: {len(self.visualizer.selected_vertices)}/3\n"
            
            if self.visualizer.edge_loop_selection_mode:
                status += "Click on open edges (magenta) to select loops"
            elif len(self.visualizer.selected_vertices) == 3:
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
    
    def select_batch_folder(self):
        """Open folder dialog to select folder for batch processing"""
        folder_path = filedialog.askdirectory(title="Select Folder with OBJ Files")
        if not folder_path:
            return
        
        self.batch_folder_path = folder_path
        self.batch_folder_label.config(text=os.path.basename(folder_path), foreground="black")
        
        # Scan for OBJ files
        obj_files = self.scan_folder_for_obj_files(folder_path)
        self.visualizer.batch_file_list = obj_files
        self.visualizer.batch_mode = len(obj_files) > 0
        
        # Update file listbox
        self.file_listbox.delete(0, tk.END)
        if len(obj_files) > 0:
            # Show relative paths from selected folder
            for file_path in obj_files:
                rel_path = os.path.relpath(file_path, folder_path)
                self.file_listbox.insert(tk.END, rel_path)
            
            # Update file count
            self.file_count_label.config(text=f"{len(obj_files)} files found")
            # Set focus to listbox for keyboard navigation
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
        self.check_and_switch_file(file_index)
    
    
    def on_file_enter(self, event=None):
        """Handle Enter key to load selected file"""
        self.on_file_selected(event)
        return "break"
    
    def check_and_switch_file(self, new_index):
        """Check for unsaved changes before switching files"""
        if new_index < 0 or new_index >= len(self.visualizer.batch_file_list):
            return
        
        # If switching to the same file, do nothing
        if new_index == self.visualizer.current_file_index:
            return
        
        # Check for unsaved changes
        if self.visualizer.has_unsaved_changes:
            response = messagebox.askyesnocancel(
                "Unsaved Changes",
                "The current file has unsaved changes.\n\n"
                "Yes: Save changes and switch\n"
                "No: Discard changes and switch\n"
                "Cancel: Stay on current file"
            )
            
            if response is None:  # Cancel
                # Restore previous selection
                if self.visualizer.current_file_index >= 0:
                    self.file_listbox.selection_clear(0, tk.END)
                    self.file_listbox.selection_set(self.visualizer.current_file_index)
                    self.file_listbox.activate(self.visualizer.current_file_index)
                    self.file_listbox.see(self.visualizer.current_file_index)
                    self.file_listbox.focus_set()
                return
            elif response is True:  # Save
                if not self.save_obj(use_edit_prefix=True):
                    # Save was cancelled or failed, don't switch
                    if self.visualizer.current_file_index >= 0:
                        self.file_listbox.selection_clear(0, tk.END)
                        self.file_listbox.selection_set(self.visualizer.current_file_index)
                        self.file_listbox.activate(self.visualizer.current_file_index)
                        self.file_listbox.see(self.visualizer.current_file_index)
                        self.file_listbox.focus_set()
                    return
        
        # Proceed to switch file
        self.navigate_to_file(new_index)
    
    def navigate_to_file(self, file_index):
        """Load and display the selected file"""
        if file_index < 0 or file_index >= len(self.visualizer.batch_file_list):
            return
        
        file_path = self.visualizer.batch_file_list[file_index]
        
        # Close current OpenGL window if open
        if self.visualizer.opengl_window is not None:
            self.visualizer.close_opengl_window()
            # Wait a bit for window to close
            self.root.update()
            time.sleep(0.1)
        
        # Load new file
        self.visualizer.obj_path = file_path
        self.visualizer.current_file_index = file_index
        
        # Update file label
        self.file_label.config(text=os.path.basename(file_path), foreground="black")
        
        # Parse the file
        if not self.visualizer.parse_obj_file():
            messagebox.showerror("Error", f"Failed to parse OBJ file:\n{file_path}")
            return
        
        # Reset to "Select Vertices" mode
        self.visualizer.edge_loop_selection_mode = False
        self.selection_mode_button.config(text="Select Vertices (Active)")
        self.selection_mode_var.set("Vertices")
        
        # Update status
        self.update_status()
        
        # Highlight current file in list and ensure it stays highlighted
        self.file_listbox.selection_clear(0, tk.END)
        self.file_listbox.selection_set(file_index)
        self.file_listbox.see(file_index)
        self.file_listbox.activate(file_index)  # Set active item
        # Set focus to listbox so arrow keys work
        self.file_listbox.focus_set()
        
        # Auto-open visualizer
        self.open_3d_view()
    
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
    
    def toggle_selection_mode(self):
        """Toggle between vertex selection and edge loop selection modes"""
        self.visualizer.edge_loop_selection_mode = not self.visualizer.edge_loop_selection_mode
        
        print(f"DEBUG: Selection mode toggled to: {'Edge Loops' if self.visualizer.edge_loop_selection_mode else 'Vertices'}")
        print(f"DEBUG: Open edges: {len(self.visualizer.open_edges)}, Edge loops: {len(self.visualizer.edge_loops)}")
        
        if self.visualizer.edge_loop_selection_mode:
            self.selection_mode_button.config(text="Select Edge Loops (Active)")
            self.selection_mode_var.set("Edge Loops")
        else:
            self.selection_mode_button.config(text="Select Vertices (Active)")
            self.selection_mode_var.set("Vertices")
        
        self.update_status()
    
    def delete_vertices(self):
        """Delete all vertices within the plane threshold and save to new file"""
        if not self.visualizer.vertices_within_threshold:
            messagebox.showwarning("Warning", "Please define a plane first by selecting 3 vertices.\nThen adjust the distance threshold to select vertices to delete.")
            return
        
        if len(self.visualizer.vertices) == 0:
            messagebox.showwarning("Warning", "No OBJ file loaded")
            return
        
        # Note: We can't accurately count what will be deleted without running the deletion logic
        # because we need to check connectivity. So we'll show an estimate.
        selection_vertices = set(self.visualizer.vertices_within_threshold)
        
        # Perform deletion (no auto-save - user will save manually)
        deleted_verts, deleted_faces, remaining_verts = self.visualizer.delete_selected_vertices()
        
        if deleted_verts is None:
            messagebox.showerror("Error", "Failed to delete vertices")
            return
        
        # Update status and reload visualization
        self.update_status()
        self.visualizer.needs_redraw = True
        
        # Automatically switch to "Select Edge Loops" mode
        if not self.visualizer.edge_loop_selection_mode:
            self.visualizer.edge_loop_selection_mode = True
            self.selection_mode_button.config(text="Select Edge Loops (Active)")
            self.selection_mode_var.set("Edge Loops")
            self.update_status()
    
    def flatten_loops(self, axis):
        """Flatten selected loops along specified axis (0=X, 1=Y, 2=Z)"""
        if len(self.visualizer.selected_loops) == 0:
            messagebox.showwarning("Warning", "Please select edge loops first.\nUse Shift+Click on open edges to select loops.")
            return
        
        axis_names = ['X', 'Y', 'Z']
        success, message = self.visualizer.flatten_selected_loops(axis)
        
        if success:
            # No confirmation - just update status
            self.update_status()
            self.visualizer.needs_redraw = True
        else:
            messagebox.showerror("Error", message)
    
    def save_obj_wrapper(self):
        """Wrapper for save button that checks batch mode"""
        use_edit_prefix = self.visualizer.batch_mode and self.visualizer.current_file_index >= 0
        self.save_obj(use_edit_prefix=use_edit_prefix)
    
    def save_obj(self, use_edit_prefix=False):
        """Save the current OBJ file
        
        Args:
            use_edit_prefix: If True, save with "_edit" prefix in same directory as original
        """
        if len(self.visualizer.vertices) == 0:
            messagebox.showwarning("Warning", "No OBJ file loaded")
            return False
        
        if not self.visualizer.obj_path:
            messagebox.showwarning("Warning", "No OBJ file loaded")
            return False
        
        # Validate geometry before saving
        is_valid, error_msg = self.visualizer.validate_geometry()
        if not is_valid:
            response = messagebox.askyesno(
                "Geometry Validation Warning",
                f"Geometry validation found issues:\n{error_msg}\n\nDo you want to save anyway?"
            )
            if not response:
                return False
        
        # Determine output path
        if use_edit_prefix:
            # Save with "_edit" prefix in same directory as original
            base_name = os.path.splitext(os.path.basename(self.visualizer.obj_path))[0]
            dir_name = os.path.dirname(self.visualizer.obj_path)
            output_path = os.path.join(dir_name, f"{base_name}_edit.obj")
        else:
            # Ask for output file location
            root = tk.Tk()
            root.withdraw()
            
            # Suggest output filename
            base_name = os.path.splitext(os.path.basename(self.visualizer.obj_path))[0]
            dir_name = os.path.dirname(self.visualizer.obj_path)
            default_name = os.path.join(dir_name, f"{base_name}_edited.obj")
            
            output_path = filedialog.asksaveasfilename(
                title="Save OBJ file",
                defaultextension=".obj",
                filetypes=[("OBJ files", "*.obj"), ("All files", "*.*")],
                initialfile=os.path.basename(default_name),
                initialdir=os.path.dirname(default_name) if os.path.dirname(default_name) else "."
            )
            
            root.destroy()
            
            if not output_path:
                return False
        
        # Write the OBJ file
        if self.visualizer.write_obj_file(output_path):
            # Reset change tracking after successful save
            self.visualizer.reset_change_tracking()
            if not use_edit_prefix:
                messagebox.showinfo("Success", f"Successfully saved OBJ file to:\n{os.path.basename(output_path)}")
            return True
        else:
            messagebox.showerror("Error", "Failed to save OBJ file")
            return False
    
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
        
        # Close existing window if open
        if self.visualizer.opengl_window is not None:
            self.visualizer.close_opengl_window()
            # Wait a bit for window to close
            self.root.update()
            time.sleep(0.2)
        
        # Reset close flag before starting new window
        self.visualizer.close_window_flag = False
        
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

