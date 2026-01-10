"""
OBJ Volume Analyzer
Scans a folder for OBJ files, calculates bounding box and volume for each,
and outputs results to a CSV file.
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
import csv
from pathlib import Path
import threading


class OBJVolumeAnalyzer:
    def __init__(self):
        self.vertices = []
        self.faces = []
        
    def parse_obj_file(self, file_path):
        """Parse OBJ file to extract vertices and faces
        
        Args:
            file_path: Path to OBJ file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.vertices = []
            self.faces = []
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
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
                            # Handle format: v/vt/vn or v//vn or v
                            indices = part.split('/')
                            if indices[0]:
                                vertex_idx = int(indices[0]) - 1  # OBJ is 1-indexed
                                if 0 <= vertex_idx < len(self.vertices):
                                    face_vertices.append(vertex_idx)
                        
                        if len(face_vertices) >= 3:
                            # Triangulate if face has more than 3 vertices
                            for i in range(1, len(face_vertices) - 1):
                                self.faces.append([face_vertices[0], face_vertices[i], face_vertices[i + 1]])
            
            if len(self.vertices) == 0:
                return False
            
            self.vertices = np.array(self.vertices)
            return True
            
        except Exception as e:
            print(f"Error parsing {file_path}: {str(e)}")
            return False
    
    def calculate_bounding_box(self):
        """Calculate bounding box dimensions
        
        Returns:
            (x, y, z) tuple of bounding box extents
        """
        if len(self.vertices) == 0:
            return (0.0, 0.0, 0.0)
        
        bbox_min = np.min(self.vertices, axis=0)
        bbox_max = np.max(self.vertices, axis=0)
        extents = bbox_max - bbox_min
        
        return (extents[0], extents[1], extents[2])
    
    def calculate_volume(self):
        """Calculate volume of closed mesh using signed volume method
        
        Returns:
            Volume as float, or None if calculation fails
        """
        if len(self.faces) == 0 or len(self.vertices) == 0:
            return None
        
        # Use signed volume method: for each triangle, calculate signed volume
        # of tetrahedron formed by origin (0,0,0) and triangle vertices
        total_volume = 0.0
        
        for face in self.faces:
            if len(face) < 3:
                continue
            
            # Get triangle vertices
            v0 = self.vertices[face[0]]
            v1 = self.vertices[face[1]]
            v2 = self.vertices[face[2]]
            
            # Calculate signed volume of tetrahedron (0, v0, v1, v2)
            # Volume = (1/6) * dot(v0, cross(v1, v2))
            # This gives signed volume based on face orientation
            cross_product = np.cross(v1, v2)
            signed_volume = np.dot(v0, cross_product) / 6.0
            total_volume += signed_volume
        
        # Return absolute volume (mesh should be closed)
        return abs(total_volume)
    
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
            print(f"Error scanning folder: {str(e)}")
        return obj_files
    
    def analyze_file(self, file_path):
        """Analyze a single OBJ file
        
        Args:
            file_path: Path to OBJ file
            
        Returns:
            Dictionary with keys: 'filename', 'volume', 'x', 'y', 'z', 'success', 'error'
        """
        result = {
            'filename': os.path.basename(file_path),
            'volume': None,
            'x': None,
            'y': None,
            'z': None,
            'success': False,
            'error': None
        }
        
        if not self.parse_obj_file(file_path):
            result['error'] = "Failed to parse OBJ file"
            return result
        
        # Calculate bounding box
        x, y, z = self.calculate_bounding_box()
        result['x'] = x
        result['y'] = y
        result['z'] = z
        
        # Calculate volume
        volume = self.calculate_volume()
        result['volume'] = volume
        
        if volume is not None:
            result['success'] = True
        else:
            result['error'] = "Failed to calculate volume"
        
        return result
    
    def analyze_folder(self, folder_path, output_csv_path, progress_callback=None):
        """Analyze all OBJ files in folder and write results to CSV
        
        Args:
            folder_path: Folder containing OBJ files
            output_csv_path: Path where CSV file should be saved
            progress_callback: Optional callback function(result) called after each file is analyzed
            
        Returns:
            (total_files, successful, failed) tuple
        """
        obj_files = self.scan_folder_for_obj_files(folder_path)
        
        if len(obj_files) == 0:
            return (0, 0, 0)
        
        results = []
        successful = 0
        failed = 0
        
        for file_path in obj_files:
            result = self.analyze_file(file_path)
            results.append(result)
            
            # Call progress callback if provided
            if progress_callback:
                progress_callback(result)
            
            if result['success']:
                successful += 1
            else:
                failed += 1
        
        # Format numeric values to avoid Excel quoting issues
        # Use a reasonable number of decimal places and clean formatting
        def format_number(value):
            """Format a number for CSV output, avoiding Excel quoting issues.
            
            Excel may quote values with many decimal places or floating point precision issues.
            This function rounds to 6 decimal places and formats cleanly to ensure Excel
            treats them as numbers, not text.
            """
            if value is None:
                return ''
            try:
                val_float = float(value)
                # Round to 6 decimal places to avoid floating point precision issues
                # that cause Excel to quote values with many decimal places like
                # 29.000001000000001 or 11.999999999999998
                rounded = round(val_float, 6)
                
                # Format with up to 6 decimal places, removing trailing zeros
                # This gives us clean numbers like "29.000001" instead of "29.000001000000001"
                formatted = f"{rounded:.6f}".rstrip('0').rstrip('.')
                
                # Handle edge cases
                if formatted == '' or formatted == '-':
                    formatted = '0'
                if formatted == '-0':
                    formatted = '0'
                
                return formatted
            except (ValueError, TypeError):
                return ''
        
        # Write results to CSV
        try:
            with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['filename', 'volume', 'x', 'y', 'z', 'status', 'error']
                # Use QUOTE_NONNUMERIC to ensure only non-numeric fields are quoted
                # But actually, we'll write numbers as strings to have full control
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
                
                writer.writeheader()
                for result in results:
                    row = {
                        'filename': result['filename'],
                        'volume': format_number(result['volume']),
                        'x': format_number(result['x']),
                        'y': format_number(result['y']),
                        'z': format_number(result['z']),
                        'status': 'Success' if result['success'] else 'Failed',
                        'error': result['error'] if result['error'] else ''
                    }
                    writer.writerow(row)
            
            return (len(obj_files), successful, failed)
            
        except Exception as e:
            raise Exception(f"Error writing CSV: {str(e)}")


class VolumeAnalyzerGUI:
    def __init__(self):
        self.analyzer = OBJVolumeAnalyzer()
        self.root = tk.Tk()
        self.root.title("OBJ Volume Analyzer")
        self.root.geometry("600x400")
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the GUI"""
        # Folder selection frame
        folder_frame = ttk.Frame(self.root, padding="10")
        folder_frame.pack(fill=tk.X)
        
        ttk.Label(folder_frame, text="Folder:").pack(side=tk.LEFT, padx=5)
        self.folder_label = ttk.Label(folder_frame, text="No folder selected", foreground="gray")
        self.folder_label.pack(side=tk.LEFT, padx=5, expand=True)
        
        ttk.Button(folder_frame, text="Select Folder", command=self.select_folder).pack(side=tk.RIGHT, padx=5)
        
        # Output CSV frame
        output_frame = ttk.Frame(self.root, padding="10")
        output_frame.pack(fill=tk.X)
        
        ttk.Label(output_frame, text="Output CSV:").pack(side=tk.LEFT, padx=5)
        self.output_label = ttk.Label(output_frame, text="No output file selected", foreground="gray")
        self.output_label.pack(side=tk.LEFT, padx=5, expand=True)
        
        ttk.Button(output_frame, text="Select Output", command=self.select_output).pack(side=tk.RIGHT, padx=5)
        
        # Status frame
        status_frame = ttk.LabelFrame(self.root, text="Status", padding="10")
        status_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.status_text = tk.Text(status_frame, height=10, wrap=tk.WORD, state=tk.DISABLED)
        self.status_text.pack(fill=tk.BOTH, expand=True)
        
        # Buttons frame
        button_frame = ttk.Frame(self.root, padding="10")
        button_frame.pack(fill=tk.X)
        
        ttk.Button(button_frame, text="Analyze Folder", command=self.analyze_folder).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clear Log", command=self.clear_log).pack(side=tk.RIGHT, padx=5)
        
        self.folder_path = None
        self.output_csv_path = None
        self.analysis_thread = None
        self.analyzing = False
    
    def log_message(self, message):
        """Append a message to the status log"""
        self.status_text.config(state=tk.NORMAL)
        self.status_text.insert(tk.END, message + "\n")
        self.status_text.see(tk.END)  # Auto-scroll to bottom
        self.status_text.config(state=tk.DISABLED)
        # Force GUI update to show log messages in real-time
        self.root.update()
    
    def clear_log(self):
        """Clear the status log"""
        self.status_text.config(state=tk.NORMAL)
        self.status_text.delete(1.0, tk.END)
        self.status_text.config(state=tk.DISABLED)
    
    def select_folder(self):
        """Open folder dialog to select folder with OBJ files"""
        folder_path = filedialog.askdirectory(title="Select Folder with OBJ Files")
        if folder_path:
            self.folder_path = folder_path
            self.folder_label.config(text=os.path.basename(folder_path), foreground="black")
            self.log_message(f"Selected folder: {folder_path}")
    
    def select_output(self):
        """Open file dialog to select output CSV file"""
        output_path = filedialog.asksaveasfilename(
            title="Save Results CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if output_path:
            self.output_csv_path = output_path
            self.output_label.config(text=os.path.basename(output_path), foreground="black")
            self.log_message(f"Output CSV: {output_path}")
    
    def analyze_folder(self):
        """Analyze all OBJ files in selected folder (runs in background thread)"""
        if not self.folder_path:
            messagebox.showwarning("Warning", "Please select a folder first")
            return
        
        if not self.output_csv_path:
            messagebox.showwarning("Warning", "Please select an output CSV file first")
            return
        
        if self.analyzing:
            messagebox.showwarning("Warning", "Analysis is already in progress")
            return
        
        self.clear_log()
        self.log_message("Starting analysis...")
        self.log_message(f"Scanning folder: {self.folder_path}")
        
        # Start analysis in background thread to keep GUI responsive
        self.analyzing = True
        self.analysis_thread = threading.Thread(target=self._analyze_folder_thread, daemon=True)
        self.analysis_thread.start()
    
    def _analyze_folder_thread(self):
        """Background thread method to perform the analysis"""
        try:
            # First, scan for files to show count before processing
            obj_files = self.analyzer.scan_folder_for_obj_files(self.folder_path)
            total_files = len(obj_files)
            
            # Update GUI from main thread - scan results
            def update_scan_result():
                if total_files == 0:
                    self.log_message("No OBJ files found in selected folder.")
                    self.analyzing = False
                    messagebox.showwarning("Warning", "No OBJ files found in the selected folder.")
                else:
                    self.log_message(f"Found {total_files} OBJ file(s) to analyze.")
                    self.log_message("Processing files...")
                    self.log_message("")  # Empty line for readability
            
            self.root.after(0, update_scan_result)
            
            if total_files == 0:
                return
            
            # Define progress callback to log each file as it's processed
            # This callback will be called from the background thread
            def log_progress(result):
                """Log progress for each analyzed file"""
                # Create a closure to capture the result
                def update_gui():
                    if result['success']:
                        # Format: filename | X: x_dim | Y: y_dim | Z: z_dim | Volume: volume
                        volume_str = f"{result['volume']:.6f}" if result['volume'] is not None else "N/A"
                        x_str = f"{result['x']:.6f}" if result['x'] is not None else "N/A"
                        y_str = f"{result['y']:.6f}" if result['y'] is not None else "N/A"
                        z_str = f"{result['z']:.6f}" if result['z'] is not None else "N/A"
                        
                        self.log_message(f"  ✓ {result['filename']} | X: {x_str} | Y: {y_str} | Z: {z_str} | Volume: {volume_str}")
                    else:
                        # Log failure with error message
                        error_msg = result.get('error', 'Unknown error')
                        self.log_message(f"  ✗ {result['filename']} | Failed: {error_msg}")
                
                # Schedule GUI update in main thread - after(0) executes as soon as possible
                self.root.after(0, update_gui)
            
            # Analyze folder with progress callback
            # This runs in the background thread, so GUI stays responsive
            _, successful, failed = self.analyzer.analyze_folder(
                self.folder_path, 
                self.output_csv_path, 
                progress_callback=log_progress
            )
            
            # Update GUI with final results from main thread
            def show_results():
                self.log_message("")  # Empty line for readability
                self.log_message("-" * 60)
                self.log_message(f"Analysis complete!")
                self.log_message(f"Total files found: {total_files}")
                self.log_message(f"Successfully analyzed: {successful}")
                self.log_message(f"Failed: {failed}")
                self.log_message(f"Results saved to: {self.output_csv_path}")
                
                messagebox.showinfo("Analysis Complete", 
                                  f"Analyzed {total_files} OBJ files\n\n"
                                  f"Successfully processed: {successful}\n"
                                  f"Failed: {failed}\n\n"
                                  f"Results saved to:\n{self.output_csv_path}")
                self.analyzing = False
            
            self.root.after(0, show_results)
            
        except Exception as e:
            def show_error():
                error_msg = f"Error during analysis: {str(e)}"
                self.log_message(error_msg)
                messagebox.showerror("Error", error_msg)
                self.analyzing = False
            
            self.root.after(0, show_error)
    
    def run(self):
        """Run the GUI"""
        self.root.mainloop()


def main():
    app = VolumeAnalyzerGUI()
    app.run()


if __name__ == "__main__":
    main()

