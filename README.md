# OBJ Scaler with Bounding Box Visualization

A fast Python tool for scaling OBJ files with interactive 3D visualization using PyOpenGL.

## Features

- Load OBJ files via tkinter file dialog
- Automatic detection of associated MTL and texture files
- Calculate and display bounding box extents
- Interactive 3D visualization with PyOpenGL (rotatable, zoomable)
- Scale OBJ files based on target dimensions
- Preserve all OBJ data (faces, normals, UVs, materials)
- Copy MTL and texture files to output location

## Installation

### Standard Python (CPython)

1. Create and activate a virtual environment:
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows PowerShell
# or
source venv/bin/activate     # Linux/Mac
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### PyPy (Faster Performance)

For significantly faster performance, use PyPy:

1. Download and install PyPy from https://www.pypy.org/download.html

2. Create a virtual environment with PyPy:
```bash
pypy -m venv venv
.\venv\Scripts\Activate.ps1  # Windows PowerShell
# or
source venv/bin/activate     # Linux/Mac
```

3. Install dependencies (PyPy compatible):
```bash
pip install -r requirements.txt
```

**Note:** PyPy provides 2-10x speedup for most Python code, especially for file parsing and NumPy operations.

## Usage

Run the script:
```bash
python obj_scaler.py
# or with PyPy:
pypy obj_scaler.py
```

### Workflow

1. Click "Select OBJ" to choose your OBJ file
2. View bounding box information in the GUI
3. Click "Show Visualization" to see an interactive 3D view:
   - **Left mouse drag**: Rotate the model
   - **Mouse wheel**: Zoom in/out
4. Enter target dimensions (X, Y, or Z) in the input fields
5. Click "Scale and Save" to generate the scaled OBJ file

The scaled file will be saved as `[original_name]_scaled.obj` in the same directory as the input file.

## Dependencies

- `numpy` - Array operations for vertices
- `PyOpenGL` - OpenGL bindings for Python
- `PyOpenGL-accelerate` - Accelerated OpenGL operations
- `glfw` - Window and input handling for OpenGL

## Performance

- **CPython**: Standard Python performance
- **PyPy**: 2-10x faster for file parsing and processing
- **PyOpenGL**: Hardware-accelerated 3D rendering (much faster than matplotlib for large meshes)

## Requirements

- Python 3.7+ or PyPy 3.7+
- OpenGL-compatible graphics card
- Windows, Linux, or macOS


