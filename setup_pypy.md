# PyPy Setup Instructions

PyPy can provide 2-10x performance improvement over standard CPython for this application.

## Installing PyPy

### Windows
1. Download PyPy from: https://www.pypy.org/download.html
2. Choose the Windows installer for Python 3.10 or 3.11
3. Run the installer and note the installation path (typically `C:\pypy3.x\`)

### Linux
```bash
# Ubuntu/Debian
sudo apt-get install pypy3

# Or download from pypy.org
```

### macOS
```bash
# Using Homebrew
brew install pypy3

# Or download from pypy.org
```

## Setting Up Virtual Environment with PyPy

1. Create virtual environment using PyPy:
```bash
# Windows
C:\pypy3.x\pypy.exe -m venv venv
.\venv\Scripts\Activate.ps1

# Linux/Mac
pypy3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
# Windows
.\venv\Scripts\python.exe obj_scaler.py

# Linux/Mac
python obj_scaler.py
```

## Performance Benefits

PyPy provides significant speedups for:
- File parsing (OBJ file reading)
- NumPy array operations
- String processing
- General Python code execution

Expected improvements: **2-10x faster** depending on the workload.

## Compatibility Notes

- All dependencies (numpy, PyOpenGL, glfw) work with PyPy
- PyPy uses a JIT compiler, so first run may be slightly slower
- Subsequent runs will be much faster due to JIT optimization


