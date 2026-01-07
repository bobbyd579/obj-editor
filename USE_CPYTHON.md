# Using CPython Instead of PyPy

## Quick Start

The project has been set up to use **CPython** (regular Python) instead of PyPy for better OpenGL/GLFW compatibility and performance.

### To Run the Application:

1. **Activate the CPython virtual environment:**
   ```powershell
   .\venv_cpython\Scripts\Activate.ps1
   ```

2. **Run the application:**
   ```powershell
   python obj_plane_visualizer.py
   ```

### Why CPython?

- **Better OpenGL compatibility** - CPython has better support for OpenGL/GLFW bindings
- **More stable performance** - No compatibility issues with C extensions
- **Faster rendering** - Better optimized for graphics libraries

### If You Get Permission Errors:

If PowerShell blocks the activation script, run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then try activating again.

---

**Note:** The old PyPy virtual environment (`venv`) is still available if you want to switch back, but CPython (`venv_cpython`) is recommended for this application.

