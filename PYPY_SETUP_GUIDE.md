# PyPy Virtual Environment Setup Guide

## Step 1: Find Your PyPy Installation

PyPy is typically installed in one of these locations on Windows:

1. **C:\pypy3.x\** (where x is the version number)
2. **C:\Program Files\pypy3.x\**
3. **C:\Users\YourName\AppData\Local\Programs\pypy3.x\**
4. **Wherever you chose during installation**

### Finding PyPy:

**Option A: Check the Start Menu**
- Open Start Menu
- Search for "pypy" or "PyPy"
- Right-click on it → "Open file location"
- This will show you where PyPy is installed

**Option B: Check Common Locations**
Open PowerShell and run:
```powershell
Get-ChildItem "C:\" -Filter "pypy*" -Directory -ErrorAction SilentlyContinue
Get-ChildItem "C:\Program Files" -Filter "pypy*" -Directory -ErrorAction SilentlyContinue
Get-ChildItem "$env:LOCALAPPDATA\Programs" -Filter "pypy*" -Directory -ErrorAction SilentlyContinue
```

**Option C: Search Your Computer**
- Open File Explorer
- Go to C:\
- Search for "pypy.exe" or "pypy3.exe"

Once you find it, note the full path. For example: `C:\pypy3.10\pypy.exe`

## Step 2: Verify PyPy Works

Open PowerShell and navigate to your PyPy installation directory, then test it:

```powershell
# Replace with your actual PyPy path
C:\pypy3.10\pypy.exe --version
```

You should see something like: `Python 3.10.x (PyPy ...)`

## Step 3: Navigate to Your Project

Open PowerShell and go to your project directory:

```powershell
cd C:\Users\bobby\OBJ-scaler
```

## Step 4: Create Virtual Environment with PyPy

**IMPORTANT:** First, remove the old CPython virtual environment if it exists:

```powershell
# Remove old venv (if it exists)
Remove-Item -Recurse -Force venv -ErrorAction SilentlyContinue
```

Now create a new virtual environment using PyPy:

```powershell
# Replace C:\pypy3.10\pypy.exe with your actual PyPy path
C:\pypy3.10\pypy.exe -m venv venv
```

**Alternative:** If PyPy is in your PATH:
```powershell
pypy -m venv venv
# or
pypy3 -m venv venv
```

## Step 5: Activate the Virtual Environment

```powershell
.\venv\Scripts\Activate.ps1
```

If you get an execution policy error, run this first:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then try activating again:
```powershell
.\venv\Scripts\Activate.ps1
```

You should see `(venv)` at the beginning of your prompt.

## Step 6: Verify You're Using PyPy

Check which Python is being used:

```powershell
python --version
where.exe python
```

You should see PyPy in the version output and the path should point to your venv.

## Step 7: Upgrade pip

```powershell
python -m pip install --upgrade pip
```

## Step 8: Install Dependencies

```powershell
pip install -r requirements.txt
```

This will install:
- numpy
- PyOpenGL
- PyOpenGL-accelerate
- glfw

## Step 9: Test the Application

```powershell
python obj_scaler.py
```

## Troubleshooting

### "pypy is not recognized"
- PyPy is not in your PATH
- Use the full path: `C:\pypy3.10\pypy.exe -m venv venv`

### "Execution Policy" error when activating
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### "No module named venv"
- Make sure you're using the correct PyPy executable
- Try: `C:\pypy3.10\pypy.exe -m ensurepip --default-pip` first

### Packages won't install
- Make sure you activated the venv (you should see `(venv)` in prompt)
- Try: `python -m pip install -r requirements.txt`

## Quick Reference Commands

```powershell
# Navigate to project
cd C:\Users\bobby\OBJ-scaler

# Create venv (replace path with your PyPy location)
C:\pypy3.10\pypy.exe -m venv venv

# Activate venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Run application
python obj_scaler.py
```


