# Complete PyPy Installation and Setup Guide

## Part 1: Installing PyPy

### Step 1: Run the PyPy Installer

1. **Locate the downloaded PyPy installer**
   - It's likely in your Downloads folder
   - File name will be something like: `pypy3.10-v7.3.x-win64.exe` or similar

2. **Run the installer**
   - Double-click the installer file
   - Follow the installation wizard

3. **During installation, note these options:**
   - **Installation location**: Usually `C:\pypy3.10\` or `C:\Program Files\pypy3.10\`
   - **Add to PATH**: It's helpful to check this option if available
   - **Create shortcuts**: Optional but helpful

4. **Complete the installation**
   - Click "Finish" when done

### Step 2: Verify PyPy Installation

Open PowerShell and test if PyPy is installed:

```powershell
pypy --version
```

OR

```powershell
pypy3 --version
```

If you see a version number (like `Python 3.10.x (PyPy ...)`), PyPy is installed correctly!

**If you get "not recognized":**
- PyPy wasn't added to PATH
- We'll use the full path instead (we'll find it in the next step)

---

## Part 2: Finding PyPy Location (if not in PATH)

If PyPy isn't recognized, find where it was installed:

### Method 1: Check Installation Location
- The installer usually shows the path during installation
- Default locations:
  - `C:\pypy3.10\`
  - `C:\pypy3.11\`
  - `C:\Program Files\pypy3.10\`

### Method 2: Search Your Computer
1. Open File Explorer
2. Go to `C:\`
3. Search for `pypy.exe`
4. Note the full path

### Method 3: Check Start Menu
1. Open Start Menu
2. Search for "pypy"
3. Right-click → "Open file location"
4. This shows where PyPy is installed

---

## Part 3: Setting Up Virtual Environment

Once PyPy is installed, follow these steps:

### Step 1: Open PowerShell in Your Project Directory

```powershell
cd C:\Users\bobby\OBJ-scaler
```

### Step 2: Remove Old Virtual Environment (if exists)

```powershell
Remove-Item -Recurse -Force venv -ErrorAction SilentlyContinue
```

### Step 3: Create Virtual Environment with PyPy

**If PyPy is in your PATH:**
```powershell
pypy -m venv venv
```

**OR if that doesn't work:**
```powershell
pypy3 -m venv venv
```

**If PyPy is NOT in PATH, use full path:**
```powershell
# Replace with your actual PyPy path
C:\pypy3.10\pypy.exe -m venv venv
```

### Step 4: Activate Virtual Environment

```powershell
.\venv\Scripts\Activate.ps1
```

**If you get an execution policy error:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then try activating again:
```powershell
.\venv\Scripts\Activate.ps1
```

You should see `(venv)` at the start of your prompt.

### Step 5: Verify You're Using PyPy

```powershell
python --version
```

You should see PyPy in the output, not regular Python.

### Step 6: Upgrade pip

```powershell
python -m pip install --upgrade pip
```

### Step 7: Install Dependencies

```powershell
pip install -r requirements.txt
```

### Step 8: Test the Application

```powershell
python obj_scaler.py
```

---

## Quick Checklist

- [ ] Install PyPy from downloaded file
- [ ] Verify PyPy works: `pypy --version`
- [ ] Navigate to project: `cd C:\Users\bobby\OBJ-scaler`
- [ ] Remove old venv (if exists)
- [ ] Create new venv: `pypy -m venv venv` (or use full path)
- [ ] Activate venv: `.\venv\Scripts\Activate.ps1`
- [ ] Verify PyPy: `python --version`
- [ ] Install deps: `pip install -r requirements.txt`
- [ ] Test: `python obj_scaler.py`

---

## Need Help?

After you install PyPy, let me know:
1. Where it was installed (the path)
2. Whether `pypy --version` works in PowerShell

Then I can help you with the exact commands for your setup!


