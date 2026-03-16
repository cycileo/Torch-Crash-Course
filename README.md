## Setup Instructions

### 1. Install `uv`
Install the `uv` package manager if you haven't already:

**macOS / Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```


### 2. Set up the environment
Create the virtual environment and install all dependencies:
```bash
uv sync
```

### 3. Usage
#### Using with VS Code (Recommended)
1. Open `main.ipynb` or `slm_explorer.ipynb`.
2. Click 'Select Kernel' in the top right corner.
3. Choose 'Python Environments...' -> './.venv/bin/python'.