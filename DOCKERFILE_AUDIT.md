# Dockerfile Audit: TradeExecGym FastAPI + Gradio App
## HuggingFace Spaces Compatibility Analysis

---

## CRITICAL ISSUES (Will Cause Failures)

### 1. **CRITICAL: Shell Script Creation Syntax Error**
**Severity:** BUILD FAILURE  
**Line:** `RUN echo '#!/bin/bash\n\...` block  
**Problem:** The `echo` command with `\n` and line continuations using `\` does NOT interpret `\n` as newlines. The backslashes remain literal in the file, creating invalid bash syntax with literal `\n` characters.

**Evidence:**
```
# What gets created (BROKEN):
#!/bin/bash\n\
export PYTHONPATH="/app:${PYTHONPATH}"\n\
...
# This causes: "syntax error near unexpected token `('"
```

**Fix:** Use `cat` with here-document instead:
```dockerfile
RUN cat > start.sh << 'EOF'
#!/bin/bash
export PYTHONPATH="/app:${PYTHONPATH}"
echo "Starting TradeExecGym Backend (FastAPI) on port 7865..."
uvicorn server.app:app --host 0.0.0.0 --port 7865 &

echo "Starting TradeExecGym Dashboard (Gradio) on primary port 7860..."
python ui/app.py --port ${PORT:-7860}
EOF
chmod +x start.sh
```

---

### 2. **CRITICAL: Missing results/ Directory Permissions**
**Severity:** RUNTIME FAILURE  
**Problem:** The Dockerfile doesn't create or chmod the `/app/results/` directory. When inference.py tries to write to it as non-root user (UID 1000 on HF Spaces), it will fail with permission denied.

**Fix:**
```dockerfile
RUN mkdir -p /app/results && chmod 777 /app/results
```

---

### 3. **CRITICAL: Unsafe Background Process in start.sh**
**Severity:** RUNTIME FAILURE  
**Problem:** The start.sh script runs uvicorn in background with `&` then immediately starts Gradio. If uvicorn fails to start (port already in use, import error, etc.), the script continues anyway and Gradio starts without the backend, causing cascade failures.

**Fix:** Add error checking:
```bash
#!/bin/bash
export PYTHONPATH="/app:${PYTHONPATH}"
echo "Starting TradeExecGym Backend (FastAPI) on port 7865..."
uvicorn server.app:app --host 0.0.0.0 --port 7865 &
BACKEND_PID=$!
sleep 2  # Allow backend time to start

if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo "ERROR: FastAPI backend failed to start"
    exit 1
fi

echo "Starting TradeExecGym Dashboard (Gradio) on primary port 7860..."
exec python ui/app.py --port ${PORT:-7860}  # exec replaces the shell so signals work
```

---

## HIGH-PRIORITY ISSUES (Likely to Cause Failures)

### 4. **HIGH: torch>=2.1.0 Size and Build Time**
**Severity:** POTENTIAL BUILD TIMEOUT  
**Problem:** 
- torch 2.1.0 uncompressed: ~2.0 GB
- Installation time: 10-15 minutes
- Combined with scipy, pandas, stable-baselines3: total >2.5 GB
- HF Spaces build timeout: ~1 hour, storage limits: 15-20 GB
- Using loose version constraint `>=2.1.0` risks installing even larger future versions

**Impact:** Build may timeout, exceed space quota, or take excessively long.

**Fix:**
```toml
# Pin to specific version and use CPU-only wheel
torch==2.1.0
```

Or for even smaller footprint:
```toml
torch==2.0.0  # ~1.8 GB instead of 2.0 GB
```

---

### 5. **HIGH: Uncertain/Unverifiable Dependency**
**Severity:** BUILD FAILURE  
**Problem:** `openenv-core>=0.1.0` is not a standard package in PyPI. It may:
- Not exist in PyPI (typo?)
- Be a private package requiring authentication
- Be an internal package not available publicly

**Fix:** Verify this package exists and is publicly available:
```bash
pip index versions openenv-core
```
Or replace with correct package name if it's a typo.

---

### 6. **HIGH: Missing .dockerignore File**
**Severity:** BUILD INEFFICIENCY & SECURITY  
**Problem:** 
- `COPY . .` copies ALL files including `.git/`, `__pycache__/`, `.env`, logs, venv/
- Mentioned file: `ui/farmsim_project_gradio_app.py` (accidentally committed, wastes ~100KB)
- Increases image layers and build context size
- May expose secrets or credentials

**Fix:** Create `.dockerignore`:
```
.git
.gitignore
__pycache__
.pytest_cache
.venv
venv
env
*.pyc
*.pyo
*.egg-info
.DS_Store
.env
.env.local
*.log
node_modules
build
dist
*.egg
results/
ui/farmsim_project_gradio_app.py
```

---

## MEDIUM-PRIORITY ISSUES (Edge Cases & Optimization)

### 7. **MEDIUM: ui/app.py May Not Accept --port Argument**
**Severity:** RUNTIME FAILURE (if unimplemented)  
**Problem:** The start.sh script passes `--port ${PORT:-7860}` to `ui/app.py`, but the audit doesn't confirm this script accepts this argument. If it doesn't, Gradio will fail to start.

**Fix:** Verify ui/app.py has argument parsing:
```python
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=7860)
args = parser.parse_args()
# Then use: app.launch(server_name='0.0.0.0', server_port=args.port, ...)
```

---

### 8. **MEDIUM: server/app.py Must Be Importable**
**Severity:** RUNTIME FAILURE  
**Problem:** The start.sh runs `uvicorn server.app:app` but doesn't verify the module structure:
- File must be: `/app/server/app.py` or `/app/server/__init__.py`
- Must contain an object named `app` (FastAPI instance)
- PYTHONPATH is set, but module discovery depends on `__init__.py` files

**Fix:** Ensure proper package structure:
```
/app/
  server/
    __init__.py
    app.py  # contains: app = FastAPI()
  ui/
    __init__.py
    app.py
```

---

### 9. **MEDIUM: matplotlib Headless Mode Not Set**
**Severity:** RUNTIME FAILURE  
**Problem:** ui/app.py imports `matplotlib.pyplot as plt`. In HF Spaces (headless environment), this needs to use a non-interactive backend:

**Fix:** Add to Dockerfile before ui/app.py runs:
```dockerfile
ENV MPLBACKEND=Agg
```

Or in ui/app.py at top:
```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
```

---

### 10. **MEDIUM: Gradio Port Binding Issue**
**Severity:** POTENTIAL RUNTIME FAILURE  
**Problem:** HF Spaces may have firewall rules preventing binding to `0.0.0.0` on non-standard ports. The primary app must listen on the injected `PORT` variable (default 7860). FastAPI backend on 7865 may be blocked.

**Fix:** Ensure FastAPI only listens internally:
```bash
# In start.sh:
uvicorn server.app:app --host 127.0.0.1 --port 7865 &  # localhost only
```

Then ui/app.py must call backend via `http://localhost:7865` not `0.0.0.0`.

---

### 11. **MEDIUM: Missing Graceful Shutdown**
**Severity:** CLEANUP ISSUE  
**Problem:** When HF Spaces stops the container, backgrounded uvicorn may not shut down cleanly. No signal handlers in start.sh.

**Fix:** Add signal handling:
```bash
#!/bin/bash
trap "kill $BACKEND_PID 2>/dev/null; exit 0" SIGTERM SIGINT

# ... rest of script
wait $BACKEND_PID  # Keep script running
```

---

## LOW-PRIORITY ISSUES (Warnings & Best Practices)

### 12. **LOW: Loose version constraints on dependencies**
**Severity:** REPRODUCIBILITY ISSUE  
**Problem:** Many dependencies use `>=` without upper bound (e.g., `fastapi>=0.115.0`, `pydantic>=2.9.2`), risking breaking changes in future versions.

**Recommendation:** Pin major versions:
```toml
fastapi==0.115.0
uvicorn==0.30.6
pydantic==2.9.2
numpy==1.26.x  # Pin compatible versions
```

---

### 13. **LOW: build-essential May Be Unnecessary**
**Severity:** IMAGE SIZE  
**Problem:** `build-essential` adds ~600 MB. It's only needed if compiling native extensions. If none of the dependencies require it, it can be removed.

**Fix:** After installing dependencies, verify if needed:
```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    # Remove build-essential if scipy/numpy wheels are pre-built
    && rm -rf /var/lib/apt/lists/*

# Install deps
RUN uv pip install --system --no-cache -e .

# OPTIONAL: Remove build tools after if not needed
# RUN apt-get remove -y build-essential
```

---

### 14. **LOW: No Health Check**
**Severity:** OBSERVABILITY  
**Problem:** No HEALTHCHECK defined. HF Spaces may mark unhealthy spaces as crashed.

**Fix:**
```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1
```

---

### 15. **LOW: uv Binary Installation Method**
**Severity:** SECURITY/RELIABILITY  
**Problem:** Copying `uv` binary from `ghcr.io/astral-sh/uv:latest` is fragile:
- `latest` tag can change unexpectedly
- Binary may not be compatible with all architectures
- No version pinning

**Fix:**
```dockerfile
# Use specific version
COPY --from=ghcr.io/astral-sh/uv:0.1.0 /uv /uvx /bin/
```

Or use pip:
```dockerfile
RUN pip install --no-cache-dir uv==0.1.0
RUN uv pip install --system -e .
```

---

### 16. **LOW: No User Isolation Setup**
**Severity:** SECURITY/COMPATIBILITY  
**Problem:** Dockerfile runs as root, but HF Spaces executes as `appuser` (UID 1000). While not a build failure, relying on implicit user switching is fragile.

**Optional Fix:**
```dockerfile
# Create user (optional, HF Spaces will still use appuser)
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser
```

---

## SUMMARY TABLE

| Issue # | Severity | Type | Status |
|---------|----------|------|--------|
| 1 | 🔴 CRITICAL | Build | Shell script creation broken |
| 2 | 🔴 CRITICAL | Runtime | Missing results/ dir perms |
| 3 | 🔴 CRITICAL | Runtime | Unsafe background process |
| 4 | 🟠 HIGH | Build/Timeout | torch size & timeout risk |
| 5 | 🟠 HIGH | Build | Unknown dependency openenv-core |
| 6 | 🟠 HIGH | Build/Sec | Missing .dockerignore |
| 7 | 🟡 MEDIUM | Runtime | Port arg not validated |
| 8 | 🟡 MEDIUM | Runtime | Module structure not verified |
| 9 | 🟡 MEDIUM | Runtime | matplotlib headless mode |
| 10 | 🟡 MEDIUM | Runtime | Gradio port binding |
| 11 | 🟡 MEDIUM | Cleanup | No graceful shutdown |
| 12 | 🔵 LOW | Reproduction | Loose version constraints |
| 13 | 🔵 LOW | Size | build-essential overhead |
| 14 | 🔵 LOW | Observability | No healthcheck |
| 15 | 🔵 LOW | Reliability | uv version unpinned |
| 16 | 🔵 LOW | Security | No user isolation |

---

## RECOMMENDED FIXES (Priority Order)

**MUST FIX (for build/runtime):**
1. Fix shell script creation (Issue #1)
2. Create results/ directory with proper permissions (Issue #2)
3. Add error handling to start.sh (Issue #3)
4. Verify openenv-core exists in PyPI (Issue #5)
5. Pin torch to specific version (Issue #4)

**STRONGLY RECOMMENDED:**
6. Create .dockerignore (Issue #6)
7. Add MPLBACKEND=Agg for matplotlib (Issue #9)
8. Verify ui/app.py accepts --port argument (Issue #7)
9. Implement graceful shutdown (Issue #11)

**NICE TO HAVE:**
10. Verify server/app.py module structure (Issue #8)
11. Add HEALTHCHECK (Issue #14)
12. Pin uv version (Issue #15)
