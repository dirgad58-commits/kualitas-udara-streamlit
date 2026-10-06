"""
Main entry point for Streamlit Community Cloud and local serving.
Automatically launches dashboard/app.py to ensure backward compatibility
when Streamlit Cloud defaults to root 'app.py'.
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

dashboard_script = os.path.join(BASE_DIR, "dashboard", "app.py")

if not os.path.exists(dashboard_script):
    raise FileNotFoundError(f"Dashboard script not found at {dashboard_script}")

# Execute dashboard/app.py with proper __file__ context
with open(dashboard_script, "r", encoding="utf-8") as f:
    code = compile(f.read(), dashboard_script, "exec")
    exec(code, {**globals(), "__file__": dashboard_script})
