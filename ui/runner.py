import sys, time, subprocess, shlex, tkinter as tk
from .config import CATS

def run_cmd(mode, algo, file_path, extra_args, out_widget, status_var, btn, root):
    btn.config(state="disabled")
    out_widget.delete("1.0", tk.END)
    status_var.set("running...")
    root.update_idletasks()

    if mode not in CATS or algo not in CATS[mode]:
        out_widget.insert(tk.END, "Invalid mode/algo\n")
        status_var.set("idle")
        btn.config(state="normal")
        return

    script = CATS[mode][algo]
    args = []
    if file_path.strip():
        args.append(file_path.strip())
    if extra_args.strip():
        args.extend(shlex.split(extra_args, posix=False))

    cmd = [sys.executable, script, *args]

    t = time.time()
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True)
        elapsed = time.time() - t
        if proc.stdout:
            out_widget.insert(tk.END, proc.stdout)
        if proc.stderr:
            out_widget.insert(tk.END, "\n[stderr]\n" + proc.stderr)
        status_var.set(f"done in {elapsed:.3f}s (exit {proc.returncode})")
    except Exception as e:
        status_var.set("error")
        out_widget.insert(tk.END, f"Error: {e}")
    finally:
        btn.config(state="normal")
