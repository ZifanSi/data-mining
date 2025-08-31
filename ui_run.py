import sys, time, subprocess, shlex, tkinter as tk
from tkinter import filedialog, ttk

CATS = {
    "cls": {
        "id3": "models/classification/id3.py",
        "c45": "models/classification/c45.py",
        "c50": "models/classification/c50.py",
        "nb":  "models/classification/naive_bayes.py",
        "knn": "models/classification/knn.py",
        "perc":"models/classification/perceptron.py",
        "logr":"models/classification/logistic_regression.py",
        "lda": "models/classification/lda.py",
        "svm": "models/classification/svm.py",
    },
    "clu": {
        "km":  "models/cluster/kmeans.py",
        "kmed":"models/cluster/kmedians.py",
        "pam": "models/cluster/pam.py",
        "hier":"models/cluster/hier.py",
        "fcm": "models/cluster/fcm.py",
        "gmm": "models/cluster/gmm_em.py",
        "pca": "models/cluster/pca.py",
        "bi":  "models/cluster/bi.py",
    },
    "pat": {
        "ap":  "models/pattern/apriori.py",
        "fp":  "models/pattern/fp_growth.py",
        "hm":  "models/pattern/hmine.py",
        "ec":  "models/pattern/eclat.py",
    },
}

def run_cmd(mode, algo, file_path, extra_args, out_widget, status_var, btn):
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
        # preserve quotes; on Windows use posix=False
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

# --- UI ---
root = tk.Tk()
root.title("Mini ML Runner")

frm = ttk.Frame(root, padding=8); frm.grid(sticky="nsew")
root.rowconfigure(0, weight=1); root.columnconfigure(0, weight=1)
for i in range(6): frm.columnconfigure(i, weight=0)
frm.columnconfigure(5, weight=1)

ttk.Label(frm, text="mode").grid(row=0, column=0, sticky="w")
mode_var = tk.StringVar(value="cls")
mode_box = ttk.Combobox(frm, textvariable=mode_var, values=list(CATS.keys()), width=6, state="readonly")
mode_box.grid(row=0, column=1, sticky="w", padx=(4,12))

ttk.Label(frm, text="algo").grid(row=0, column=2, sticky="w")
algo_var = tk.StringVar()
algo_box = ttk.Combobox(frm, textvariable=algo_var, width=8, state="readonly")
algo_box.grid(row=0, column=3, sticky="w", padx=(4,12))

def refresh_algos(*_):
    algos = list(CATS.get(mode_var.get(), {}).keys())
    algo_box["values"] = algos
    if algos:
        algo_var.set(algos[0])
mode_box.bind("<<ComboboxSelected>>", refresh_algos)
refresh_algos()

ttk.Label(frm, text="file").grid(row=1, column=0, sticky="w")
file_var = tk.StringVar()
file_entry = ttk.Entry(frm, textvariable=file_var)
file_entry.grid(row=1, column=1, columnspan=4, sticky="ew", padx=4)
def browse():
    p = filedialog.askopenfilename(title="Select data file", filetypes=[("JSON","*.json"), ("All files","*.*")])
    if p: file_var.set(p)
ttk.Button(frm, text="...", width=3, command=browse).grid(row=1, column=5, sticky="e")

ttk.Label(frm, text="args").grid(row=2, column=0, sticky="w")
args_var = tk.StringVar()
args_entry = ttk.Entry(frm, textvariable=args_var)
args_entry.grid(row=2, column=1, columnspan=5, sticky="ew", padx=4)

out = tk.Text(frm, height=18, wrap="word")
out.grid(row=3, column=0, columnspan=6, sticky="nsew", pady=(8,4))
frm.rowconfigure(3, weight=1)

status_var = tk.StringVar(value="idle")
status = ttk.Label(frm, textvariable=status_var)
status.grid(row=4, column=0, columnspan=4, sticky="w", pady=(4,0))

run_btn = ttk.Button(frm, text="Run", command=lambda: run_cmd(mode_var.get(), algo_var.get(),
                                                              file_var.get(), args_var.get(),
                                                              out, status_var, run_btn))
run_btn.grid(row=4, column=5, sticky="e", pady=(4,0))

root.mainloop()
