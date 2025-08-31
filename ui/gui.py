import tkinter as tk
from tkinter import filedialog, ttk
from .config import CATS
from .runner import run_cmd

def main():
    root = tk.Tk()
    root.title("Mini ML Runner")

    # --- notebook with two tabs ------------------------------------------------
    nb = ttk.Notebook(root)
    nb.grid(sticky="nsew")
    root.rowconfigure(0, weight=1)
    root.columnconfigure(0, weight=1)

    distil_tab = ttk.Frame(nb, padding=8)
    viz_tab = ttk.Frame(nb, padding=8)
    nb.add(distil_tab, text="Data Distillation")
    nb.add(viz_tab, text="Visualization")

    # ===========================================================================
    # TAB 1: Data Distillation (original UI)
    # ===========================================================================
    frm = distil_tab
    for i in range(6): frm.columnconfigure(i, weight=0)
    frm.columnconfigure(5, weight=1)
    frm.rowconfigure(3, weight=1)

    # --- mode & algo ---
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
        else:
            algo_var.set("")
    mode_box.bind("<<ComboboxSelected>>", refresh_algos)
    refresh_algos()

    # --- file ---
    ttk.Label(frm, text="file").grid(row=1, column=0, sticky="w")
    file_var = tk.StringVar()
    ttk.Entry(frm, textvariable=file_var).grid(row=1, column=1, columnspan=4, sticky="ew", padx=4)

    def browse():
        p = filedialog.askopenfilename(title="Select data file",
                                       filetypes=[("JSON","*.json"), ("All files","*.*")])
        if p: file_var.set(p)
    ttk.Button(frm, text="...", width=3, command=browse).grid(row=1, column=5, sticky="e")

    # --- args ---
    ttk.Label(frm, text="args").grid(row=2, column=0, sticky="w")
    args_var = tk.StringVar()
    ttk.Entry(frm, textvariable=args_var).grid(row=2, column=1, columnspan=5, sticky="ew", padx=4)

    # --- output ---
    out = tk.Text(frm, height=18, wrap="word")
    out.grid(row=3, column=0, columnspan=6, sticky="nsew", pady=(8,4))

    # --- status ---
    status_var = tk.StringVar(value="idle")
    ttk.Label(frm, textvariable=status_var).grid(row=4, column=0, columnspan=4, sticky="w", pady=(4,0))

    # --- run ---
    run_btn = ttk.Button(frm, text="Run", command=lambda: run_cmd(
        mode_var.get(), algo_var.get(), file_var.get(), args_var.get(),
        out, status_var, run_btn, root))
    run_btn.grid(row=4, column=5, sticky="e", pady=(4,0))

    # ===========================================================================
    # TAB 2: Visualization (simple chart runner)
    #    - mode fixed to "viz"
    #    - algo = chart type
    #    - builds args string for your existing run_cmd
    # ===========================================================================
    vfrm = viz_tab
    for i in range(6): vfrm.columnconfigure(i, weight=0)
    vfrm.columnconfigure(5, weight=1)
    vfrm.rowconfigure(4, weight=1)

    # file
    ttk.Label(vfrm, text="file").grid(row=0, column=0, sticky="w")
    vfile_var = tk.StringVar()
    ttk.Entry(vfrm, textvariable=vfile_var).grid(row=0, column=1, columnspan=4, sticky="ew", padx=4)

    def vbrowse():
        p = filedialog.askopenfilename(title="Select data file",
                                       filetypes=[("CSV/JSON","*.csv *.json"), ("All files","*.*")])
        if p: vfile_var.set(p)
    ttk.Button(vfrm, text="...", width=3, command=vbrowse).grid(row=0, column=5, sticky="e")

    # chart type
    ttk.Label(vfrm, text="chart").grid(row=1, column=0, sticky="w")
    vchart_var = tk.StringVar(value="hist")
    vchart_box = ttk.Combobox(vfrm, textvariable=vchart_var,
                              values=["hist", "bar", "line", "scatter"],
                              width=10, state="readonly")
    vchart_box.grid(row=1, column=1, sticky="w", padx=(4,12))

    # x / y columns
    ttk.Label(vfrm, text="x").grid(row=1, column=2, sticky="w")
    vx_var = tk.StringVar()
    ttk.Entry(vfrm, textvariable=vx_var, width=12).grid(row=1, column=3, sticky="w", padx=(4,12))

    ttk.Label(vfrm, text="y").grid(row=1, column=4, sticky="w")
    vy_var = tk.StringVar()
    ttk.Entry(vfrm, textvariable=vy_var, width=12).grid(row=1, column=5, sticky="ew", padx=(4,0))

    # extra args (bins, group, filters, etc.)
    ttk.Label(vfrm, text="extra args").grid(row=2, column=0, sticky="w")
    vargs_var = tk.StringVar()  # e.g. "--bins 30 --group dept"
    ttk.Entry(vfrm, textvariable=vargs_var).grid(row=2, column=1, columnspan=5, sticky="ew", padx=4)

    # output area
    vout = tk.Text(vfrm, height=16, wrap="word")
    vout.grid(row=4, column=0, columnspan=6, sticky="nsew", pady=(8,4))

    # status + run
    vstatus_var = tk.StringVar(value="idle")
    ttk.Label(vfrm, textvariable=vstatus_var).grid(row=5, column=0, columnspan=4, sticky="w", pady=(4,0))

    def build_viz_args():
        parts = []
        if vx_var.get().strip(): parts += ["--x", vx_var.get().strip()]
        if vy_var.get().strip(): parts += ["--y", vy_var.get().strip()]
        if vargs_var.get().strip(): parts += vargs_var.get().strip().split()
        return " ".join(parts)

    viz_run_btn = ttk.Button(vfrm, text="Run", command=lambda: run_cmd(
        "viz", vchart_var.get(), vfile_var.get(), build_viz_args(),
        vout, vstatus_var, viz_run_btn, root))
    viz_run_btn.grid(row=5, column=5, sticky="e", pady=(4,0))

    root.mainloop()
