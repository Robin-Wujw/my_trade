"""
上涨波段分位计算器 - 窗口版

输出内容：
  1. 当前价位于上涨波段的百分位
  2. 上涨波段的 50% / 62.5% / 75% 分位对应价格
  3. 最高价→当前价 回调波段的 50% 分位价格
"""

import tkinter as tk
from tkinter import ttk, messagebox

from wave_utils import calc_wave_pct, level_price, wave_levels


# ── GUI ───────────────────────────────────────────────────

class WavePositionApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("分位计算器")
        root.geometry("460x600")
        root.resizable(False, False)

        bg = "#f5f5f5"
        root.configure(bg=bg)

        main = tk.Frame(root, bg=bg, padx=20, pady=16)
        main.pack(fill="both", expand=True)

        # ── 标题 ──
        tk.Label(main, text="分位计算器", font=("DengXian Light", 16, "bold"),
                 bg=bg, fg="#333").pack(pady=(0, 16))

        # ── 输入 ──
        inp = tk.LabelFrame(main, text="  参数输入  ", bg=bg, fg="#555",
                            font=("DengXian Light", 10), padx=14, pady=12)
        inp.pack(fill="x")

        fields = [
            ("上涨波段低点", "low", "5.00"),
            ("上涨波段高点", "high", "15.00"),
            ("回调波段低点", "retracement", "10.00"),
            ("现价或目标价", "price", "13.50"),
        ]
        self.vars = {}

        for label, key, ph in fields:
            row = tk.Frame(inp, bg=bg)
            row.pack(fill="x", pady=3)
            tk.Label(row, text=label, width=12, anchor="e",
                     font=("DengXian Light", 10), bg=bg).pack(side="left", padx=(0, 8))
            var = tk.StringVar()
            self.vars[key] = var
            entry = tk.Entry(row, textvariable=var, width=14,
                             font=("Consolas", 11), justify="center")
            entry.insert(0, ph)
            entry.bind("<FocusIn>",
                       lambda e, v=var, p=ph: v.set("") if v.get() == p else None)
            entry.bind("<Return>", lambda e: self.calculate())
            entry.pack(side="left")

        btn_row = tk.Frame(inp, bg=bg)
        btn_row.pack(pady=(8, 0))
        tk.Button(btn_row, text="计算", command=self.calculate,
                  font=("DengXian Light", 11, "bold"), bg="#1976D2", fg="white",
                  relief="flat", padx=36, pady=4, cursor="hand2").pack()

        # ── 结果 ──
        res = tk.LabelFrame(main, text="  计算结果  ", bg=bg, fg="#555",
                            font=("DengXian Light", 10), padx=14, pady=12)
        res.pack(fill="x", pady=(10, 8))

        self.result_text = tk.Text(res, height=20, width=44, bg="white",
                                   font=("Consolas", 10), relief="solid",
                                   borderwidth=1, state="disabled", wrap="none")
        self.result_text.pack()

        self._set_placeholder_text('请输入参数后点击"计算"')

    def _set_placeholder_text(self, text: str):
        self.result_text.config(state="normal")
        self.result_text.delete("1.0", "end")
        self.result_text.insert("1.0", text)
        self.result_text.config(state="disabled")

    def calculate(self):
        try:
            low = float(self.vars["low"].get().strip())
            high = float(self.vars["high"].get().strip())
            retracement = float(self.vars["retracement"].get().strip())
            price = float(self.vars["price"].get().strip())
        except ValueError:
            messagebox.showerror("输入错误", "请填写有效数字")
            return

        try:
            levels = wave_levels(low, high, price, retracement)
            pct = levels["wave_pct"]
        except ValueError as e:
            messagebox.showerror("计算错误", str(e))
            return

        half = levels["level_50"]
        half_1_1 = round(half * 1.1, 2)
        p625 = levels["level_625"]
        p75 = levels["level_75"]
        retrace_50 = levels["retracement_50"]
        rise = round((high / low - 1) * 100, 1)

        lines = [
            "━" * 38,
            f"  波段区间    {low:.2f}  →  {high:.2f}   (+{rise}%)",
            f"  当前价格    {price:.2f}",
            "─" * 38,
            f"  上涨波段分位    {pct}%",
            "",
            f"  50%   分位价格  {half:.2f}",
            f"  50%   × 1.1     {half_1_1:.2f}",
            f"  62.5% 分位价格  {p625:.2f}",
            f"  75%   分位价格  {p75:.2f}",
            "",
            f"  回调波段 50% 分位  {retrace_50:.2f}",
            "━" * 38,
        ]

        self.result_text.config(state="normal")
        self.result_text.delete("1.0", "end")
        self.result_text.insert("1.0", "\n".join(lines))
        self.result_text.config(state="disabled")


def main():
    root = tk.Tk()
    WavePositionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
