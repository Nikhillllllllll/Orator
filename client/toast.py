"""Floating status HUD for the Wisper client.

Run as a subprocess:  python toast.py "<message>" "<bg_hex>"

It shows a small borderless, always-on-top banner near the bottom of the screen
and stays until the process is terminated. The window is created *non-activating*
on macOS (``noActivates``) so it never steals keyboard focus from the app you're
dictating into — otherwise it would break frontmost-app detection and auto-paste.
"""

import sys
import tkinter as tk


def main() -> None:
    message = sys.argv[1] if len(sys.argv) > 1 else "Wisper"
    bg = sys.argv[2] if len(sys.argv) > 2 else "#222222"

    root = tk.Tk()
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    try:
        root.attributes("-alpha", 0.94)
    except tk.TclError:
        pass
    # macOS: non-activating, tooltip-style window — must not take focus.
    try:
        root.tk.call(
            "::tk::unsupported::MacWindowStyle", "style", root._w, "help", "noActivates"
        )
    except tk.TclError:
        pass

    frame = tk.Frame(root, bg=bg, padx=24, pady=14, highlightthickness=0)
    frame.pack()
    tk.Label(
        frame,
        text=message,
        fg="#ffffff",
        bg=bg,
        font=("Helvetica Neue", 16, "bold"),
    ).pack()

    root.update_idletasks()
    w, h = root.winfo_width(), root.winfo_height()
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"+{(sw - w) // 2}+{sh - h - 140}")

    root.mainloop()


if __name__ == "__main__":
    main()
