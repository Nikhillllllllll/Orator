"""Floating record button — runs as its own process (like the status toast).

Isolating tkinter in a subprocess keeps a GUI hiccup from taking down the
dictation client. The client and this button talk through two tiny files: the
client writes the current status for us to display, and we write a "toggle"
command back when the button is clicked.

    python overlay.py <cmd_file> <status_file>

`tkinter` is imported inside `run_button`, so the module stays import-safe on a
headless box (CI) where there's no display.
"""

import os
import sys
from pathlib import Path

# status -> (glyph, background color)
STYLES = {
    "idle": ("●", "#3a3a3a"),
    "recording": ("●", "#c0392b"),
    "processing": ("…", "#b9770e"),
}


def style_for(status: str) -> tuple[str, str]:
    """Glyph + background color for a status (falls back to idle)."""
    return STYLES.get(status, STYLES["idle"])


def run_button(cmd_file: Path, status_file: Path) -> None:
    """Show the floating button until the window closes / the process is killed.

    Clicking writes "toggle" to `cmd_file`; the button's color tracks the status
    the client writes to `status_file`.
    """
    import tkinter as tk

    root = tk.Tk()
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    try:
        root.attributes("-alpha", 0.95)
    except tk.TclError:
        pass
    # macOS: float without ever becoming the active window, so clicking the
    # button doesn't pull focus off the app being dictated into.
    try:
        root.tk.call(
            "::tk::unsupported::MacWindowStyle", "style", root._w, "help", "noActivates"
        )
    except tk.TclError:
        pass

    glyph, color = style_for("idle")
    button = tk.Label(
        root, text=glyph, fg="white", bg=color,
        font=("Helvetica Neue", 22, "bold"), padx=18, pady=10, cursor="hand2",
    )
    button.pack()

    def on_click(_event):
        # Atomic write (temp + rename) so the client never reads a half-written
        # command file and discards it.
        try:
            tmp = cmd_file.with_suffix(".tmp")
            tmp.write_text("toggle")
            os.replace(tmp, cmd_file)
        except OSError:
            pass

    button.bind("<Button-1>", on_click)

    # Right-drag to reposition the puck.
    drag = {"x": 0, "y": 0}
    button.bind("<ButtonPress-3>", lambda e: drag.update(x=e.x, y=e.y))
    button.bind(
        "<B3-Motion>",
        lambda e: root.geometry(
            f"+{root.winfo_x() + e.x - drag['x']}+{root.winfo_y() + e.y - drag['y']}"
        ),
    )

    # Initial position: bottom-right.
    root.update_idletasks()
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    w, h = root.winfo_width(), root.winfo_height()
    root.geometry(f"+{sw - w - 40}+{sh - h - 120}")

    def refresh() -> None:
        try:
            status = status_file.read_text().strip()
        except OSError:
            status = "idle"
        text, bg = style_for(status)
        button.config(text=text, bg=bg)
        root.after(120, refresh)

    refresh()
    root.mainloop()


if __name__ == "__main__":
    run_button(Path(sys.argv[1]), Path(sys.argv[2]))
