"""Always-on-top floating record button — an alternative to the hotkey.

Click to start dictation, click again to stop (two clicks, then the cleaned
text is pasted). The puck reflects state by color: grey idle, red recording,
amber processing. On macOS it's created non-activating (like the status toast)
so clicking it does NOT steal focus from the app you're dictating into — which
would otherwise send the auto-paste to the wrong window.

`tkinter` is imported inside `run_overlay`, so this module stays import-safe on
a headless box (CI) where there's no display to attach to.
"""

from collections.abc import Callable

# status -> (glyph, background color)
STYLES = {
    "idle": ("●", "#3a3a3a"),
    "recording": ("●", "#c0392b"),
    "processing": ("…", "#b9770e"),
}


def style_for(status: str) -> tuple[str, str]:
    """Glyph + background color for a status (falls back to idle)."""
    return STYLES.get(status, STYLES["idle"])


def run_overlay(toggle: Callable[[], None], status: Callable[[], str]) -> None:
    """Show the floating button and block until the window closes / Ctrl+C.

    `toggle` starts or stops recording; `status` returns the current status
    string, polled to keep the button in sync with the background threads that
    actually drive recording.
    """
    import tkinter as tk

    root = tk.Tk()
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    try:
        root.attributes("-alpha", 0.95)
    except tk.TclError:
        pass
    # macOS: float without ever becoming the active window (so focus — and thus
    # the auto-paste target — stays on the app being dictated into).
    try:
        root.tk.call(
            "::tk::unsupported::MacWindowStyle", "style", root._w, "help", "noActivates"
        )
    except tk.TclError:
        pass

    glyph, color = style_for(status())
    button = tk.Label(
        root, text=glyph, fg="white", bg=color,
        font=("Helvetica Neue", 22, "bold"), padx=18, pady=10, cursor="hand2",
    )
    button.pack()
    button.bind("<Button-1>", lambda _e: toggle())

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

    # Poll status -> repaint. The periodic callback also keeps the interpreter
    # ticking so Ctrl+C is delivered while mainloop is running.
    def refresh() -> None:
        text, bg = style_for(status())
        button.config(text=text, bg=bg)
        root.after(120, refresh)

    refresh()
    root.mainloop()
