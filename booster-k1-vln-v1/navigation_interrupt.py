"""One interrupt per navigation run, with final cleanup protected."""

import signal


class NavigationInterrupt:
    def __init__(self):
        self.requested = False
        self.cleaning_up = False

    def __call__(self, signum, frame):
        if self.requested:
            return
        self.requested = True
        if not self.cleaning_up:
            raise KeyboardInterrupt


def begin_navigation_cleanup() -> None:
    """Protect cleanup when running under the server's interrupt handler."""
    handler = signal.getsignal(signal.SIGINT)
    if isinstance(handler, NavigationInterrupt):
        handler.cleaning_up = True
