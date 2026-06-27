import random
import sys
import time


class PerformanceMonitor:
    def __init__(self, print_every: int = 100):
        self.average_events = {}            # event_name -> EMA of elapsed seconds
        self.total_events = {}              # event_name -> total elapsed seconds
        self._lines_printed = 0     # rows written by the last print_stats (for overwrite)
        self.iterations = 0
        self.print_every = print_every

    def __call__(self, event_name):
        self._current = event_name
        return self

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.perf_counter() - self._start
        prev = self.average_events.get(self._current)
        # seed on first observation, otherwise exponential moving average
        self.average_events[self._current] = elapsed if prev is None else 0.9 * prev + 0.1 * elapsed
        self.total_events[self._current] = self.total_events.get(self._current, 0) + elapsed
        return False

    @staticmethod
    def _common_prefix(names):
        """Longest shared prefix, trimmed back to a word boundary, for clean labels."""
        if len(names) < 2:
            return ""
        prefix = names[0]
        for name in names[1:]:
            while not name.startswith(prefix):
                prefix = prefix[:-1]
                if not prefix:
                    return ""
        return prefix[:prefix.rfind("_") + 1]   # keep only up to the last '_'

    def print_stats(self):
        # move the cursor back up over the previously printed block and clear it,
        # so repeated calls redraw in place instead of scrolling the terminal.
        self.iterations += 1
        if self.iterations % self.print_every != 0:
            return
        if self._lines_printed:
            sys.stdout.write(f"\033[{self._lines_printed}F\033[J")

        # sort by event name (stable) rather than by timing — sorting by the EMA
        # value reorders rows every call as timings drift, which makes the
        # in-place redraw flicker.
        rows = sorted(self.average_events.items(), key=lambda kv: kv[0])
        prefix = self._common_prefix([name for name, _ in rows])
        labels = [name[len(prefix):] for name, _ in rows]

        name_w = max((len(l) for l in labels), default=len("event"))
        name_w = max(name_w, len("event"), len("total"))
        avg_w, tot_w = 14, 14
        header = f"{'event':<{name_w}}   {'ms/call (EMA)':>{avg_w}}   {'total s':>{tot_w}}"
        sep = "-" * len(header)

        lines = [header, sep]
        avg_sum = 0.0
        tot_sum = 0.0
        for label, (name, t) in zip(labels, rows):
            tot = self.total_events.get(name, 0.0)
            avg_sum += t
            tot_sum += tot
            lines.append(f"{label:<{name_w}}   {t * 1e3:>{avg_w}.4f}   {tot:>{tot_w}.4f}")
        lines.append(sep)
        lines.append(f"{'total':<{name_w}}   {avg_sum * 1e3:>{avg_w}.4f}   {tot_sum:>{tot_w}.4f}")

        sys.stdout.write("\n".join(lines) + "\n")
        sys.stdout.flush()
        self._lines_printed = len(lines)

        # totals are per simulation loop only — clear after each report
        self.total_events = {}

    def reset(self):
        self.average_events = {}
        self.total_events = {}
        self._lines_printed = 0