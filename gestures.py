class GestureFSM:
    def __init__(self, start_thr, peak_thr, end_thr,
                 min_start=3, min_end=3, refrac=8, name="gesture"):
        self.s, self.c, self.r = "idle", 0, 0
        self.start_thr, self.peak_thr, self.end_thr = start_thr, peak_thr, end_thr
        self.min_start, self.min_end, self.refrac = min_start, min_end, refrac
        self.name, self.last_peak_t = name, None

    def step(self, value, t):
        if self.r: self.r -= 1
        if self.s == "idle":
            self.c = self.c+1 if value > self.start_thr else 0
            if self.c >= self.min_start: self.s, self.c = "start", 0
        elif self.s == "start":
            if value > self.peak_thr: self.s; self.last_peak_t = t; self.s = "peak"
            elif value < self.end_thr: self.s = "idle"
        elif self.s == "peak":
            self.c = self.c+1 if value < self.end_thr else 0
            if self.c >= self.min_end:
                self.s, self.c, self.r = "idle", 0, self.refrac
                return {"event":"count", "gesture":self.name, "t":self.last_peak_t}
        return None
