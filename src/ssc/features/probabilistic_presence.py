from __future__ import annotations

from dataclasses import dataclass
import math


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass
class ProbPresenceConfig:
    p_init: float = 0.10
    p_show: float = 0.50
    p_hide: float = 0.40
    p_det: float = 0.80
    p_miss: float = 0.05
    p_miss_strong: float = 0.005
    bbox_ema_alpha: float = 0.15
    bbox_hold_frames: int = 3
    log_odds_min: float = -8.0
    log_odds_max: float = 3.0

    # Project note:
    # We used a log-odds / occupancy-grid style update so the probability is smooth but still responsive.
    # References:
    # - Probabilistic Robotics (log-odds form): https://mitpress.mit.edu/9780262201629/probabilistic-robotics/
    # - Occupancy mapping lecture notes: http://ais.informatik.uni-freiburg.de/teaching/ss16/robotics/slides/12-occupancy-mapping.pdf
    # We also cap log_odds_max so it doesn't saturate near 1.0 (that made boxes linger too long).
    # The log-odds implementation was AI-assisted (GPT-5.2 series).
    # Reason: to speed up documentation; the parameters/logic were reviewed and tuned by us.

    def __post_init__(self) -> None:
        assert 0.0 < self.p_init < 1.0
        assert 0.0 < self.p_show < 1.0
        assert 0.0 < self.p_hide < 1.0
        assert 0.0 < self.p_det < 1.0
        assert 0.0 < self.p_miss < 1.0
        assert 0.0 < self.p_miss_strong < 1.0
        assert self.p_show >= self.p_hide
        assert int(self.bbox_hold_frames) >= 0
        assert self.log_odds_min < self.log_odds_max


@dataclass
class ProbPresenceState:
    log_odds: float
    bbox: tuple[int, int, int, int] | None = None

    @property
    def probability(self) -> float:
        return _sigmoid(self.log_odds)


class ProbabilisticPresenceTracker:
    def __init__(self, cfg: ProbPresenceConfig = ProbPresenceConfig()):
        self.cfg = cfg
        self._prior_log_odds = self._logit(cfg.p_init)
        self._state = ProbPresenceState(log_odds=self._prior_log_odds, bbox=None)
        self._miss_count = 0

    def reset(self) -> None:
        self._prior_log_odds = self._logit(self.cfg.p_init)
        self._state = ProbPresenceState(log_odds=self._prior_log_odds, bbox=None)
        self._miss_count = 0

    @staticmethod
    def _logit(p: float) -> float:
        p = _clamp(p, 1e-6, 1.0 - 1e-6)
        return float(math.log(p / (1.0 - p)))

    def update(
        self,
        *,
        detected_bbox: tuple[int, int, int, int] | None,
        in_roi: bool = True,
    ) -> ProbPresenceState:
        if not in_roi:
            self._state.log_odds = self._prior_log_odds
            self._state.bbox = None
            self._miss_count = 0
            return self._state

        if detected_bbox is not None:
            self._miss_count = 0
            self._state.log_odds += self._logit(self.cfg.p_det) - self._prior_log_odds
            self._state.bbox = self._ema_bbox(self._state.bbox, detected_bbox)
        else:
            self._miss_count += 1
            miss_p = self.cfg.p_miss_strong if self._state.probability >= self.cfg.p_show else self.cfg.p_miss
            self._state.log_odds += self._logit(miss_p) - self._prior_log_odds

            if self._state.probability < self.cfg.p_hide and self._miss_count > self.cfg.bbox_hold_frames:
                self._state.bbox = None

        self._state.log_odds = _clamp(self._state.log_odds, self.cfg.log_odds_min, self.cfg.log_odds_max)
        return self._state

    def should_draw(self) -> bool:
        return self._state.bbox is not None and self._state.probability >= self.cfg.p_show

    def bbox(self) -> tuple[int, int, int, int] | None:
        return self._state.bbox

    def probability(self) -> float:
        return self._state.probability

    def _ema_bbox(
        self,
        prev: tuple[int, int, int, int] | None,
        new: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int]:
        if prev is None:
            return new

        a = float(self.cfg.bbox_ema_alpha)
        px, py, pw, ph = prev
        nx, ny, nw, nh = new

        x = int(round((1.0 - a) * px + a * nx))
        y = int(round((1.0 - a) * py + a * ny))
        w = int(round((1.0 - a) * pw + a * nw))
        h = int(round((1.0 - a) * ph + a * nh))
        return (x, y, w, h)
