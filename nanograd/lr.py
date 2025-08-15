from abc import ABC, abstractmethod
from itertools import chain
from math import pi, cos, inf


class Schedule(ABC):

    def __init__(self, t_max: int | float = inf):
        self.t = -1
        self.t_max = t_max
        self.a = 0.

    @property
    def done(self) -> bool:
        return self.t >= self.t_max

    @abstractmethod
    def _step(self):
        pass

    def step(self):
        self.t += 1
        if not self.done:
            self._step()

    def __call__(self) -> float:
        self.step()
        return self.a

    def __add__(self, other):
        if not isinstance(other, Schedule):
            return NotImplemented
        le = self.schedules if isinstance(self, ChainedSchedule) else [self]
        ri = other.schedules if isinstance(other, ChainedSchedule) else [other]
        return ChainedSchedule(le + ri)


class ChainedSchedule(Schedule):

    def __init__(self, schedules: list[Schedule]):
        if any(s.t_max == inf for s in schedules):
            t_max = inf
        else:
            t_max = sum(s.t_max for s in schedules)
        super().__init__(t_max=t_max)
        self.j = 0
        self.a = schedules[0].a
        self.schedules = schedules

    def _step(self):
        schedule = self.schedules[self.j]
        if schedule.done and self.j < len(self.schedules) - 1:
            self.j += 1
            schedule = self.schedules[self.j]
        schedule.step()
        self.a = schedule.a


class ConstantSchedule(Schedule):

    def __init__(self, a: float):
        super().__init__()
        self.t = -1
        self.a = a

    def _step(self):
        pass


class LinearSchedule(Schedule):

    def __init__(self, t_max: int, a_max: float, a_min: float):
        super().__init__(t_max)
        self.t = -1
        self.t_max = t_max
        self.a_min = a_min
        self.d = (a_max - a_min) / t_max
        self.a = a_max + self.d

    def _step(self):
        self.a -= self.d


class CosineSchedule(Schedule):

    def __init__(self, t_max: int, a_max: float, a_min: float):
        super().__init__(t_max)
        self.t = -1
        self.t_max = t_max
        self.a_min = a_min
        self.c = 0.5 * (a_max - a_min)

    def _step(self):
        self.a = self.a_min + self.c * (1 + cos(pi * self.t / self.t_max))


class ExponentialSchedule(Schedule):

    def __init__(self, t_max: int, a_max: float, a_min: float, gamma: float):
        super().__init__(t_max)
        self.t = -1
        self.t_max = t_max
        self.g = gamma
        self.d = 1 / gamma
        self.a_min = a_min
        self.c = a_max - a_min

    def _step(self):
        self.d *= self.g
        self.a = self.a_min + self.c * self.d
