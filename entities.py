# entities.py
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

@dataclass
class Ship:
    id: int
    route: List[str]
    big: bool
    looped_route: bool
    current_port: Optional[str] = None
    ready_to_depart: bool = False
    direction: int = 1
    current_leg: int = 0
    containers: int = 0
    contLoad: int = 0
    contUnload: int = 0
    best_time: List[float] = field(default_factory=list)
    cur_time: List[float] = field(default_factory=list)
    all_time: List[float] = field(default_factory=list)
    scenarioID: int = 0
    berth_id: Optional[int] = None

    def peek_next_ports(self, k: int = 3) -> List[str]:
        n = len(self.route)
        idx = self.current_leg
        dir = self.direction
        ports: List[str] = []
        for _ in range(k):
            if self.looped_route:
                next_idx = (idx + 1) % n
                next_dir = 1
            else:
                ti = idx + dir
                if ti >= n:
                    next_dir = -1
                    next_idx = idx + next_dir
                elif ti < 0:
                    next_dir = 1
                    next_idx = idx + next_dir
                else:
                    next_dir = dir
                    next_idx = ti
            ports.append(self.route[next_idx])
            idx, dir = next_idx, next_dir
        return ports

    def advance_leg(self):
        n = len(self.route)
        if n == 0:
            raise ValueError(f"Ship {self.id} has an empty route")

        if self.looped_route:
            next_leg = (self.current_leg + 1) % n
            next_dir = 1
        else:
            next_leg = self.current_leg + self.direction
            if next_leg >= n:
                next_dir = -1
                next_leg = self.current_leg + next_dir
            elif next_leg < 0:
                next_dir = 1
                next_leg = self.current_leg + next_dir
            else:
                next_dir = self.direction

        if not (0 <= next_leg < n):
            raise ValueError(
                f"Ship {self.id} computed invalid leg {next_leg} "
                f"(route length {n})"
            )

        self.current_leg = next_leg
        self.direction   = next_dir
        self.current_port = self.route[self.current_leg]
        self.ready_to_depart = False


class Port:
    def __init__(self, name: str):
        self.name: str = name
        self.free_berth_ids: List[int] = [1, 2]
        self.queue: List[int] = []

    def arrive(self, ship_id: int) -> Optional[int]:
        if self.free_berth_ids:
            berth_id = self.free_berth_ids.pop(0)
            return berth_id
        self.queue.append(ship_id)
        return None

    def release(self, choice: Optional[int] = None) -> Tuple[Optional[int], Optional[int]]:
        missing = {1,2} - set(self.free_berth_ids)
        if missing:
            freed = missing.pop()
            self.free_berth_ids.append(freed)
            self.free_berth_ids.sort()
        else:
            freed = None

        if self.queue:
            idx = choice if choice in range(len(self.queue)) else 0
            next_ship = self.queue.pop(idx)
            return next_ship, freed

        return None, freed

    def berth_size(self, berth_id: int) -> str:
        return 'big' if berth_id == 1 else 'small'

@dataclass
class PathSegment:
    frm: str
    to: str
    distance: float
    storm: Optional[str] = None

@dataclass
class StormZone:
    name: str
    months: Tuple[int, int]