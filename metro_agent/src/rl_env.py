import random
import numpy as np
from src.engine.engine import Engine


class MiniMetroRLEnv:
    def __init__(self, dt_ms=16, fail_threshold=30, congest_threshold=15):
        self.dt_ms = dt_ms
        self.fail_threshold = fail_threshold
        self.congest_threshold = congest_threshold
        self.engine = None
        self.t = 0

    def reset(self, seed=0):
        random.seed(seed)
        np.random.seed(seed)

        self.engine = Engine()
        self.t = 0

        return self._get_obs(), self._get_info()

    def step(self, action_id):
        self._apply_action(action_id)

        self.engine.increment_time(self.dt_ms)
        self.t += 1

        info = self._get_info()

        terminated = info["failed"]
        reward = -info["total_waiting"]

        return self._get_obs(), reward, terminated, False, info

    def _station_degree(self, station):
    # count how many paths include this station
        deg = 0
        for path in self.engine._components.paths:
            if station in path.stations:
                deg += 1
        return deg

    # -------------------------

    def _get_obs(self):
        # simplest observation: queue sizes at stations
        stations = self.engine._components.stations
        paths = self.engine._components.paths

        queues = [st.occupation for st in stations]
        degrees = [self._station_degree(st) for st in stations]

        num_paths = len(paths)

        return np.array(
            queues + degrees + [num_paths],
            dtype=np.float32
        )

    def _get_info(self):
        stations = self.engine._components.stations
        queues = [st.occupation for st in stations]
        caps   = [st.capacity for st in stations]

        max_queue = max(queues) if queues else 0
        total_waiting = sum(queues)

        # congestion event: any station over a chosen threshold
        congested = any(q > self.congest_threshold for q in queues)

        # failure condition (for survival time):
        # simplest + very defensible: station overflow
        failed = any(q > c for q, c in zip(queues, caps))

        return {
            "max_queue": max_queue,
            "total_waiting": total_waiting,
            "congested": congested,
            "failed": failed,
            "time": self.engine._components.status.game_time,
        }

    def _apply_action(self, action_id: int):
        # start simple: do nothing baseline
        pm = self.engine.path_manager
        stations = self.engine._components.stations
        paths = self.engine._components.paths

        if action_id == 0:
            return  # do nothing

        # ---------------------------------------
        # Action 1: create new connection between 2 most congested stations
        if action_id == 1 and len(stations) >= 2:
            # sort stations by queue size
            sorted_st = sorted(stations, key=lambda s: s.occupation, reverse=True)
            s1, s2 = sorted_st[:2]

            path = pm.create_path(s1)
            pm.add_station(s2)
            pm.finalize_path()
            return

        # ---------------------------------------
        # Action 2: extend busiest path toward most congested station
        if action_id == 2 and paths:
            busiest_path = max(paths, key=lambda p: sum(st.occupation for st in p.stations))
            worst_station = max(stations, key=lambda s: s.occupation)

            pm.extend_path(busiest_path, worst_station)
            return

        # ---------------------------------------
        # Action 3: remove most overloaded path
        if action_id == 3 and paths:
            overloaded_path = max(paths, key=lambda p: sum(st.occupation for st in p.stations))
            pm.remove_path(overloaded_path)
            return

        # ---------------------------------------
        # Action 4: connect least connected station
        if action_id == 4 and paths:
            # station with fewest path memberships
            least_connected = min(stations, key=lambda s: len(s.paths))
            target_path = min(paths, key=lambda p: len(p.stations))

            pm.extend_path(target_path, least_connected)
            return