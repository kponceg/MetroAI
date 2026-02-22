import random
import numpy as np

from src.engine.engine import Engine
import gymnasium as gym
from gymnasium import spaces

from src.entity import station

BaseEnv = gym.Env

class MiniMetroRLEnv(BaseEnv):
    def __init__(self,
                 dt_ms = 16,
                 fail_threshold = 10, # used to determined whether the game ends
                 congest_threshold = 8, # changes to determined congest of station
                 # for rl
                 max_stations = 15, # dimension
                 max_episode_steps = 2000, # max steps
                 occupancy_rate_capped = 5.0, # cap occ ratio prevent dominating
                 # reward weight
                 w_queue = 1, # control crowding
                 w_waiting = 0.01,
                 build_bonus = 5.0, # reward for building a path
                 invalid_action_penalty = 2.0, # prevent useless action
                 terminal_fail_penalty = 20.0,
                 ):
        self.dt_ms = dt_ms
        self.fail_threshold = fail_threshold
        self.congest_threshold = congest_threshold
        self.engine = None
        self.t = 0
        self.max_stations = max_stations
        self.max_episode_steps = max_episode_steps
        self.occupancy_rate_capped = occupancy_rate_capped
        # reward weights
        self.w_queue = w_queue
        self.w_waiting = w_waiting
        self.build_bonus = build_bonus
        self.invalid_action_penalty = invalid_action_penalty
        self.terminal_fail_penalty = terminal_fail_penalty

        self.last_total_waiting = 0.0
        self.last_num_paths = 0
        self.n_actions = 5

        observation_dim = 2 * self.max_stations + 3
        higher_bound = np.array([self.occupancy_rate_capped] * self.max_stations +
                                [1000] * self.max_stations +
                                [1000.0, 1000.0, 1.0], dtype=np.float32)
        lower_bound = np.zeros(observation_dim)
        self.observation_space = spaces.Box(low=lower_bound, high=higher_bound, dtype = np.float32)
        self.action_space = spaces.Discrete(self.n_actions) # 0 ... 4


    def reset(self, seed=0):
        random.seed(seed)
        np.random.seed(seed)

        self.engine = Engine()
        self.t = 0

        obs = self._get_obs()
        info = self._get_info()
        self.last_total_waiting = float(info['total_waiting'])
        self.last_num_paths = info.get('num_paths', 0)

        return obs, info

    def step(self, action):
        #self._apply_action(action)
        invalid = not self._apply_action(action) # prevent breaking

        self.engine.increment_time(self.dt_ms)
        self.t += 1
        info = self._get_info()

        info["invalid_action"] = invalid

        terminated = info["failed"] # bool
        truncated = self.t >= self.max_episode_steps # max time

        # reward
        total_waiting = info["total_waiting"]
        num_paths = info.get("num_paths", 0)
        #max_queue = info["max_queue"]

        delta_wait = self.last_total_waiting - total_waiting # less people waiting to more rewards
        reward = delta_wait
        reward -= self.w_waiting * total_waiting

        delta_paths = num_paths - self.last_num_paths
        if delta_paths > 0:
            reward += self.build_bonus * delta_paths

        if invalid:
            reward -= self.invalid_action_penalty
        if terminated:
            reward -= self.terminal_fail_penalty

        self.last_total_waiting = total_waiting
        self.last_num_paths = num_paths

        return self._get_obs(), float(reward), terminated, truncated, info

    def _station_degree(self, station):
        # count how many paths include this station
        deg = 0
        for path in self.engine._components.paths:
            if station in path.stations:
                deg += 1
        return deg

# -------------------------
    def _num_paths(self) -> int:
        return len(self.engine._components.paths)

# -------------------------

    def _get_obs(self):
        # simplest observation: queue sizes at stations
        stations = self.engine._components.stations
        paths = self.engine._components.paths

        num_stations = len(stations) # for future increasing stations
        num_paths = len(paths)
        stations = stations[:self.max_stations] # pending to review

        queues = [] # occ_rate
        degrees = []
        for st in stations:
            occ = st.occupation
            capacity = st.capacity
            r = occ / capacity
            queues.append(min(r, self.occupancy_rate_capped))
            degrees.append(self._station_degree(st))
        #queues = [st.occupation for st in stations]
        #degrees = [self._station_degree(st) for st in stations]

        pad = self.max_stations - len(stations) # padding
        if pad >0:
            queues.extend([0.0] * pad)
            degrees.extend([0.0] * pad)

        t_norm = self.t / self.max_episode_steps # early/late game


        return np.array(
            queues + degrees + [num_paths, num_stations, t_norm],
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
        # 1. stations overflow or 2. too many station waiting (max queue reach fail_threshold)
        overflow = any(q > c for q, c in zip(queues, caps))
        threshold_fail = max_queue > self.fail_threshold
        failed = overflow or threshold_fail

        return {
            "max_queue": max_queue,
            "total_waiting": total_waiting,
            "congested": congested,
            "failed": failed,
            "time": self.engine._components.status.game_time,
            "num_paths": self._num_paths()
        }

    def _apply_action(self, action_id: int) -> bool:
        # start simple: do nothing baseline
        if action_id < 0 or action_id >= self.n_actions:
            return False

        pm = self.engine.path_manager
        stations = self.engine._components.stations
        paths = self.engine._components.paths

        if action_id == 0:
            return True # do nothing

        # ---------------------------------------
        # Action 1: create new connection between 2 most congested stations
        if action_id == 1 and len(stations) >= 2:
            before = self._num_paths()
            # sort stations by queue size
            sorted_st = sorted(stations, key=lambda s: s.occupation, reverse=True)
            s1, s2 = sorted_st[:2]
            try:
                w = pm.start_path_on_station(s1)
                if w is None:
                    return False # reach limited

                creating = pm._creating_or_expanding_path
                if creating is None:
                    return False

                creating.add_station_to_path(s2)
                creating.try_to_end_path_on_station(s2)

                after = self._num_paths()
                return after > before  # true if new line has created

            except Exception:
                return False
            finally:
                # close
                pm._creating_or_expanding_path = None


        # ---------------------------------------
        # Action 2: extend busiest path toward most congested station
        if action_id == 2 and paths:
            try:
                busiest_path = max(paths, key=lambda p: sum(st.occupation for st in p.stations))
                worst_station = max(stations, key=lambda s: s.occupation)

                pm.extend_path(busiest_path, worst_station)
                return True
            except Exception:
                return False

        # ---------------------------------------
        # Action 3: remove most overloaded path
        if action_id == 3 and paths:
            try:
                overloaded_path = max(paths, key=lambda p: sum(st.occupation for st in p.stations))
                pm.remove_path(overloaded_path)
                return True
            except Exception:
                return False

        # ---------------------------------------
        # Action 4: connect least connected station
        if action_id == 4 and paths:
            # station with fewest path memberships
            try:
                least_connected = min(stations, key=lambda s: len(s.paths))
                target_path = min(paths, key=lambda p: len(p.stations))

                pm.extend_path(target_path, least_connected)
                return True
            except Exception:
                return False