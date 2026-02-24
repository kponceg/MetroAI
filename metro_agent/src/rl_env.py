import random
import numpy as np

from src.engine.engine import Engine
import gymnasium as gym
from gymnasium import spaces

BaseEnv = gym.Env

class MiniMetroRLEnv(BaseEnv):
    def __init__(self,
                 dt_ms = 16,
                 fail_threshold = 10, # used to determined whether the game ends
                 congest_threshold = 9, # changes to determined congest of station
                 # for rl
                 max_stations = 10, # dimension
                 max_episode_steps = 20000, # max steps
                 occupancy_rate_capped = 5.0, # cap occ ratio prevent dominating
                 # TODO: tune weights
                 # reward weight
                 w_queue = 0.5, # control crowding
                 w_waiting = 0.02,
                 build_bonus = 2.0, # reward for building a path
                 expand_bonus = 3.0,
                 remove_penalty = 1.0,
                 action_cost = 0.01,
                 path_cost = 0.01,
                 congest_penalty_scale = 0.1, # not used for now
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
        self.expand_bonus = expand_bonus
        self.remove_penalty = remove_penalty
        self.action_cost = action_cost
        self.path_cost = path_cost
        self.congest_penalty_scale = congest_penalty_scale
        self.invalid_action_penalty = invalid_action_penalty
        self.terminal_fail_penalty = terminal_fail_penalty
        self.timeout_ms = 2000 # used to determine game end by overflow
        self.last_total_waiting = 0.0
        self.last_num_paths = 0
        self.n_actions = 5 # 5
        self._last_action = None # none/create/expand/remove
        self.invalid_streak = 0
        self.max_invalid_streak = 50  # cap to avoid exploding penalties
        self.invalid_streak_scale = 0.5  # tune: 0.3–1.0

        observation_dim = 2 * self.max_stations + 4
        higher_bound = np.array(
            [self.occupancy_rate_capped] * self.max_stations +
            [1000] * self.max_stations +
            [1000.0, 1000.0, 1.0, 1.0],  # extra 1.0 for paths_left
            dtype=np.float32
        )
        lower_bound = np.zeros(observation_dim, dtype=np.float32)
        self.observation_space = spaces.Box(low=lower_bound, high=higher_bound, dtype = np.float32)
        self.action_space = spaces.MultiDiscrete([self.n_actions, self.max_stations, self.max_stations]) # 0 ... 4
        # private
        self._station_rank = {}  # station.id -> first-seen index
        self._station_ids = []  # list of station.id in first-seen order
        self._timeout_ms_by_station_id = {}

    def reset(self, seed=0):
        self.invalid_streak = 0
        random.seed(seed)
        np.random.seed(seed)

        self.engine = Engine()
        self.t = 0
        self._station_rank.clear()
        self._station_ids.clear()

        obs = self._get_obs()
        info = self._get_info()
        self.last_total_waiting = float(info['total_waiting'])
        self.last_num_paths = info.get('num_paths', 0)
        self._timeout_ms_by_station_id = {st.id: 0 for st in self.engine._components.stations}

        return obs, info

    def step(self, action):

        a = np.asarray(action, dtype=int)
        act = int(a.flatten()[0])

        # reset, will be set inside apply action
        self._last_action = None

        success = self._apply_action(action)

        self.engine.increment_time(self.dt_ms)

        # update overflow timers
        for st in self.engine._components.stations:
            self._timeout_ms_by_station_id.setdefault(st.id, 0)

            q = float(st.occupation)
            cap = float(st.capacity)

            if q >= cap:  # reach the 12 limit start count down
                self._timeout_ms_by_station_id[st.id] += self.dt_ms
            else:
                self._timeout_ms_by_station_id[st.id] = 0

        self.t += 1
        info = self._get_info()

        # check termination
        invalid = not success
        info["invalid_action"] = invalid
        
        if invalid:
            self.invalid_streak = min(self.invalid_streak + 1, self.max_invalid_streak)
        else:
            self.invalid_streak = 0
        info["invalid_streak"] = self.invalid_streak

        terminated = bool(info["failed"])
        truncated = self.t >= self.max_episode_steps # max time

        # reward
        total_waiting = info["total_waiting"]
        num_paths = info.get("num_paths", 0)

        # reward on reduce waiting
        delta_wait = self.last_total_waiting - total_waiting # less people waiting to more rewards
        reward = delta_wait - self.w_waiting * total_waiting

        # regularization cost
        if act != 0:
            reward -= self.action_cost
        reward -= self.path_cost * num_paths

        # reward/penalty on operations
        if self._last_action == "create":
            reward += self.build_bonus
        elif self._last_action == "expand":
            reward += self.expand_bonus
        elif self._last_action == "remove":
            reward -= self.remove_penalty

        # penalties
        if invalid:
            # base penalty
            penalty = self.invalid_action_penalty

            # chained invalid penalty (quadratic growth is strong; linear is gentler)
            penalty += self.invalid_streak_scale * (self.invalid_streak ** 2)

            reward -= penalty
        if terminated:
            reward -= self.terminal_fail_penalty

        # PENDING TO REVIEW
        #max_queue = float(info.get("max_queue", 0.0))
        #over = max(0.0, max_queue - float(self.congest_threshold))
        #reward -= self.congest_penalty_scale * over

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
    def _sorted_stations(self) -> list:
        """
        return a sorted list of stations, and prevent reshuffle in future increasing stations
        """
        current = list(self.engine._components.stations)

        for st in current:
            station_id = st.id
            if station_id not in self._station_ids:
                self._station_rank[station_id] = len(self._station_ids)
                self._station_ids.append(station_id)

        station_map = {st.id: st for st in current}

        sorted_stations = [station_map[sid] for sid in self._station_ids if sid in station_map]
        return sorted_stations

    # pending to review
    def _sorted_paths(self) -> list:
        """
        return a sorted list of paths by id?
        """
        paths = list(self.engine._components.paths)
        try:
            paths.sort(key=lambda p: getattr(p, "path_order", getattr(p, "order", 0)))
        except Exception:
            pass
        return paths
    # -------------------------

    def _get_obs(self):
        # simplest observation: queue sizes at stations
        stations = self._sorted_stations() #self.engine._components.stations
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

        pad = self.max_stations - len(stations) # padding
        if pad >0:
            queues.extend([0.0] * pad)
            degrees.extend([0.0] * pad)

        t_norm = self.t / self.max_episode_steps # early/late game

        pm = self.engine.path_manager
        max_paths = pm.max_num_paths
        paths_left = (max_paths - num_paths) / max(1, max_paths)

        return np.array(
            queues + degrees + [num_paths, num_stations, paths_left, t_norm],
            dtype=np.float32
        )

    def _get_info(self):
        stations = self.engine._components.stations
        queues = [st.occupation for st in stations]
        # caps   = [st.capacity for st in stations]

        max_queue = max(queues) if queues else 0
        total_waiting = sum(queues)

        # congestion event: any station over a chosen threshold
        # congest_ratio = 0.75 # PENDING TO REVIEW;
        # congested = any(q >= congest_ratio * cap for q, cap in zip(queues, caps))
        congested = any(q > self.congest_threshold for q in queues)

        # failure condition (for survival time):
        # 1. stations overflow or 2. too many station waiting (max queue reach fail_threshold)
        overflow = any(ms >= self.timeout_ms for ms in self._timeout_ms_by_station_id.values())
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

    def _apply_action(self, action: int) -> bool:
        """
        Parameter:
        action: (action_id, station_i, station_j) from MultiDiscrete

        Return:
        true -> if the action is executed successfully
        false -> if the action is invalid
        """
        try:
            a = np.asarray(action, dtype=int)
            if a.ndim == 0:
                action_id, i, j = int(a), 0, 0
            elif a.ndim == 1:
                action_id, i, j = int(a[0]), int(a[1]), int(a[2])
            else:
                # VecEnv (n_envs, 3)
                action_id, i, j = int(a[0, 0]), int(a[0, 1]), int(a[0, 2])
        except Exception:
            return False

        def path_load(p):
            return sum(st.occupation for st in p.stations)

        pm = self.engine.path_manager
        stations = self._sorted_stations()
        paths = self.engine._components.paths
        n_stations = min(self.max_stations, len(stations))
        n_paths = min(self.max_stations, len(paths))

        # ---------------------------------------
        # Action 0: Do Nothing
        if action_id == 0:
            return True

        # ---------------------------------------
        # Action 1: create new connection between 2 most congested stations
        if action_id == 1:
            if len(paths) >= pm.max_num_paths:
                return False
            if n_stations < 2:
                return False
            if i < 0 or j < 0 or i >= n_stations or j >= n_stations or i == j:
                return False

            s1 = stations[i]
            s2 = stations[j]

            before = self._num_paths()
            # sort stations by queue size
            # sorted_st = sorted(stations, key=lambda s: s.occupation, reverse=True)
            # s1, s2 = sorted_st[:2]
            try:
                w = pm.start_path_on_station(s1)
                if w is None:
                    return False # reach resources limit

                creating = pm._creating_or_expanding_path
                if creating is None:
                    return False

                creating.add_station_to_path(s2)
                creating.try_to_end_path_on_station(s2)

                after = self._num_paths()

            except Exception as e:
                print(f"Action1 Error: {e}")
                return False
            finally:
                # close
                pm._creating_or_expanding_path = None

            if after > before: # true if new line has created
                self._last_action = "create"
                return True
            else:
                return False



        # ---------------------------------------
        # Action 2: extend busiest path toward most congested station
        if action_id == 2:
            if n_paths == 0 or n_stations < 2:
                return False

            # HARD bounds check
            if i < 0 or j < 0 or i >= n_paths or j >= n_stations:
                return False

            chosen_path = paths[i]
            target = stations[j]

            # don't add duplicate station
            if target in chosen_path.stations:
                return False

            anchors = []
            if chosen_path.stations:
                anchors.append(chosen_path.stations[-1])
                anchors.append(chosen_path.stations[0])

            for anchor in anchors:
                if anchor is None:
                    continue

                candidate_paths = pm.get_paths_with_station(anchor)
                if not candidate_paths or chosen_path not in candidate_paths:
                    continue

                local_idx = candidate_paths.index(chosen_path)
                before_len = len(chosen_path.stations)

                try:
                    editor = pm.start_expanding_path_on_station(anchor, local_idx)
                    if editor is None:
                        continue

                    expanding = pm._creating_or_expanding_path
                    if not expanding:
                        continue

                    expanding.add_station_to_path(target)

                finally:
                    pm._creating_or_expanding_path = None

                if len(chosen_path.stations) > before_len:
                    self._last_action = "expand"
                    return True

            return False

        # ---------------------------------------
        # Action 3: remove least useful path and add new connection between top congested stations
        if action_id == 3:
            if n_paths == 0 or n_stations < 2:
                return False

            # choose a path to remove:
            # use i if valid, otherwise remove lowest-load path
            if 0 <= i < n_paths:
                path_remove = paths[i]
            else:
                path_remove = min(paths, key=path_load)

            before = self._num_paths()

            try:
                pm.remove_path(path_remove)

                # choose stations to connect (use i,j if valid, else top-2 congested)
                if 0 <= i < n_stations and 0 <= j < n_stations and i != j:
                    s1, s2 = stations[i], stations[j]
                else:
                    sorted_st = sorted(stations[:n_stations], key=lambda s: s.occupation, reverse=True)
                    s1, s2 = sorted_st[0], sorted_st[1]

                w = pm.start_path_on_station(s1)
                if w is None:
                    return False

                creating = pm._creating_or_expanding_path
                if creating is None:
                    return False

                creating.add_station_to_path(s2)
                creating.try_to_end_path_on_station(s2)

                after = self._num_paths()

            except Exception:
                return False
            finally:
                pm._creating_or_expanding_path = None

            # succeeded if we didn't permanently lose a path
            if after >= before:
                self._last_action = "replace"   
                return True
            return False

        # ---------------------------------------
        # Action 4: connect the least-connected high-demand station to a path that includes a congested station
        if action_id == 4:
            if n_paths == 0 or n_stations == 0:
                return False

            # HARD bounds
            if i < 0 or j < 0 or i >= n_paths or j >= n_stations:
                return False

            chosen_path = paths[i]
            target = stations[j]

            if target in chosen_path.stations:
                return False

            # anchor must be a station on chosen_path
            anchor = chosen_path.last_station
            candidate_paths = pm.get_paths_with_station(anchor)
            if not candidate_paths or chosen_path not in candidate_paths:
                return False

            local_idx = candidate_paths.index(chosen_path)
            before_len = len(chosen_path.stations)
            try:
                editor = pm.start_expanding_path_on_station(anchor, local_idx)
                if editor is None:
                    return False

                expanding = pm._creating_or_expanding_path
                if not expanding:
                    return False

                expanding.add_station_to_path(target)

            finally:
                pm._creating_or_expanding_path = None

            if len(chosen_path.stations) > before_len:
                self._last_action = "expand"
                return True
            return False
    
