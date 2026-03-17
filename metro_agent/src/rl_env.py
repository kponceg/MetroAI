import random
import numpy as np

from src.engine.engine import Engine
from src.config import Config, max_num_paths, station_capacity, station_shape_type_list

import gymnasium as gym
from gymnasium import spaces

BaseEnv = gym.Env

class MiniMetroRLEnv(BaseEnv):
    metadata = {'render_modes': []}

    def __init__(self,
                 dt_ms = 16,
                 decision_interval_ms = 250,
                 warning_ratio = 0.75, # congest_ratio

                 # rl
                 max_stations = Config.num_stations, # dimension
                 max_episode_steps = 4000,
                 occupancy_rate_capped = 5.0, # cap occ ratio prevent dominating

                 # TODO: tune reward weights
                 w_waiting = 0.01,
                 build_bonus = 0.3, # reward for building a path
                 expand_bonus = 1.0,
                 remove_penalty = 0.5,
                 action_cost = 0.05,
                 path_cost = 0.05,

                 survival_reward = 0.1,
                 w_score = 1.0,
                 w_queue_delta = 0.5,
                 w_critical = 0.5,

                 invalid_action_penalty = 0.5, # prevent useless action
                 terminal_fail_penalty = 30.0,
                 ):
        self.dt_ms = dt_ms # engine dt ms
        self.decision_interval_ms = decision_interval_ms
        self.warning_ratio = warning_ratio

        self.engine = None
        self.t = 0
        self.elapsed_ms = 0
        self.n_actions = 5

        # rl
        self.max_stations = max_stations
        self.max_paths = max_num_paths
        self.station_capacity = station_capacity
        self.spawn_interval_ms = int(Config.passenger_spawning.interval_step * 1000)
        self.max_episode_steps = max_episode_steps
        self.occupancy_rate_capped = occupancy_rate_capped
        self.dest_shape_types = list(station_shape_type_list)
        self.num_dest_shape_types = len(self.dest_shape_types)

        # reward weights
        self.w_waiting = w_waiting
        self.build_bonus = build_bonus
        self.expand_bonus = expand_bonus
        self.remove_penalty = remove_penalty
        self.action_cost = action_cost
        self.w_score = w_score
        self.w_queue_delta = w_queue_delta
        self.w_critical = w_critical

        self.path_cost = path_cost
        self.survival_reward = survival_reward

        self.edit_cooldown_ms = 1500
        self.remove_cooldown_ms = 4000
        self.min_path_age_ms = 5000

        self._edit_cooldown_left_ms = 0
        self._remove_cooldown_left_ms = 0
        self._path_birth_ms = {}

        self.invalid_action_penalty = invalid_action_penalty
        self.terminal_fail_penalty = terminal_fail_penalty
        self.timeout_ms = 10000 # used to determine game end by overflow; 10 sec

        self.last_total_waiting = 0.0
        self.last_score = 0.0
        self.last_max_queue = 0.0

        self._last_action = "none" # none/create/expand/remove
        self.invalid_streak = 0
        self.max_invalid_streak = 50  # cap to avoid exploding penalties
        self.invalid_streak_scale = 0.5  # tune: 0.3–1.0

        # --- modified the new observation space
        # station features:
        # 1) queue ratio: congest rate
        # abandoned 2) degree norm: connection info
        # 3) on_any_path: station on any path
        # abandoned 4) is_endpoint: is this station at the end of the path
        # abandoned 5) warning_flag: almost full, need action
        # abandoned 6) critical_flag: is full, need action immediately
        # 7) destination shape distribution at this station
        # 8) station shape
        station_feat_dim = (2 + 2 * self.num_dest_shape_types) *  self.max_stations #
        # station_feat_dim = (6 + self.num_dest_shape_types) * self.max_stations

        # path features for each path slot:
        # 1) path_exists
        # 2) start_idx_norm :
        # 3) end_idx_norm : with start idx norm, tells agent where is the path
        # 4) path_len_norm : long or short path
        # 5) path_load_norm : tell the agent the path is busy or not
        # abandoned 6) can_expand_left: can path extend to its left?
        # abandoned 7) can_expand_right: can path extend to its left?
        # abandoned 8) left_endpoint_queue_ratio： how congested are the left and right ends
        # abandoned 9) right_endpoint_queue_ratio
        # abandoned 10) left_endpoint_degree_norm： whats the degree of left and right ends
        # abandoned 11) right_endpoint_degree_norm
        path_feat_dim = 5 * self.max_paths #
        # path_feat_dim = 11 * self.max_paths

        # global features:
        # 1) num_paths_norm : current network size
        # 2) paths_left_norm : path remains
        # 3) spawn_progress: 0 to 1 progress of next passenger spawning
        # 4) max_queue_ratio: tell the agent how congest is now
        # 5) mean_queue_ratio
        # 6) num_unserved_stations_norm: num of stations not connected
        global_feat_dim = 6

        observation_dim = station_feat_dim + path_feat_dim + global_feat_dim

        higher_bound = np.array(
            # station features
            [self.occupancy_rate_capped] * self.max_stations +  # queue ratio
            #[1.0] * self.max_stations +  # degree norm
            [1.0] * self.max_stations +  # on_any_path

            #[1.0] * self.max_stations +  # is_endpoint
            #[1.0] * self.max_stations +  # warning
            #[1.0] * self.max_stations +  # critical

            [1.0] * self.max_stations * self.num_dest_shape_types + # station self shap
            [1.0] * self.max_stations * self.num_dest_shape_types +  # destination-shape distribution

            # path features
            [1.0] * self.max_paths +  # exists
            [1.0] * self.max_paths +  # start idx norm
            [1.0] * self.max_paths +  # end idx norm
            [1.0] * self.max_paths +  # path len norm
            [1.0] * self.max_paths +  # path load norm

            #[1.0] * self.max_paths +  # can_expand_left
            #[1.0] * self.max_paths +  # can_expand_right
            #[self.occupancy_rate_capped] * self.max_paths +  # left endpoint queue
            #[self.occupancy_rate_capped] * self.max_paths +  # right endpoint queue
            #[1.0] * self.max_paths +  # left endpoint degree
            #[1.0] * self.max_paths +  # right endpoint degree

            # global features
            [1.0, 1.0, 1.0, self.occupancy_rate_capped, self.occupancy_rate_capped, 1.0],
            dtype=np.float32
        )
        lower_bound = np.zeros(observation_dim, dtype=np.float32)

        self.observation_space = spaces.Box(low=lower_bound, high=higher_bound, dtype = np.float32)
        self.action_space = spaces.MultiDiscrete([self.n_actions, self.max_stations, self.max_stations]) # 0 ... 4

        # private
        self._station_rank = {}  # station.id -> first-seen index
        self._station_ids = []  # list of station.id in first-seen order
        self._timeout_ms_by_station_id = {}

    def reset(self, *, seed = None, options = None):
        super().reset(seed = seed)

        self.invalid_streak = 0
        random.seed(seed)
        np.random.seed(seed)

        self.engine = Engine()
        self.t = 0
        self.elapsed_ms = 0
        self._station_rank.clear()
        self._station_ids.clear()
        self._timeout_ms_by_station_id = {st.id: 0 for st in self.engine._components.stations}
        self._last_action = "none"

        self._edit_cooldown_left_ms = 0
        self._remove_cooldown_left_ms = 0
        self._path_birth_ms = {}

        obs = self._get_obs()
        info = self._get_info()
        self.last_total_waiting = float(info['total_waiting'])
        self.last_score = float(info["score"])

        self.last_max_queue = float(info["max_queue"])

        return obs, info

    def step(self, action):
        a = np.asarray(action, dtype=int)
        act = int(a.flatten()[0])
        self._last_action = "none" # reset, will be set inside apply action

        success = self._apply_action(action)
        self._advance_game(self.decision_interval_ms)

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
        score = float(info["score"])
        num_paths = info.get("num_paths", 0)
        #num_critical = int(info["num_critical"])
        #num_warning = int(info["num_warning"])
        max_queue = float(info["max_queue"])

        #delta_maxq = self.last_max_queue - max_queue
        #delta_maxq = float(np.clip(delta_maxq, -2.0, 2.0))

        # reward on reduce waiting
        #delta_wait = self.last_total_waiting - total_waiting # less people waiting to more rewards
        #delta_wait = float(np.clip(delta_wait, -5.0, 5.0))
        delta_score = score - self.last_score

        current_paths = self._sorted_paths()[:self.max_paths]
        avg_path_len = np.mean([self._path_len_norm(p) for p in current_paths]) if len(current_paths) > 0 else 0.0


        reward = 0.0
        reward += delta_score
        reward -= 0.02 * total_waiting
        reward -= 0.1 * avg_path_len

        #reward += self.survival_reward
        #reward += self.w_score * delta_score
        #reward += self.w_queue_delta * delta_wait

        #reward -= self.w_waiting * total_waiting
        #reward -= self.w_critical * num_critical
        #reward -= self.path_cost * num_paths

        #reward += 0.2 * delta_maxq

        # regularization cost
        #if act != 0:
        #    reward -= self.action_cost

        # global penalty on congestion
        #reward -= 0.03 * num_warning

        # a little more when agent choose to do nothing
        #if act == 0 and (num_warning > 0 or num_critical > 0):
        #    reward -= 0.5
        if act == 0 and total_waiting > 0:
            reward -= 0.5

        # reward/penalty on operations
        #if self._last_action == "create":
        #    reward += self.build_bonus
        #elif self._last_action == "expand":
        #    reward += self.expand_bonus
        #elif self._last_action == "remove":
        #    reward -= self.remove_penalty

        #reward -= self.action_cost

        # penalties
        if invalid:
            reward -= 1
            if act == 1:
                reward -= 0.5
        #    reward -= self.invalid_action_penalty
        #    reward -= 0.05 * self.invalid_streak

        if terminated:
            reward -= self.terminal_fail_penalty

        self.last_total_waiting = total_waiting
        self.last_score = score
        self.last_max_queue = max_queue

        return self._get_obs(), float(reward), terminated, truncated, info

# -------------------------
# private help functions
    def _station_degree(self, station):
        # count how many paths include this station
        deg = 0
        for path in self.engine._components.paths:
            if station in path.stations:
                deg += 1
        return deg

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

    def _station_on_any_path(self, station) -> float:
        for p in self.engine._components.paths:
            if station in p.stations:
                return 1.0
        return 0.0

    def _station_is_endpoint(self, station) -> float:
        for p in self.engine._components.paths:
            if not getattr(p, "stations", None):
                continue
            if station == p.stations[0] or station == p.stations[-1]:
                return 1.0
        return 0.0

    def _station_index_norm(self, station) -> float:
        sid = station.id
        if sid not in self._station_rank:
            return 0.0
        if self.max_stations <= 1:
            return 0.0
        idx = min(self._station_rank[sid], self.max_stations - 1)
        return idx / float(self.max_stations - 1)

    def _path_load_norm(self, path) -> float:
        sts = getattr(path, "stations", [])
        if not sts:
            return 0.0
        total_occ = float(sum(float(st.occupation) for st in sts))
        denom = max(1.0, float(len(sts) * self.station_capacity))
        return min(total_occ / denom, 1.0)

    def _path_len_norm(self, path) -> float:
        sts = getattr(path, "stations", [])
        if not sts:
            return 0.0
        return min(len(sts) / max(1.0, float(self.max_stations)), 1.0)

    def _advance_game(self, total_ms: int) -> None:
        remaining = int(total_ms)
        while remaining > 0:
            dt = min(self.dt_ms, remaining)
            self.engine.increment_time(dt)
            self.elapsed_ms += dt
            self._update_overflow_timers(dt)

            self._edit_cooldown_left_ms = max(0, self._edit_cooldown_left_ms - dt)
            self._remove_cooldown_left_ms = max(0, self._remove_cooldown_left_ms - dt)
            remaining -= dt

    def _update_overflow_timers(self, dt_ms: int) -> None:
        for st in self.engine._components.stations:
            self._timeout_ms_by_station_id.setdefault(st.id, 0)
            q = float(st.occupation)
            cap = float(st.capacity)
            if q >= cap:
                self._timeout_ms_by_station_id[st.id] += dt_ms
            else:
                self._timeout_ms_by_station_id[st.id] = 0

    def _path_can_expand_from_anchor(self, path, anchor) -> float:
        sts = getattr(path, "stations", [])
        if not sts:
            return 0.0
        if anchor != sts[0] and anchor != sts[-1]:
            return 0.0

        # if all stations are already on this path, no obvious expansion target
        current_ids = {st.id for st in sts}
        all_stations = self._sorted_stations()[:self.max_stations]
        for st in all_stations:
            if st.id not in current_ids:
                return 1.0
        return 0.0

    def _station_queue_ratio(self, station) -> float:
        occ = float(station.occupation)
        cap = float(station.capacity)
        if cap <= 0:
            return 0.0
        return min(occ / cap, self.occupancy_rate_capped)

    def _station_degree_norm(self, station) -> float:
        return self._station_degree(station) / max(1.0, float(self.max_paths))

    def _station_dest_shape_distribution(self, station) -> list[float]:
        counts = {shape_type: 0.0 for shape_type in self.dest_shape_types}

        passengers = getattr(station, "passengers", [])
        total = float(len(passengers))
        if total <= 0:
            return [0.0 for _ in self.dest_shape_types]

        for passenger in passengers:
            dst_type = passenger.destination_shape.type
            if dst_type in counts:
                counts[dst_type] += 1.0

        return [counts[shape_type] / total for shape_type in self.dest_shape_types]

    def _sorted_paths(self) -> list:
        paths = list(self.engine._components.paths)
        try:
            paths.sort(key=lambda p: getattr(p, "path_order", getattr(p, "order", 0)))
        except Exception:
            pass
        return paths

    def _path_key(self, path):
        return getattr(path, "id", id(path))

    def _path_is_busy(self, path) -> bool:
        for metro in getattr(path, "metros", []):
            # there are passengers on the train
            if getattr(metro, "passengers", None):
                if len(metro.passengers) > 0:
                    return True
            # metro is still on the way
            if getattr(metro, "current_station", None) is None:
                return True
        return False

    def _station_self_shape(self, station) -> list[float]:
        st_type = station.shape.type
        return [1.0 if st_type == shape_type else 0.0 for shape_type in self.dest_shape_types]

    # -------------------------

    def _get_obs(self):
        stations = self._sorted_stations()[:self.max_stations]
        paths = self._sorted_paths()[:self.max_paths]

        max_queue = 0.0

        # station features
        queues = [] # occ_rate
        #degrees = []
        on_path = []

        #endpoints = []
        #warning_flags = []
        #critical_flags = []

        # one list per destination shape type
        dest_shape_features = [[] for _ in range(self.num_dest_shape_types)]
        self_shape_features = [[] for _ in range(self.num_dest_shape_types)]

        for st in stations:
            occ = float(st.occupation)
            capacity = float(st.capacity)
            r = occ / capacity if capacity > 0 else 0.0

            queues.append(min(r, self.occupancy_rate_capped))
            #degrees.append(self._station_degree(st) / max(1.0, float(self.max_paths)))
            on_path.append(self._station_on_any_path(st))

            #endpoints.append(self._station_is_endpoint(st))
            #warning_flags.append(1.0 if occ >= self.warning_ratio * capacity else 0.0)
            #critical_flags.append(1.0 if occ >= capacity else 0.0)
            dest_dist = self._station_dest_shape_distribution(st)
            for k, value in enumerate(dest_dist):
                dest_shape_features[k].append(value)

            self_shape = self._station_self_shape(st)
            for k, value in enumerate(self_shape):
                self_shape_features[k].append(value)

            max_queue = max(max_queue, occ)

        pad = self.max_stations - len(stations) # padding
        if pad >0:
            queues.extend([0.0] * pad)
            #degrees.extend([0.0] * pad)
            on_path.extend([0.0] * pad)

            #endpoints.extend([0.0] * pad)
            #warning_flags.extend([0.0] * pad)
            #critical_flags.extend([0.0] * pad)
            for k in range(self.num_dest_shape_types):
                dest_shape_features[k].extend([0.0] * pad)
                self_shape_features[k].extend([0.0] * pad)

        # path features
        path_exists = []
        path_start_idx = []
        path_end_idx = []
        path_len = []
        path_load = []

        #can_expand_left = []
        #can_expand_right = []
        #left_endpoint_queue = []
        #right_endpoint_queue = []
        #left_endpoint_degree = []
        #right_endpoint_degree = []

        for p in paths:
            sts = getattr(p, "stations", [])
            if not sts:
                path_exists.append(0.0)
                path_start_idx.append(0.0)
                path_end_idx.append(0.0)
                path_len.append(0.0)
                path_load.append(0.0)

                #can_expand_left.append(0.0)
                #can_expand_right.append(0.0)
                #left_endpoint_queue.append(0.0)
                #right_endpoint_queue.append(0.0)
                #left_endpoint_degree.append(0.0)
                #right_endpoint_degree.append(0.0)
                continue

            start_st = sts[0]
            end_st = sts[-1]

            path_exists.append(1.0)
            path_start_idx.append(self._station_index_norm(start_st))
            path_end_idx.append(self._station_index_norm(end_st))
            path_len.append(min(len(sts) / max(1.0, float(self.max_stations)), 1.0))
            path_load.append(self._path_load_norm(p))

            #can_expand_left.append(self._path_can_expand_from_anchor(p, start_st))
            #can_expand_right.append(self._path_can_expand_from_anchor(p, end_st))
            #left_endpoint_queue.append(self._station_queue_ratio(start_st))
            #right_endpoint_queue.append(self._station_queue_ratio(end_st))
            #left_endpoint_degree.append(self._station_degree_norm(start_st))
            #right_endpoint_degree.append(self._station_degree_norm(end_st))

        pad_p = self.max_paths - len(paths)
        if pad_p > 0:
            path_exists.extend([0.0] * pad_p)
            path_start_idx.extend([0.0] * pad_p)
            path_end_idx.extend([0.0] * pad_p)
            path_len.extend([0.0] * pad_p)
            path_load.extend([0.0] * pad_p)

            #can_expand_left.extend([0.0] * pad_p)
            #can_expand_right.extend([0.0] * pad_p)
            #left_endpoint_queue.extend([0.0] * pad_p)
            #right_endpoint_queue.extend([0.0] * pad_p)
            #left_endpoint_degree.extend([0.0] * pad_p)
            #right_endpoint_degree.extend([0.0] * pad_p)

        # global features
        num_paths = self._num_paths()
        paths_left = max(0, self.max_paths - num_paths)

        num_paths_norm = num_paths / max(1, self.max_paths)
        paths_left_norm = paths_left / max(1, self.max_paths)

        ms_until_next_spawn = float(self.engine._passenger_spawner.ms_until_next_spawn)
        spawn_progress = 1.0 - np.clip(ms_until_next_spawn / max(1.0, self.spawn_interval_ms), 0.0, 1.0)

        mean_queue_ratio = float(np.mean(queues[:len(stations)])) if len(stations) > 0 else 0.0
        num_unserved = sum(1.0 - x for x in on_path[:len(stations)])
        num_unserved_stations_norm = num_unserved / max(1.0, float(self.max_stations))

        max_queue_ratio = min(max_queue / max(1.0, self.station_capacity), self.occupancy_rate_capped)

        return np.array(
            queues +
            #degrees +
            on_path +
            #endpoints +
            #warning_flags +
            #critical_flags +
            sum(self_shape_features, []) +
            sum(dest_shape_features, []) +
            path_exists +
            path_start_idx +
            path_end_idx +
            path_len +
            path_load +
            #can_expand_left +
            #can_expand_right +
            #left_endpoint_queue +
            #right_endpoint_queue +
            #left_endpoint_degree +
            #right_endpoint_degree +
            [num_paths_norm, paths_left_norm, spawn_progress, max_queue_ratio, mean_queue_ratio, num_unserved_stations_norm],
            dtype=np.float32
        )

    def _get_info(self):
        stations = self.engine._components.stations
        queues = [st.occupation for st in stations]
        caps   = [st.capacity for st in stations]

        max_queue = max(queues) if queues else 0
        total_waiting = sum(queues)
        num_critical = sum(q >= c for q, c in zip(queues, caps))
        num_warning = sum(q >= self.warning_ratio * c for q, c in zip(queues, caps))

        # failure condition (for survival time):
        # 1. stations overflow or 2. too many station waiting (max queue reach fail_threshold)
        overflow = any(ms >= self.timeout_ms for ms in self._timeout_ms_by_station_id.values())
        if num_critical:
            remaining = [max(0, self.timeout_ms - ms) for ms in self._timeout_ms_by_station_id.values() if ms > 0]
            min_overflow_remaining_ms = min(remaining) if remaining else self.timeout_ms
        else:
            min_overflow_remaining_ms = self.timeout_ms

        #threshold_fail = max_queue > self.fail_threshold
        #failed = overflow or threshold_fail

        return {
            "max_queue": max_queue,
            "total_waiting": total_waiting,
            "num_warning": num_warning,
            "num_critical": num_critical,
            "congested": num_critical > 0,
            "failed": overflow,
            "score": self.engine._components.status.score,
            "engine_ticks": self.engine._components.status.game_time,
            "elapsed_ms": int(self.elapsed_ms),
            "decision_step": int(self.t),
            "num_paths": self._num_paths(),
            "paths_left": max(0, self.max_paths - self._num_paths()),
            "ms_until_next_spawn": float(self.engine._passenger_spawner.ms_until_next_spawn),
            "min_overflow_remaining_ms": int(min_overflow_remaining_ms),
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

        def valid_path_idx(idx: int) -> bool:
            return 0 <= idx < n_paths

        def unique_keep_order(seq):
            out = []
            seen = set()
            for x in seq:
                key = id(x)
                if key not in seen:
                    out.append(x)
                    seen.add(key)
            return out

        def station_priority(st):
            # high demand + low degree first
            return (-float(st.occupation), self._station_degree(st))



        pm = self.engine.path_manager
        stations = self._sorted_stations()
        paths = self._sorted_paths()
        n_stations = min(self.max_stations, len(stations))
        n_paths = min(self.max_paths, len(paths))

        # ---------------------------------------
        # Action 0: Do Nothing
        if action_id == 0:
            self._last_action = "none"
            return True

        # ---------------------------------------
        # Action 1: create new connection between 2 most congested stations
        # ---------------------------------------
        if action_id == 1:
            if len(paths) >= pm.max_num_paths:
                return False
            if n_stations < 2:
                return False
            if i < 0 or j < 0 or i >= n_stations or j >= n_stations or i == j:
                return False

            # candidate first station
            s1_candidates = []
            if 0 <= i < n_stations:
                s1_candidates.append(stations[i])

            remaining_s1 = sorted(stations[:n_stations], key=station_priority)
            s1_candidates = unique_keep_order(s1_candidates + remaining_s1)

            # try ordered station pairs
            for s1 in s1_candidates:
                s2_candidates = []
                if 0 <= j < n_stations and stations[j] is not s1:
                    s2_candidates.append(stations[j])

                # prefer different, also high-demand / low-degree
                remaining_s2 = [st for st in sorted(stations[:n_stations], key=station_priority) if st is not s1]
                s2_candidates = unique_keep_order(s2_candidates + remaining_s2)

                for s2 in s2_candidates:
                    before = self._num_paths()

                    try:
                        w = pm.start_path_on_station(s1)
                        if w is None:
                            continue # reach resources limit

                        creating = pm._creating_or_expanding_path
                        if creating is None:
                            continue

                        creating.add_station_to_path(s2)
                        creating.try_to_end_path_on_station(s2)

                        after = self._num_paths()

                    except Exception as e:
                        return False
                    finally:
                        # close
                        pm._creating_or_expanding_path = None

                    if after > before: # true if new line has created
                        new_path = max(self.engine._components.paths, key=lambda p: getattr(p, "path_order", 0))
                        self._path_birth_ms[self._path_key(new_path)] = self.elapsed_ms
                        self._edit_cooldown_left_ms = self.edit_cooldown_ms
                        self._last_action = "create"
                        return True

            # no legal creation found then fallback to action 0
            # self._last_action = "none"
            return False

        # ---------------------------------------
        # Action 2: extend busiest path toward most congested station
        # ---------------------------------------
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

                except Exception:
                    pass

                finally:
                    pm._creating_or_expanding_path = None

                if len(chosen_path.stations) > before_len:
                    self._edit_cooldown_left_ms = self.edit_cooldown_ms
                    self._last_action = "expand"
                    return True

            return False

        # ---------------------------------------
        if action_id == 3:
            if n_paths == 0 or n_stations < 2:
                return False

            if self._edit_cooldown_left_ms > 0:
                return False

            if self._remove_cooldown_left_ms > 0:
                return False

            # choose a path to remove:
            # use i if valid, otherwise remove lowest-load path
            if 0 <= i < n_paths:
                path_remove = paths[i]
            else:
                path_remove = min(paths, key=path_load)

            path_age = self.elapsed_ms - self._path_birth_ms.get(self._path_key(path_remove), self.elapsed_ms)

            if path_age < self.min_path_age_ms:
                return False

            if self._path_is_busy(path_remove):
                return False

            before = self._num_paths()

            # Backup path so we can restore on failure
            backup_stations = list(getattr(path_remove, "stations", []))
            if len(backup_stations) < 2:
                return False  # can't safely remove/restore a degenerate path

            def _build_path_from_station_list(st_list):
                w = pm.start_path_on_station(st_list[0])
                if w is None:
                    return False
                creating = pm._creating_or_expanding_path
                if creating is None:
                    return False

                # Add intermediate stations
                for st in st_list[1:]:
                    creating.add_station_to_path(st)

                # Ensure it ends
                creating.try_to_end_path_on_station(st_list[-1])
                return True

            # choose stations to connect (use i,j if valid, else top-2 congested)
            if 0 <= i < n_stations and 0 <= j < n_stations and i != j:
                s1, s2 = stations[i], stations[j]
            else:
                sorted_st = sorted(stations[:n_stations], key=lambda s: s.occupation, reverse=True)
                s1, s2 = sorted_st[0], sorted_st[1]

            removed = False
            created_new = False
            restored_old = False

            try:
                # 1) Remove old path (free a slot), BUT we have a backup.
                pm.remove_path(path_remove)
                removed = True

                # 2) Try to create the new path
                w = pm.start_path_on_station(s1)
                if w is None:
                    return False

                creating = pm._creating_or_expanding_path
                if creating is None:
                    return False

                creating.add_station_to_path(s2)
                creating.try_to_end_path_on_station(s2)
                created_new = True

            except Exception:
                return False

            finally:
                # Always clear any half-open creation state
                pm._creating_or_expanding_path = None

                # 3) If we removed the old path but failed to create the new one, restore.
                if removed and not created_new:
                    try:
                        # Make sure creating state is clean before restore
                        pm._creating_or_expanding_path = None
                        restored_old = _build_path_from_station_list(backup_stations)
                    except Exception:
                        restored_old = False
                    finally:
                        pm._creating_or_expanding_path = None

            after = self._num_paths()

            # Success if we ended up with at least as many paths as before.
            # - created_new True => usually after == before
            # - created_new False + restored_old True => after == before
            if created_new and after >= before:
                self._remove_cooldown_left_ms = self.remove_cooldown_ms
                self._edit_cooldown_left_ms = self.edit_cooldown_ms
                self._last_action = "remove"
                return True

            # If we got here, we permanently lost a path
            return False

        # ---------------------------------------
        # Action 4: x
        # i/j are soft hints only; if they are bad, we auto-search a legal expansion.
        # ---------------------------------------
        if action_id == 4:
            if n_paths == 0 or n_stations == 0:
                return False

            # candidate paths
            path_candidates = []
            if valid_path_idx(i):
                path_candidates.append(paths[i])

            # then try busiest / most pressured paths
            remaining_paths = sorted(paths[:n_paths], key=path_load, reverse=True)
            path_candidates = unique_keep_order(path_candidates + remaining_paths)

            # candidate targets
            target_candidates = []
            if 0 <= j < n_stations:
                target_candidates.append(stations[j])

            # then try high-demand, low-degree stations
            remaining_targets = sorted(stations[:n_stations], key=station_priority)
            target_candidates = unique_keep_order(target_candidates + remaining_targets)

            # try all reasonable (path, anchor, target) combinations
            for chosen_path in path_candidates:
                anchors = []
                if chosen_path.stations:
                    anchors.append(chosen_path.stations[-1])
                    anchors.append(chosen_path.stations[0])

                anchors = unique_keep_order([a for a in anchors if a is not None])

                for target in target_candidates:
                    if target in chosen_path.stations:
                        continue

                    for anchor in anchors:
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

                        except Exception:
                            pass

                        finally:
                            pm._creating_or_expanding_path = None

                        if len(chosen_path.stations) > before_len:
                            self._last_action = "expand"
                            return True

            # no legal guided expansion found -> fallback to action 0
            #self._last_action = "none"
            #return True
            return False