# ship_scheduler/env.py
import heapq
import pandas as pd
import ast
from typing import Any, Dict, Tuple
from datetime import datetime, timedelta
from .config import DISTANCES_CSV, STORMS_CSV, SHIPS_CSV, SHIPMENTS_CSV
from .entities import Ship, Port

class ShipSchedulingEnv:
    def __init__(self, alpha=0.1, beta=1.0, gamma=0.5, years: float = 4.0):
        self.ALPHA = alpha
        self.BETA  = beta
        self.GAMMA = gamma
        self.years = years
        self.distances = pd.read_csv(DISTANCES_CSV)
        self.distances['storms'] = self.distances['storms'].apply(
            lambda v: ast.literal_eval(v) if isinstance(v, str) else v
        )
        self.storms = pd.read_csv(STORMS_CSV, parse_dates=['start','end'])
        self.shipments_df = pd.read_csv(SHIPMENTS_CSV)

        self.start_datetime = datetime(2022, 1, 1)
        self.end_datetime   = self.start_datetime + timedelta(days=365*years)
        self.max_hours = (self.end_datetime - self.start_datetime).total_seconds() / 3600.0

        self.time = 0.0
        self.event_queue = []
        self.ports = {}
        self.ships = {}
        self.log = []
        self.last_freed_berth = None
        self.reset()

    def reset(self) -> Dict[str, Any]:
        self.time = 0.0
        self.log.clear()

        df = pd.read_csv(SHIPS_CSV)
        df['route'] = df['route'].apply(ast.literal_eval)
        self.ships.clear()


        for _, row in df.iterrows():
            ship = Ship(**row)
            ship.current_port = ship.route[0]
            scenario = self.shipments_df.iloc[ship.scenarioID]
            ship.contUnload = int(scenario[f"Ship_{ship.id}_unload"])
            ship.contLoad   = int(scenario[f"Ship_{ship.id}_load"])
            # timing lists
            ship.cur_time   = []
            ship.best_time  = []
            ship.all_time   = []
            ship._arrival_time = None
            ship._service_start = None
            ship.segment_times = []
            ship.prev_elapsed  = 0.0
            self.ships[ship.id] = ship

        self.time_in_queue   = {sid: 0.0 for sid in self.ships}
        self.time_in_service = {sid: 0.0 for sid in self.ships}
        self.time_in_storm   = {sid: 0.0 for sid in self.ships}

        ports = ["Antwerp","Suez","Singapore","Shanghai","Panama"]
        self.ports = {name: Port(name) for name in ports}

        self.event_queue.clear()
        for sid in self.ships:
            heapq.heappush(self.event_queue, (0.0, 'arrival', sid))

        first = next(iter(self.ships))

        return self._get_state('arrival', first)

    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict]:
        timestamp, ev, sid = heapq.heappop(self.event_queue)
        self.time = timestamp
        if self.time >= self.max_hours:
            return self._get_state(ev, sid), 0.0, True, {}

        ship = self.ships[sid]
        reward = 0.0

        if ev == 'departure':
            ship.berth_id = None
            # mark start of a new loop when at first port
            if ship.current_leg == 0:
                ship.cur_time.clear()
                ship.segment_times.clear()
                ship.prev_elapsed = 0.0
                ship.loop_start_time = self.time
                ship.cur_time.append(0.0)

            old_port = ship.current_port
            n = len(ship.route)
            if ship.looped_route:
                nxt = (ship.current_leg + 1) % n
                ndir = 1
            else:
                tmp = ship.current_leg + ship.direction
                if tmp >= n:
                    ndir = -1
                    nxt  = ship.current_leg + ndir
                elif tmp < 0:
                    ndir = 1
                    nxt  = ship.current_leg + ndir
                else:
                    ndir = ship.direction
                    nxt  = tmp
            target_port = ship.route[nxt]

            travel, storm_time = self._compute_travel_time_between(old_port, target_port)
            ship.advance_leg()

            # final‐loop reward when back at start
            if ship.current_leg == 0:
                elapsed    = self.time - ship.loop_start_time
                ship.cur_time.append(elapsed)
                total_time = sum(ship.cur_time)

                if ship.best_time:
                    reward = reward + (sum(ship.best_time) - total_time)
                if not ship.best_time or total_time < sum(ship.best_time):
                    ship.best_time = ship.cur_time.copy()

                ship.all_time.append(total_time)
                ship.cur_time.clear()

            self.log.append((self.time, sid, 'DEPARTED', old_port, 0))

            # storm penalty right at departure
            sim_dt       = self.start_datetime + timedelta(hours=self.time)
            mask         = (
                (self.distances['route_from']==old_port)
                & (self.distances['route_to']==ship.current_port)
            )
            storms_lists = self.distances.loc[mask, 'storms']
            storm_flag   = 0
            for lst in storms_lists:
                for z in lst or []:
                    dfz = self.storms[self.storms['zone']==z]
                    if ((dfz['start']<=sim_dt)&(dfz['end']>=sim_dt)).any():
                        storm_flag = 1
                        break
                if storm_flag:
                    
                    break
            reward = reward - self.BETA * storm_flag
            if storm_flag:
                self.log.append((self.time, sid, 'STORM_ENCOUNTER', ship.current_port, 0))
                self.time_in_storm[sid] += storm_time

            delay_h = float(action.get('delay', 0.0))
            ship.ready_to_depart = False
            arrive_t = self.time + delay_h + travel
            heapq.heappush(self.event_queue, (arrive_t, 'arrival', sid))
            self.log.append((self.time, sid, 'ENROUTE', ship.current_port, 0))

        elif ev == 'arrival':
            port = self.ports[ship.current_port]
            ship._arrival_time = self.time

            # per‐leg speed reward
            if hasattr(ship, 'loop_start_time'):
                elapsed = self.time - ship.loop_start_time
                ship.cur_time.append(elapsed)

                seg_time = elapsed - ship.prev_elapsed
                ship.segment_times.append(seg_time)

                if len(ship.segment_times) > 1:
                    prev_avg    = sum(ship.segment_times[:-1]) / (len(ship.segment_times)-1)
                    improvement = prev_avg - seg_time
                    reward      = reward + self.GAMMA * improvement

                ship.prev_elapsed = elapsed

            queue_len = len(port.queue)
            berth_id  = port.arrive(sid)
            ship.berth_id = berth_id

            if berth_id is None:
                # queue penalty 
                reward = reward - self.ALPHA * queue_len
                self.log.append((self.time, sid, 'QUEUED', ship.current_port, 0))
            else:
                queue_dur = self.time - ship._arrival_time
                self.time_in_queue[sid] += queue_dur
                ship._service_start = self.time
                self.log.append((self.time, sid, 'BERTHED', ship.current_port, berth_id))
                service_h = self._compute_service_time(ship, port, berth_id)
                self.log.append((self.time, sid, 'SERVICE_STARTED', ship.current_port, berth_id))
                heapq.heappush(self.event_queue, (self.time + service_h, 'service_complete', sid))

        elif ev == 'service_complete':
            service_dur = self.time - ship._service_start
            self.time_in_service[sid] += service_dur
            port = self.ports[ship.current_port]
            self.log.append((self.time, sid, 'SERVICE_COMPLETE', ship.current_port, ship.berth_id))
            berth_choice = action.get('berth_choice', None)
            next_sid, freed_id = port.release(choice=berth_choice)
            self.last_freed_berth = freed_id
            if next_sid is not None:
                self.ships[next_sid].berth_id = freed_id
                queue_dur = self.time - ship._arrival_time
                self.time_in_queue[sid] += queue_dur
                self.ships[next_sid]._service_start = self.time
                self.log.append((self.time, next_sid, 'BERTHED', self.ships[next_sid].current_port , freed_id))
                service_h = self._compute_service_time(ship, port, freed_id)
                self.log.append((self.time, next_sid, 'SERVICE_STARTED', self.ships[next_sid].current_port , freed_id))
                service_h2 = self._compute_service_time(self.ships[next_sid], port, freed_id)
                heapq.heappush(self.event_queue, (self.time + service_h2, 'service_complete', next_sid))

            ship.scenarioID += 1
            if ship.scenarioID < len(self.shipments_df):
                row = self.shipments_df.iloc[ship.scenarioID]
                ship.contUnload = int(row[f"Ship_{sid}_unload"])
                ship.contLoad   = int(row[f"Ship_{sid}_load"])

            ship.ready_to_depart = True
            heapq.heappush(self.event_queue, (self.time, 'departure', sid))

        state = self._get_state(ev, sid)
        done = self.time >= self.max_hours
        return state, reward, done, {}

    def _get_state(self, event_type: str, ship_id: int) -> Dict[str, Any]:
        if event_type == 'departure':
            return self._get_departure_state(ship_id)
        return self._get_berth_state(ship_id)

    def _get_departure_state(self, ship_id: int) -> Dict[str, Any]:
        ship = self.ships[ship_id]

        future_ports = ship.peek_next_ports(3)
        future_queues = [ len(self.ports[p].queue) for p in future_ports ]

        sim_dt = self.start_datetime + timedelta(hours=self.time)
        future_storms = []
        for p in future_ports:
            mask = (
                (self.distances['route_from'] == ship.current_port) &
                (self.distances['route_to']   == p)
            )
            storms_lists = self.distances.loc[mask, 'storms']
            zones = {z for lst in storms_lists for z in (lst or [])}
            active = False
            for z in zones:
                dfz = self.storms[self.storms['zone'] == z]
                if ((dfz['start'] <= sim_dt) & (dfz['end'] >= sim_dt)).any():
                    active = True
                    break
            future_storms.append(active)

        elapsed = sum(ship.cur_time)
        best    = sum(ship.best_time)
        gap     = best - elapsed
        ratio   = elapsed / best if best > 0 else 0.0

        return {
            'decision_type':  'departure',
            'time':           self.time,
            'ship_id':        ship_id,
            'future_queues':  future_queues,
            'future_storms':  future_storms,
            'best_time':      best,
            'elapsed':        elapsed,
            'gap_to_best':    gap,
            'progress_ratio': ratio,
        }

    def _get_berth_state(self, ship_id: int) -> Dict[str, Any]:
        ship = self.ships[ship_id]
        port_name = ship.current_port
        port = self.ports[port_name]
        freed   = self.last_freed_berth  
        waiting = []
        sim_dt = self.start_datetime + timedelta(hours=self.time)

        for sid in port.queue:
            s = self.ships[sid]

            if s.best_time and s.cur_time:
                idx = len(s.cur_time) - 1
                if idx < len(s.best_time):
                    delay = s.cur_time[idx] - s.best_time[idx]
                else:
                    delay = 0.0
            else:
                delay = 0.0

            # storm ahead
            next_port = s.route[s.current_leg]
            mask = ((self.distances['route_from'] == s.current_port)
                    & (self.distances['route_to'] == next_port))
            storms_lists = self.distances.loc[mask, 'storms']
            zones = {z for lst in storms_lists for z in (lst or [])}
            storm_ahead = any(
                ((self.storms.query("zone==@z")['start'] <= sim_dt)
                & (self.storms.query("zone==@z")['end']   >= sim_dt)).any()
                for z in zones
            )

            service_h = self._compute_service_time(s, port, freed)

            waiting.append({
                'ship_id':                sid,
                'delay':                  delay,
                'storm_ahead':            storm_ahead,
                'service_time':  service_h
            })

        dep = self._get_departure_state(ship_id)

        return {
            'decision_type':  'berth',
            'time':           self.time,
            'ship_id':        ship_id,
            'future_queues':  dep['future_queues'],
            'future_storms':  dep['future_storms'],
            'waiting_ships':  waiting,
            }


    def _compute_service_time(self, ship: Ship, port: Port, berth_id: int) -> float:
        unload, load = ship.contUnload, ship.contLoad
        small, large = sorted((unload, load))
        ops = [3] * small + [2] * (large - small)  # minutes per container

        # number of cranes at this berth
        num_cranes = 5 if port.berth_size(berth_id) == 'big' else 3

        # distribute operations round-robin across cranes
        service_plan = {cr: [] for cr in range(1, num_cranes + 1)}
        for idx, dur in enumerate(ops):
            cr = (idx % num_cranes) + 1
            service_plan[cr].append(dur)

        # slowest crane workload
        total_minutes = max(sum(times) for times in service_plan.values())
        return total_minutes / 60.0  # convert to hours

    def _compute_travel_time_between(self, frm: str, to: str) -> float:
        mask = (
            (self.distances['route_from'] == frm) &
            (self.distances['route_to']   == to)
        )
        subset = self.distances.loc[mask]
        if subset.empty:
            raise ValueError(f"No distance entry for leg {frm!r} → {to!r}")
        row = subset.iloc[0]

        total_distance = row['total_distance']
        storm_ratio    = row['storm_ratio']
        sim_dt         = self.start_datetime + timedelta(hours=self.time)

        zones = row['storms']
        storm_active = False
        for z in zones:
            dfz = self.storms[self.storms['zone'] == z]
            if ((dfz['start'] <= sim_dt) & (dfz['end'] >= sim_dt)).any():
                storm_active = True
                break

        normal_kmh = 20 * 1.852
        storm_kmh  =  5 * 1.852

        if storm_active:
            storm_dist = total_distance * storm_ratio
            clear_dist = total_distance * (1 - storm_ratio)
            storm_time = storm_dist / storm_kmh
            clear_time = clear_dist / normal_kmh
            return storm_time + clear_time, storm_time
        else:
            total_time = total_distance / normal_kmh
            return total_time, 0.0

    def get_log_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.log, columns=['time','ship_id','state','port', 'berth_id'])
