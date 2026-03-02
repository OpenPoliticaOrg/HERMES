#!/usr/bin/env python3
"""
Simulator for packet/message passing on directed networks.

Supports:
- Receding-horizon min-cost LP routing baseline (`min_cost_lp`)
- Queue-differential backpressure routing (`backpressure`)
- Context-conditioned traffic/capacity/loss profiles
- Metrics JSON + optional matplotlib dashboard
"""

import argparse
import heapq
import json
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linprog


@dataclass
class Edge:
    idx: int
    edge_id: str
    src: int
    dst: int
    capacity: int
    delay: int
    loss: float


@dataclass
class Commodity:
    idx: int
    commodity_id: str
    src: int
    dst: int
    rate: float
    kind: str
    ttl_steps: int
    copies: int


def _resolve(root, value):
    p = Path(value)
    if p.is_absolute():
        return p
    return root / p


def _clip01(x):
    return min(max(float(x), 0.0), 1.0)


def _weighted_lookup(mapping, key, default):
    if not isinstance(mapping, dict):
        return default
    if key in mapping:
        return mapping[key]
    if "*" in mapping:
        return mapping["*"]
    return default


def _resolve_context(schedule, step, default_context):
    if not schedule:
        return default_context
    chosen = default_context
    for start_step, context_name in schedule:
        if step >= start_step:
            chosen = context_name
    return chosen


def _parse_context_schedule(raw_schedule, default_context):
    if not isinstance(raw_schedule, dict):
        return [(0, default_context)]
    out = []
    for k, v in raw_schedule.items():
        out.append((int(k), str(v)))
    out.sort(key=lambda x: x[0])
    if not out:
        out = [(0, default_context)]
    return out


def _dijkstra(num_nodes, adjacency, src):
    dist = [float("inf")] * num_nodes
    dist[src] = 0.0
    heap = [(0.0, src)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, w in adjacency[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist


class NetworkSpec:
    def __init__(self, config):
        self.nodes = list(config.get("nodes", []))
        if len(self.nodes) == 0:
            raise ValueError("Config must include non-empty `nodes`.")
        self.node_to_idx = {n: i for i, n in enumerate(self.nodes)}
        self.num_nodes = len(self.nodes)

        edge_payload = config.get("edges", [])
        if len(edge_payload) == 0:
            raise ValueError("Config must include non-empty `edges`.")

        self.edges = []
        self.out_edges = [[] for _ in range(self.num_nodes)]
        reverse_adj = [[] for _ in range(self.num_nodes)]

        for i, e in enumerate(edge_payload):
            src_name = e["src"]
            dst_name = e["dst"]
            if src_name not in self.node_to_idx or dst_name not in self.node_to_idx:
                raise ValueError(f"Unknown edge endpoint in edge {e}")
            src = self.node_to_idx[src_name]
            dst = self.node_to_idx[dst_name]
            edge_id = str(e.get("id", f"{src_name}->{dst_name}"))
            cap = int(max(0, e.get("capacity", 0)))
            delay = int(max(1, e.get("delay", 1)))
            loss = _clip01(e.get("loss", 0.0))
            edge = Edge(
                idx=i,
                edge_id=edge_id,
                src=src,
                dst=dst,
                capacity=cap,
                delay=delay,
                loss=loss,
            )
            self.edges.append(edge)
            self.out_edges[src].append(i)
            reverse_adj[dst].append((src, float(delay)))

        commodity_payload = config.get("traffic", [])
        if len(commodity_payload) == 0:
            raise ValueError("Config must include non-empty `traffic`.")
        self.commodities = []
        for k, c in enumerate(commodity_payload):
            src_name = c["src"]
            dst_name = c["dst"]
            if src_name not in self.node_to_idx or dst_name not in self.node_to_idx:
                raise ValueError(f"Unknown traffic endpoint in commodity {c}")
            comm = Commodity(
                idx=k,
                commodity_id=str(c.get("id", f"{src_name}->{dst_name}")),
                src=self.node_to_idx[src_name],
                dst=self.node_to_idx[dst_name],
                rate=float(max(0.0, c.get("rate", 0.0))),
                kind=str(c.get("kind", "generic")),
                ttl_steps=(
                    int(c.get("ttl_steps"))
                    if c.get("ttl_steps", None) is not None
                    else -1
                ),
                copies=int(max(1, c.get("copies", 1))),
            )
            self.commodities.append(comm)
        self.num_commodities = len(self.commodities)

        # Precompute shortest path distances to each commodity destination.
        # dist_to_dst[k][node] = shortest delay from node to dst_k
        self.dist_to_dst = []
        for comm in self.commodities:
            # Dijkstra on reversed graph from destination.
            dist = _dijkstra(self.num_nodes, reverse_adj, comm.dst)
            self.dist_to_dst.append(dist)

        self.contexts = config.get("contexts", {})
        self.default_context = str(config.get("default_context", "default"))
        self.context_schedule = _parse_context_schedule(
            config.get("context_schedule", {}), self.default_context
        )
        self.simulation_cfg = config.get("simulation", {})
        self.policies_cfg = config.get("policies", {})


class MessagePassingSimulator:
    def __init__(
        self,
        spec,
        policy_name,
        steps,
        seed,
        traffic_mode="poisson",
        lp_send_reward=1000.0,
        lp_alpha_delay=1.0,
        lp_alpha_distance=1.0,
        lp_alpha_downstream_queue=0.05,
        bp_delay_weight=0.2,
    ):
        self.spec = spec
        self.policy_name = policy_name
        self.steps = int(max(1, steps))
        seed = int(seed)
        # Separate RNG streams so arrival processes remain comparable across policies.
        self.rng_arrival = np.random.default_rng(seed)
        self.rng_loss = np.random.default_rng(seed + 1000003)
        self.traffic_mode = str(traffic_mode)

        self.lp_send_reward = float(lp_send_reward)
        self.lp_alpha_delay = float(lp_alpha_delay)
        self.lp_alpha_distance = float(lp_alpha_distance)
        self.lp_alpha_downstream_queue = float(lp_alpha_downstream_queue)
        self.bp_delay_weight = float(bp_delay_weight)

        self.queues = [
            [deque() for _ in range(self.spec.num_commodities)]
            for _ in range(self.spec.num_nodes)
        ]
        self.in_transit = defaultdict(list)  # arrival_step -> [(node, k, packet), ...]
        self.det_remainder = np.zeros(self.spec.num_commodities, dtype=np.float64)
        self.message_seq_by_comm = [0] * self.spec.num_commodities

        self.injected_total = 0
        self.delivered_total = 0
        self.dropped_total = 0
        self.latencies = []

        self.injected_by_comm = [0] * self.spec.num_commodities
        self.delivered_by_comm = [0] * self.spec.num_commodities
        self.dropped_by_comm = [0] * self.spec.num_commodities
        self.latencies_by_comm = [[] for _ in range(self.spec.num_commodities)]
        self.injected_unique_by_comm = [0] * self.spec.num_commodities
        self.delivered_unique_by_comm = [0] * self.spec.num_commodities
        self.timely_unique_deliveries_by_comm = [0] * self.spec.num_commodities
        self.duplicate_deliveries_by_comm = [0] * self.spec.num_commodities
        self.stale_unique_deliveries_by_comm = [0] * self.spec.num_commodities
        self.stale_packet_deliveries_by_comm = [0] * self.spec.num_commodities
        self.latencies_unique_by_comm = [[] for _ in range(self.spec.num_commodities)]
        self.delivered_message_ids_by_comm = [set() for _ in range(self.spec.num_commodities)]

        self.series = {
            "context": [],
            "injected": [],
            "injected_unique": [],
            "delivered": [],
            "dropped": [],
            "backlog": [],
            "in_transit": [],
        }

    def _context_cfg(self, context_name):
        return self.spec.contexts.get(context_name, {})

    def _effective_edge_capacity(self, edge, context_cfg):
        scale = float(_weighted_lookup(context_cfg.get("edge_capacity_scale", {}), edge.edge_id, 1.0))
        return max(0, int(round(edge.capacity * scale)))

    def _effective_edge_loss(self, edge, context_cfg):
        add = float(_weighted_lookup(context_cfg.get("edge_loss_add", {}), edge.edge_id, 0.0))
        scale = float(_weighted_lookup(context_cfg.get("edge_loss_scale", {}), edge.edge_id, 1.0))
        return _clip01(edge.loss * scale + add)

    def _effective_rate(self, comm, context_cfg):
        scale = float(_weighted_lookup(context_cfg.get("traffic_scale", {}), comm.commodity_id, 1.0))
        return max(0.0, comm.rate * scale)

    def _queue_lengths(self):
        q = np.zeros((self.spec.num_nodes, self.spec.num_commodities), dtype=np.int64)
        for n in range(self.spec.num_nodes):
            for k in range(self.spec.num_commodities):
                q[n, k] = len(self.queues[n][k])
        return q

    def _record_delivery(self, step, comm_idx, packet):
        birth_step, msg_id = packet
        comm = self.spec.commodities[comm_idx]

        self.delivered_total += 1
        self.delivered_by_comm[comm_idx] += 1
        latency = max(0, int(step - birth_step))
        self.latencies.append(latency)
        self.latencies_by_comm[comm_idx].append(latency)

        ttl = comm.ttl_steps if comm.ttl_steps >= 0 else None
        if ttl is not None and latency > ttl:
            self.stale_packet_deliveries_by_comm[comm_idx] += 1

        delivered_ids = self.delivered_message_ids_by_comm[comm_idx]
        if msg_id in delivered_ids:
            self.duplicate_deliveries_by_comm[comm_idx] += 1
            return

        delivered_ids.add(msg_id)
        self.delivered_unique_by_comm[comm_idx] += 1
        self.latencies_unique_by_comm[comm_idx].append(latency)
        if ttl is not None:
            if latency <= ttl:
                self.timely_unique_deliveries_by_comm[comm_idx] += 1
            else:
                self.stale_unique_deliveries_by_comm[comm_idx] += 1
        else:
            self.timely_unique_deliveries_by_comm[comm_idx] += 1

    def _deliver_in_transit(self, step):
        delivered_now = 0
        arrivals = self.in_transit.pop(step, [])
        for node, k, packet in arrivals:
            comm = self.spec.commodities[k]
            if node == comm.dst:
                delivered_now += 1
                self._record_delivery(step, k, packet)
            else:
                self.queues[node][k].append(packet)
        return delivered_now

    def _inject_traffic(self, step, context_cfg):
        injected_now = 0
        injected_unique_now = 0
        for comm in self.spec.commodities:
            lam = self._effective_rate(comm, context_cfg)
            if self.traffic_mode == "deterministic":
                self.det_remainder[comm.idx] += lam
                n = int(np.floor(self.det_remainder[comm.idx]))
                self.det_remainder[comm.idx] -= n
            else:
                n = int(self.rng_arrival.poisson(lam))
            if n <= 0:
                continue
            q = self.queues[comm.src][comm.idx]
            for _ in range(n):
                msg_idx = self.message_seq_by_comm[comm.idx]
                self.message_seq_by_comm[comm.idx] += 1
                msg_id = f"{comm.commodity_id}:{msg_idx}"
                for _copy in range(comm.copies):
                    q.append((step, msg_id))
                    injected_now += 1
                    self.injected_total += 1
                    self.injected_by_comm[comm.idx] += 1
                injected_unique_now += 1
                self.injected_unique_by_comm[comm.idx] += 1
        return injected_now, injected_unique_now

    def _min_cost_lp_flows(self, qlen, edge_caps):
        var_keys = []
        c = []
        for edge in self.spec.edges:
            u, v = edge.src, edge.dst
            for comm in self.spec.commodities:
                if qlen[u, comm.idx] <= 0:
                    continue
                dist_v = self.spec.dist_to_dst[comm.idx][v]
                if not np.isfinite(dist_v):
                    continue
                downstream_q = int(qlen[v, :].sum())
                hop_cost = (
                    self.lp_alpha_delay * float(edge.delay)
                    + self.lp_alpha_distance * float(dist_v)
                    + self.lp_alpha_downstream_queue * float(downstream_q)
                )
                # Big transmit reward drives the solver to ship as many packets as feasible.
                obj = hop_cost - self.lp_send_reward
                # Bonus for directly reaching destination.
                if v == comm.dst:
                    obj -= 0.5 * self.lp_send_reward
                var_keys.append((edge.idx, comm.idx))
                c.append(obj)

        if len(var_keys) == 0:
            return {}

        n_vars = len(var_keys)
        edge_rows = []
        edge_b = []
        for edge in self.spec.edges:
            cap = edge_caps[edge.idx]
            if cap <= 0:
                continue
            row = np.zeros(n_vars, dtype=np.float64)
            used = False
            for j, (e_idx, _) in enumerate(var_keys):
                if e_idx == edge.idx:
                    row[j] = 1.0
                    used = True
            if used:
                edge_rows.append(row)
                edge_b.append(float(cap))

        queue_rows = []
        queue_b = []
        for n in range(self.spec.num_nodes):
            for comm in self.spec.commodities:
                avail = int(qlen[n, comm.idx])
                if avail <= 0:
                    continue
                row = np.zeros(n_vars, dtype=np.float64)
                used = False
                for j, (e_idx, k_idx) in enumerate(var_keys):
                    e = self.spec.edges[e_idx]
                    if e.src == n and k_idx == comm.idx:
                        row[j] = 1.0
                        used = True
                if used:
                    queue_rows.append(row)
                    queue_b.append(float(avail))

        A_ub = None
        b_ub = None
        if edge_rows or queue_rows:
            A_ub = np.vstack(edge_rows + queue_rows)
            b_ub = np.array(edge_b + queue_b, dtype=np.float64)

        bounds = [(0.0, None)] * n_vars
        res = linprog(
            c=np.array(c, dtype=np.float64),
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=bounds,
            method="highs",
        )
        if not res.success or res.x is None:
            return self._shortest_path_greedy_flows(qlen, edge_caps)

        x = np.maximum(res.x, 0.0)
        x_floor = np.floor(x).astype(np.int64)

        # Feasible integer projection using largest fractional parts.
        out = {var_keys[j]: int(x_floor[j]) for j in range(n_vars) if x_floor[j] > 0}
        residual_edge = {e.idx: int(edge_caps[e.idx]) for e in self.spec.edges}
        residual_q = qlen.copy().astype(np.int64)
        for (e_idx, k_idx), amount in out.items():
            residual_edge[e_idx] -= amount
            src = self.spec.edges[e_idx].src
            residual_q[src, k_idx] -= amount

        frac_order = np.argsort(-(x - x_floor))
        for j in frac_order:
            frac = x[j] - x_floor[j]
            if frac <= 1e-9:
                break
            e_idx, k_idx = var_keys[int(j)]
            src = self.spec.edges[e_idx].src
            if residual_edge[e_idx] <= 0 or residual_q[src, k_idx] <= 0:
                continue
            out[(e_idx, k_idx)] = out.get((e_idx, k_idx), 0) + 1
            residual_edge[e_idx] -= 1
            residual_q[src, k_idx] -= 1

        return {k: v for k, v in out.items() if v > 0}

    def _shortest_path_greedy_flows(self, qlen, edge_caps):
        # Fallback if LP fails: route by shortest next hop with edge capacities.
        out = {}
        residual_q = qlen.copy().astype(np.int64)
        residual_edge = dict(edge_caps)
        for edge in self.spec.edges:
            if residual_edge[edge.idx] <= 0:
                continue
            u, v = edge.src, edge.dst
            candidates = []
            for comm in self.spec.commodities:
                if residual_q[u, comm.idx] <= 0:
                    continue
                d = self.spec.dist_to_dst[comm.idx][v]
                if not np.isfinite(d):
                    continue
                candidates.append((float(d), comm.idx))
            candidates.sort(key=lambda x: x[0])
            rem = residual_edge[edge.idx]
            for _, k_idx in candidates:
                if rem <= 0:
                    break
                send = int(min(rem, residual_q[u, k_idx]))
                if send <= 0:
                    continue
                out[(edge.idx, k_idx)] = out.get((edge.idx, k_idx), 0) + send
                residual_q[u, k_idx] -= send
                rem -= send
            residual_edge[edge.idx] = rem
        return out

    def _backpressure_flows(self, qlen, edge_caps):
        out = {}
        residual_q = qlen.copy().astype(np.int64)
        for edge in self.spec.edges:
            rem = int(edge_caps[edge.idx])
            if rem <= 0:
                continue
            u, v = edge.src, edge.dst
            scores = []
            for comm in self.spec.commodities:
                if residual_q[u, comm.idx] <= 0:
                    continue
                if not np.isfinite(self.spec.dist_to_dst[comm.idx][v]):
                    continue
                q_v = 0 if v == comm.dst else int(qlen[v, comm.idx])
                q_u = int(residual_q[u, comm.idx])
                dist_u = self.spec.dist_to_dst[comm.idx][u]
                dist_v = self.spec.dist_to_dst[comm.idx][v]
                progress = float(dist_u - dist_v)
                pressure = float(q_u - q_v) + 0.05 * progress - self.bp_delay_weight * float(edge.delay)
                if pressure > 0:
                    scores.append((pressure, comm.idx))
            scores.sort(key=lambda x: x[0], reverse=True)
            for _, k_idx in scores:
                if rem <= 0:
                    break
                send = int(min(rem, residual_q[u, k_idx]))
                if send <= 0:
                    continue
                out[(edge.idx, k_idx)] = out.get((edge.idx, k_idx), 0) + send
                residual_q[u, k_idx] -= send
                rem -= send
        return out

    def _send_flows(self, step, flow_map, edge_losses):
        delivered_now = 0
        dropped_now = 0
        for (e_idx, k_idx), amount in flow_map.items():
            if amount <= 0:
                continue
            edge = self.spec.edges[e_idx]
            q = self.queues[edge.src][k_idx]
            send = int(min(amount, len(q)))
            if send <= 0:
                continue
            loss_prob = edge_losses[edge.idx]
            for _ in range(send):
                packet = q.popleft()
                if self.rng_loss.random() < loss_prob:
                    dropped_now += 1
                    self.dropped_total += 1
                    self.dropped_by_comm[k_idx] += 1
                    continue
                arrival_step = int(step + edge.delay)
                comm = self.spec.commodities[k_idx]
                if arrival_step <= step and edge.dst == comm.dst:
                    delivered_now += 1
                    self._record_delivery(step, k_idx, packet)
                elif arrival_step <= step:
                    self.queues[edge.dst][k_idx].append(packet)
                else:
                    self.in_transit[arrival_step].append((edge.dst, k_idx, packet))
        return delivered_now, dropped_now

    def _collect_step_stats(self, context_name, injected, injected_unique, delivered, dropped):
        backlog = 0
        for n in range(self.spec.num_nodes):
            for k in range(self.spec.num_commodities):
                backlog += len(self.queues[n][k])
        in_transit_count = sum(len(v) for v in self.in_transit.values())
        self.series["context"].append(context_name)
        self.series["injected"].append(int(injected))
        self.series["injected_unique"].append(int(injected_unique))
        self.series["delivered"].append(int(delivered))
        self.series["dropped"].append(int(dropped))
        self.series["backlog"].append(int(backlog))
        self.series["in_transit"].append(int(in_transit_count))

    def run(self):
        for step in range(self.steps):
            context_name = _resolve_context(
                self.spec.context_schedule, step, self.spec.default_context
            )
            context_cfg = self._context_cfg(context_name)

            delivered_from_transit = self._deliver_in_transit(step)
            injected_now, injected_unique_now = self._inject_traffic(step, context_cfg)

            qlen = self._queue_lengths()
            edge_caps = {}
            edge_losses = {}
            for edge in self.spec.edges:
                edge_caps[edge.idx] = self._effective_edge_capacity(edge, context_cfg)
                edge_losses[edge.idx] = self._effective_edge_loss(edge, context_cfg)

            if self.policy_name == "min_cost_lp":
                flow_map = self._min_cost_lp_flows(qlen, edge_caps)
            elif self.policy_name == "backpressure":
                flow_map = self._backpressure_flows(qlen, edge_caps)
            else:
                raise ValueError(f"Unknown policy: {self.policy_name}")

            delivered_from_send, dropped_now = self._send_flows(step, flow_map, edge_losses)
            delivered_now = delivered_from_transit + delivered_from_send

            self._collect_step_stats(
                context_name=context_name,
                injected=injected_now,
                injected_unique=injected_unique_now,
                delivered=delivered_now,
                dropped=dropped_now,
            )

        # Drain in-transit packets for final accounting.
        final_step = self.steps
        while self.in_transit:
            delivered = self._deliver_in_transit(final_step)
            self._collect_step_stats(
                context_name="drain",
                injected=0,
                injected_unique=0,
                delivered=delivered,
                dropped=0,
            )
            final_step += 1

        return self.summary()

    def summary(self):
        injected = int(self.injected_total)
        delivered = int(self.delivered_total)
        dropped = int(self.dropped_total)
        injected_unique = int(sum(self.injected_unique_by_comm))
        delivered_unique = int(sum(self.delivered_unique_by_comm))
        timely_unique = int(sum(self.timely_unique_deliveries_by_comm))
        duplicate_deliveries = int(sum(self.duplicate_deliveries_by_comm))
        stale_unique_deliveries = int(sum(self.stale_unique_deliveries_by_comm))
        stale_packet_deliveries = int(sum(self.stale_packet_deliveries_by_comm))
        avg_latency = float(np.mean(self.latencies)) if self.latencies else None
        p95_latency = (
            float(np.percentile(self.latencies, 95)) if len(self.latencies) > 0 else None
        )
        per_comm = {}
        for comm in self.spec.commodities:
            lat = self.latencies_by_comm[comm.idx]
            lat_unique = self.latencies_unique_by_comm[comm.idx]
            injected_unique_c = int(self.injected_unique_by_comm[comm.idx])
            delivered_unique_c = int(self.delivered_unique_by_comm[comm.idx])
            timely_unique_c = int(self.timely_unique_deliveries_by_comm[comm.idx])
            duplicate_c = int(self.duplicate_deliveries_by_comm[comm.idx])
            stale_unique_c = int(self.stale_unique_deliveries_by_comm[comm.idx])
            stale_packet_c = int(self.stale_packet_deliveries_by_comm[comm.idx])
            per_comm[comm.commodity_id] = {
                "kind": comm.kind,
                "ttl_steps": (None if comm.ttl_steps < 0 else int(comm.ttl_steps)),
                "copies": int(comm.copies),
                "injected": int(self.injected_by_comm[comm.idx]),
                "delivered": int(self.delivered_by_comm[comm.idx]),
                "dropped": int(self.dropped_by_comm[comm.idx]),
                "delivery_ratio": (
                    float(self.delivered_by_comm[comm.idx]) / float(self.injected_by_comm[comm.idx])
                    if self.injected_by_comm[comm.idx] > 0
                    else None
                ),
                "injected_unique": injected_unique_c,
                "delivered_unique": delivered_unique_c,
                "unique_delivery_ratio": (
                    float(delivered_unique_c) / float(injected_unique_c)
                    if injected_unique_c > 0
                    else None
                ),
                "timely_unique_deliveries": timely_unique_c,
                "timely_unique_ratio": (
                    float(timely_unique_c) / float(injected_unique_c)
                    if injected_unique_c > 0
                    else None
                ),
                "duplicate_deliveries": duplicate_c,
                "duplicate_per_unique": (
                    float(duplicate_c) / float(max(delivered_unique_c, 1))
                    if delivered_unique_c > 0
                    else None
                ),
                "stale_unique_deliveries": stale_unique_c,
                "stale_unique_ratio": (
                    float(stale_unique_c) / float(max(delivered_unique_c, 1))
                    if delivered_unique_c > 0
                    else None
                ),
                "stale_packet_deliveries": stale_packet_c,
                "avg_latency": float(np.mean(lat)) if lat else None,
                "p95_latency": float(np.percentile(lat, 95)) if lat else None,
                "avg_unique_latency": float(np.mean(lat_unique)) if lat_unique else None,
                "p95_unique_latency": (
                    float(np.percentile(lat_unique, 95)) if lat_unique else None
                ),
            }

        handoff_ids = [
            c.idx for c in self.spec.commodities if str(c.kind).lower() == "handoff"
        ]
        alert_ids = [
            c.idx for c in self.spec.commodities if str(c.kind).lower() == "alert"
        ]
        handoff_injected_unique = int(sum(self.injected_unique_by_comm[i] for i in handoff_ids))
        handoff_delivered_unique = int(sum(self.delivered_unique_by_comm[i] for i in handoff_ids))
        handoff_timely_unique = int(
            sum(self.timely_unique_deliveries_by_comm[i] for i in handoff_ids)
        )
        alert_duplicate_deliveries = int(
            sum(self.duplicate_deliveries_by_comm[i] for i in alert_ids)
        )
        alert_delivered_unique = int(sum(self.delivered_unique_by_comm[i] for i in alert_ids))

        return {
            "policy": self.policy_name,
            "steps": int(self.steps),
            "traffic_mode": self.traffic_mode,
            "totals": {
                "injected": injected,
                "delivered": delivered,
                "dropped": dropped,
                "delivery_ratio": (float(delivered) / float(injected)) if injected > 0 else None,
                "drop_ratio": (float(dropped) / float(injected)) if injected > 0 else None,
                "injected_unique": injected_unique,
                "delivered_unique": delivered_unique,
                "unique_delivery_ratio": (
                    float(delivered_unique) / float(injected_unique)
                    if injected_unique > 0
                    else None
                ),
                "avg_latency": avg_latency,
                "p95_latency": p95_latency,
                "duplicate_deliveries": duplicate_deliveries,
                "duplicate_delivery_ratio": (
                    float(duplicate_deliveries) / float(max(delivered_unique, 1))
                    if delivered_unique > 0
                    else None
                ),
                "stale_unique_deliveries": stale_unique_deliveries,
                "stale_unique_ratio": (
                    float(stale_unique_deliveries) / float(max(delivered_unique, 1))
                    if delivered_unique > 0
                    else None
                ),
                "stale_packet_deliveries": stale_packet_deliveries,
                "timely_unique_delivery_ratio": (
                    float(timely_unique) / float(injected_unique)
                    if injected_unique > 0
                    else None
                ),
            },
            "coordination_kpis": {
                "handoff_success_rate": (
                    float(handoff_delivered_unique) / float(handoff_injected_unique)
                    if handoff_injected_unique > 0
                    else None
                ),
                "handoff_timely_success_rate": (
                    float(handoff_timely_unique) / float(handoff_injected_unique)
                    if handoff_injected_unique > 0
                    else None
                ),
                "alert_duplicate_rate": (
                    float(alert_duplicate_deliveries) / float(max(alert_delivered_unique, 1))
                    if alert_delivered_unique > 0
                    else None
                ),
            },
            "per_commodity": per_comm,
            "timeseries": self.series,
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Network message passing simulator")
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Path to repository root.",
    )
    parser.add_argument(
        "--config",
        default="data/network_message_passing/example_network.json",
        help="Simulation config JSON path (relative to repo root unless absolute).",
    )
    parser.add_argument(
        "--policy",
        choices=["min_cost_lp", "backpressure", "both"],
        default="both",
        help="Policy to run.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=-1,
        help="Override simulation steps. -1 uses config value.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="Override random seed. -1 uses config value.",
    )
    parser.add_argument(
        "--traffic-mode",
        choices=["poisson", "deterministic"],
        default=None,
        help="Arrival process mode; defaults to config value.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional output JSON path.",
    )
    parser.add_argument(
        "--plot-path",
        default=None,
        help="Optional output plot path (PNG).",
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Show matplotlib plot window.",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print full JSON payload to stdout.",
    )
    parser.add_argument(
        "--lp-send-reward",
        type=float,
        default=None,
        help="Override LP send reward.",
    )
    parser.add_argument(
        "--lp-alpha-delay",
        type=float,
        default=None,
        help="Override LP delay weight.",
    )
    parser.add_argument(
        "--lp-alpha-distance",
        type=float,
        default=None,
        help="Override LP distance-to-destination weight.",
    )
    parser.add_argument(
        "--lp-alpha-downstream-queue",
        type=float,
        default=None,
        help="Override LP downstream queue weight.",
    )
    parser.add_argument(
        "--bp-delay-weight",
        type=float,
        default=None,
        help="Override backpressure delay penalty weight.",
    )
    return parser.parse_args()


def _policy_params(spec, args):
    min_cost_cfg = spec.policies_cfg.get("min_cost_lp", {})
    backpressure_cfg = spec.policies_cfg.get("backpressure", {})

    return {
        "lp_send_reward": (
            args.lp_send_reward
            if args.lp_send_reward is not None
            else float(min_cost_cfg.get("send_reward", 1000.0))
        ),
        "lp_alpha_delay": (
            args.lp_alpha_delay
            if args.lp_alpha_delay is not None
            else float(min_cost_cfg.get("alpha_delay", 1.0))
        ),
        "lp_alpha_distance": (
            args.lp_alpha_distance
            if args.lp_alpha_distance is not None
            else float(min_cost_cfg.get("alpha_distance", 1.0))
        ),
        "lp_alpha_downstream_queue": (
            args.lp_alpha_downstream_queue
            if args.lp_alpha_downstream_queue is not None
            else float(min_cost_cfg.get("alpha_downstream_queue", 0.05))
        ),
        "bp_delay_weight": (
            args.bp_delay_weight
            if args.bp_delay_weight is not None
            else float(backpressure_cfg.get("delay_weight", 0.2))
        ),
    }


def _build_run(spec, policy, steps, seed, traffic_mode, params):
    sim = MessagePassingSimulator(
        spec=spec,
        policy_name=policy,
        steps=steps,
        seed=seed,
        traffic_mode=traffic_mode,
        lp_send_reward=params["lp_send_reward"],
        lp_alpha_delay=params["lp_alpha_delay"],
        lp_alpha_distance=params["lp_alpha_distance"],
        lp_alpha_downstream_queue=params["lp_alpha_downstream_queue"],
        bp_delay_weight=params["bp_delay_weight"],
    )
    return sim.run()


def _print_summary(result):
    totals = result["totals"]
    coord = result.get("coordination_kpis", {})
    print(f"Policy: {result['policy']}")
    print(f"  Injected:      {totals['injected']}")
    print(f"  Delivered:     {totals['delivered']}")
    print(f"  Dropped:       {totals['dropped']}")
    print(f"  Delivery ratio:{totals['delivery_ratio']}")
    print(f"  Avg latency:   {totals['avg_latency']}")
    print(f"  P95 latency:   {totals['p95_latency']}")
    print(f"  Unique ratio:  {totals.get('unique_delivery_ratio')}")
    print(f"  Stale ratio:   {totals.get('stale_unique_ratio')}")
    print(f"  Duplicates:    {totals.get('duplicate_deliveries')}")
    print(f"  Handoff succ:  {coord.get('handoff_success_rate')}")


def _plot_results(results_map, out_path=None, show=False):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    ax_throughput = axes[0][0]
    ax_backlog = axes[0][1]
    ax_cum_delivered = axes[1][0]
    ax_delivery_ratio = axes[1][1]

    for policy, result in results_map.items():
        ts = result["timeseries"]
        delivered = np.array(ts["delivered"], dtype=np.float64)
        injected = np.array(ts["injected"], dtype=np.float64)
        backlog = np.array(ts["backlog"], dtype=np.float64)

        cum_delivered = np.cumsum(delivered)
        cum_injected = np.cumsum(injected)
        ratio = np.divide(
            cum_delivered,
            np.maximum(cum_injected, 1.0),
        )

        ax_throughput.plot(delivered, label=policy)
        ax_backlog.plot(backlog, label=policy)
        ax_cum_delivered.plot(cum_delivered, label=policy)
        ax_delivery_ratio.plot(ratio, label=policy)

    ax_throughput.set_title("Delivered Packets per Step")
    ax_backlog.set_title("Backlog (Total Queue Length)")
    ax_cum_delivered.set_title("Cumulative Delivered")
    ax_delivery_ratio.set_title("Cumulative Delivery Ratio")
    for ax in [ax_throughput, ax_backlog, ax_cum_delivered, ax_delivery_ratio]:
        ax.set_xlabel("Step")
        ax.legend(loc="best")
        ax.grid(alpha=0.25)

    fig.tight_layout()

    if out_path:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(p), dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def run():
    args = parse_args()
    root = Path(args.repo_root).resolve()
    config_path = _resolve(root, args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r") as fp:
        config = json.load(fp)
    spec = NetworkSpec(config)

    sim_cfg = spec.simulation_cfg
    steps = int(args.steps if args.steps > 0 else sim_cfg.get("steps", 150))
    seed = int(args.seed if args.seed >= 0 else sim_cfg.get("seed", 7))
    traffic_mode = args.traffic_mode or str(sim_cfg.get("traffic_mode", "poisson"))
    params = _policy_params(spec, args)

    policies = [args.policy] if args.policy != "both" else ["min_cost_lp", "backpressure"]
    results = {}
    for policy in policies:
        results[policy] = _build_run(
            spec=spec,
            policy=policy,
            steps=steps,
            seed=seed,
            traffic_mode=traffic_mode,
            params=params,
        )

    payload = {
        "config_path": str(config_path),
        "steps": steps,
        "seed": seed,
        "traffic_mode": traffic_mode,
        "params": params,
        "results": results,
    }

    for policy in policies:
        _print_summary(results[policy])

    if args.output_json:
        out = _resolve(root, args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as fp:
            json.dump(payload, fp, indent=2, sort_keys=True)
        print(f"Saved JSON results to: {out}")

    if args.plot_path or args.show_plot:
        _plot_results(
            results_map=results,
            out_path=_resolve(root, args.plot_path) if args.plot_path else None,
            show=args.show_plot,
        )
        if args.plot_path:
            print(f"Saved plot to: {_resolve(root, args.plot_path)}")

    if args.print_json:
        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    run()
