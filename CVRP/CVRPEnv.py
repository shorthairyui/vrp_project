from dataclasses import dataclass
import torch
import numpy as np
import torch.nn.functional as F

from utils import augment_xy_data_by_8_fold


@dataclass
class Reset_State:
    depot_xy: torch.Tensor = None
    # shape: (batch, 1, 2)
    node_xy: torch.Tensor = None
    # shape: (batch, problem, 2)
    node_demand: torch.Tensor = None
    # shape: (batch, problem)
    ready_time: torch.Tensor = None
    # shape: (batch, problem)
    due_time: torch.Tensor = None
    # shape: (batch, problem)
    service_time: torch.Tensor = None
    # shape: (batch, problem)
    dist: torch.Tensor = None
    # shape: (batch, problem+1, problem+1)


@dataclass
class Step_State:
    selected_count: int = None
    load: torch.Tensor = None
    # shape: (batch, multi)
    current_node: torch.Tensor = None
    # shape: (batch, multi)
    ninf_mask: torch.Tensor = None
    # shape: (batch, multi, problem+1)
    finished: torch.Tensor = None
    # shape: (batch, multi)
    current_time: torch.Tensor = None
    # shape: (batch, multi)
    last_arrival_time: torch.Tensor = None
    # shape: (batch, multi)
    last_waiting_time: torch.Tensor = None
    # shape: (batch, multi)
    last_tardiness: torch.Tensor = None
    # shape: (batch, multi)


class CVRPEnv:
    def __init__(self, multi_width, device, ready_time=None, due_time=None, service_time=None,
                 tardiness_coeff=1.0, enforce_hard_time_windows=False, use_lookahead_mask=False):

        # Const @INIT
        ####################################
        self.device = device
        self.vrplib = False
        self.problem_size = None
        self.multi_width = multi_width

        self.depot_xy = None
        self.unscaled_depot_xy = None
        self.node_xy = None
        self.node_demand = None
        self.input_mask = None
        self.default_ready_time = ready_time
        self.default_due_time = due_time
        self.default_service_time = service_time
        self.depot_node_ready_time = None
        self.depot_node_due_time = None
        self.depot_node_service_time = None
        self.tardiness_coeff = tardiness_coeff
        self.enforce_hard_time_windows = enforce_hard_time_windows
        self.use_lookahead_mask = use_lookahead_mask

        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.depot_node_xy = None
        # shape: (batch, problem+1, 2)
        self.depot_node_demand = None
        # shape: (batch, problem+1)

        # Dynamic-1
        ####################################
        self.selected_count = None
        self.current_node = None
        # shape: (batch, multi)
        self.selected_node_list = None
        # shape: (batch, multi, 0~)

        # Dynamic-2
        ####################################
        self.at_the_depot = None
        # shape: (batch, multi)
        self.load = None
        # shape: (batch, multi)
        self.visited_ninf_flag = None
        # shape: (batch, multi, problem+1)
        self.ninf_mask = None
        # shape: (batch, multi, problem+1)
        self.finished = None
        # shape: (batch, multi)
        self.current_time = None
        # shape: (batch, multi)
        self.time_window_penalty = None
        # shape: (batch, multi)
        self.last_arrival_time = None
        self.last_waiting_time = None
        self.last_tardiness = None

        # states to return
        ####################################
        self.reset_state = Reset_State()
        self.step_state = Step_State()

    def load_vrplib_problem(self, instance, aug_factor=1):
        self.vrplib = True
        self.batch_size = 1
        node_coord = torch.FloatTensor(instance['node_coord']).unsqueeze(0).to(self.device)
        demand = torch.FloatTensor(instance['demand']).unsqueeze(0).to(self.device)
        demand = demand / instance['capacity']
        ready_time = instance.get('ready_time', None)
        if ready_time is None:
            default_ready = 0.0 if self.default_ready_time is None else self.default_ready_time
            ready_time = torch.as_tensor(default_ready, dtype=demand.dtype, device=self.device).expand_as(demand).clone()
        else:
            ready_time = torch.FloatTensor(ready_time).unsqueeze(0).to(self.device)

        due_time = instance.get('due_time', None)
        if due_time is None:
            default_due = float('inf') if self.default_due_time is None else self.default_due_time
            due_time = torch.as_tensor(default_due, dtype=demand.dtype, device=self.device).expand_as(demand).clone()
        else:
            due_time = torch.FloatTensor(due_time).unsqueeze(0).to(self.device)

        service_time = instance.get('service_time', None)
        if service_time is None:
            default_service = 0.0 if self.default_service_time is None else self.default_service_time
            service_time = torch.as_tensor(default_service, dtype=demand.dtype, device=self.device).expand_as(demand).clone()
        else:
            service_time = torch.FloatTensor(service_time).unsqueeze(0).to(self.device)
        self.unscaled_depot_node_xy = node_coord
        # shape: (batch, problem+1, 2)
        
        min_x = torch.min(node_coord[:, :, 0], 1)[0]
        min_y = torch.min(node_coord[:, :, 1], 1)[0]
        max_x = torch.max(node_coord[:, :, 0], 1)[0]
        max_y = torch.max(node_coord[:, :, 1], 1)[0]
        scaled_depot_node_x = (node_coord[:, :, 0] - min_x) / (max_x - min_x)
        scaled_depot_node_y = (node_coord[:, :, 1] - min_y) / (max_y - min_y)
        
        # self.depot_node_xy = self.unscaled_depot_node_xy / 1000
        self.depot_node_xy = torch.cat((scaled_depot_node_x[:, :, None]
                                        , scaled_depot_node_y[:, :, None]), dim=2)
        depot = self.depot_node_xy[:, instance['depot'], :]
        # shape: (batch, problem+1)
        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                depot = augment_xy_data_by_8_fold(depot)
                self.depot_node_xy = augment_xy_data_by_8_fold(self.depot_node_xy)
                self.unscaled_depot_node_xy = augment_xy_data_by_8_fold(self.unscaled_depot_node_xy)
                demand = demand.repeat(8, 1)
                ready_time = ready_time.repeat(8, 1)
                due_time = due_time.repeat(8, 1)
                service_time = service_time.repeat(8, 1)
            else:
                raise NotImplementedError

        self.depot_node_demand = demand
        self.depot_node_ready_time = ready_time
        self.depot_node_due_time = due_time
        self.depot_node_service_time = service_time
        self.reset_state.depot_xy = depot
        self.reset_state.node_xy = self.depot_node_xy[:, 1:, :]
        self.reset_state.node_demand = demand[:, 1:]
        self.reset_state.ready_time = ready_time[:, 1:]
        self.reset_state.due_time = due_time[:, 1:]
        self.reset_state.service_time = service_time[:, 1:]
        self.problem_size = self.reset_state.node_xy.shape[1]

        self.dist = (self.depot_node_xy[:, :, None, :] - self.depot_node_xy[:, None, :, :]).norm(p=2, dim=-1)
        # shape: (batch, problem+1, problem+1)
        self.reset_state.dist = self.dist

    def load_random_problems(self, batch, aug_factor=1):
        self.batch_size = batch['loc'].shape[0]
        node_coord = batch['loc'].to(self.device)
        demand = batch['demand'].to(self.device)
        depot = batch['depot'].to(self.device)
        ready_time = batch.get('ready_time', None)
        if ready_time is None:
            default_ready = torch.as_tensor(0.0 if self.default_ready_time is None else self.default_ready_time,
                                            device=self.device, dtype=demand.dtype)
            ready_time = default_ready.expand(self.batch_size, demand.shape[1] + 1).clone()
        else:
            ready_time = ready_time.to(self.device)
        due_time = batch.get('due_time', None)
        if due_time is None:
            default_due = torch.as_tensor(float('inf') if self.default_due_time is None else self.default_due_time,
                                          device=self.device, dtype=demand.dtype)
            due_time = default_due.expand(self.batch_size, demand.shape[1] + 1).clone()
        else:
            due_time = due_time.to(self.device)
        service_time = batch.get('service_time', None)
        if service_time is None:
            default_service = torch.as_tensor(0.0 if self.default_service_time is None else self.default_service_time,
                                              device=self.device, dtype=demand.dtype)
            service_time = default_service.expand(self.batch_size, demand.shape[1] + 1).clone()
        else:
            service_time = service_time.to(self.device)
        if len(depot.shape) == 2:
            depot = depot[:, None, :]
        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                depot = augment_xy_data_by_8_fold(depot)
                node_coord = augment_xy_data_by_8_fold(node_coord)
                demand = demand.repeat(8, 1)
                ready_time = ready_time.repeat(8, 1)
                due_time = due_time.repeat(8, 1)
                service_time = service_time.repeat(8, 1)
            else:
                raise NotImplementedError
            
        self.depot_node_xy = torch.cat((depot, node_coord), dim=1)
        self.depot_node_demand = torch.cat((torch.zeros(self.batch_size, 1).to(self.device), demand), dim=1)
        self.depot_node_ready_time = ready_time
        self.depot_node_due_time = due_time
        self.depot_node_service_time = service_time

        self.reset_state.depot_xy = depot
        self.reset_state.node_xy = self.depot_node_xy[:, 1:, :]
        self.reset_state.node_demand = demand
        self.reset_state.ready_time = ready_time[:, 1:]
        self.reset_state.due_time = due_time[:, 1:]
        self.reset_state.service_time = service_time[:, 1:]
        self.problem_size = self.reset_state.node_xy.shape[1]
        self.dist = (self.depot_node_xy[:, :, None, :] - self.depot_node_xy[:, None, :, :]).norm(p=2, dim=-1)
        # shape: (batch, problem+1, problem+1)
        self.reset_state.dist = self.dist

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, multi)
        self.selected_node_list = torch.zeros(size=(self.batch_size, self.multi_width, 0), dtype=torch.long, device=self.device)
        # shape: (batch, multi, 0~)

        self.at_the_depot = torch.ones(size=(self.batch_size, self.multi_width), dtype=torch.bool, device=self.device)
        # shape: (batch, multi)
        self.load = torch.ones(size=(self.batch_size, self.multi_width), device=self.device)
        # shape: (batch, multi)
        self.visited_ninf_flag = torch.zeros(size=(self.batch_size, self.multi_width, self.problem_size+1), device=self.device)
        # shape: (batch, multi, problem+1)
        if self.input_mask is not None:
            self.visited_ninf_flag = self.input_mask[:, None, :].expand(self.batch_size, self.multi_width, self.problem_size+1).clone()
        self.ninf_mask = torch.zeros(size=(self.batch_size, self.multi_width, self.problem_size+1), device=self.device)
        # shape: (batch, multi, problem+1)
        self.finished = torch.zeros(size=(self.batch_size, self.multi_width), dtype=torch.bool, device=self.device)
        # shape: (batch, multi)
        self.current_time = torch.zeros(size=(self.batch_size, self.multi_width), device=self.device)
        self.time_window_penalty = torch.zeros(size=(self.batch_size, self.multi_width), device=self.device)
        self.last_arrival_time = torch.zeros(size=(self.batch_size, self.multi_width), device=self.device)
        self.last_waiting_time = torch.zeros(size=(self.batch_size, self.multi_width), device=self.device)
        self.last_tardiness = torch.zeros(size=(self.batch_size, self.multi_width), device=self.device)

        reward = None
        done = False
        return self.reset_state, reward, done

    def reset_width(self, new_width):
        self.multi_width = new_width

    def pre_step(self):
        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished
        self.step_state.current_time = self.current_time
        self.step_state.last_arrival_time = self.last_arrival_time
        self.step_state.last_waiting_time = self.last_waiting_time
        self.step_state.last_tardiness = self.last_tardiness

        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected):
        # selected.shape: (batch, multi)
        # Dynamic-1
        ####################################

        prev_node = torch.zeros_like(selected) if self.current_node is None else self.current_node
        self.selected_count += 1
        self.current_node = selected
        # shape: (batch, multi)
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
        # shape: (batch, multi, 0~)

        # Dynamic-2
        ####################################
        self.at_the_depot = (selected == 0)

        demand_list = self.depot_node_demand[:, None, :].expand(self.batch_size, self.multi_width, -1)
        # shape: (batch, multi, problem+1)
        gathering_index = selected[:, :, None]
        # shape: (batch, multi, 1)
        selected_demand = demand_list.gather(dim=2, index=gathering_index).squeeze(dim=2)
        # shape: (batch, multi)
        self.load -= selected_demand
        self.load[self.at_the_depot] = 1 # refill loaded at the depot

        travel_time = self.dist[torch.arange(self.batch_size, device=self.device)[:, None], prev_node, selected]
        ready_time = self.depot_node_ready_time[torch.arange(self.batch_size, device=self.device)[:, None], selected]
        due_time = self.depot_node_due_time[torch.arange(self.batch_size, device=self.device)[:, None], selected]
        service_time = self.depot_node_service_time[torch.arange(self.batch_size, device=self.device)[:, None], selected]

        arrival_time = self.current_time + travel_time
        waiting_time = torch.clamp(ready_time - arrival_time, min=0)
        tardiness = torch.clamp(arrival_time - due_time, min=0)
        self.current_time = arrival_time + waiting_time + service_time
        # Scale late arrivals by tardiness_coeff so users can control how costly a
        # unit of lateness is relative to travel distance.
        self.time_window_penalty += self.tardiness_coeff * tardiness
        self.last_arrival_time = arrival_time
        self.last_waiting_time = waiting_time
        self.last_tardiness = tardiness

        self.visited_ninf_flag.scatter_(2, self.selected_node_list, float('-inf'))
        # shape: (batch, multi, problem+1)
        self.visited_ninf_flag[:, :, 0][~self.at_the_depot] = 0  # depot is considered unvisited, unless you are AT the depot

        self.ninf_mask = self.visited_ninf_flag.clone()
        round_error_epsilon = 1e-6
        demand_too_large = self.load[:, :, None] + round_error_epsilon < demand_list
        # shape: (batch, multi, problem+1)
        # print(self.load)
        self.ninf_mask[demand_too_large] = float('-inf')
        # shape: (batch, multi, problem+1)

        current_idx_expanded = self.current_node[:, :, None, None].expand(self.batch_size, self.multi_width, 1, self.problem_size + 1)
        dist_from_current = torch.take_along_dim(self.dist[:, None, :, :].expand(self.batch_size, self.multi_width, self.problem_size + 1, self.problem_size + 1),
                                                current_idx_expanded, dim=2).squeeze(2)
        arrival_if_travel = self.current_time[:, :, None] + dist_from_current
        time_infeasible = arrival_if_travel > self.depot_node_due_time[:, None, :]
        if self.enforce_hard_time_windows:
            self.ninf_mask[time_infeasible] = float('-inf')

        if self.use_lookahead_mask:
            base_ninf_mask = self.ninf_mask.clone()
            # finish time if we travel to each candidate j next (includes waiting + service)
            ready_time_all = self.depot_node_ready_time[:, None, :].expand(self.batch_size, self.multi_width, -1)
            service_time_all = self.depot_node_service_time[:, None, :].expand(self.batch_size, self.multi_width, -1)
            arrival_to_candidate = arrival_if_travel
            waiting_to_candidate = torch.clamp(ready_time_all - arrival_to_candidate, min=0)
            finish_candidate = arrival_to_candidate + waiting_to_candidate + service_time_all

            # travel from candidate j to any remaining node i
            dist_from_candidate = self.dist[:, None, :, :].expand(self.batch_size, self.multi_width, -1, -1)
            arrival_from_candidate = finish_candidate[:, :, :, None] + dist_from_candidate

            ready_time_exp = self.depot_node_ready_time[:, None, None, :].expand(self.batch_size, self.multi_width, self.problem_size + 1, -1)
            due_time_exp = self.depot_node_due_time[:, None, None, :].expand(self.batch_size, self.multi_width, self.problem_size + 1, -1)
            feasible_next = torch.maximum(arrival_from_candidate, ready_time_exp) <= due_time_exp

            remaining_unvisited = (self.visited_ninf_flag == 0)
            remaining_unvisited[:, :, 0] = False  # ignore depot in reachability check
            remaining_unvisited_exp = remaining_unvisited[:, :, None, :].expand_as(feasible_next)

            has_remaining = remaining_unvisited_exp.any(dim=3)
            feasible_exists = (feasible_next & remaining_unvisited_exp).any(dim=3)
            reachable = torch.where(has_remaining, feasible_exists, torch.ones_like(feasible_exists, dtype=torch.bool))

            lookahead_block = ~reachable & (~self.finished[:, :, None])
            self.ninf_mask = self.ninf_mask.masked_fill(lookahead_block, float('-inf'))

            # fallback: if lookahead masked everything for a sample, revert to base mask
            fully_blocked = torch.isinf(self.ninf_mask).all(dim=2)
            if fully_blocked.any():
                self.ninf_mask[fully_blocked] = base_ninf_mask[fully_blocked]

        newly_finished = (self.visited_ninf_flag == float('-inf')).all(dim=2)
        # shape: (batch, multi)
        self.finished = self.finished + newly_finished
        # shape: (batch, multi)

        # do not mask depot for finished episode.
        self.ninf_mask[:, :, 0][self.finished] = 0

        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished
        self.step_state.current_time = self.current_time
        self.step_state.last_arrival_time = self.last_arrival_time
        self.step_state.last_waiting_time = self.last_waiting_time
        self.step_state.last_tardiness = self.last_tardiness
        # returning values
        hard_violation = None
        if self.enforce_hard_time_windows:
            hard_violation = (tardiness > 0).any()
            if hard_violation:
                self.finished = torch.ones_like(self.finished, dtype=torch.bool, device=self.device)

        done = self.finished.all()
        if done:
            if hard_violation:
                reward = torch.full(size=(self.batch_size, self.multi_width), fill_value=-1e6, device=self.device)
            elif self.vrplib == True:
                reward = self.compute_unscaled_reward()
            else:
                reward = self._get_reward()
        else:
            reward = None

        return self.step_state, reward, done

    def _get_reward(self):
        gathering_index = self.selected_node_list[:, :, :, None].expand(-1, -1, -1, 2)
        # shape: (batch, multi, selected_list_length, 2)
        all_xy = self.depot_node_xy[:, None, :, :].expand(-1, self.multi_width, -1, -1)
        # shape: (batch, multi, problem+1, 2)

        ordered_seq = all_xy.gather(dim=2, index=gathering_index)
        # shape: (batch, multi, selected_list_length, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq-rolled_seq)**2).sum(3).sqrt()
        # shape: (batch, multi, selected_list_length)

        travel_distances = segment_lengths.sum(2)
        total_penalty = 0
        if self.time_window_penalty is not None:
            total_penalty = self.time_window_penalty
        # shape: (batch, multi)
        return -(travel_distances + total_penalty)

    def compute_unscaled_reward(self, solutions=None, rounding=True):
        if solutions is None:
            solutions = self.selected_node_list
        gathering_index = solutions[:, :, :, None].expand(-1, -1, -1, 2)
        # shape: (batch, multi, selected_list_length, 2)
        all_xy = self.unscaled_depot_node_xy[:, None, :, :].expand(-1, self.multi_width, -1, -1)
        # shape: (batch, multi, problem+1, 2)

        ordered_seq = all_xy.gather(dim=2, index=gathering_index)
        # shape: (batch, multi, selected_list_length, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)

        segment_lengths = ((ordered_seq-rolled_seq)**2).sum(3).sqrt()
        if rounding == True:
            segment_lengths = torch.round(segment_lengths)
        # shape: (batch, multi, selected_list_length)

        travel_distances = segment_lengths.sum(2)
        if solutions is self.selected_node_list:
            time_penalty = self.time_window_penalty if self.time_window_penalty is not None else 0
        else:
            time_penalty = self._compute_time_penalty(solutions)
        # shape: (batch, multi)
        return -(travel_distances + time_penalty)

    def _compute_time_penalty(self, solutions):
        batch_size, multi_width, _ = solutions.shape
        current_time = torch.zeros(size=(batch_size, multi_width), device=self.device)
        penalty = torch.zeros(size=(batch_size, multi_width), device=self.device)
        prev_node = torch.zeros(size=(batch_size, multi_width), dtype=torch.long, device=self.device)
        ready_time = self.depot_node_ready_time
        due_time = self.depot_node_due_time
        service_time = self.depot_node_service_time

        for step in range(solutions.shape[2]):
            selected = solutions[:, :, step]
            travel_time = self.dist[torch.arange(batch_size, device=self.device)[:, None], prev_node, selected]
            arrival_time = current_time + travel_time
            waiting_time = torch.clamp(ready_time[torch.arange(batch_size, device=self.device)[:, None], selected] - arrival_time, min=0)
            tardiness = torch.clamp(arrival_time - due_time[torch.arange(batch_size, device=self.device)[:, None], selected], min=0)
            penalty += self.tardiness_coeff * tardiness
            current_time = arrival_time + waiting_time + service_time[torch.arange(batch_size, device=self.device)[:, None], selected]
            prev_node = selected
        return penalty

    
    def get_cur_feature(self):
        if self.current_node is None:
            return None, None, None, None
        
        current_node = self.current_node[:, :, None, None].expand(self.batch_size, self.multi_width, 1, self.problem_size + 1)

        # Compute the relative distance
        cur_dist = torch.take_along_dim(self.dist[:, None, :, :].expand(self.batch_size, self.multi_width, self.problem_size + 1, self.problem_size + 1), 
                                        current_node, dim=2).squeeze(2)
        # shape: (batch, multi, problem)
        # print(cur_dist[0])
        expanded_xy = self.depot_node_xy[:, None, :, :].expand(self.batch_size, self.multi_width, self.problem_size + 1, 2)
        relative_xy = expanded_xy - torch.take_along_dim(expanded_xy, self.current_node[:, :, None, None].expand(
            self.batch_size, self.multi_width, 1, 2), dim=2)
        # shape: (batch, problem, 2)

        relative_x = relative_xy[:, :, :, 0]
        relative_y = relative_xy[:, :, :, 1]

        # Compute the relative coordinates
        cur_theta = torch.atan2(relative_y, relative_x)
        # shape: (batch, multi, problem)

        # Compute the normalized demand. inf generated by division will be masked. 
        demand_list = self.depot_node_demand[:, None, :].expand(self.batch_size, self.multi_width, -1)
        norm_demand = demand_list / self.load[:, :, None]

        return cur_dist, cur_theta, relative_xy, norm_demand
