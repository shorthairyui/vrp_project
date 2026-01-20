import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from models import *
from models import _get_encoding


class CVRPModel(nn.Module):

    def __init__(self, **model_params):
        nn.Module.__init__(self)
        self.model_params = model_params

        self.encoder = CVRP_Encoder(**model_params)
        self.decoder = CVRP_Decoder(**model_params)
        self.encoded_nodes = None
        # shape: (batch, problem, embedding)
        self.use_tw_node_dynamic_embed = self.model_params.get('use_tw_node_dynamic_embed', False)
        self.dynamic_feature_dim = 3
        self._time_eps = 1e-6
        self.ready_time_with_depot = None
        self.due_time_with_depot = None
        self.service_time_with_depot = None

    def pre_forward(self, reset_state):
        depot_xy = reset_state.depot_xy
        # shape: (batch, 1, 2)
        node_xy = reset_state.node_xy
        # shape: (batch, problem, 2)
        node_demand = reset_state.node_demand
        # shape: (batch, problem)
        dist = reset_state.dist
        # shape: (batch, problem+1, problem+1)
        node_xy_demand = torch.cat((node_xy, node_demand[:, :, None]), dim=2)
        # shape: (batch, problem, 3)
        self.encoded_nodes = self.encoder(depot_xy, node_xy_demand, dist)
        # shape: (batch, problem+1, embedding)
        if self.use_tw_node_dynamic_embed:
            self.ready_time_with_depot = reset_state.ready_time_with_depot
            self.due_time_with_depot = reset_state.due_time_with_depot
            self.service_time_with_depot = reset_state.service_time_with_depot
        self.decoder.set_kv(self.encoded_nodes)

    def compute_tw_node_dynamic_features(self, state, cur_dist):
        """
        Build normalized per-node dynamic TW features for the current decoding step.
        """
        if cur_dist is None:
            return None

        ready_time = state.ready_time_with_depot if state.ready_time_with_depot is not None else self.ready_time_with_depot
        due_time = state.due_time_with_depot if state.due_time_with_depot is not None else self.due_time_with_depot
        service_time = state.service_time_with_depot if state.service_time_with_depot is not None else self.service_time_with_depot
        if ready_time is None or due_time is None or service_time is None:
            return None

        with torch.no_grad():
            batch_size, multi_width, _ = cur_dist.shape
            ready_time = ready_time[:, None, :].expand(batch_size, multi_width, -1)
            due_time = due_time[:, None, :].expand(batch_size, multi_width, -1)
            service_time = service_time[:, None, :].expand(batch_size, multi_width, -1)

            arrival_time = state.current_time[:, :, None] + cur_dist
            start_time = torch.maximum(arrival_time, ready_time)
            slack_arrival = due_time - arrival_time
            slack_start = due_time - start_time

            finite_due_time = torch.where(torch.isinf(due_time), due_time.new_zeros(()), due_time)
            time_scale = finite_due_time.max(dim=2)[0].clamp(min=self._time_eps)[:, :, None]

            slack_arrival = torch.where(torch.isinf(slack_arrival), time_scale, slack_arrival)
            slack_start = torch.where(torch.isinf(slack_start), time_scale, slack_start)

            features = torch.stack(
                (
                    slack_arrival / time_scale,
                    slack_start / time_scale,
                    cur_dist / time_scale,
                ),
                dim=-1,
            )
        return features.detach()

    def one_step_rollout(self, state, cur_dist, cur_theta, xy, norm_demand, eval_type):
        device = state.ninf_mask.device
        batch_size = state.ninf_mask.shape[0]
        multi_width = state.ninf_mask.shape[1]
        problem_size = state.ninf_mask.shape[2] - 1
        
        if state.selected_count == 0:  # First Move, depot
            selected = torch.zeros(size=(batch_size, multi_width), dtype=torch.long, device=device)
            prob = torch.ones(size=(batch_size, multi_width), device=device)

        elif state.selected_count == 1:  # Second Move, POMO
            selected = torch.tensor(random.sample(range(0, problem_size), multi_width), device=device)[
                           None, :] \
                    .expand(batch_size, multi_width)
            # shape: (batch, pomo+1)
            prob = torch.ones(size=(batch_size, multi_width), device=device)

        else:
            if self.use_tw_node_dynamic_embed:
                x_dyn = self.compute_tw_node_dynamic_features(state, cur_dist)
            else:
                x_dyn = None
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
            # shape: (batch, pomo+1, embedding)
            probs = self.decoder(
                encoded_last_node,
                state.load,
                cur_dist,
                cur_theta,
                xy,
                norm_demand=norm_demand,
                ninf_mask=state.ninf_mask,
                x_dyn=x_dyn,
            )
            # shape: (batch, pomo+1, problem+1)

            if eval_type == 'sample':
                # print(probs.isnan().any())
                with torch.no_grad():
                    selected = probs.reshape(batch_size * multi_width, -1).multinomial(1) \
                        .squeeze(dim=1).reshape(batch_size, multi_width)
                # shape: (batch, pomo+1)
                prob = torch.take_along_dim(probs, selected[:, :, None], dim=2).reshape(batch_size, multi_width)
                # shape: (batch, pomo+1)
                if not (prob != 0).all():   # avoid sampling prob 0
                    prob += 1e-6

            else:
                selected = probs.argmax(dim=2)
                # shape: (batch, pomo+1)
                prob = None  # value not needed. Can be anything.

        return selected, prob
    

class CVRPModel_local(nn.Module):

    def __init__(self, **model_params):
        nn.Module.__init__(self)
        self.model_params = model_params
        self.local_policy = local_policy_att(model_params, idx=0)

    def pre_forward(self, reset_state):
        pass

    def one_step_rollout(self, state, cur_dist, cur_theta, xy, norm_demand, eval_type):
        device = state.ninf_mask.device
        batch_size = state.ninf_mask.shape[0]
        multi_width = state.ninf_mask.shape[1]
        problem_size = state.ninf_mask.shape[2] - 1
        
        if state.selected_count == 0:  # First Move, depot
            selected = torch.zeros(size=(batch_size, multi_width), dtype=torch.long, device=device)
            prob = torch.ones(size=(batch_size, multi_width), device=device)

        elif state.selected_count == 1:  # Second Move, POMO
            selected = torch.tensor(random.sample(range(0, problem_size), multi_width), device=device)[
                           None, :] \
                    .expand(batch_size, multi_width)
            # shape: (batch, pomo+1)
            prob = torch.ones(size=(batch_size, multi_width), device=device)

        else:
            u_local = self.local_policy(dist=cur_dist, theta=cur_theta, xy=xy, norm_demand=norm_demand, ninf_mask=state.ninf_mask)
            # shape: (batch, pomo+1, problem+1)
            logit_clipping = self.model_params['logit_clipping']
            score_clipped = logit_clipping * torch.tanh(u_local)

            score_masked = score_clipped + state.ninf_mask

            probs = F.softmax(score_masked, dim=2)
            # shape: (batch, pomo, problem)

            if eval_type == 'sample':
                with torch.no_grad():
                    selected = probs.reshape(batch_size * multi_width, -1).multinomial(1) \
                        .squeeze(dim=1).reshape(batch_size, multi_width)
                # shape: (batch, pomo+1)
                prob = torch.take_along_dim(probs, selected[:, :, None], dim=2).reshape(batch_size, multi_width)
                # shape: (batch, pomo+1)
                if not (prob != 0).all():   # avoid sampling prob 0
                    prob += 1e-6

            else:
                selected = probs.argmax(dim=2)
                # shape: (batch, pomo+1)
                prob = None  # value not needed. Can be anything.

        return selected, prob
    
