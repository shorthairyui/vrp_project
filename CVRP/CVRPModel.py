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
        self.dynamic_feature_dim = 6
        if self.use_tw_node_dynamic_embed:
            hidden_dim = self.model_params.get('tw_node_dynamic_hidden_dim', self.model_params['embedding_dim'])
            self.tw_node_dynamic_embedder = TwNodeDynamicEmbedder(
                in_dim=self.dynamic_feature_dim,
                embedding_dim=self.model_params['embedding_dim'],
                hidden_dim=hidden_dim,
            )
            self.tw_node_fusion_norm = nn.LayerNorm(self.model_params['embedding_dim'])
        self._time_eps = 1e-6
        self.ready_time_with_depot = None
        self.due_time_with_depot = None
        self.service_time_with_depot = None
        self.tw_node_dynamic_embed_chunked = self.model_params.get('tw_node_dynamic_embed_chunked', False)

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

        batch_size, multi_width, _ = cur_dist.shape
        ready_time = ready_time[:, None, :].expand(batch_size, multi_width, -1)
        due_time = due_time[:, None, :].expand(batch_size, multi_width, -1)
        service_time = service_time[:, None, :].expand(batch_size, multi_width, -1)

        arrival_time = state.current_time[:, :, None] + cur_dist
        waiting_time = torch.clamp(ready_time - arrival_time, min=0)
        start_time = torch.maximum(arrival_time, ready_time)
        finish_time = start_time + service_time
        slack = due_time - start_time

        finite_due_time = torch.where(torch.isinf(due_time), due_time.new_zeros(()), due_time)
        time_scale = finite_due_time.max(dim=2)[0].clamp(min=self._time_eps)[:, :, None]
        time_scale_expanded = time_scale

        slack = torch.where(torch.isinf(slack), time_scale_expanded, slack)

        visited = state.visited if state.visited is not None else (state.ninf_mask == float('-inf'))
        visited = visited.float()

        features = torch.stack(
            (
                arrival_time / time_scale_expanded,
                waiting_time / time_scale_expanded,
                start_time / time_scale_expanded,
                finish_time / time_scale_expanded,
                slack / time_scale_expanded,
                visited,
            ),
            dim=-1,
        )
        return features

    def _compute_tw_node_fused_embeddings(self, state, cur_dist):
        x_dyn = self.compute_tw_node_dynamic_features(state, cur_dist)
        if x_dyn is None:
            return None, None, None
        h_dyn = self.tw_node_dynamic_embedder(x_dyn)
        static_expanded = self.encoded_nodes[:, None, :, :].expand(-1, cur_dist.size(1), -1, -1)
        h_tilde = self.tw_node_fusion_norm(static_expanded + h_dyn)
        return h_tilde, x_dyn, h_dyn

    def _slice_step_state(self, state, pomo_idx):
        sliced = type(state)()
        sliced.ready_time_with_depot = state.ready_time_with_depot
        sliced.due_time_with_depot = state.due_time_with_depot
        sliced.service_time_with_depot = state.service_time_with_depot
        sliced.current_time = state.current_time[:, pomo_idx:pomo_idx + 1]
        sliced.visited = state.visited[:, pomo_idx:pomo_idx + 1] if state.visited is not None else None
        sliced.ninf_mask = state.ninf_mask[:, pomo_idx:pomo_idx + 1]
        return sliced

    def _compute_probs_with_chunked_dynamic_kv(self, state, cur_dist, cur_theta, xy, norm_demand):
        probs_per_pomo = []
        for pomo_idx in range(cur_dist.size(1)):
            state_slice = self._slice_step_state(state, pomo_idx)
            cur_dist_slice = cur_dist[:, pomo_idx:pomo_idx + 1, :]
            cur_theta_slice = cur_theta[:, pomo_idx:pomo_idx + 1, :]
            xy_slice = xy[:, pomo_idx:pomo_idx + 1, :, :]
            norm_demand_slice = norm_demand[:, pomo_idx:pomo_idx + 1, :]
            fused_nodes, _, _ = self._compute_tw_node_fused_embeddings(state_slice, cur_dist_slice)
            if fused_nodes is None:
                fused_nodes = self.encoded_nodes
            self.decoder.set_kv(fused_nodes)
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node[:, pomo_idx:pomo_idx + 1])
            probs = self.decoder(
                encoded_last_node,
                state.load[:, pomo_idx:pomo_idx + 1],
                cur_dist_slice,
                cur_theta_slice,
                xy_slice,
                norm_demand=norm_demand_slice,
                ninf_mask=state.ninf_mask[:, pomo_idx:pomo_idx + 1, :],
            )
            probs_per_pomo.append(probs)
        return torch.cat(probs_per_pomo, dim=1)

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
            if self.use_tw_node_dynamic_embed and self.tw_node_dynamic_embed_chunked:
                probs = self._compute_probs_with_chunked_dynamic_kv(state, cur_dist, cur_theta, xy, norm_demand)
            else:
                if self.use_tw_node_dynamic_embed:
                    fused_nodes, _, _ = self._compute_tw_node_fused_embeddings(state, cur_dist)
                    if fused_nodes is None:
                        fused_nodes = self.encoded_nodes
                    self.decoder.set_kv(fused_nodes)
                encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
                # shape: (batch, pomo+1, embedding)
                probs = self.decoder(encoded_last_node, state.load, cur_dist, cur_theta, xy, norm_demand=norm_demand, ninf_mask=state.ninf_mask)
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
    
