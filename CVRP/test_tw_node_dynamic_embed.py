import pytest

torch = pytest.importorskip("torch", reason="PyTorch is required for dynamic TW embedding tests.")

from CVRPEnv import CVRPEnv
from CVRPModel import CVRPModel


def _build_model_params():
    return {
        'ensemble': False,
        'distance_penalty': False,
        'positional': False,
        'xi': -1,
        'local_size': [3],
        'ensemble_size': 1,
        'demand': True,
        'euclidean': True,
        'embedding_dim': 32,
        'encoder_layer_num': 1,
        'head_num': 4,
        'qkv_dim': 8,
        'logit_clipping': 10,
        'ff_hidden_dim': 64,
        'local_att_hidden_dim': 16,
        'local_att_head_num': 2,
        'local_att_qkv_dim': 4,
        'use_tw_node_dynamic_embed': True,
    }


def test_tw_node_dynamic_embedding_demo(capsys):
    batch = {
        'loc': torch.tensor([[[0.0, 0.5], [0.2, 1.0], [0.8, 0.2]]], dtype=torch.float),
        'demand': torch.tensor([[0.1, 0.2, 0.1]], dtype=torch.float),
        'depot': torch.tensor([[0.0, 0.0]], dtype=torch.float),
        'ready_time': torch.tensor([[0.0, 0.0, 0.5, 0.2]], dtype=torch.float),
        'due_time': torch.tensor([[2.0, 1.0, 1.5, 1.8]], dtype=torch.float),
        'service_time': torch.tensor([[0.0, 0.05, 0.05, 0.1]], dtype=torch.float),
    }

    env = CVRPEnv(multi_width=1, device='cpu')
    env.load_random_problems(batch, aug_factor=1)
    reset_state, _, _ = env.reset()

    model = CVRPModel(**_build_model_params())
    model.pre_forward(reset_state)

    # take an initial depot move so that current_node is defined
    env.pre_step()
    env.step(torch.zeros((1, 1), dtype=torch.long))
    state, _, _ = env.pre_step()
    cur_dist, cur_theta, xy, norm_demand = env.get_cur_feature()

    x_dyn = model.compute_tw_node_dynamic_features(state, cur_dist)
    assert x_dyn is not None
    assert x_dyn.shape[-1] == model.dynamic_feature_dim

    encoded_last_node = model.encoded_nodes.gather(
        dim=1,
        index=state.current_node[:, :, None].expand(1, 1, model.encoded_nodes.size(-1)),
    )
    probs = model.decoder(
        encoded_last_node,
        state.load,
        cur_dist,
        cur_theta,
        xy,
        norm_demand=norm_demand,
        ninf_mask=state.ninf_mask,
        x_dyn=x_dyn,
    )
    assert probs.shape[-1] == env.problem_size + 1

    # Demo printouts for quick inspection during development
    print("X_dyn step features (batch=0, pomo=0):", x_dyn[0, 0])
    print("probs mean/std:", probs.mean().item(), probs.std().item())

    captured = capsys.readouterr()
    assert "X_dyn step features" in captured.out
