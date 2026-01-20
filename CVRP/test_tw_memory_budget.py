import pytest

torch = pytest.importorskip("torch", reason="PyTorch is required for memory budget tests.")

from models import CVRP_Decoder


def test_dynamic_attention_memory_budget():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for memory budget test.")

    device = torch.device("cuda")
    batch, pomo, nodes, emb = 4, 64, 100, 128
    head_num = 8
    qkv_dim = 16

    model_params = {
        'embedding_dim': emb,
        'head_num': head_num,
        'qkv_dim': qkv_dim,
        'logit_clipping': 10,
        'distance_penalty': False,
        'ensemble': False,
        'lazy_mask_alpha': 1.0,
        'lazy_mask_q_dim': 16,
        'lazy_mask_dyn_dim': 3,
        'lazy_mask_hidden_dim': 64,
    }

    decoder = CVRP_Decoder(**model_params).to(device)
    encoded_nodes = torch.randn(batch, nodes, emb, device=device)
    decoder.set_kv(encoded_nodes)

    encoded_last_node = torch.randn(batch, pomo, emb, device=device)
    load = torch.rand(batch, pomo, device=device)
    cur_dist = torch.rand(batch, pomo, nodes, device=device)
    cur_theta = torch.rand(batch, pomo, nodes, device=device)
    xy = torch.rand(batch, pomo, nodes, 2, device=device)
    norm_demand = torch.rand(batch, pomo, nodes, device=device)
    ninf_mask = torch.zeros(batch, pomo, nodes, device=device)
    x_dyn = torch.rand(batch, pomo, nodes, 3, device=device)

    torch.cuda.reset_peak_memory_stats(device)
    _ = decoder(
        encoded_last_node,
        load,
        cur_dist,
        cur_theta,
        xy,
        norm_demand=norm_demand,
        ninf_mask=ninf_mask,
        x_dyn=x_dyn,
    )
    peak = torch.cuda.max_memory_allocated(device)
    print(f"peak_bytes={peak}")
