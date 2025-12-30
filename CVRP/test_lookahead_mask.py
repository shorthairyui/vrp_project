import torch

from CVRPEnv import CVRPEnv


def test_lookahead_masks_dead_end_candidate():
    device = torch.device('cpu')
    env = CVRPEnv(multi_width=1, device=device, use_lookahead_mask=True)

    # depot at (0,0), c1 far away, c2 close to depot.
    # Visiting c1 first makes c2 unreachable before its tight due time.
    batch = {
        'loc': torch.tensor([[
            [0.0, 0.0],   # depot
            [10.0, 0.0],  # c1
            [0.5, 0.0],   # c2
        ]], dtype=torch.float),
        'demand': torch.zeros(1, 2, dtype=torch.float),
        'depot': torch.tensor([[[0.0, 0.0]]], dtype=torch.float),
        'ready_time': torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float),
        'due_time': torch.tensor([[1e6, 100.0, 1.0]], dtype=torch.float),  # c2 has tight due time
        'service_time': torch.zeros(1, 3, dtype=torch.float),
    }

    env.load_random_problems(batch)
    env.reset()

    # First move is the depot itself (as in the rollout policy).
    env.step(torch.zeros((1, 1), dtype=torch.long, device=device))

    # After the first step, lookahead should mask the dead-end candidate c1 (index 1)
    # because visiting c1 first would make c2 infeasible.
    assert torch.isinf(env.ninf_mask[0, 0, 1])
    assert not torch.isinf(env.ninf_mask[0, 0, 2])
