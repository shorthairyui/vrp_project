import pytest

torch = pytest.importorskip("torch", reason="PyTorch is required for time window environment tests.")

from CVRPEnv import CVRPEnv


def test_time_window_updates_and_penalties():
    batch = {
        'loc': torch.tensor([[[0.0, 0.5], [0.0, 1.0]]], dtype=torch.float),
        'demand': torch.tensor([[0.2, 0.2]], dtype=torch.float),
        'depot': torch.tensor([[0.0, 0.0]], dtype=torch.float),
        'ready_time': torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float),
        'due_time': torch.tensor([[float('inf'), 0.4, 2.0]], dtype=torch.float),
        'service_time': torch.tensor([[0.0, 0.1, 0.0]], dtype=torch.float),
    }

    env = CVRPEnv(multi_width=1, device='cpu', tardiness_coeff=5.0)
    env.load_random_problems(batch, aug_factor=1)
    reset_state, reward, done = env.reset()

    assert reward is None and done is False
    assert torch.allclose(reset_state.ready_time.squeeze(0), torch.tensor([0.0, 1.0]))
    assert torch.allclose(reset_state.due_time.squeeze(0), torch.tensor([0.4, 2.0]))

    env.pre_step()

    _, reward, done = env.step(torch.tensor([[1]]))
    assert reward is None and done is False
    assert torch.isclose(env.last_arrival_time.squeeze(), torch.tensor(0.5))
    assert torch.isclose(env.last_tardiness.squeeze(), torch.tensor(0.1))

    _, reward, done = env.step(torch.tensor([[2]]))
    assert reward is None and done is False
    assert torch.isclose(env.current_time.squeeze(), torch.tensor(1.1))

    _, reward, done = env.step(torch.tensor([[0]]))
    assert done is True
    assert torch.isclose(env.time_window_penalty.squeeze(), torch.tensor(0.5))
    assert torch.is_tensor(reward)
