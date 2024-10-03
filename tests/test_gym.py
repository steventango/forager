import gymnasium as gym

import forager.gym


def test_gym_forager_v1():
    env = gym.make("forager/Forager-v1")
    observation, info = env.reset(seed=0)
    assert observation.shape == (15, 15, 2)
    assert info == {}
    observation, reward, terminated, truncated, info = env.step(0)
    assert observation.shape == (15, 15, 2)
    assert reward == 0.0
    assert not terminated
    assert not truncated
    assert info == {}
    env.close()


def test_gym_forager_v2():
    env = gym.make("forager/Forager-v2")
    observation, info = env.reset(seed=0)
    assert observation.shape == (3, 3, 3)
    assert info == {}
    observation, reward, terminated, truncated, info = env.step(0)
    assert observation.shape == (3, 3, 3)
    assert reward == -1.0
    assert not terminated
    assert not truncated
    assert info == {}
    env.close()


def test_gym_forager_render():
    env = gym.make("forager/Forager-v1", render_mode="rgb_array")
    env.reset(seed=0)
    frame = env.render()
    assert frame.shape == (512, 512, 3)
    env.close()
