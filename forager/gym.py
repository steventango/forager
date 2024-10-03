from dataclasses import dataclass, field
from typing import Dict

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register

from forager.colors import Palette
from forager.config import ForagerConfig
from forager.Env import ForagerEnv
from forager.interface import Action
from forager.objects import Flower, Thorns, Wall
from forager.observations import get_color_vision


@dataclass
class ForagerGymConfig(ForagerConfig):
    object_freqs: Dict[str, float] = field(default_factory=dict)


class ForagerGymEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(self, config: ForagerGymConfig | None = None, *, config_path: str | None = None, render_mode=None):
        self.config = config
        self.config_path = config_path

        observation, _ = self.reset(config.seed, {"config": config, "config_path": config_path})

        low = int(observation.min())
        high = int(observation.max())
        self.observation_space = spaces.Box(low, high, shape=observation.shape, dtype=int)

        self.action_space = spaces.Discrete(4)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        if options is None:
            options = {}

        if config := options.get("config"):
            self.config = config
        if config_path := options.get("config_path"):
            self.config_path = config_path
        self.config.seed = seed
        self.env = ForagerEnv(self.config, config_path=self.config_path)

        for object_type in self.config.object_types:
            object_freq = self.config.object_freqs[object_type]
            self.env.generate_objects(freq=object_freq, name=object_type)

        observation = self.env.start()
        info = self._get_info()
        return observation, info

    def step(self, action: Action):
        observation, reward = self.env.step(action)
        info = self._get_info()
        return observation, reward, False, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            from PIL import Image
            frame = get_color_vision(
                self.env._state,
                self.env._size,
                self.env._ap_size,
                self.env._obj_store.idx_to_name,
                self.env._obj_store.name_to_color,
            )
            img = Image.fromarray(frame)
            img = img.resize((512, 512), Image.Resampling.NEAREST)
            import numpy as np
            frame = np.array(img)
            return frame


colors_v1 = Palette(2)
colors_v1.register("flower", (0, 255, 0))
colors_v1.register("thorns", (255, 0, 0))
config_v1 = ForagerGymConfig(
    size=15,
    object_types={
        "flower": Flower,
        "thorns": Thorns,
    },
    colors=colors_v1,
    observation_mode="objects",
    aperture=15,
    object_freqs={
        "flower": 0.05,
        "thorns": 0.1,
    },
)

register(
    id="forager/Forager-v1",
    entry_point="forager.gym:ForagerGymEnv",
    kwargs={"config": config_v1},
)


colors_v2 = Palette(3)
colors_v1.register("flower", (0, 255, 0))
colors_v1.register("thorns", (255, 0, 0))
colors_v1.register("walls", (0, 0, 0))
config_v2 = ForagerGymConfig(
    size=10,
    object_types={
        "flower": Flower,
        "thorns": Thorns,
        "wall": Wall,
    },
    colors=colors_v2,
    observation_mode="objects",
    aperture=3,
    object_freqs={
        "flower": 0.01,
        "thorns": 0.1,
        "wall": 0.2,
    },
)

register(
    id="forager/Forager-v2",
    entry_point="forager.gym:ForagerGymEnv",
    kwargs={"config": config_v2},
)
