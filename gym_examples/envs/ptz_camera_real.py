import cv2
import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
from PIL import Image


class PtzCameraRealEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self,
                 frames_dir,
                 render_mode=None,
                 num_grid_x=9,
                 num_grid_y=5,
                 num_grid_viewport_x=5,
                 num_grid_viewport_y=3):
        self.frames = sorted(frames_dir.glob('*.png'))
        self.num_grid_x = num_grid_x
        self.num_grid_y = num_grid_y
        self.num_grid_viewport_x = num_grid_viewport_x
        self.num_grid_viewport_y = num_grid_viewport_y

        # Calculate grid size in piexls. If grid size not divisible, discard
        # pixels in the lower and right edges
        self.img_w, self.img_h, = Image.open(str(self.frames[0])).size
        self.grid_size_x = int(self.img_w / num_grid_x)
        self.grid_size_y = int(self.img_h / num_grid_y)
        self.viewport_size_x = num_grid_viewport_x * self.grid_size_x
        self.viewport_size_y = num_grid_viewport_y * self.grid_size_y

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=np.array([self.viewport_size_y, self.viewport_size_x, 3]),
            dtype=np.uint8)

        # "no-op", "right", "up", "left", "down"
        self.action_space = spaces.Discrete(5)

        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
            4: np.array([0, 0]),
        }
        self.frame_id = 0

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return self.get_view_of_viewport(self.viewport_grid_loc)

    def get_view_of_viewport(self, vp):
        """
        Can be used by the oracle to cheat by looking outside the viewport.
        """

        # Crop out the viewport
        x = vp[0] * self.grid_size_x
        y = vp[1] * self.grid_size_y
        img = self.img[y: y + self.num_grid_viewport_y * self.grid_size_y,
                       x: x + self.num_grid_viewport_x * self.grid_size_x]
        return img

    def _get_info(self):
        return {
            'vp': self.viewport_grid_loc,
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # The grid id for the upper-right corner of the viewport
        self.viewport_grid_loc = (
            int((self.num_grid_x - self.num_grid_viewport_x) / 2),
            int((self.num_grid_y - self.num_grid_viewport_y) / 2),
        )

        self._load_img()
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        self.frame_id += 1
        return observation, info

    def step(self, action):
        self._load_img()
        self._move_viewport(action)

        observation = self._get_obs()
        reward = 0  # TODO
        terminated = self.frame_id == len(self.frames) - 1
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        self.frame_id += 1
        return observation, reward, terminated, False, info

    def _load_img(self):
        img_path = self.frames[self.frame_id]
        self.img = np.array(Image.open(str(img_path)))

    def _move_viewport(self, action):
        direction = self._action_to_direction[action]
        self.viewport_grid_loc = np.array(self.viewport_grid_loc) + direction
        self.viewport_grid_loc = (
            np.clip(self.viewport_grid_loc[0], 0, self.num_grid_x - self.num_grid_viewport_x),
            np.clip(self.viewport_grid_loc[1], 0, self.num_grid_y - self.num_grid_viewport_y),
        )

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        window_size = (self.img_w, self.img_h)

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(window_size)
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.surfarray.make_surface(np.transpose(self.img, (1, 0, 2)))

        self._draw_gridlines_onto_canvas(canvas)
        self._draw_viewport_box_onto_canvas(canvas)

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                pygame.surfarray.pixels3d(canvas), axes=(1, 0, 2)
            )

    def _draw_gridlines_onto_canvas(self, canvas):
        for i in range(self.num_grid_y):
            pygame.draw.line(
                canvas,
                0,
                (0, self.grid_size_y * i),
                (self.img_w, self.grid_size_y * i),
                width=1,
            )
        for i in range(self.num_grid_x):
            pygame.draw.line(
                canvas,
                0,
                (self.grid_size_x * i, 0),
                (self.grid_size_x * i, self.img_h),
                width=1,
            )

    def _draw_viewport_box_onto_canvas(self, canvas):
        """
        Draw a bounding box corresponding to the viewport. Only to be used for
        human visualization, not used for agent observation.
        """
        x = self.viewport_grid_loc[0] * self.grid_size_x
        y = self.viewport_grid_loc[1] * self.grid_size_y
        w = self.num_grid_viewport_x * self.grid_size_x
        h = self.num_grid_viewport_y * self.grid_size_y
        lines = [
            [(x, y), (x, y + h)],
            [(x + w, y), (x + w, y + h)],
            [(x, y), (x + w, y)],
            [(x, y + h), (x + w, y + h)],
        ]

        for l in lines:
            start_pos, end_pos = l
            pygame.draw.line(
                canvas,
                (0, 0, 255),
                start_pos,
                end_pos,
                width=4,
            )
        return canvas

    def get_panoramic_wh(self):
        return (self.img_w, self.img_h)

    def get_grid_wh(self):
        return (self.grid_size_x, self.grid_size_y)

    def get_num_grids_wh(self):
        """
        Returns the whole frame size in unit of grids.
        """
        return (self.num_grid_x, self.num_grid_y)

    def get_num_grids_viewport_wh(self):
        """
        Returns the viewport size in unit of grids.
        """
        return (self.num_grid_viewport_x, self.num_grid_viewport_y)
