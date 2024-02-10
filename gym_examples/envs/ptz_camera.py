import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces


class SquareObj():

    def __init__(self, loc_x, loc_y, size, vel_x):
        # Loc refers to the upper-right corner
        self.loc_x = loc_x
        self.loc_y = loc_y
        self.size = size
        self.vel_x = vel_x


class PtzCameraEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, num_grid_x=9, num_grid_y=5,
                 num_grid_viewport_x=5, num_grid_viewport_y=3,
                 grid_size=50, lane_width=20, obj_margin=2):
        # The size of the square grid
        self.num_grid_x = num_grid_x
        self.num_grid_y = num_grid_y
        self.size_x = grid_size * self.num_grid_x
        self.size_y = grid_size * self.num_grid_y

        # The size of the viewport
        self.num_grid_viewport_x = num_grid_viewport_x
        self.num_grid_viewport_y = num_grid_viewport_y

        # The height & width of each grid in number of pixels
        self.grid_size = grid_size

        # The size of each object in pixels
        self.lane_width = lane_width

        # The margin between an object to each side of the lane
        self.obj_margin = obj_margin

        # The grid id for the upper-right corner of the viewport
        self.viewport_grid_loc = (
            int((self.num_grid_x - self.num_grid_viewport_x) / 2),
            int((self.num_grid_y - self.num_grid_viewport_y) / 2),
        )

        self.obj_size = lane_width - obj_margin * 2

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=np.array([self.num_grid_viewport_y * self.grid_size,
                           self.num_grid_viewport_x * self.grid_size,
                           3]),
            dtype=np.uint8)
        print('observation_space: ', self.observation_space)

        self.objects = []

        # "no-op", "right", "up", "left", "down"
        self.action_space = spaces.Discrete(5)

        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
            4: np.array([0, 0]),
        }

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
        canvas = self._init_canvas_with_frame_content()
        frame = np.transpose(
            np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
        )

        # Crop out the viewport
        x = self.viewport_grid_loc[0] * self.grid_size
        y = self.viewport_grid_loc[1] * self.grid_size
        frame = frame[y: y + self.num_grid_viewport_y * self.grid_size,
                      x: x + self.num_grid_viewport_x * self.grid_size]
        print(frame.shape)
        return frame

    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Agent location refers to the grid id of the upper-right corner of the
        # viewport
        self._agent_location = (int((self.num_grid_x - self.num_grid_viewport_x) / 2),
                                int((self.num_grid_y - self.num_grid_viewport_y) / 2))

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        self._move_viewport(action)
        self._move_objects()
        self._gc_objects()
        self._spawn_objects()

        observation = self._get_obs()
        reward = 0
        terminated = False
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

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

    def _move_objects(self):
        for o in self.objects:
            o.loc_x += o.vel_x

    def _gc_objects(self):
        def is_out(o):
            if o.vel_x > 0:
                return o.loc_x >= self.size_x
            else:
                return o.loc_x + o.size <= 0
        self.objects = [o for o in self.objects
                        if not is_out(o)]

    def _spawn_objects(self):
        num_lanes = int(self.size_y / self.lane_width)

        for i in range(num_lanes):
            if np.random.random() > 0.02:
                continue

            loc_y = self.lane_width * i + self.obj_margin
            if i < int(num_lanes / 2):
                # Traveling right
                loc_x = 0 + self.obj_margin
                vel_x = self._get_initial_vel()
            else:
                # Traveling left
                loc_x = self.obj_size + self.grid_size * self.num_grid_x + self.obj_margin
                vel_x = -1 * self._get_initial_vel()
            o = SquareObj(loc_x, loc_y, self.obj_size, vel_x)
            self.objects.append(o)

    def _get_initial_vel(self):
        return np.random.normal(loc=10, scale=1)

    def _render_frame(self):
        window_size = (self.size_x, self.size_y)

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(window_size)
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = self._init_canvas_with_frame_content()
        canvas = self._draw_viewport_box_onto_canvas(canvas)

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
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def _init_canvas_with_frame_content(self):
        window_size = (self.size_x, self.size_y)

        canvas = pygame.Surface(window_size)
        canvas.fill((255, 255, 255))

        # Draw objects
        for o in self.objects:
            pygame.draw.rect(
                canvas,
                (255, 0, 0),
                pygame.Rect(o.loc_x, o.loc_y, o.size, o.size),
            )

        # Gridlines
        for i in range(self.num_grid_y):
            pygame.draw.line(
                canvas,
                0,
                (0, self.grid_size * i),
                (self.size_x, self.grid_size * i),
                width=1,
            )
        for i in range(self.num_grid_x):
            pygame.draw.line(
                canvas,
                0,
                (self.grid_size * i, 0),
                (self.grid_size * i, self.size_y),
                width=1,
            )
        return canvas

    def _draw_viewport_box_onto_canvas(self, canvas):
        """
        Draw a bounding box corresponding to the viewport. Only to be used for
        human visualization, not used for agent observation.
        """
        x = self.viewport_grid_loc[0] * self.grid_size
        y = self.viewport_grid_loc[1] * self.grid_size
        w = self.num_grid_viewport_x * self.grid_size
        h = self.num_grid_viewport_y * self.grid_size
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
                (255, 0, 0),
                start_pos,
                end_pos,
                width=2,
            )
        return canvas

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
