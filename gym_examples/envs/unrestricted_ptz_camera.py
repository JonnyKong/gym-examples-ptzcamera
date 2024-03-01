from .ptz_camera import PtzCameraEnv
import numpy as np
from gymnasium import spaces


class UnrestrictedPtzCameraEnv(PtzCameraEnv):
    def __init__(
        self,
        render_mode=None,
        num_grid_x=9,
        num_grid_y=5,
        num_grid_viewport_x=5,
        num_grid_viewport_y=3,
        grid_size=50,
        lane_width=25,
        obj_margin=2,
    ):
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

        self.obj_size = lane_width - obj_margin * 2

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=np.array(
                [
                    self.num_grid_viewport_y * self.grid_size,
                    self.num_grid_viewport_x * self.grid_size,
                    3,
                ]
            ),
            dtype=np.uint8,
        )

        self.objects = []
        self.vp_2_objcnt = {}

        self.action_space = spaces.MultiDiscrete(
            [self.num_grid_viewport_x, self.num_grid_viewport_y]
        )

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

    def _move_viewport(self, action):
        self.viewport_grid_loc = (action[0], action[1])
