import numpy as np
from gymnasium import spaces
from PIL import Image

from .ptz_camera_real import PtzCameraRealEnv


class UnrestrictedPtzCameraRealEnv(PtzCameraRealEnv):
    """
    A PTZ camera env that is not restricted to move one step at a time. Each
    action is the (x, y) coordinate of the top-left corner, in unit of grid.
    """

    def __init__(
        self,
        frames_dir,
        render_mode=None,
        num_grid_x=9,
        num_grid_y=5,
        num_grid_viewport_x=5,
        num_grid_viewport_y=3,
    ):
        self.frames = sorted(frames_dir.glob("*.png"))
        self.num_grid_x = num_grid_x
        self.num_grid_y = num_grid_y
        self.num_grid_viewport_x = num_grid_viewport_x
        self.num_grid_viewport_y = num_grid_viewport_y

        # Calculate grid size in piexls. If grid size not divisible, discard
        # pixels in the lower and right edges
        (
            self.img_w,
            self.img_h,
        ) = Image.open(str(self.frames[0])).size
        self.grid_size_x = int(self.img_w / num_grid_x)
        self.grid_size_y = int(self.img_h / num_grid_y)
        self.viewport_size_x = num_grid_viewport_x * self.grid_size_x
        self.viewport_size_y = num_grid_viewport_y * self.grid_size_y

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=np.array([self.viewport_size_y, self.viewport_size_x, 3]),
            dtype=np.uint8,
        )

        self.action_space = spaces.MultiDiscrete(
            [self.num_grid_viewport_x, self.num_grid_viewport_y]
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        """
        If human-rendering is used, `self.window` will be a reference to the
        window that we draw to. `self.clock` will be a clock that is used to
        ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

        self.frame_id = 0

    def _move_viewport(self, action):
        self.viewport_grid_loc = action
