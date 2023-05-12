### Variant of CarRacing environment that generates a full map of the track
### instead of a limited FOV view.
### AND also includes offline rendering with a particular road_poly and car state.
# Changes to render()
#   - Keep a fixed zoom
#   - Keep a fixed view of the track (doesn't scroll)
#   - Don't change angle of view as car rotates
#   - Don't render the indicators (speed, etc.)

## Rohan Banerjee
## Very similar to class in rohanb2018/world-models,
## except here we also have "state_pixels_offline" mode to do offline rendering.

from gym.envs.box2d.car_racing import CarRacing, WINDOW_W, WINDOW_H, ZOOM, SCALE, VIDEO_W, VIDEO_H, STATE_W, STATE_H, PLAYFIELD
from gym.envs.box2d.car_dynamics import Car

import pyglet
pyglet.options["debug_gl"] = False
from pyglet import gl

import math
import numpy as np


class FullMapCarRacing(CarRacing):
    def __init__(self, verbose=1):
        super().__init__(verbose)

    def render(self, mode="human", road_poly=None, car_state=None):
        """
        "state_pixels_offline" Args:
            road_poly: list of road polygons
            car_state: (x,y,v,theta): state of the car: can also be None, in which case we only render the map
        """
        assert mode in ["human", "state_pixels", "state_pixels_offline", "rgb_array"]
        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(WINDOW_W, WINDOW_H)
            self.score_label = pyglet.text.Label(
                "0000",
                font_size=36,
                x=20,
                y=WINDOW_H * 2.5 / 40.00,
                anchor_x="left",
                anchor_y="center",
                color=(255, 255, 255, 255),
            )
            self.transform = rendering.Transform()

        if "t" not in self.__dict__:
            return  # reset() not called yet

        # Change: keep a fixed zoom
        zoom = 0.2 * SCALE
        self.transform.set_scale(zoom, zoom)
        # Change: keeping a fixed view of the track.
        self.transform.set_translation(WINDOW_W / 2, WINDOW_H / 2)
        # Change: not changing angle as car rotates.
        self.transform.set_rotation(0)

        if mode=="state_pixels_offline":
            if car_state is not None:
                (x,y,v,theta) = car_state
                # Create a fake car at the desired location.
                car = Car(self.world,theta,x,y)
                car.draw(self.viewer, "state_pixels" not in mode)
                car.destroy()
        else:
            car = self.car
            car.draw(self.viewer, "state_pixels" not in mode)

        arr = None
        win = self.viewer.window
        win.switch_to()
        win.dispatch_events()

        win.clear()
        t = self.transform
        if mode == "rgb_array":
            VP_W = VIDEO_W
            VP_H = VIDEO_H
        elif mode == "state_pixels" or mode == "state_pixels_offline":
            VP_W = WINDOW_W
            VP_H = WINDOW_H
        else:
            pixel_scale = 1
            if hasattr(win.context, "_nscontext"):
                pixel_scale = (
                    win.context._nscontext.view().backingScaleFactor()
                )  # pylint: disable=protected-access
            VP_W = int(pixel_scale * WINDOW_W)
            VP_H = int(pixel_scale * WINDOW_H)

        gl.glViewport(0, 0, VP_W, VP_H)
        t.enable()

        if mode=="state_pixels_offline":
            # Not the best solution, but will work for now.
            # Set the current road_poly to the given road_poly
            old_road_poly = self.road_poly
            self.road_poly = road_poly
            self.render_road()
            self.road_poly = old_road_poly
        else:
            self.render_road()

        for geom in self.viewer.onetime_geoms:
            geom.render()
        self.viewer.onetime_geoms = []
        t.disable()
        # Change: removing the indicators because they affect the map.

        if mode == "human":
            win.flip()
            return self.viewer.isopen

        image_data = (
            pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
        )
        arr = np.fromstring(image_data.get_data(), dtype=np.uint8, sep="")
        arr = arr.reshape(VP_H, VP_W, 4)
        arr = arr[::-1, :, 0:3]

        return arr

if __name__=="__main__":
    from pyglet.window import key

    a = np.array([0.0, 0.0, 0.0])

    def key_press(k, mod):
        global restart
        if k == 0xFF0D:
            restart = True
        if k == key.LEFT:
            a[0] = -1.0
        if k == key.RIGHT:
            a[0] = +1.0
        if k == key.UP:
            a[1] = +1.0
        if k == key.DOWN:
            a[2] = +0.8  # set 1.0 for wheels to block to zero rotation

    def key_release(k, mod):
        if k == key.LEFT and a[0] == -1.0:
            a[0] = 0
        if k == key.RIGHT and a[0] == +1.0:
            a[0] = 0
        if k == key.UP:
            a[1] = 0
        if k == key.DOWN:
            a[2] = 0

    env = FullMapCarRacing()
    env.render()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release
    record_video = False
    if record_video:
        from gym.wrappers.monitor import Monitor

        env = Monitor(env, "/tmp/video-test", force=True)
    isopen = True
    while isopen:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        while True:
            s, r, done, info = env.step(a)
            total_reward += r
            if steps % 200 == 0 or done:
                print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
                print("step {} total_reward {:+0.2f}".format(steps, total_reward))
            steps += 1
            isopen = env.render()
            if done or restart or isopen == False:
                break
    env.close()