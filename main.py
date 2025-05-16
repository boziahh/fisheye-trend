import moderngl
import moderngl_window as mglw
from PIL import Image
import numpy as np
import pyglet


def load_texture(ctx, path):
    image = Image.open(path).convert("RGBA").transpose(Image.FLIP_TOP_BOTTOM)
    texture = ctx.texture(image.size, 4, image.tobytes())
    texture.build_mipmaps()
    return texture


class FisheyeApp(mglw.WindowConfig):
    window_size = (800, 800)
    title = "GPU Fisheye Outline"
    aspect_ratio = None
    resizable = False
    window_cls = 'moderngl_window.context.pyglet.Window'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.eye_texture = load_texture(self.ctx, "eye.png")
        self.outline_texture = load_texture(self.ctx, "outline.png")

        vertices = np.array([
            -1.0, -1.0, 0.0, 0.0,
             1.0, -1.0, 1.0, 0.0,
            -1.0,  1.0, 0.0, 1.0,
             1.0,  1.0, 1.0, 1.0,
        ], dtype="f4")

        self.quad = self.ctx.buffer(vertices)
        self.prog = self.ctx.program(
            vertex_shader=vertex_shader,
            fragment_shader=fragment_shader
        )
        self.vao = self.ctx.vertex_array(
            self.prog, [(self.quad, "2f 2f", "in_pos", "in_uv")]
        )

        self.virtual_mouse = np.array([400.0, 400.0], dtype=np.float32)
        self.real_mouse = np.array([400.0, 400.0], dtype=np.float32)
        self.target = self._new_target()
        self.direction = np.zeros(2, dtype=np.float32)

        self.speed = 250
        self.smoothing = 0.08
        self.paused = False
        self.mode = "random"

        self.wnd._window.push_handlers(
            on_key_press=self._on_key_press,
            on_mouse_motion=self._on_mouse_motion
        )

    def _on_key_press(self, symbol, _):
        if symbol == pyglet.window.key.SPACE:
            self.paused = not self.paused
        elif symbol == pyglet.window.key.M:
            self.mode = "mouse" if self.mode == "random" else "random"

    def _on_mouse_motion(self, x, y, *_):
        self.real_mouse[:] = x, y

    def _new_target(self):
        return np.random.uniform([0, 0], self.window_size).astype(np.float32)

    def _update_virtual_mouse(self, dt):
        if self.paused:
            return

        if self.mode == "mouse":
            self.virtual_mouse[:] = np.clip(self.real_mouse, [0, 0], self.window_size)
            return

        to_target = self.target - self.virtual_mouse
        dist_sq = np.dot(to_target, to_target)

        if dist_sq < 1e-6:
            dir_norm = np.zeros(2)
        else:
            dir_norm = to_target / np.sqrt(dist_sq)

        self.direction = (1 - self.smoothing) * self.direction + self.smoothing * dir_norm
        if np.linalg.norm(self.direction) > 0:
            self.direction /= np.linalg.norm(self.direction)

        self.virtual_mouse += self.direction * self.speed * dt
        self.virtual_mouse[:] = np.clip(self.virtual_mouse, [0, 0], self.window_size)

        if np.dot(self.target - self.virtual_mouse, self.target - self.virtual_mouse) < 400:
            self.target = self._new_target()

    def on_render(self, time, frame_time):
        self._update_virtual_mouse(frame_time)
        self.ctx.clear()

        mx, my = self.virtual_mouse
        w, h = self.wnd.buffer_width, self.wnd.buffer_height
        normalized_mouse = (mx / w, 1.0 - my / h)

        self.eye_texture.use(0)
        self.outline_texture.use(1)
        self.prog['eye_texture'] = 0
        self.prog['outline_texture'] = 1
        self.prog['mouse_pos'].value = normalized_mouse
        self.prog['strength'].value = 0.2

        self.vao.render(moderngl.TRIANGLE_STRIP)


vertex_shader = """
#version 330
in vec2 in_pos;
in vec2 in_uv;
out vec2 uv;
void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
    uv = in_uv;
}
"""

fragment_shader = """
#version 330
uniform sampler2D eye_texture;
uniform sampler2D outline_texture;
uniform vec2 mouse_pos;
uniform float strength;
in vec2 uv;
out vec4 fragColor;
void main() {
    vec4 base = texture(eye_texture, uv);
    vec2 offset = uv - mouse_pos;
    float r = length(offset);
    float r2 = r + (r * r - r) * strength;
    float scale = r > 0.0 ? r2 / r : 1.0;
    offset *= scale;
    vec2 d_uv = mouse_pos + offset;
    vec4 ov = texture(outline_texture, d_uv);
    fragColor = mix(base, ov, ov.a);
}
"""

if __name__ == "__main__":
    mglw.run_window_config(FisheyeApp)
