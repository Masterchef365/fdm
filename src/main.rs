use fdm::Fdm;
use idek::{
    prelude::*,
    winit::event::{ElementState, Event as WinitEvent, VirtualKeyCode, WindowEvent},
    IndexBuffer,
};
use num_complex::Complex32;
use rand::{distributions::Uniform, prelude::*};

fn main() -> Result<()> {
    launch::<(), FdmVisualizer>(Settings::default().vr_if_any_args())
}

struct FdmVisualizer {
    verts: VertexBuffer,
    amp_verts: VertexBuffer,
    indices: IndexBuffer,
    line_shader: Shader,

    fdm: Fdm,

    pause: bool,

    camera: MultiPlatformCamera,
}

fn init_fdm() -> Fdm {
    let n_cells = 50_000;
    let width = 10.;

    let dx = width / n_cells as f32;
    let t = 0.0;
    let a = Complex32::from_polar(2., 0.);
    let h = 1.;
    let m = 1.;

    let init: Vec<Complex32> = (0..n_cells)
        .map(|idx| {
            let x = (idx as f32 - n_cells as f32 / 2.) * dx;
            wave_packet(x, t, a, h, m)
        })
        .collect();

    Fdm::new(&init, dx)
}

impl App for FdmVisualizer {
    fn init(ctx: &mut Context, platform: &mut Platform, _: ()) -> Result<Self> {
        let fdm = init_fdm();

        let vertices = fdm_vertices(&fdm);
        let verts = ctx.vertices(&vertices, true)?;

        let amp_vertices = amp_vertices(&fdm);
        let amp_verts = ctx.vertices(&amp_vertices, true)?;

        let indices: Vec<u32> = (1..vertices.len() * 2 - 1)
            .map(|i| (i / 2) as u32)
            .collect();
        let indices = ctx.indices(&indices, false)?;

        Ok(Self {
            pause: true,
            amp_verts,
            fdm,
            line_shader: ctx.shader(
                DEFAULT_VERTEX_SHADER,
                DEFAULT_FRAGMENT_SHADER,
                Primitive::Lines,
            )?,
            verts,
            indices,
            camera: MultiPlatformCamera::new(platform),
        })
    }

    fn frame(&mut self, ctx: &mut Context, _: &mut Platform) -> Result<Vec<DrawCmd>> {
        if !self.pause {
            for _ in 0..3 {
                self.fdm.step(0.0000001, |x: f32| Complex32::new(x, 0.));
            }
        }
        dbg!(self.fdm.grid().iter().map(|e| e.norm_sqr()).sum::<f32>());

        let vertices = fdm_vertices(&self.fdm);
        ctx.update_vertices(self.verts, &vertices)?;

        let amp_vertices = amp_vertices(&self.fdm);
        ctx.update_vertices(self.amp_verts, &amp_vertices)?;

        Ok(vec![
            DrawCmd::new(self.amp_verts)
                .indices(self.indices)
                .shader(self.line_shader)
                .transform([
                    [1., 0., 0., 0.],
                    [0., 1., 0., 0.],
                    [0., 0., 1., 0.],
                    [0., 0., -2., 1.],
                ]),
            DrawCmd::new(self.verts)
                .indices(self.indices)
                .shader(self.line_shader),
        ])
    }

    fn event(
        &mut self,
        ctx: &mut Context,
        platform: &mut Platform,
        mut event: Event,
    ) -> Result<()> {
        if self.camera.handle_event(&mut event) {
            ctx.set_camera_prefix(self.camera.get_prefix())
        }

        if let Event::Winit(event) = &event {
            if let WinitEvent::WindowEvent { event, .. } = event {
                if let WindowEvent::KeyboardInput { input, .. } = event {
                    if let Some(key) = input.virtual_keycode {
                        if input.state == ElementState::Released {
                            match key {
                                VirtualKeyCode::Space => self.pause = !self.pause,
                                VirtualKeyCode::R => self.fdm = init_fdm(),
                                _ => (),
                            }
                        }
                    }
                }
            }
        }

        idek::close_when_asked(platform, &event);
        Ok(())
    }
}

fn fdm_vertices(fdm: &Fdm) -> Vec<Vertex> {
    let x_map = |i: usize| (i as f32 * fdm.dx()) * 2. - 1.;

    fdm.grid()
        .iter()
        .enumerate()
        .map(|(i, u)| Vertex {
            pos: [x_map(i), u.re, u.im],
            color: [1., u.re.abs(), u.im.abs()],
        })
        .collect()
}

fn amp_vertices(fdm: &Fdm) -> Vec<Vertex> {
    let x_map = |i: usize| (i as f32 * fdm.dx()) * 2. - 1.;

    fdm.grid()
        .iter()
        .enumerate()
        .map(|(i, u)| Vertex {
            pos: [x_map(i), u.norm_sqr(), 0.],
            color: [0.1, 0.4, 1.],
        })
        .collect()
}

/// Output the given wave function value for a wave packet at w(x, t) with parameters
/// a: The square of the width of the packet
/// h: Planck's constant (adjustable for viewing purposes)
/// m: The mass of the particle
/// https://en.wikipedia.org/wiki/Wave_packet
fn wave_packet(x: f32, t: f32, a: Complex32, h: f32, m: f32) -> Complex32 {
    let ihtm = Complex32::i() * h * t / m;
    (a / (a + ihtm)).powf(3. / 2.) * (-x * x / (2. * (a + ihtm))).exp()
}
