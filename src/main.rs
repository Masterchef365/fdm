use fdm::Fdm;
use idek::{prelude::*, IndexBuffer, winit::event::{Event as WinitEvent, WindowEvent, KeyboardInput, ElementState, VirtualKeyCode}};
use num_complex::Complex32;

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
    let n_cells = 1000;
    let init: Vec<Complex32> = (0..n_cells)
        .map(|x| {
            //if (x >= x_len / 3) && (x <= 2 * x_len / 3) {
            if x == n_cells / 2 {
                Complex32::new(100.0, 0.)
            } else {
                Complex32::new(0., 0.)
            }
        })
        .collect();

    Fdm::new(&init, 1.)
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
                self.fdm.step(0.000001, |x: f32| Complex32::new(x, 0.));
            }
        }

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
            .shader(self.line_shader)
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
