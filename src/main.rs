use anyhow::{Context as AnyhowContext, Result};
use fdm::{inner_size, Array2D, Fdm};
use idek::{
    prelude::*,
    winit::event::{ElementState, Event as WinitEvent, VirtualKeyCode, WindowEvent},
    IndexBuffer,
};
use num_complex::Complex32;
use png::{BitDepth, ColorType};
use std::fs::File;

type Args = String;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let image_path = args.next().context("Requires image path")?;
    launch::<Args, FdmVisualizer>(Settings::default().args(image_path))
}

struct FdmVisualizer {
    verts: VertexBuffer,
    indices: IndexBuffer,
    point_shader: Shader,

    init_bufs: [Array2D; 3],
    dx: f32,

    fdms: [Fdm; 3],

    pause: bool,

    camera: MultiPlatformCamera,
}

const SCALE: f32 = 10.;

fn scene(fdm: &[Fdm; 3]) -> Vec<Vertex> {
    let scale = 1.0;
    fdm_vertices(
        &fdm,
        |[r, g, b]| {
            (
                //(r * r + g * g + b * b).norm(),
                (r + g + b).norm(),
                [r.norm_sqr(), g.norm_sqr(), b.norm_sqr()]
                    .map(|v| v * 10.),
            )
        },
        scale,
    )
}

impl App<Args> for FdmVisualizer {
    fn init(ctx: &mut Context, platform: &mut Platform, image_path: Args) -> Result<Self> {
        let init_bufs = png_wave(&image_path)?;

        let dx = SCALE / init_bufs[0].width() as f32;
        let fdms = init_bufs.clone().map(|init_buf| Fdm::new(init_buf, dx));

        let verts = scene(&fdms);

        let indices = linear_indices(verts.len());

        let verts = ctx.vertices(&verts, true)?;

        let indices = ctx.indices(&indices, false)?;

        Ok(Self {
            dx,
            init_bufs,
            pause: true,
            verts,
            fdms,
            point_shader: ctx.shader(
                DEFAULT_VERTEX_SHADER,
                DEFAULT_FRAGMENT_SHADER,
                Primitive::Points,
            )?,
            indices,
            camera: MultiPlatformCamera::new(platform),
        })
    }

    fn frame(&mut self, ctx: &mut Context, _: &mut Platform) -> Result<Vec<DrawCmd>> {
        if !self.pause {
            for fdm in &mut self.fdms {
                fdm.step(1. / 2., |_: f32| Complex32::new(0., 0.));
            }
            self.refresh_vertices(ctx)?;
        }

        Ok(vec![
            DrawCmd::new(self.verts)
                .indices(self.indices)
                .shader(self.point_shader), //.transform(translate(-SCALE - 1., 0., 0.)),
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

        #[allow(irrefutable_let_patterns)]
        if let Event::Winit(event) = &event {
            if let WinitEvent::WindowEvent { event, .. } = event {
                if let WindowEvent::KeyboardInput { input, .. } = event {
                    if let Some(key) = input.virtual_keycode {
                        if input.state == ElementState::Released {
                            match key {
                                VirtualKeyCode::Space => self.pause = !self.pause,
                                VirtualKeyCode::R => {
                                    self.fdms =
                                        self.init_bufs.clone().map(|buf| Fdm::new(buf, self.dx));
                                    self.refresh_vertices(ctx)?;
                                }
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

impl FdmVisualizer {
    pub fn refresh_vertices(&mut self, ctx: &mut Context) -> Result<()> {
        ctx.update_vertices(self.verts, &scene(&self.fdms))?;
        Ok(())
    }
}

fn fdm_vertices(
    fdms: &[Fdm; 3],
    display: fn([Complex32; 3]) -> (f32, [f32; 3]),
    scale: f32,
) -> Vec<Vertex> {
    let grid = fdms[0].grid();
    let mut vertices = Vec::with_capacity(grid.width() * grid.height());
    for j in 0..grid.height() {
        for i in 0..grid.width() {
            let values = [0, 1, 2].map(|idx| fdms[idx].grid()[(i, j)]);
            let (y, color) = display(values);
            let x = i as f32 * fdms[0].dx();
            let z = j as f32 * fdms[0].dx();
            let pos = [x, y, z].map(|v| v * scale);
            vertices.push(Vertex::new(pos, color));
        }
    }
    vertices
}

fn linear_indices(len: usize) -> Vec<u32> {
    (0..len as u32).collect()
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

fn wave_packet_2d(width: usize, scale: f32, t: f32, a: Complex32, h: f32, m: f32) -> Array2D {
    let mut grid = Array2D::new(width, width);
    let (nx, ny) = inner_size(&grid);

    let coord_map = |v: usize| ((v as f32 / width as f32) * 2. - 1.) * scale;

    for j in 1..nx {
        for i in 1..ny {
            let [x, y] = [i, j].map(coord_map);
            let r = (x * x + y * y).sqrt();
            grid[(i, j)] = wave_packet(r, t, a, h, m);
        }
    }

    grid
}

fn translate(x: f32, y: f32, z: f32) -> [[f32; 4]; 4] {
    [
        [1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.],
        [x, y, z, 1.],
    ]
}

fn png_wave(path: &str) -> Result<[Array2D; 3]> {
    let decoder = png::Decoder::new(File::open(path)?);
    let mut reader = decoder.read_info()?;
    let mut image_data: Vec<u8> = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut image_data)?;
    let image_data = &image_data[..info.buffer_size()];

    assert_eq!(info.bit_depth, BitDepth::Eight);
    assert_eq!(info.color_type, ColorType::Rgb);

    let width = (info.width.max(info.height) + 2) as usize;

    let mut grids = [(); 3].map(|_| Array2D::new(width, width));

    for (j, row) in image_data.chunks_exact(info.line_size).enumerate() {
        for (i, pixel) in row.chunks_exact(3).enumerate().take(width) {
            for (channel, grid) in pixel.iter().zip(&mut grids) {
                let value = *channel as f32 / 256.;
                grid[(i + 1, j + 1)] = Complex32::new(value, 0.);
            }
        }
    }

    Ok(grids)
}
