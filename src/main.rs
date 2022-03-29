use fdm::{Fdm, Array2D, inner_size};
use idek::{
    prelude::*,
    winit::event::{ElementState, Event as WinitEvent, VirtualKeyCode, WindowEvent},
    IndexBuffer,
};
use anyhow::{Result, Context as AnyhowContext};
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
    re_verts: VertexBuffer,
    im_verts: VertexBuffer,
    amp_verts: VertexBuffer,
    indices: IndexBuffer,
    point_shader: Shader,

    init_buf: Array2D,
    dx: f32,

    fdm: Fdm,

    pause: bool,

    camera: MultiPlatformCamera,
}

const SCALE: f32 = 10.;
/*
fn init_fdm() -> Fdm {
    let t = 0.0;
    let a = Complex32::from_polar(1., 0.);
    let h = 1.;
    let m = 1.;

    let mut init = wave_packet_2d(width, SCALE, t, a, h, m);
    init.data_mut().iter_mut().for_each(|c| *c *= 5.);

    //init[(width/2, width/2)] = Complex32::new(1000., 0.);

    Fdm::new(init, dx)
}
*/

fn scene(fdm: &Fdm) -> [Vec<Vertex>; 3] {
    //dbg!(fdm.grid().data().iter().map(|c| c.norm_sqr()).sum::<f32>());

    let scale = 1.0;
    [
        fdm_vertices(&fdm, |cpx| (cpx.re, [0., 0.3, 1.]), scale),
        fdm_vertices(&fdm, |cpx| (cpx.im, [1., 0.3, 0.]), scale),
        fdm_vertices(&fdm, |cpx| (cpx.norm_sqr(), [1.; 3]), scale)
    ]
}

impl App<Args> for FdmVisualizer {
    fn init(ctx: &mut Context, platform: &mut Platform, image_path: Args) -> Result<Self> {
        let init_buf = png_wave(&image_path)?;

        let dx = SCALE / init_buf.width() as f32;
        let fdm = Fdm::new(init_buf.clone(), dx);

        let [re_verts, im_verts, amp_verts] = scene(&fdm);
        let indices = linear_indices(amp_verts.len());

        let re_verts = ctx.vertices(&re_verts, true)?;
        let im_verts = ctx.vertices(&im_verts, true)?;
        let amp_verts = ctx.vertices(&amp_verts, true)?;

        let indices = ctx.indices(&indices, false)?;

        Ok(Self {
            dx,
            init_buf,
            pause: true,
            amp_verts,
            fdm,
            point_shader: ctx.shader(
                DEFAULT_VERTEX_SHADER,
                DEFAULT_FRAGMENT_SHADER,
                Primitive::Points,
            )?,
            re_verts,
            im_verts,
            indices,
            camera: MultiPlatformCamera::new(platform),
        })
    }

    fn frame(&mut self, ctx: &mut Context, _: &mut Platform) -> Result<Vec<DrawCmd>> {
        if !self.pause {
            self.fdm.step(1./2., |_: f32| Complex32::new(0., 0.));
            self.refresh_vertices(ctx)?;
        }

        Ok(vec![
            DrawCmd::new(self.amp_verts)
                .indices(self.indices)
                .shader(self.point_shader)
                .transform(translate(-SCALE - 1., 0., 0.)),

            DrawCmd::new(self.re_verts)
                .indices(self.indices)
                .shader(self.point_shader),

            DrawCmd::new(self.im_verts)
                .indices(self.indices)
                .shader(self.point_shader),

            DrawCmd::new(self.im_verts)
                .indices(self.indices)
                .shader(self.point_shader)
                .transform(translate(-SCALE - 1., 0., -SCALE - 1.)),

            DrawCmd::new(self.re_verts)
                .indices(self.indices)
                .shader(self.point_shader)
                .transform(translate(0., 0., -SCALE - 1.)),
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
                                VirtualKeyCode::R => {
                                    self.fdm = Fdm::new(self.init_buf.clone(), self.dx);
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
        let [re_verts, im_verts, amp_verts] = scene(&self.fdm);
        ctx.update_vertices(self.im_verts, &im_verts)?;
        ctx.update_vertices(self.re_verts, &re_verts)?;
        ctx.update_vertices(self.amp_verts, &amp_verts)?;
        Ok(())
    }
}

fn fdm_vertices(fdm: &Fdm, display: fn(Complex32) -> (f32, [f32; 3]), scale: f32) -> Vec<Vertex> {
    let grid = fdm.grid();
    let mut vertices = Vec::with_capacity(grid.width() * grid.height());
    for j in 0..grid.height() {
        for i in 0..grid.width() {
            let (y, color) = display(grid[(i, j)]);
            let x = i as f32 * fdm.dx();
            let z = j as f32 * fdm.dx();
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

fn png_wave(path: &str) -> Result<Array2D> {
    let decoder = png::Decoder::new(File::open(path)?);
    let mut reader = decoder.read_info()?;
    let mut image_data: Vec<u8> = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut image_data)?;
    let image_data = &image_data[..info.buffer_size()];

    assert_eq!(info.bit_depth, BitDepth::Eight);
    assert_eq!(info.color_type, ColorType::Rgb);

    let width = (info.width.max(info.height) + 2) as usize;

    let mut grid = Array2D::new(width, width);

    for (j, row) in image_data.chunks_exact(info.line_size).enumerate() {
        for (i, pixel) in row.chunks_exact(3).enumerate().take(width) {
            let value = pixel.iter().map(|&v| v as f32 / 256.).sum::<f32>().sqrt();
            grid[(i+1, j+1)] = Complex32::new(value, 0.);
        }
    }

    Ok(grid)
}
