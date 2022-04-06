use std::{
    char::MAX,
    fs::File,
    io::{BufWriter, Write},
    num::NonZeroU32,
};

use fdm::{inner_size, Array2D, Fdm};
use idek::{
    prelude::*,
    winit::event::{ElementState, Event as WinitEvent, VirtualKeyCode, WindowEvent},
    IndexBuffer,
};
use num_complex::Complex32;
use rayon::prelude::*;

fn main() -> Result<()> {
    launch::<(), FdmVisualizer>(Settings::default().vr_if_any_args().msaa_samples(8))
}

struct FdmVisualizer {
    re_verts: VertexBuffer,
    im_verts: VertexBuffer,
    amp_verts: VertexBuffer,
    indices: IndexBuffer,
    point_shader: Shader,

    fdm: Fdm,

    pause: bool,

    camera: MultiPlatformCamera,
}

const SCALE: f32 = 10.;
fn init_fdm() -> Fdm {
    let width = 250;

    let dx = SCALE / width as f32;
    let t = 0.0;
    let a = Complex32::from_polar(1., 0.);
    let h = 1.;
    let m = 1.;

    let mut init = wave_packet_2d(width, SCALE, t, a, h, m);
    init.data_mut().iter_mut().for_each(|c| *c *= 5.);

    //init[(width/2, width/2)] = Complex32::new(200., 0.);

    Fdm::new(init, dx)
}

fn scene(fdm: &Fdm) -> Vec<Vertex> {
    //dbg!(fdm.grid().data().iter().map(|c| c.norm_sqr()).sum::<f32>());

    let scale = 1.0;
    fdm_vertices(&fdm, |cpx| (cpx.norm_sqr(), [1.; 3]), scale)
}

impl App for FdmVisualizer {
    fn init(ctx: &mut Context, platform: &mut Platform, _: ()) -> Result<Self> {
        let fdm = init_fdm();

        let amp_verts = scene(&fdm);
        let indices = linear_indices(amp_verts.len());

        let re_verts = ctx.vertices(&amp_verts, true)?;
        let im_verts = ctx.vertices(&amp_verts, true)?;
        let amp_verts = ctx.vertices(&amp_verts, true)?;

        let indices = ctx.indices(&indices, false)?;

        Ok(Self {
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
            self.fdm.step(1. / 2., |_: f32| Complex32::new(0., 0.));
            self.refresh_vertices(ctx);
        }

        Ok(vec![
            DrawCmd::new(self.amp_verts)
                .indices(self.indices)
                .shader(self.point_shader)
                .transform(translate(-SCALE - 1., 0., 0.)),
            /*DrawCmd::new(self.re_verts)
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
                .transform(translate(0., 0., -SCALE - 1.)),*/
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
                                    self.fdm = init_fdm();
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
        let amp_verts = scene(&self.fdm);
        //ctx.update_vertices(self.im_verts, &im_verts)?;
        //ctx.update_vertices(self.re_verts, &re_verts)?;
        ctx.update_vertices(self.amp_verts, &amp_verts)?;
        Ok(())
    }
}

fn mandelbrot(c: Complex32, max_iters: u32) -> Option<NonZeroU32> {
    let mut z = c;
    for i in 1..=max_iters {
        z = z * z + c;
        if z.norm_sqr() >= 4. {
            return NonZeroU32::new(i);
        }
    }

    None
}

fn mandelbrot_image_aa(
    width: usize,
    height: usize,
    scale: f32,
    origin: (f32, f32),
    max_iters: u32,
    aa_divs: i32,
    color_fn: impl Sync + Fn(Option<NonZeroU32>) -> [u8; 3],
) -> Vec<u8> {
    let (orig_x, orig_y) = origin;

    let mut output = vec![0; width * height * 3];

    let subpix_width = aa_divs * 2 + 1;
    let subpix_area = subpix_width * subpix_width;

    output
        .par_chunks_exact_mut(width * 3)
        .enumerate()
        .for_each(|(row_idx, row)| {
            for col_idx in 0..width {
                let mut color = [0i32; 3];
                for i in -aa_divs..=aa_divs {
                    for j in -aa_divs..=aa_divs {
                        let x = (col_idx as i32 * subpix_width + i) as f32
                            / (width as i32 * subpix_width) as f32;
                        let y = (row_idx as i32 * subpix_width + j as i32) as f32
                            / (height as i32 * subpix_width) as f32;

                        let c = Complex32::new(
                            (x * 2. - 1.) * scale + orig_x,
                            (y * 2. - 1.) * scale + orig_y,
                        );

                        color
                            .iter_mut()
                            .zip(color_fn(mandelbrot(c, max_iters)))
                            .for_each(|(c, m)| *c += m as i32);
                    }
                }

                let color = color.map(|c| (c / subpix_area) as u8);
                row[col_idx * 3..][..3].copy_from_slice(&color);
            }
        });

    output
}

fn mandelbrot_image(
    width: usize,
    height: usize,
    scale: f32,
    origin: (f32, f32),
    max_iters: u32,
    color: impl Fn(Option<NonZeroU32>) -> [u8; 3],
) -> Vec<u8> {
    let (orig_x, orig_y) = origin;

    let mut output = Vec::with_capacity(width * height * 3);

    for row in 0..height {
        for col in 0..width {
            let x = col as f32 / width as f32;
            let y = row as f32 / height as f32;

            let c = Complex32::new(
                (x * 2. - 1.) * scale + orig_x,
                (y * 2. - 1.) * scale + orig_y,
            );

            output.extend_from_slice(&color(mandelbrot(c, max_iters)));
        }
    }

    output
}

fn mandelbrot_color(i: Option<NonZeroU32>, max_iters: u32) -> [u8; 3] {
    match i {
        Some(i) => {
            let v = ((255 * i.get()) / max_iters) as u8;
            let v = v.saturating_mul(2);
            [
                v,
                v.saturating_add(10),
                v.saturating_mul(2).saturating_add(3),
            ]
        }
        None => [0; 3],
    }
}

fn no_main() -> Result<()> {
    let path = "out.ppm";
    let mut file = BufWriter::new(File::create(path)?);
    let max_iters = 255;
    let (width, height) = (5_000, 5_000);

    println!("Generating");
    let time = std::time::Instant::now();
    let image = mandelbrot_image_aa(width, height, 2., (-1. / 4., 0.), max_iters, 4, |i| {
        mandelbrot_color(i, max_iters)
    });
    println!("Finished in {}s, writing...", time.elapsed().as_secs_f32());

    writeln!(file, "P6")?;
    writeln!(file, "{} {}", width, height)?;
    writeln!(file, "255")?;
    file.write_all(&image)?;

    Ok(())
}

fn fdm_vertices(fdm: &Fdm, display: fn(Complex32) -> (f32, [f32; 3]), scale: f32) -> Vec<Vertex> {
    let grid = fdm.grid();

    let max_iters = 50;

    let mut vertices = Vec::with_capacity(grid.width() * grid.height());
    for j in 0..grid.height() {
        for i in 0..grid.width() {
            let c = grid[(i, j)];
            let (y, color) = display(c);
            let y = y * 5.;
            let x = i as f32 * fdm.dx();
            let z = j as f32 * fdm.dx();
            let pos = [x, y, z].map(|v| v * scale);

            let m = mandelbrot(c, max_iters);
            let color = mandelbrot_color(m, max_iters);
            let color = color.map(|c| c as f32 / 256.);

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