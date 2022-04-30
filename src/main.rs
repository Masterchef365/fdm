use std::sync::mpsc::{Sender, self, Receiver};

use fdm::{inner_size, Array2D, Fdm};
use idek::{
    prelude::*,
    winit::event::{ElementState, Event as WinitEvent, VirtualKeyCode, WindowEvent},
    IndexBuffer,
};
use num_complex::Complex32;

fn main() -> Result<()> {
    launch::<(), FdmVisualizer>(Settings::default().vr_if_any_args().msaa_samples(8))
}

struct FdmVisualizer {
    re_verts: VertexBuffer,
    im_verts: VertexBuffer,
    amp_verts: VertexBuffer,
    indices: IndexBuffer,
    point_shader: Shader,

    rx: Receiver<Array2D>,
    grid: Array2D,

    pause: bool,

    camera: MultiPlatformCamera,
}

fn sim_thread(tx: Sender<Array2D>) {
    let mut fdm = init_fdm();

    loop {
        fdm.step(1. / 2.);
        tx.send(fdm.grid().clone()).expect("Render hungup");
    }
}

const SCALE: f32 = 10.;
const WIDTH: usize = 100;
const DX: f32 = SCALE as f32 / WIDTH as f32;

fn init_fdm() -> Fdm {
    let width = 100;

    let t = 0.0;
    let a = Complex32::from_polar(1., 0.);
    let h = 1.;
    let m = 1.;

    let mut init = wave_packet_2d(width, SCALE, t, a, h, m);
    init.data_mut().iter_mut().for_each(|c| *c *= 5.);

    Fdm::new(init, DX)
}

fn scene(grid: &Array2D) -> [Vec<Vertex>; 3] {
    //dbg!(fdm.grid().data().iter().map(|c| c.norm_sqr()).sum::<f32>());

    let scale = 1.0;
    [
        fdm_vertices(grid, |cpx| (cpx.re, [0., 0.3, 1.]), scale, DX),
        fdm_vertices(grid, |cpx| (cpx.im, [1., 0.3, 0.]), scale, DX),
        fdm_vertices(grid, |cpx| (cpx.norm_sqr(), [1.; 3]), scale, DX),
    ]
}

impl App for FdmVisualizer {
    fn init(ctx: &mut Context, platform: &mut Platform, _: ()) -> Result<Self> {
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(|| sim_thread(tx));

        let grid = rx.recv()?;

        let [re_verts, im_verts, amp_verts] = scene(&grid);
        let indices = linear_indices(amp_verts.len());

        let re_verts = ctx.vertices(&re_verts, true)?;
        let im_verts = ctx.vertices(&im_verts, true)?;
        let amp_verts = ctx.vertices(&amp_verts, true)?;

        let indices = ctx.indices(&indices, false)?;

        Ok(Self {
            pause: true,
            amp_verts,
            rx,
            grid,
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
        if let Ok(grid) = self.rx.try_recv() {
            self.grid = grid;
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

        /*
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
        */
        self.refresh_vertices(ctx)?;

        idek::close_when_asked(platform, &event);
        Ok(())
    }
}

impl FdmVisualizer {
    pub fn refresh_vertices(&mut self, ctx: &mut Context) -> Result<()> {
        let [re_verts, im_verts, amp_verts] = scene(&self.grid);
        ctx.update_vertices(self.im_verts, &im_verts)?;
        ctx.update_vertices(self.re_verts, &re_verts)?;
        ctx.update_vertices(self.amp_verts, &amp_verts)?;
        Ok(())
    }
}

fn fdm_vertices(
    grid: &Array2D,
    display: fn(Complex32) -> (f32, [f32; 3]),
    scale: f32,
    dx: f32,
) -> Vec<Vertex> {
    let mut vertices = Vec::with_capacity(grid.width() * grid.height());
    for j in 0..grid.height() {
        for i in 0..grid.width() {
            let (y, color) = display(grid[(i, j)]);
            let pos = [i as f32 * dx, y, j as f32 * dx].map(|v| v * scale);
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
