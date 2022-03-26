use fdm::fdm;
use idek::{prelude::*, IndexBuffer};
use std::time::Instant;
use num_complex::Complex32;

fn main() -> Result<()> {
    let dt = 0.01;
    let x_len = 1000;
    let init: Vec<Complex32> = (0..x_len)
        .map(|x| {
            //if (x >= x_len / 3) && (x <= 2 * x_len / 3) {
            if x == x_len / 2 {
                Complex32::new(0.5, 2.5)
            } else {
                Complex32::new(0., 0.)
            }
        })
        .collect();

    let time_len = 8000;

    let start = Instant::now();

    let mesh = fdm(10., dt * time_len as f32, 0.5, x_len, time_len, &init);

    println!("FDM took {} ms", start.elapsed().as_secs_f32() * 1000.);

    let sim = SimData {
        x_len,
        time_len,
        data: mesh,
    };

    launch::<SimData, TriangleApp>(Settings::default().vr_if_any_args().args(sim))
}

#[derive(Default)]
pub struct SimData {
    pub x_len: usize,
    pub time_len: usize,
    pub data: Vec<Complex32>,
}

struct TriangleApp {
    verts: VertexBuffer,
    indices: IndexBuffer,
    line_shader: Shader,

    frame: usize,
    sim: SimData,

    camera: MultiPlatformCamera,
}

impl App<SimData> for TriangleApp {
    fn init(ctx: &mut Context, platform: &mut Platform, sim: SimData) -> Result<Self> {
        let vertices = sim_vertices(&sim, 0);

        let verts = ctx.vertices(&vertices, true)?;
        let indices: Vec<u32> = (1..vertices.len() * 2 - 1)
            .map(|i| (i / 2) as u32)
            .collect();
        let indices = ctx.indices(&indices, false)?;

        Ok(Self {
            line_shader: ctx.shader(
                DEFAULT_VERTEX_SHADER,
                DEFAULT_FRAGMENT_SHADER,
                Primitive::Lines,
            )?,
            verts,
            indices,
            sim,
            frame: 0,
            camera: MultiPlatformCamera::new(platform),
        })
    }

    fn frame(&mut self, ctx: &mut Context, _: &mut Platform) -> Result<Vec<DrawCmd>> {
        self.frame = (self.frame + 1) % self.sim.time_len;

        let vertices = sim_vertices(&self.sim, self.frame);
        ctx.update_vertices(self.verts, &vertices)?;

        Ok(vec![DrawCmd::new(self.verts)
            .indices(self.indices)
            .shader(self.line_shader)])
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
        idek::close_when_asked(platform, &event);
        Ok(())
    }
}

fn sim_vertices(sim: &SimData, t: usize) -> Vec<Vertex> {
    let row = sim
        .data
        .chunks_exact(sim.x_len)
        .skip(t)
        .next()
        .expect("Invalid row");

    let x_map = |i: usize| (i as f32 / sim.x_len as f32) * 2. - 1.;

    row.iter()
        .enumerate()
        .map(|(i, u)| Vertex {
            pos: [x_map(i), -u.re, u.im],
            color: [1.; 3],
        })
        .collect()
}
