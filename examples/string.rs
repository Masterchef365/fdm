use fdm::fdm;
use idek::{prelude::*, IndexBuffer};

fn main() -> Result<()> {
    let dt = 0.01;
    let init = [1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0.];
    let time_len = 500;
    let x_len = init.len();

    let mesh = fdm(10., dt * time_len as f32, 1.0, x_len, time_len, &init);

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
    pub data: Vec<f32>,
}

struct TriangleApp {
    verts: VertexBuffer,
    indices: IndexBuffer,
    line_shader: Shader,

    frame: usize,
    sim: SimData,
}

impl App<SimData> for TriangleApp {
    fn init(ctx: &mut Context, _: &mut Platform, sim: SimData) -> Result<Self> {
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
        })
    }

    fn frame(&mut self, ctx: &mut Context, _: &mut Platform) -> Result<Vec<DrawCmd>> {
        self.frame = (self.frame + 1) % self.sim.time_len;
        
        let vertices = sim_vertices(&self.sim, self.frame);
        ctx.update_vertices(self.verts, &vertices)?;

        Ok(vec![DrawCmd::new(self.verts).indices(self.indices).shader(self.line_shader)])
    }

    fn event(&mut self, ctx: &mut Context, platform: &mut Platform, event: Event) -> Result<()> {
        idek::simple_ortho_cam_ctx(ctx, platform);
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
            pos: [x_map(i), -*u, 0.],
            color: [1.; 3],
        })
        .collect()
}
