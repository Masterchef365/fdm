use fdm::Fdm;
use idek::{prelude::*, IndexBuffer};
use num_complex::Complex32;

fn main() -> Result<()> {
    /*
    let dt = 0.01;
    let x_len = 1000;

    let time_len = 8000;

    let sim = SimData {
        x_len,
        time_len,
        data: mesh,
    };
    */

    launch::<(), FdmVisualizer>(Settings::default().vr_if_any_args())
}

struct FdmVisualizer {
    verts: VertexBuffer,
    indices: IndexBuffer,
    line_shader: Shader,

    fdm: Fdm,

    camera: MultiPlatformCamera,
}

impl App for FdmVisualizer {
    fn init(ctx: &mut Context, platform: &mut Platform, _: ()) -> Result<Self> {
        let n_cells = 1000;
        let init: Vec<Complex32> = (0..n_cells)
            .map(|x| {
                //if (x >= x_len / 3) && (x <= 2 * x_len / 3) {
                if x == n_cells / 2 {
                    Complex32::new(0.5, -0.5)
                } else {
                    Complex32::new(0., 0.)
                }
            })
            .collect();

        let fdm = Fdm::new(&init, 1., 1., 1.);

        let vertices = fdm_vertices(&fdm);

        let verts = ctx.vertices(&vertices, true)?;
        let indices: Vec<u32> = (1..vertices.len() * 2 - 1)
            .map(|i| (i / 2) as u32)
            .collect();
        let indices = ctx.indices(&indices, false)?;

        Ok(Self {
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
        self.fdm.step(0.00000001, |x: f32| Complex32::new(x, 0.));

        let vertices = fdm_vertices(&self.fdm);
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

fn fdm_vertices(fdm: &Fdm) -> Vec<Vertex> {
    let x_map = |i: usize| (i as f32 * fdm.dx()) * 2. - 1.;

    fdm.grid()
        .iter()
        .enumerate()
        .map(|(i, u)| Vertex {
            pos: [x_map(i), -u.re, u.im],
            color: [1.; 3],
        })
        .collect()
}
