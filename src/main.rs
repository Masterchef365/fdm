// TODO: Show a 2D animation of the FDM at work!
use fdm::fdm;

fn main() {
    let init = [1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0.];
    let dt = 0.1;
    let n_t = 500;

    let mesh = fdm(10., dt * n_t as f32, 1.0, init.len(), n_t, &init);

    for row in mesh.chunks_exact(init.len()) {
        for item in row {
            print!("{:+.03}, ", item);
        }
        println!();
    }
}
