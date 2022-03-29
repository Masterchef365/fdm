mod array2d;
use num_complex::Complex32;
pub type Array2D = array2d::Array2D<Complex32>;

const fn zero() -> Complex32 {
    Complex32::new(0., 0.)
}

pub struct Fdm {
    last: Array2D,
    current: Array2D,
    scratch: Array2D,
    dx: f32,
}

impl Fdm {
    /// Creates a new FDM with the given width (in the X direction) and initial grid contents `init`
    pub fn new(init: Array2D, dx: f32) -> Self {
        Self {
            last: Array2D::new(init.width(), init.height()),
            scratch: Array2D::new(init.width(), init.height()),
            current: init,
            dx,
        }
    }

    pub fn step(&mut self, dt: f32, _v: impl Fn(f32) -> Complex32) {
        let (nx, ny) = inner_size(&self.current);

        std::mem::swap(&mut self.last, &mut self.current);
        const K: usize = 80;

        // here hbar / 2m = 1
        let r = -Complex32::i() * dt;// / (self.dx * self.dx);

        for _ in 0..K {
            for i in 1..=nx {
                for j in 1..=ny {
                    let sum = self.current[(i - 1, j)]
                        + self.current[(i + 1, j)]
                        + self.current[(i, j - 1)]
                        + self.current[(i, j + 1)];

                    self.scratch[(i, j)] = (self.last[(i, j)] - r * sum) / (1. - 4. * r);
                }
            }

            std::mem::swap(&mut self.current, &mut self.scratch);
        }
    }

    pub fn grid(&self) -> &Array2D {
        &self.current
    }

    pub fn grid_mut(&mut self) -> &mut Array2D {
        &mut self.current
    }

    pub fn dx(&self) -> f32 {
        self.dx
    }
}

pub fn inner_size(x: &Array2D) -> (usize, usize) {
    debug_assert!(x.width() >= 2);
    debug_assert!(x.height() >= 2);

    (x.width() - 2, x.height() - 2)
}
