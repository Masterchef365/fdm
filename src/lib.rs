use num_complex::Complex32;

const fn zero() -> Complex32 {
    Complex32::new(0., 0.)
}

pub struct Fdm {
    last: Vec<Complex32>,
    current: Vec<Complex32>,
    scratch: Vec<Complex32>,
    dx: f32,
    h: f32,
    m: f32,
}

impl Fdm {
    /// Creates a new FDM with the given width (in the X direction) and initial grid contents `init`
    pub fn new(init: &[Complex32], width: f32, h: f32, m: f32) -> Self {
        Self {
            current: init.to_vec(),
            last: vec![zero(); init.len()],
            scratch: vec![zero(); init.len()],
            dx: width / init.len() as f32,
            h,
            m,
        }
    }

    pub fn step(&mut self, dt: f32, v: impl Fn(f32) -> Complex32) {
        std::mem::swap(&mut self.last, &mut self.current);
        const K: usize = 20;

        let r = dt / (self.dx * self.dx);

        for _ in 0..K {
            for ((idx, (last, scratch)), current) in self
                .last
                .iter()
                .zip(&mut self.scratch)
                .enumerate()
                .skip(1)
                .zip(self.current.windows(3))
            {

                let cfd = current[0] - 2. * current[1] + current[2];
                *scratch = current[1] + r * Complex32::i() * (*last - cfd);

                /*
                let cfd = central_finite_difference((last[0], last[1], last[2]));

                *current = last[1]
                    + (dt / (self.dx * self.dx)) * (self.h / (2. * self.m)) * Complex32::i() * cfd;
                */
                //let x = idx as f32 * self.dx;
                //- dt / self.h * Complex32::i() * v(x) * last[1];
            }

            std::mem::swap(&mut self.current, &mut self.scratch);
        }
    }

    pub fn grid(&self) -> &[Complex32] {
        &self.current
    }

    pub fn dx(&self) -> f32 {
        self.dx
    }
}

/// Compute the Central Finite Distance from the given neighborhood
fn central_finite_difference(
    (left, center, right): (Complex32, Complex32, Complex32),
) -> Complex32 {
    left - 2. * center + right
}
