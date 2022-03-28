use num_complex::Complex32;

const fn zero() -> Complex32 {
    Complex32::new(0., 0.)
}

pub struct Fdm {
    last: Vec<Complex32>,
    current: Vec<Complex32>,
    scratch: Vec<Complex32>,
    dx: f32,
}

impl Fdm {
    /// Creates a new FDM with the given width (in the X direction) and initial grid contents `init`
    pub fn new(init: &[Complex32], dx: f32) -> Self {
        Self {
            current: init.to_vec(),
            last: vec![zero(); init.len()],
            scratch: vec![zero(); init.len()],
            dx,
        }
    }

    pub fn step(&mut self, dt: f32, _v: impl Fn(f32) -> Complex32) {
        std::mem::swap(&mut self.last, &mut self.current);
        const K: usize = 21;

        // here hbar / 2m = 1
        let r = -Complex32::i() * dt / (self.dx * self.dx);

        for _ in 0..K {
            for ((last, scratch), current) in self
                .last
                .iter()
                .zip(&mut self.scratch)
                .skip(1)
                .zip(self.current.windows(3))
            {
                *scratch = (*last - r * (current[0] + current[2])) / (1. - 2. * r);
            }

            std::mem::swap(&mut self.current, &mut self.scratch);
        }
    }

    pub fn grid(&self) -> &[Complex32] {
        &self.current
    }

    pub fn grid_mut(&mut self) -> &mut [Complex32] {
        &mut self.current
    }

    pub fn dx(&self) -> f32 {
        self.dx
    }
}
