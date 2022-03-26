use num_complex::Complex32;

const fn zero() -> Complex32 {
    Complex32::new(0., 0.)
}

pub struct Fdm {
    last: Vec<Complex32>,
    current: Vec<Complex32>,
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
            dx: width / init.len() as f32,
            h,
            m,
        }
    }

    pub fn step(&mut self, dt: f32, v: impl Fn(f32) -> Complex32) {
        std::mem::swap(&mut self.last, &mut self.current);
        for ((idx, current), last) in self
            .current
            .iter_mut()
            .enumerate()
            .skip(1)
            .zip(self.last.windows(3))
        {
            let x = idx as f32 * self.dx;

            let cfd = central_finite_difference((last[0], last[1], last[2]));

            *current = last[1]
                + (dt / (self.dx * self.dx)) * (self.h / (2. * self.m)) * Complex32::i() * cfd
                - dt / self.h * Complex32::i() * v(x) * last[1];
        }
    }

    pub fn grid(&self) -> &[Complex32] {
        &self.current
    }

    pub fn dx(&self) -> f32 {
        self.dx
    }
}

// TODO:
// * Implement arbitrary boundary conditions
// * More dimensions
/// max_x: Length of the positional domain
/// max_t: Length of the time domain
/// c: Wave speed
/// n_x: Number of discrete spatial steps
/// n_t: Number of discrete time steps
/// init: Initial condition at t = 0
pub fn nope(
    max_x: f32,
    max_t: f32,
    c: f32,
    n_x: usize,
    n_t: usize,
    init: &[Complex32],
) -> Vec<Complex32> {
    // Input checks
    assert_eq!(init.len(), n_x);

    // Calculate simulation parameters
    let dx = max_x / n_x as f32;
    let dt = max_t / n_t as f32;
    let courant = c * dt / dx;

    // Initialize mesh, copy init values
    let mut mesh = vec![zero(); n_x * n_t];
    mesh[..n_x].as_mut().copy_from_slice(init);

    // Fill in the second time step special case
    let (mesh_t_0, mesh_t_1) = mesh[..n_x * 2].split_at_mut(n_x);
    for (i, u) in mesh_t_1.iter_mut().enumerate() {
        let neighbors @ (_, center, _) = neighborhood(&mesh_t_0, i);
        *u = center
            - 0.5 * courant * courant * central_finite_difference(neighbors) * Complex32::i();
    }

    // Compute the remaining time steps
    for t in 2..n_t {
        // Compute the three windows: Current, previous, and the previous to that
        let t_window = &mut mesh[(t - 2) * n_x..(t + 1) * n_x];
        let (prev_rows, current_row) = t_window.split_at_mut(n_x * 2);
        let (prev_prev_row, prev_row) = prev_rows.split_at_mut(n_x);

        // Solve a new row
        for (i, u) in current_row.iter_mut().enumerate() {
            let down = prev_prev_row[i];
            let neighbors @ (_, center, _) = neighborhood(prev_row, i);
            *u = -down
                + 2. * center
                + courant * courant * central_finite_difference(neighbors) * Complex32::i();
        }
    }

    mesh
}

/// Find the three x neighboring pixels
fn neighborhood(row: &[Complex32], i: usize) -> (Complex32, Complex32, Complex32) {
    let left = match i.checked_sub(1) {
        Some(q) => row[q],
        None => zero(),
    };

    let center = row[i];

    let right = match i + 1 < row.len() {
        true => row[i + 1],
        false => zero(),
    };

    (left, center, right)
}

/// Compute the Central Finite Distance from the given neighborhood
fn central_finite_difference(
    (left, center, right): (Complex32, Complex32, Complex32),
) -> Complex32 {
    left + -2. * center + right
}
