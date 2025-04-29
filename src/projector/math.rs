// src/projector/math.rs
//! Vector math operations for 3D geometry calculations.
//! Provides basic operations like dot product, vector scaling, subtraction, reflection, etc.

/// Calculates the dot product of two vector slices.
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Subtracts vector slice `b` from `a` and stores the result in `out`.
pub fn vec_sub(a: &[f32], b: &[f32], out: &mut [f32]) {
    for i in 0..out.len().min(a.len()).min(b.len()) {
        out[i] = a[i] - b[i];
    }
}

/// Scales a vector slice `a` by a scalar `s` and stores the result in `out`.
pub fn vec_scale(a: &[f32], s: f32, out: &mut [f32]) {
    for i in 0..out.len().min(a.len()) {
        out[i] = a[i] * s;
    }
}

/// Calculates the reflection of vector `neg_l` (pointing away from surface) about normal `n`.
/// Result is stored in `out`.
pub fn reflect(neg_l: &[f32], n: &[f32], out: &mut [f32]) {
    // Assumes neg_l, n, out are 3D vectors
    if out.len() != 3 || neg_l.len() != 3 || n.len() != 3 {
        // Handle error or return early if dimensions mismatch
        return;
    }
    let dot_nl_neg = dot_product(n, neg_l);
    // R = L - 2 * dot(N, L) * N => Using neg_l = -L => R = -neg_l - 2 * dot(N, -neg_l) * N
    // Simplified R = 2 * dot(N, neg_l) * N - neg_l, which points away from the surface
    let mut temp = [0.0f32; 3];
    vec_scale(n, 2.0 * dot_nl_neg, &mut temp);
    vec_sub(&temp, neg_l, out);
}

/// Normalizes a 3D vector in-place.
pub fn normalize_vec3(v: &mut [f32; 3]) {
    let length_squared = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    if length_squared > 0.0 {
        let inv_length = 1.0 / length_squared.sqrt();
        v[0] *= inv_length;
        v[1] *= inv_length;
        v[2] *= inv_length;
    }
}
