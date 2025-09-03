use std::f32::consts::PI;
use num_complex::{Complex32, ComplexFloat};
use rayon::prelude::*;

#[cfg(test)]
mod tests {
    use num_complex::Complex32;
    use crate::{fit_init, gauss_newton, inverse_normal_matrix};

    #[test]
    fn t_fit_init() {

        let p0 = 0.;
        let p1 = -3.1; // works for phase slopes less than pi
        let k = [0.,1.,2.];
        let res_tol = 1e-4;
        let max_iter = 30;

        // some random values so the fit isn't perfect
        let noise = vec![
            Complex32::new(0.004, -0.003),
            Complex32::new(0.002, -0.004),
            Complex32::new(-0.005, 0.),
        ];

        // magnitude values for the weighted LS
        let mag = vec![1.,0.5,0.25];

        // simulate complex signal
        let m:Vec<_> =  k.iter().zip(noise.iter().zip(&mag)).map(|(&k,(n,&m))|{
            m * Complex32::new(0.,k * p1 + p0).exp() + n
        }).collect();

        // perform initial fit
        let mut fit = fit_init(&m);

        println!("init fit: {:?}",fit);

        let err = ((fit[0] - p0).powi(2) + (fit[1] - p1).powi(2)).sqrt();
        //println!("err = {err}");
        //assert!(err < 1e-5);

        let matrix_entries = inverse_normal_matrix(&m);
        let mut w = vec![Complex32::ZERO; m.len()];
        let mut r = w.clone();

        let tol = res_tol * fit[1].abs();
        println!("mat entries = {:?}", matrix_entries);
        let err = gauss_newton(&m,&matrix_entries,&mut w,&mut r,&mut fit,tol,max_iter);

        println!("fit = {:?}",fit);
        println!("err = {err}");




    }

}

pub fn fit_freq(img:&[Complex32], dims:&[usize], tol:f32, max_iter:usize) -> (Vec<f32>, Vec<f32>) {

    assert_eq!(dims.len(), 4, "input must be 4 dimensional");
    let n_echoes = dims[3];
    assert!(n_echoes >= 3, "input must have at least 3 echoes");
    let n_vox:usize = dims[0..3].iter().product();

    let mut fit = vec![0f32;n_vox];
    let mut err = vec![0f32;n_vox];

    let vols:Vec<&[Complex32]> = img.chunks_exact(n_vox).collect();

    fit.par_iter_mut().zip(err.par_iter_mut()).enumerate().for_each(|(i,(fit,err))|{

        //println!("i = {i}");
        let mut w = vec![Complex32::ZERO;n_echoes];
        let mut r = vec![Complex32::ZERO;n_echoes];
        let mut m = vec![Complex32::ZERO;n_echoes];

        // gather echo data for voxel i
        m.iter_mut().zip(&vols).for_each(|(m,echo)| *m = echo[i] );

        let inv_mat = inverse_normal_matrix(&m);
        let mut x = fit_init(&m);
        *err = gauss_newton(&m, &inv_mat, &mut w, &mut r, &mut x, tol, max_iter);
        *fit = x[1]
    });

    (fit,err)

}


fn gauss_newton(m:&[Complex32], inv_mat_entries:&[f32], w:&mut [Complex32], r:&mut [Complex32], coeffs:&mut [f32], tol:f32, max_iter:usize) -> f32 {

    let ai11 = inv_mat_entries[0];
    let ai12 = inv_mat_entries[1];
    let ai22 = inv_mat_entries[2];

    let mut loop_count = 0;

    let mut p0 = coeffs[0];
    let mut p1 = coeffs[1];

    //println!("m = {:?}",m);

    loop {

        // update the solution w
        w.iter_mut().zip(m.iter()).enumerate().for_each(|(k,(w,m))|{
            *w = m.norm() * Complex32::new(0.,k as f32 * p1 + p0).exp();
        });

        // update the residual
        r.iter_mut().zip(w.iter().zip(m.iter())).for_each(|(r,(&w,&m))| *r = m - w);

        // update right-hand side variables
        let pr1:Complex32 = w.iter().zip(r.iter()).map(|(w,&r)|{
            (Complex32::i() * w).conj() * r
        }).sum();

        let pr2:Complex32 = w.iter().zip(r.iter()).enumerate().map(|(k,(w,&r))|{
            (Complex32::i() * w).conj() * r * k as f32
        }).sum();

        // solve for the updates
        let dp0 = (ai11 * pr1 + ai12 * pr2).re;
        let dp1 = (ai12 * pr1 + ai22 * pr2).re;

        // check if the update was small relative to tolerance to terminate
        loop_count += 1;
        if dp1.abs() < tol || loop_count > max_iter {
            break
        }

        // do update
        p0 += dp0;
        p1 += dp1;
    }

    // assign values
    coeffs[0] = p0;
    coeffs[1] = wrap_to_pi(p1);

    // fit std err
    ai22.sqrt()

}




/// computes the entries for the inverse normal matrix for the gauss-newton updates
/// inputs are complex signal m, and coefficients are written to `coeffs` as \[a11,a12,a22\]
fn inverse_normal_matrix(m:&[Complex32]) -> [f32;3] {
    let a11:f32 = m.iter().map(|x| x.norm_sqr()).sum();
    let a12:f32 = m.iter().enumerate().map(|(k,x)| x.norm_sqr() * k as f32).sum();
    let a22:f32 = m.iter().enumerate().map(|(k,x)| x.norm_sqr() * k.pow(2) as f32).sum();

    let d = a11 * a22 - a12 * a12;
    let det = 1./d;

    let ai11 = det * a22;
    let ai12 = -det * a12;
    let ai22 = det * a11;

    [ai11, ai12, ai22]

}



/// returns the initial linear frequency (slope) and phase (intercept) fit from the first 3 echoes.
/// `phase`: per-echo phase values in \[-pi, pi\], length >= 3
/// `fit`: output slice where fit\[0\] = p0 (intercept), fit\[1\] = p1 (slope per echo index)
///
/// Model (using echo indices k = 0,1,2):
///   phi_k = p0 + p1 * k
pub fn fit_init(m: &[Complex32]) -> [f32;2] {
    use std::f32::consts::PI;

    assert!(m.len() >= 3, "need at least 3 echoes");

    let two_pi = 2.0_f32 * PI;

    // 1) Take first three echoes' phases
    let mut y0 = m[0].to_polar().1;
    let mut y1 = m[1].to_polar().1;
    let mut y2 = m[2].to_polar().1;

    // 2) Establish baseline step c (between echo1 and echo0) closest to 0 modulo 2π
    let c_raw = y1 - y0;
    let c_candidates = [c_raw - two_pi, c_raw, c_raw + two_pi];
    let mut c = c_candidates[0];
    let mut best = c.abs();
    for &cc in &c_candidates[1..] {
        let a = cc.abs();
        if a < best {
            best = a;
            c = cc;
        }
    }

    // 3) Local unwrap across echoes using c as the "expected" step
    // n = 0: compare (y1 - y0) to c, adjust tail [y1..] if deviation crosses ±π
    let mut adjust_tail = |yn: f32, yn1: &mut f32, tail: &mut [f32]| {
        let cd = (*yn1 - yn) - c; // deviation from baseline c
        if cd < -PI {
            *yn1 += two_pi;
            for t in tail {
                *t += two_pi;
            }
        } else if cd > PI {
            *yn1 -= two_pi;
            for t in tail {
                *t -= two_pi;
            }
        }
    };
    // n = 0 (affects y1 and y2)
    {
        let mut tail = [&mut y2];
        adjust_tail(y0, &mut y1, unsafe { std::slice::from_mut(tail[0]) });
    }
    // n = 1 (affects y2)
    adjust_tail(y1, &mut y2, &mut []);

    // 4) LS fit over k = 0,1,2 (closed form for equally spaced samples)
    // slope p1 = (y2 - y0)/2, intercept p0 = mean(y) - p1
    let p1 = 0.5 * (y2 - y0);
    let mean_y = (y0 + y1 + y2) / 3.0;
    let mut p0 = mean_y - p1;

    let p1 = wrap_to_pi(p1);

    // (p0 need not be wrapped, but you can if you like)
    // let p0 = wrap_to_pi(p0);

    [p0,p1]

}

pub fn wrap_to_pi(mut x: f32) -> f32 {
    let two_pi = 2.0 * PI;
    // use Euclidean remainder to handle negatives too
    x = (x + PI) % two_pi;
    if x < 0.0 {
        x += two_pi;
    }
    x - PI
}