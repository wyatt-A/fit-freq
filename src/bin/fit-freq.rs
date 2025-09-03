use std::ffi::c_double;
use std::path::{Path, PathBuf};
use std::time::Instant;
use clap::Parser;
use array_lib::{io_nrrd, nrrd_rs, ArrayDim};
use array_lib::io_nrrd::{write_nrrd, Encoding, NRRD};
use array_lib::nrrd_rs::header_defs::Kind;
use array_lib::nrrd_rs::read_header;
use num_complex::{Complex32, ComplexFloat};
use fit_freq::fit_freq;
use rayon::prelude::*;

#[derive(Parser,Debug)]
struct Args {

    /// input complex .nhdr or .nrrd
    #[clap(short, long)]
    input: Vec<PathBuf>,

    /// output directory
    #[clap(short, long)]
    output_dir: PathBuf,

    #[clap(long)]
    mag_prefix: Option<String>,
    #[clap(long)]
    freq_prefix: Option<String>,
    #[clap(long)]
    std_err_prefix: Option<String>,

    /// relative fit tolerance
    fit_tol: Option<f32>,

    /// max iteration count per voxel
    max_iter: Option<usize>,
}

fn main() -> Result<(), String> {

    let args = Args::parse();

    if args.input.len() < 3 {
        Err("you must provide at least 3 echoes".to_string())?
    }

    let n_echoes = args.input.len();
    // read headers to ensure all sizes are consistent
    let (dims,ref_header) = check_sizes(&args.input);
    if dims[0] != 2 {
        Err("expecting first dimension to be 2 for real and imaginary".to_string())?
    }
    let dims = dims[1..].to_vec();
    let vol_size:usize = dims.iter().product();

    let mut echo_data = vec![Complex32::ZERO; vol_size * n_echoes];


    echo_data.par_chunks_exact_mut(vol_size).zip(&args.input).for_each(|(vol,path)|{
        println!("loading {} ...",path.display());
        let (c_data,..) = load_complex_nhdr(path);
        vol.copy_from_slice(&c_data);
    });

    let max = echo_data.par_iter().map(|x| x.norm()).max_by(|a,b|a.partial_cmp(b).expect("failed to compare values")).unwrap();
    let scale = 1./max;
    echo_data.par_iter_mut().for_each(|x|*x = x.scale(scale));

    println!("calculating magnitude ...");
    // calculate magnitude
    let mut magn = vec![0f32; vol_size];
    let echoes:Vec<&[Complex32]> = echo_data.chunks_exact(vol_size).collect();

    magn.par_iter_mut().enumerate().for_each(|(i,x)| {
        // calculate rms magnitude over echoes
        *x = echoes.iter().map(|echo|{
            echo[i].re as f64 * echo[i].re as f64 + echo[i].im as f64 * echo[i].im as f64
        }).sum::<f64>().sqrt() as f32
    });

    println!("writing magnitude ...");

    let mag_filename = args.freq_prefix.as_ref().map(|s|s.as_str()).unwrap_or("magn");
    write_nrrd(args.output_dir.join(mag_filename),&magn,ArrayDim::from_shape(&dims),None,false,Encoding::raw);

    let tol = args.fit_tol.unwrap_or(1e-4);
    let max_iter = args.max_iter.unwrap_or(30);

    println!("running fitter ... ");
    let full_dims = [dims[0],dims[1],dims[2],n_echoes];
    let now = Instant::now();
    let (freq,std_err) = fit_freq(&echo_data, &full_dims, tol, max_iter);
    let dur = now.elapsed().as_secs_f64();
    println!("fitting completed in {} sec",dur);

    let vol_dims = ArrayDim::from_shape(&full_dims[0..3]);

    println!("writing outputs");

    let freq_filename = args.freq_prefix.as_ref().map(|s|s.as_str()).unwrap_or("freq");
    let std_err_filename = args.std_err_prefix.as_ref().map(|s|s.as_str()).unwrap_or("std_err");

    write_nrrd(args.output_dir.join(freq_filename), &freq, vol_dims, None, false, Encoding::raw);
    write_nrrd(args.output_dir.join(std_err_filename), &std_err, vol_dims, None, false, Encoding::raw);

    Ok(())
}

pub fn check_sizes(inputs:&[PathBuf]) -> (Vec<usize>,NRRD) {
    let mut dims:Option<Vec<usize>> = None;
    let mut nhdr:Option<NRRD> = None;
    for input in inputs {
        let nrrd = read_header(input);
        if dims.is_none() {
            dims = Some(nrrd.sizes.shape().to_vec());
            nhdr = Some(nrrd.clone());
        }
        if let Some(dims) = &dims {
            assert_eq!(dims,nrrd.sizes.shape())
        }
    }
    (dims.unwrap(),nhdr.unwrap())
}

fn load_complex_nhdr(nhdr:impl AsRef<Path>) -> (Vec<Complex32>, ArrayDim, NRRD) {
    // load the first volume to get dimensions
    let (data,dims,header) = io_nrrd::read_nrrd::<f32>(nhdr.as_ref());

    let dims_ = dims.shape_ns();
    if dims_[0] != 2 {
        panic!("expected first dim to be 2")
    }

    // convert to complex
    let cdata:Vec<_> = data.par_chunks_exact(2).map(|pair|{
        Complex32::new(pair[0],pair[1])
    }).collect();

    let dims = ArrayDim::from_shape(&dims.shape_ns()[1..]);

    (cdata,dims,header)

}