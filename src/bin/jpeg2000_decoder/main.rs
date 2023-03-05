//! # jpeg2000-decoder  -- Decoder program for JPEG 2000 files.
//!
//  Animats
//  March, 2023
//
use anyhow::{anyhow, Error};
use image::DynamicImage;
use image::GenericImageView;
use std::fs::File;
use std::io::BufReader;
use std::io::Read;
use jpeg2k_sandboxed::{Jpeg2kSandboxed, DecodeParameters, J2KImage, DecodeImageRequest};

////mod decode;
pub mod fetch;
////use decode::estimate_initial_read_size;

/// Arguments to the program
#[derive(Clone, Debug, Default)]
struct ArgInfo {
    /// Source URL
    pub in_url: String,
    /// Destination file
    pub out_file: String,
    /// Maximum output image dimension, in pixels
    pub max_size: usize,
    /// Reduction factor
    pub reduction_factor: u8,
    /// If true, ignore above fields and read LLSD commands from input.
    pub llsd_mode: bool,
    /// Verbose mode. Goes to standard error if LLSD mode.
    pub verbose: bool,
    /// User agent for HTTP requests.
    pub user_agent: String,
}

//
//  parseargs -- parse command line args
//
//  Sets options, returns file to process
//
fn parseargs() -> ArgInfo {
    let mut arginfo = ArgInfo {
        max_size: 1000000000,
        ..Default::default()
    };
    {
        //  This block limits scope of borrows by ap.refer() method
        use argparse::{ArgumentParser, Store}; // only visible here
        let mut ap = ArgumentParser::new();
        ap.set_description("Decoder for JPEG 2000 files.");
        ap.refer(&mut arginfo.in_url)
            .add_option(&["-i", "--infile"], Store, "Input URL or file.");
        ap.refer(&mut arginfo.out_file)
            .add_option(&["-o", "--outfile"], Store, "Output file.");
        ap.refer(&mut arginfo.reduction_factor)
            .add_option(&["-r", "--reduction"], Store, "Reduction factor.");
        ap.refer(&mut arginfo.max_size).add_option(
            &["--maxsize"],
            Store,
            "Maximum dimension of output image",
        );
        ap.refer(&mut arginfo.verbose)
            .add_option(&["-v", "--verbose"], Store, "Verbose mode.");
        ap.refer(&mut arginfo.llsd_mode)
            .add_option(&["--llsd"], Store, "LLSD mode");
        ap.parse_args_or_exit();
    }
    //  Check for required args
    if !arginfo.llsd_mode {
        if arginfo.in_url.is_empty() || arginfo.out_file.is_empty() {
            eprintln!("If LLSD mode is off, an input URL and an output file must be specified");
            std::process::exit(1);
        }
    }
    arginfo
}

/// LLSD mode
fn run_llsd_mode(verbose: bool) -> Result<(), Error> {
    todo!()
}

/// Decompress one URL or file mode.
fn decompress_one_url(
    in_url: &str,
    out_file: &str,
    max_size: usize,
    reduction: u8,
    verbose: bool,
) -> Result<(), Error> {
    // Initial dumb version.
    let file_bytes_guess = max_size * max_size * 4 + 200; // guess file size needed.
    let in_file = File::open(in_url)?;
    let mut buf_reader = BufReader::new(in_file);
    let mut contents = Vec::new();
    buf_reader.read_to_end(&mut contents)?;
    let contents = if contents.len() > file_bytes_guess {
        println!(
            "Truncating file from {} bytes to {} bytes",
            contents.len(),
            file_bytes_guess
        );
        contents[0..file_bytes_guess].to_vec()
    } else {
        contents
    };
    let decoder = Jpeg2kSandboxed::new().expect("Unable to create sandboxed decoder");
    ////let decode_parameters = DecodeParameters::new().reduce(reduction.into());
    let decode_parameters = DecodeParameters { reduce: reduction.into(), .. Default::default() };
    let req = DecodeImageRequest::new_with(contents, decode_parameters);
    let jp2_image: J2KImage = decoder.decode(&req)?;
    ////println!("Decode parameters: {:?}", decode_parameters);
    ////let jp2_image = J2KImage::from_bytes_with(&contents, decode_parameters)?;
    println!("Input file {}: {:?}", in_url, jp2_image);
    ////let jp2_image = Image::from_file(in_url)?; // load from file (not URL)
    /*
        //  ***TEMP*** timing test - result is about 30ms per image.
        let now = std::time::Instant::now();
        const TRIES: usize = 1000;
        for _ in 0..1000 {
            let decode_parameters = DecodeParameters::new();
            let jp2_image = Image::from_bytes_with(&contents, decode_parameters)?;
            let img: DynamicImage = (&jp2_image).try_into()?;  // convert
        }
        let elapsed = now.elapsed().as_secs_f32() / (TRIES as f32);
        println!("Decompression time: {} secs.", elapsed);
    */

    let img: DynamicImage = jp2_image.try_into().map_err(|_| anyhow!("JPEG 2000 conversion error, no other data available"))?; // convert
    println!(
        "Output file {}: ({}, {})",
        out_file,
        img.width(),
        img.height()
    );
    img.save(out_file)?; // save as PNG file
    Ok(())
}

/// Main program
fn main() {
    let args = parseargs();
    eprintln!("args: {:?}", args); // ***TEMP***
    let status = if args.llsd_mode {
        run_llsd_mode(args.verbose)
    } else {
        decompress_one_url(
            args.in_url.as_str(),
            args.out_file.as_str(),           
            args.max_size,
            args.reduction_factor,
            args.verbose,
        )
    };
    if let Err(e) = status {
        eprintln!("Error: {:?}", e);
        std::process::exit(1);
    }
}
