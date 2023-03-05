//! # jpeg2000-decoder  -- Decoder program for JPEG 2000 files.
//!
//  Animats
//  March, 2023
//
use anyhow::{anyhow, Error};
use image::DynamicImage;
use image::GenericImageView;
use jpeg2k_sandboxed::{DecodeImageRequest, DecodeParameters, J2KImage, Jpeg2kSandboxed};
use std::fs::File;
use std::io::BufReader;
use std::io::Read;

/// Arguments to the program
#[derive(Clone, Debug, Default)]
struct ArgInfo {
    /// Source file
    pub in_file: String,
    /// Destination file
    pub out_file: String,
    /// Maximum output image dimension, in pixels
    pub max_size: usize,
    /// Reduction factor
    pub reduction_factor: u8,
    /// Verbose mode. Goes to standard error if LLSD mode.
    pub verbose: bool,
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
        ap.refer(&mut arginfo.in_file)
            .add_option(&["-i", "--infile"], Store, "Input URL or file.");
        ap.refer(&mut arginfo.out_file)
            .add_option(&["-o", "--outfile"], Store, "Output file.");
        ap.refer(&mut arginfo.reduction_factor).add_option(
            &["-r", "--reduction"],
            Store,
            "Reduction factor.",
        );
        ap.refer(&mut arginfo.max_size).add_option(
            &["--maxsize"],
            Store,
            "Maximum dimension of output image",
        );
        ap.refer(&mut arginfo.verbose)
            .add_option(&["-v", "--verbose"], Store, "Verbose mode.");
        ap.parse_args_or_exit();
    }
    //  Check for required args
    if arginfo.in_file.is_empty() || arginfo.out_file.is_empty() {
        eprintln!("An input file and an output file must be specified.");
        std::process::exit(1);
    }
    arginfo
}

/// Decompress one file.
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
    let decode_parameters = DecodeParameters {
        reduce: reduction.into(),
        ..Default::default()
    };
    let req = DecodeImageRequest::new_with(contents, decode_parameters);
    let jp2_image: J2KImage = decoder.decode(&req)?;

    let img: DynamicImage = jp2_image
        .try_into()
        .map_err(|_| anyhow!("JPEG 2000 conversion error, no other data available"))?; // convert
    if verbose {
        println!("Input file {}", in_url);
        println!(
            "Output file {}: ({}, {})",
            out_file,
            img.width(),
            img.height()
        );
    }
    img.save(out_file)?; // save as PNG file
    Ok(())
}

/// Main program
fn main() {
    let args = parseargs();
    ////eprintln!("args: {:?}", args); // ***TEMP***
    let status = decompress_one_url(
        args.in_file.as_str(),
        args.out_file.as_str(),
        args.max_size,
        args.reduction_factor,
        args.verbose,
    );
    if let Err(e) = status {
        eprintln!("Decoder error: {:?}", e);
        std::process::exit(1);
    }
}
