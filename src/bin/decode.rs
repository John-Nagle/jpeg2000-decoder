//! # decode.rs  -- Decoder tools for JPEG 2000 files.
//!
//  Animats
//  February, 2023
//
//! ## Information stored in the header of a JPEG 2000 file
//!
//! Dump from a sample image, using jpeg2k's dump program:
//!    Image { x_offset: 0, y_offset: 0, width: 768, height: 512, color_space: SRGB, numcomps: 3, comps: [
//!    ImageComponent { dx: 1, dy: 1, w: 768, h: 512, x0: 0, y0: 0, prec: 8, bpp: 0, sgnd: 0, resno_decoded: 5, factor: 0, data: 0x7fd1554eb010,  alpha: 0 }, 
//!    ImageComponent { dx: 1, dy: 1, w: 768, h: 512, x0: 0, y0: 0, prec: 8, bpp: 0, sgnd: 0, resno_decoded: 5, factor: 0, data: 0x7fd15536a010, alpha: 0 }, 
//!    ImageComponent { dx: 1, dy: 1, w: 768, h: 512, x0: 0, y0: 0, prec: 8, bpp: 0, sgnd: 0, resno_decoded: 5, factor: 0, data: 0x7fd1551e9010, alpha: 0 }] }
//!
//! So this is a 3-component image, RGB (not RGBA).
//! * prec -- bits per pixel per component.
//! * bpp -- not used, deprecated. Ref: https://github.com/uclouvain/openjpeg/pull/1383
//! * resno_decoded -- Not clear, should be the number of discard levels available.

use jpeg2k::{DecodeParameters};
use fetch::{fetch_asset};
/*
use anyhow::{Error};
use jpeg2k::*;
use image::{DynamicImage};
use std::fs::File;
use std::io::Read;
use std::io::BufReader;
use image::GenericImageView;
*/

/// JPEG 2000 image currently being fetched.
#[derive(Default)]
struct FetchedImage {
    /// First bytes of the input file, if previously fetched.
    beginning_bytes: Vec<u8>,
    /// Image as read, but not exported
    image_opt: Option<jpeg2k::Image>,
}
          
impl FetchedImage {
    /// Fetch image from server at indicated size.
    pub fn fetch(&mut self, agent: &ureq::Agent, url: &str, max_size_opt: Option<u32>) -> Result<(), ureq::Error>  {
        if self.image_opt.is_none() {
            let bounds: Option<(u32, u32)> = if let Some(max_size) = max_size_opt {
                Some((0, estimate_initial_read_size(max_size)))    // first guess
            } else {
                None
            };
            let decode_parameters = DecodeParameters::new();
            self.beginning_bytes = fetch_asset(agent, url, bounds)?;           // fetch the asset           
            self.image_opt = Some(jpeg2k::Image::from_bytes_with(&self.beginning_bytes, decode_parameters)?);
            Ok(())     
        } else {
            todo!();    // ***MORE***
        }
    }
}
    

/// Maximum image dimension allowed
const JPEG_2000_COMPRESSION_FACTOR: f32 = 0.9;    // conservative estimate of how much JPEG 2000 reduces size
const BYTES_PER_PIXEL: u32 = 4;             // assume RGBA, 8 bits
const MINIMUM_SIZE_TO_READ: u32 = 4096;   // smaller than this and JPEG 2000 files break down.
const LARGEST_IMAGE_DIMENSION: u32 = 8192;  // 8192 x 8192 should be big enough for anyone
/// Estimate amount of data to read for a desired resolution.
/// This should overestimate, so we read enough.
///
/// Returns (max bytes, discard level).
/// Discard level 0 is full size, 1 is 1/4 size, etc. 
pub fn estimate_read_size(image_size: (u32, u32), bytes_per_pixel: u32, max_dim: u32) -> (u32, u32) {
    assert!(max_dim > 0);       // would cause divide by zero
    let reduction_ratio = (image_size.0.max(image_size.1)) as u32 / (max_dim as u32);
    if reduction_ratio < 2 {
        return (u32::MAX, 0)      // full size
    }
    //  Not full size, will be reducing.
    let in_pixels = image_size.0 * image_size.1;	
    let out_pixels = in_pixels / (reduction_ratio * reduction_ratio);   // number of pixels desired in output
    println!("Reduction ratio: {}, out pixels = {}", reduction_ratio, out_pixels);    // ***TEMP***
    //  Read this many bytes and decode.
    let max_bytes = (((out_pixels * bytes_per_pixel) as f32) * JPEG_2000_COMPRESSION_FACTOR) as u32;
    let max_bytes = max_bytes.max(MINIMUM_SIZE_TO_READ);
    //  Reduction ratio 1 -> discard level 0, 4->1, 16->2, etc. Round down.
    let discard_level = calc_discard_level(reduction_ratio);  // ***SCALE***
    (max_bytes, discard_level)
}
	
///  Reduction ratio 1 -> discard level 0, 2->1, 3->2, etc. Round up. Just log2.
//  Yes, there is a cleverer way to do this by shifting and masking.
fn calc_discard_level(reduction_ratio: u32) -> u32 {
    assert!(reduction_ratio > 0);
    for i in 0..16 {
        if 2_u32.pow(i) as u32 	>= reduction_ratio {
            return i.try_into().expect("calc discard level overflow")
        }
    }
    panic!("Argument to calc_discard_level is out of range.");
}

/// Estimate when we don't know what the image size is.
pub fn estimate_initial_read_size(max_dim: u32) -> u32 {
    let square = |x| x*x;            // ought to be built in
    if max_dim > LARGEST_IMAGE_DIMENSION {      // to avoid overflow
        u32::MAX                    // no limit
    } else {
        ((square(max_dim as f32) * BYTES_PER_PIXEL as f32 * JPEG_2000_COMPRESSION_FACTOR) as u32).max(MINIMUM_SIZE_TO_READ)
    }
}

#[test]
/// Sanity check on estimator math
fn test_calc_discard_level() {
    assert_eq!(calc_discard_level(1), 0);
    assert_eq!(calc_discard_level(2), 1);
    assert_eq!(calc_discard_level(3), 2);
    assert_eq!(calc_discard_level(4), 2);
    assert_eq!(calc_discard_level(5), 3);
    assert_eq!(calc_discard_level(8), 3);
    assert_eq!(calc_discard_level(16), 4);
    assert_eq!(calc_discard_level(17), 5);
    assert_eq!(calc_discard_level(63), 6);
    assert_eq!(calc_discard_level(64), 6);
    assert_eq!(calc_discard_level(65), 7);
}
#[test]
/// Sanity check on estimator math.
/// These assume the values of the constants above.
fn test_estimate_read_size() {
    //  Don't know size of JPEG 2000 image.
    assert_eq!(estimate_initial_read_size(1), MINIMUM_SIZE_TO_READ);
    assert_eq!(estimate_initial_read_size(64), 14745);  // given constant values above, 90% of output image area.
    assert_eq!(estimate_initial_read_size(32), 4096);  // given constant values above, 90% of output image area.
    //  Know size of JPEG 2000 image.
    assert_eq!(estimate_read_size((64,64), BYTES_PER_PIXEL, 64), (u32::MAX, 0));
    assert_eq!(estimate_read_size((64,64), BYTES_PER_PIXEL, 32), (4096, 1));  // 2:1 reduction
    assert_eq!(estimate_read_size((512,512), BYTES_PER_PIXEL, 32), (4096, 4)); // 16:1 reduction, discard level 4
    assert_eq!(estimate_read_size((512,512), BYTES_PER_PIXEL, 64), (14745, 3)); // 8:1 reduction, discard level 3
    assert_eq!(estimate_read_size((512,256), BYTES_PER_PIXEL, 64), (7372, 3)); // 8:1 reduction, discard level 3
    assert_eq!(estimate_read_size((512,256), BYTES_PER_PIXEL, 512), (u32::MAX, 0)); // no reduction, full size.
}
