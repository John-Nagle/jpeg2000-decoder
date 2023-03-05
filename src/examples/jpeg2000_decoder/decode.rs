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
use anyhow::{anyhow};
use crate::fetch::{build_agent, fetch_asset, err_is_retryable};
use jpeg2k_sandboxed::{Jpeg2kSandboxed, DecodeParameters, J2KImage, DecodeImageRequest};
use std::convert;
/*
use anyhow::{Error};
use jpeg2k::*;
use image::{DynamicImage};
use std::fs::File;
use std::io::Read;
use std::io::BufReader;
use image::GenericImageView;
*/

/// Things that can go wrong with an asset.
#[derive(Debug)]
pub enum AssetError {
    /// HTTP and network errors
    Http(ureq::Error),
    /// Decoder errors
    Jpeg(anyhow::Error),
    /// Content errors
    Content(String),
}

impl AssetError {
    /// Is this error retryable?
    pub fn is_retryable(&self) -> bool {
        match self {
            AssetError::Http(e) => err_is_retryable(e),
            AssetError::Jpeg(_) => false,
            AssetError::Content(_) => false,
        }
    }
}

//
//  Encapsulate errors from each of the lower level error types
//
impl convert::From<ureq::Error> for AssetError {
    fn from(err: ureq::Error) -> AssetError {
        AssetError::Http(err)
    }
}
/// The decoder currently returns an empty as an error type. 
impl convert::From<()> for AssetError {
    fn from(err: ()) -> AssetError {
        AssetError::Jpeg(anyhow!("JPEG decompression error, no other info available"))
    }
}

/// The decoder currently returns an empty as an error type. 
impl convert::From<anyhow::Error> for AssetError {
    fn from(err: anyhow::Error) -> AssetError {
        AssetError::Jpeg(err)
    }
}

/// Data about the image
#[derive(Debug)]
pub struct ImageStats {
    /// Bytes per pixel, rounded up from bits.
    bytes_per_pixel: u8,
    /// Original dimensions of image.
    dimensions: (u32, u32),
}


/// JPEG 2000 image currently being fetched.
//  Not saving beginning bytes for re-use because the decompressor consumes the input vector.
#[derive(Default)]
pub struct FetchedImage {
    /// Image as read, but not exported
    image_opt: Option<J2KImage>,
}

impl FetchedImage {
    /// Fetch image from server at indicated size.
    fn fetch(
        &mut self,
        agent: &ureq::Agent,
        decoder: &Jpeg2kSandboxed,
        url: &str,
        max_size_opt: Option<u32>,
    ) -> Result<(), AssetError> {
        if self.image_opt.is_none() {
            //  No previous info. Fetch with guess as to size.
            let bounds: Option<(u32, u32)> = if let Some(max_size) = max_size_opt {
                Some((0, estimate_initial_read_size(max_size))) // first guess
            } else {
                None
            };
            let decode_parameters = DecodeParameters::default(); // default decode, best effort
            let content = fetch_asset(agent, url, bounds)?; // fetch the asset
            let req = DecodeImageRequest::new_with(content, decode_parameters);
            let decode_result = decoder.decode(&req);
            log::debug!("Decompressed with no discard level. {}, file bounds {:?}, error: {}", url, bounds, decode_result.is_err());
            match decode_result {
                Ok(v) => self.image_opt = Some(v),
                Err(e) => return Err(e.into()),
            };
            self.sanity_check()                     // sanity check before decode
        } else {
            //  We have a previous image and can be more accurate.
            let stats = self.get_image_stats().unwrap();    // should alwasy get, we just tested for image presence.
            let (bounds, discard_level) = if let Some(max_size) = max_size_opt {
                let (max_bytes, discard_level) = estimate_read_size(stats.dimensions, stats.bytes_per_pixel, max_size);
                (Some((0, max_bytes)), discard_level)  // calc bounds to read
            } else {
                (None, 0)                                    // caller wants full size
            };
            //  Now fetch. Currently, from beginning, but we could optimize and reuse the first part.
            let content = fetch_asset(agent, url, bounds)?; // fetch the asset
            let decode_parameters = DecodeParameters { reduce: discard_level, .. Default::default() }; // default decode, best effort
            let req = DecodeImageRequest::new_with(content, decode_parameters);
            let decode_result = decoder.decode(&req);
            log::debug!("Decompressed texture {}, file bounds {:?}, discard level {}, error: {}", url, bounds, discard_level, decode_result.is_err());
            match decode_result {
                Ok(v) => self.image_opt = Some(v),
                Err(e) => return Err(e.into()),
            };
            self.sanity_check()                     // sanity check before decode
        }
    }
    
    /// Image sanity check. Size, precision, etc.
    fn sanity_check(&self) -> Result<(), AssetError> {
        if let Some(img) = &self.image_opt {
            if img.orig_width < 1 || img.orig_width > LARGEST_IMAGE_DIMENSION
            || img.orig_height < 1 || img.orig_height > LARGEST_IMAGE_DIMENSION {
                return Err(AssetError::Content(format!("Image dimensions ({},{}) out of range", img.orig_width, img.orig_height)));
            }
            if img.num_components < 1 || img.num_components > 4 {
                return Err(AssetError::Content(format!("Image component count {} of range", img.num_components)));
            }
            //  Can't check precision in the sandboxed version, because precision is not exported.
            Ok(())
        } else {
            Err(AssetError::Content(format!("Image not fetched")))
        }
    }
    
    /// Statistics about the image
    fn get_image_stats(&self) -> Option<ImageStats> {
        if let Some(img) = &self.image_opt {
            /*
            let mut bits_per_pixel = 0;
            for component in img.components().iter() {
                bits_per_pixel += component.precision()
            }
            */
            Some(ImageStats {
                dimensions: (img.orig_width, img.orig_height),
                ////bytes_per_pixel: ((bits_per_pixel + 7) / 8) as u8,
                bytes_per_pixel: img.num_components as u8, // assumes bit depth of 8
            })
        } else {
            None
        }
    }
}

/// Conservative estimate of how much JPEG 2000 reduces size
const JPEG_2000_COMPRESSION_FACTOR: f32 = 0.9;
/// Below 1024, JPEG 2000 files tend to break down. This is one packet with room for HTTP headers.
const MINIMUM_SIZE_TO_READ: u32 = 1024;
/// 8192 x 8192 should be a big enough texture for anyone
const LARGEST_IMAGE_DIMENSION: u32 = 8192;

/// Estimate amount of data to read for a desired resolution.
/// This should overestimate, so we read enough.
///
/// Returns (max bytes, discard level).
/// Discard level 0 is full size, 1 is 1/4 size, etc.
pub fn estimate_read_size(
    image_size: (u32, u32),
    bytes_per_pixel: u8,
    max_dim: u32,
) -> (u32, u32) {
    assert!(max_dim > 0); // would cause divide by zero
    let reduction_ratio = (image_size.0.max(image_size.1)) as u32 / (max_dim as u32);
    if reduction_ratio < 2 {
        return (u32::MAX, 0); // full size
    }
    //  Not full size, will be reducing.
    let in_pixels = image_size.0 * image_size.1;
    let out_pixels = in_pixels / (reduction_ratio * reduction_ratio); // number of pixels desired in output
    
       //  Read this many bytes and decode.
    let max_bytes = (((out_pixels as f32) * (bytes_per_pixel as f32)) * JPEG_2000_COMPRESSION_FACTOR) as u32;
    let max_bytes = max_bytes.max(MINIMUM_SIZE_TO_READ);
    //  Reduction ratio 1 -> discard level 0, 4->1, 16->2, etc. Round down.
    let discard_level = calc_discard_level(reduction_ratio); // ***SCALE***
    println!(
        "Reduction ratio: {}, discard level {}, bytes to read = {}",
        reduction_ratio, discard_level, max_bytes
    ); // ***TEMP***
    (max_bytes, discard_level)
}

///  Reduction ratio 1 -> discard level 0, 2->1, 3->2, etc. Round up. Just log2.
//  Yes, there is a cleverer way to do this by shifting and masking.
fn calc_discard_level(reduction_ratio: u32) -> u32 {
    assert!(reduction_ratio > 0);
    for i in 0..16 {
        if 2_u32.pow(i) as u32 >= reduction_ratio {
            return i.try_into().expect("calc discard level overflow");
        }
    }
    panic!("Argument to calc_discard_level is out of range.");
}

/// Estimate when we don't know what the image size is.
pub fn estimate_initial_read_size(max_dim: u32) -> u32 {
    const BYTES_PER_PIXEL: u8 = 4;  // worst case estimate
    let square = |x| x * x; // ought to be built in
    if max_dim > LARGEST_IMAGE_DIMENSION {
        // to avoid overflow
        u32::MAX // no limit
    } else {
        ((square(max_dim as f32) * BYTES_PER_PIXEL as f32 * JPEG_2000_COMPRESSION_FACTOR) as u32)
            .max(MINIMUM_SIZE_TO_READ)
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
    /// Assume RGBA, 8 bits   
    const BYTES_PER_PIXEL: u8 = 4;
    //  Don't know size of JPEG 2000 image.
    assert_eq!(estimate_initial_read_size(1), MINIMUM_SIZE_TO_READ);
    assert_eq!(estimate_initial_read_size(64), 14745); // given constant values above, 90% of output image area.
    assert_eq!(estimate_initial_read_size(32), MINIMUM_SIZE_TO_READ.max(3686)); // given constant values above, 90% of output image area.
                                                      //  Know size of JPEG 2000 image.
    assert_eq!(
        estimate_read_size((64, 64), BYTES_PER_PIXEL, 64),
        (u32::MAX, 0)
    );
    assert_eq!(estimate_read_size((64, 64), BYTES_PER_PIXEL, 32), (MINIMUM_SIZE_TO_READ.max(3686), 1)); // 2:1 reduction
    assert_eq!(
        estimate_read_size((512, 512), BYTES_PER_PIXEL, 32),
        (MINIMUM_SIZE_TO_READ.max(3686), 4)
    ); // 16:1 reduction, discard level 4
    assert_eq!(
        estimate_read_size((512, 512), BYTES_PER_PIXEL, 64),
        (14745, 3)
    ); // 8:1 reduction, discard level 3
    assert_eq!(
        estimate_read_size((512, 256), BYTES_PER_PIXEL, 64),
        (7372, 3)
    ); // 8:1 reduction, discard level 3
    assert_eq!(
        estimate_read_size((512, 256), BYTES_PER_PIXEL, 512),
        (u32::MAX, 0)
    ); // no reduction, full size.
}

#[test]
fn fetch_test_texture() {
    use crate::DynamicImage;
    use image::GenericImageView;
    const TEXTURE_DEFAULT: &str = "89556747-24cb-43ed-920b-47caed15465f"; // plywood in both Second Life and Open Simulator
    const TEXTURE_CAP: &str = "http://asset-cdn.glb.agni.lindenlab.com";
    const USER_AGENT: &str = "Test asset fetcher. Contact info@animats.com if problems.";
    const TEXTURE_OUT_SIZE: Option<u32> = Some(16);
    let decoder = Jpeg2kSandboxed::new().expect("Unable to create sandboxed decoder");
    let url = format!("{}/?texture_id={}", TEXTURE_CAP, TEXTURE_DEFAULT);
    println!("Asset url: {}", url);
    let agent = build_agent(USER_AGENT, 1);
    let mut image = FetchedImage::default();
    image.fetch(&agent, &decoder, &url, TEXTURE_OUT_SIZE).expect("Fetch failed");
    assert!(image.image_opt.is_some()); // got image
    println!("Image stats: {:?}", image.get_image_stats());
    let img: DynamicImage = image.image_opt.unwrap()
        .try_into()
        .expect("Conversion failed"); // convert

    let out_file = "/tmp/testimg.png"; // Linux only
    println!(
        "Output file {}: ({}, {})",
        out_file,
        img.width(),
        img.height()
    );
    img.save(out_file).expect("File save failed"); // save as PNG file
}

#[test]
fn fetch_multiple_textures_serial() {
    use crate::DynamicImage;
    use image::GenericImageView;
    use std::io::BufRead;
    ////const TEST_UUIDS: &str = "samples/smalluuidlist.txt"; // test of UUIDs, relative to manifest dir
    const TEST_UUIDS: &str = "samples/bugislanduuidlist.txt"; // test of UUIDs at Bug Island, some of which have problems.
    const USER_AGENT: &str = "Test asset fetcher. Contact info@animats.com if problems.";
    fn fetch_test_texture(agent: &ureq::Agent, decoder: &Jpeg2kSandboxed, uuid: &str, max_size: u32) {
        const TEXTURE_CAP: &str = "http://asset-cdn.glb.agni.lindenlab.com";
        ////const TEXTURE_OUT_SIZE: Option<u32> = Some(2048);
        let url = format!("{}/?texture_id={}", TEXTURE_CAP, uuid);
        println!("Asset url: {}", url);
        let now = std::time::Instant::now();
        let mut image = FetchedImage::default();
        // First fetch
        image.fetch(&agent, decoder, &url, Some(16)).expect("Fetch failed");
        let fetch_time = now.elapsed();
        let now = std::time::Instant::now();
        assert!(image.image_opt.is_some()); // got image
        println!("Image stats: {:?}", image.get_image_stats());
        //  Second fetch, now that we have header info
        image.fetch(&agent, decoder, &url, Some(max_size)).expect("Fetch failed");
        let img: DynamicImage = image.image_opt.unwrap()
            .try_into()
            .expect("Conversion failed"); // convert
        let decode_time = now.elapsed();
        let now = std::time::Instant::now();

        let out_file = format!("/tmp/TEST-{}.png", uuid); // Linux only
        println!(
            "Output file {}: ({}, {})",
            out_file,
            img.width(),
            img.height()
        );
        img.save(out_file).expect("File save failed"); // save as PNG file
        let save_time = now.elapsed();
        println!("File {} fetch: {:#?}, decode {:#?}: save: {:#?}", uuid, fetch_time.as_secs_f32(), decode_time.as_secs_f32(), save_time.as_secs_f32());
    }
    println!("---Fetch multiple textures serial start---");
    let decoder = Jpeg2kSandboxed::new().expect("Unable to create sandboxed decoder");
    //  Try all the files in the list
    let basedir = env!["CARGO_MANIFEST_DIR"];           // where the manifest is
    let file = std::fs::File::open(format!("{}/{}", basedir, TEST_UUIDS)).expect("Unable to open file of test UUIDs");
    let reader = std::io::BufReader::new(file);
    const TEXTURE_OUT_SIZE: u32 = 128;
    let agent = build_agent(USER_AGENT, 1);
    for line in reader.lines() { 
        let line = line.expect("Error reading UUID file");
        let line = line.trim();
        if line.is_empty() { continue }
        if line.starts_with('#') { continue }
        println!("{}", line);
        fetch_test_texture(&agent, &decoder, line, TEXTURE_OUT_SIZE);
    }
}
