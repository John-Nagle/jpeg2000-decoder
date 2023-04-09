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

use crate::fetch::{build_agent, fetch_asset, err_is_retryable};
use crate::{PvQueue, PvQueueLink};
use image::DynamicImage;
use image::GenericImageView;
use jpeg2k::DecodeParameters;
use anyhow::{anyhow, Error};
use std::convert;

/// Things that can go wrong with an asset.
#[derive(Debug)]
pub enum AssetError {
    /// HTTP and network errors
    Http(ureq::Error),
    /// Decoder errors
    Jpeg(jpeg2k::error::Error),
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
impl convert::From<jpeg2k::error::Error> for AssetError {
    fn from(err: jpeg2k::error::Error) -> AssetError {
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
#[derive(Default)]
pub struct FetchedImage {
    /// First bytes of the input file, if previously fetched.
    beginning_bytes: Vec<u8>,
    /// Image as read, but not exported
    image_opt: Option<jpeg2k::Image>,
}

impl FetchedImage {

    /// Fetch texture image from server at requested size.
    ///
    /// 1. Fetch header and small size.
    /// 2. Fetch at requested size using truncated file.
    /// 3. Re-fetch at full size if truncated file fails to decode.
    pub fn fetch(
        &mut self,
        agent: &ureq::Agent,
        url: &str,
        max_size_opt: Option<u32>,
        bottleneck_opt: Option<&PvQueueLink>,
    ) -> Result<(), AssetError> {
        const INITIAL_FETCH_SIZE: u32 = 16;
        // First fetch, at 16x16 pixels, to get size info.
        self.fetch_and_decode_single(&agent, &url, Some(INITIAL_FETCH_SIZE), bottleneck_opt)?;
        assert!(self.image_opt.is_some()); // got image
        if let Some(max_size) = max_size_opt {
            if max_size <= INITIAL_FETCH_SIZE {
                return Ok(())      // already big enough
            }
        }
        if self.image_opt.as_ref().unwrap().orig_width() <= INITIAL_FETCH_SIZE 
        && self.image_opt.as_ref().unwrap().orig_height() <= INITIAL_FETCH_SIZE {
            return Ok(());                         // as big as it will get
        }
        //  Second fetch, now that we have header info.
        self.fetch_and_decode_single(&agent, &url, max_size_opt, bottleneck_opt)
    }

    /// Fetch image from server at indicated size.
    fn fetch_and_decode_single(
        &mut self,
        agent: &ureq::Agent,
        url: &str,
        max_size_opt: Option<u32>,
        bottleneck_opt: Option<&PvQueueLink>,
    ) -> Result<(), AssetError> {
        if self.image_opt.is_none() {
            //  No previous info. Fetch with guess as to size.
            let bounds: Option<(usize, usize)> = if let Some(max_size) = max_size_opt {
                Some((0, estimate_initial_read_size(max_size))) // first guess
            } else {
                None
            };
            ////println!("Bounds: {:?}", bounds); // ***TEMP***
            let decode_parameters = DecodeParameters::new(); // default decode, best effort
            self.beginning_bytes = fetch_asset(agent, url, bounds)?; // fetch the asset
            let decode_result = {
                //  Bottleneck the decode operation.
                let _lok = if let Some(bottleneck) = bottleneck_opt {
                    Some(PvQueue::lock(bottleneck))
                } else {
                    None
                };
                profiling::scope!("J2K decode");
                jpeg2k::Image::from_bytes_with(&self.beginning_bytes, decode_parameters)
            };
            match decode_result {
                Ok(v) => self.image_opt = Some(v),
                Err(e) => return Err(e.into()),
            };
            ////self.image_opt = Some(jpeg2k::Image::from_bytes_with(&self.beginning_bytes, decode_parameters).map_err(into)?);
            self.sanity_check()                     // sanity check before decode
        } else {
            //  We have a previous image and can be more accurate.
            let stats = self.get_image_stats().unwrap();    // should always get, we just tested for image presence.
            let discard_level = if let Some(max_size) = max_size_opt {
                let (max_bytes, discard_level) = estimate_read_size(stats.dimensions, stats.bytes_per_pixel, max_size);
                //  Disable partial reading - it's not working reliably. It only saves us reading a few hundred bytes, anyway.
                ////self.beginning_bytes.append(&mut fetch_asset(agent, url, Some((self.beginning_bytes.len(), max_bytes as usize)))?); // fetch the rest of the asset
                self.beginning_bytes = fetch_asset(agent, url, Some((0, max_bytes as usize)))?; // fetch the entire asset, beginning to end.
                discard_level // calc bounds to read
            } else {
                self.beginning_bytes = fetch_asset(agent, url, None)?; // fetch the entire asset, beginning to end.
                0                                   // caller wants full size
            };
            //  Now fetch. Currently, from beginning, but we could optimize and reuse the first part.
            ////self.beginning_bytes = fetch_asset(agent, url, bounds)?; // fetch the asset
            let decode_parameters = DecodeParameters::new().reduce(discard_level); // decoded to indicated level
            //  Decoder bottleneck - avoid too many simultaneous CPU-bound tasks
            let decode_result = {
                let _lok = if let Some(bottleneck) = bottleneck_opt {
                    Some(PvQueue::lock(bottleneck))
                } else {
                    None
                };
                profiling::scope!("J2K decode");
                jpeg2k::Image::from_bytes_with(&self.beginning_bytes, decode_parameters)
            };
            match decode_result {
                Ok(v) => self.image_opt = Some(v),
                Err(e) => return Err(e.into()),
            };
            self.sanity_check()                     // sanity check after decode
        }
    }
    
    /// Get decoded image
    pub fn get_dynamic_image(&self, bottleneck_opt: Option<&PvQueueLink>) -> Result<DynamicImage, Error> {
        //  Apply concurrency bottleneck
        let _lok = if let Some(bottleneck) = bottleneck_opt {
            Some(PvQueue::lock(bottleneck))
        } else {
            None
        };
        profiling::scope!("J2K to image");
        if let Some(image) = &self.image_opt {
            image.try_into().map_err(|e| anyhow!("Error converting JPEG 2000 image to final output: {:?}", e))
        } else {
            Err(anyhow!("No image was decoded."))   // error by caller, should not have called
        }
    }
    
    /// Image sanity check. Size, precision, etc.
    fn sanity_check(&self) -> Result<(), AssetError> {
        if let Some(img) = &self.image_opt {
            if img.orig_width() < 1 || img.orig_width() > LARGEST_IMAGE_DIMENSION
            || img.orig_height() < 1 || img.orig_height() > LARGEST_IMAGE_DIMENSION {
                return Err(AssetError::Content(format!("Image dimensions ({},{}) out of range", img.orig_width(), img.orig_height())));
            }
            if img.components().is_empty() || img.components().len() > 4 {
                return Err(AssetError::Content(format!("Image component count {} of range", img.components().len())));
            }
            for component in img.components().iter() {
                //  Component precision is in bits
                if component.precision() < 1 || component.precision() > 16 {
                    return Err(AssetError::Content(format!("Image component precision {} of range", component.precision())));
                }
            }                
            Ok(())
        } else {
            Err(AssetError::Content(format!("Image not fetched")))
        }
    }
    
    /// Statistics about the image
    fn get_image_stats(&self) -> Option<ImageStats> {
        if let Some(img) = &self.image_opt {
            let mut bits_per_pixel = 0;
            for component in img.components().iter() {
                bits_per_pixel += component.precision()
            }
            Some(ImageStats {
                dimensions: (img.orig_width(), img.orig_height()),
                bytes_per_pixel: ((bits_per_pixel + 7) / 8) as u8,
            })
        } else {
            None
        }
    }
}

/// Conservative estimate of how much JPEG 2000 reduces size
const JPEG_2000_COMPRESSION_FACTOR: f32 = 0.9;

/// Below 1024, JPEG 2000 files tend to break down. This is one packet with room for HTTP headers.
const MINIMUM_SIZE_TO_READ: usize = 1024;
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
) -> (usize, u32) {
    assert!(max_dim > 0); // would cause divide by zero
    let reduction_ratio = (image_size.0.max(image_size.1)) as u32 / (max_dim as u32);
    if reduction_ratio < 2 {
        return (usize::MAX, 0); // full size
    }
    //  Not full size, will be reducing.
    let in_pixels = image_size.0 * image_size.1;
    let out_pixels = in_pixels / (reduction_ratio * reduction_ratio); // number of pixels desired in output
    
       //  Read this many bytes and decode.
    let max_bytes = (((out_pixels as f32) * (bytes_per_pixel as f32)) * JPEG_2000_COMPRESSION_FACTOR) as usize;
    let max_bytes = max_bytes.max(MINIMUM_SIZE_TO_READ);
    //  Reduction ratio 1 -> discard level 0, 4->1, 16->2, etc. Round down.
    let discard_level = calc_discard_level(reduction_ratio); // ***SCALE***
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
pub fn estimate_initial_read_size(max_dim: u32) -> usize {
    const BYTES_PER_PIXEL: u8 = 4;  // worst case estimate
    let square = |x| x * x; // ought to be built in
    if max_dim > LARGEST_IMAGE_DIMENSION {
        // to avoid overflow
        usize::MAX // no limit
    } else {
        ((square(max_dim as f32) * BYTES_PER_PIXEL as f32 * JPEG_2000_COMPRESSION_FACTOR) as usize)
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
        (usize::MAX, 0)
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
        (usize::MAX, 0)
    ); // no reduction, full size.
}

#[test]
fn fetch_test_texture() {
    use image::DynamicImage;
    use image::GenericImageView;
    const TEXTURE_DEFAULT: &str = "89556747-24cb-43ed-920b-47caed15465f"; // plywood in both Second Life and Open Simulator
    const TEXTURE_CAP: &str = "http://asset-cdn.glb.agni.lindenlab.com";
    const USER_AGENT: &str = "Test asset fetcher. Contact info@animats.com if problems.";
    const TEXTURE_OUT_SIZE: Option<u32> = Some(16);
    let url = format!("{}/?texture_id={}", TEXTURE_CAP, TEXTURE_DEFAULT);
    println!("Asset url: {}", url);
    let agent = build_agent(USER_AGENT, 1);
    let mut image = FetchedImage::default();
    image.fetch_and_decode_single(&agent, &url, TEXTURE_OUT_SIZE, None).expect("Fetch failed");
    assert!(image.image_opt.is_some()); // got image
    println!("Image stats: {:?}", image.get_image_stats());
    let img: DynamicImage = (&image.image_opt.unwrap())
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
    use image::DynamicImage;
    use image::GenericImageView;
    use std::io::BufRead;
    ////const TEST_UUIDS: &str = "samples/smalluuidlist.txt"; // test of UUIDs, relative to manifest dir
    const TEST_UUIDS: &str = "samples/bugislanduuidlist.txt"; // test of UUIDs at Bug Island, some of which have problems.
    const USER_AGENT: &str = "Test asset fetcher. Contact info@animats.com if problems.";
    fn fetch_test_texture(agent: &ureq::Agent, uuid: &str, max_size: u32) {
        const TEXTURE_CAP: &str = "http://asset-cdn.glb.agni.lindenlab.com";
        ////const TEXTURE_OUT_SIZE: Option<u32> = Some(2048);
        let url = format!("{}/?texture_id={}", TEXTURE_CAP, uuid);
        println!("Asset url: {}", url);
        let now = std::time::Instant::now();
        let mut image = FetchedImage::default();
        // First fetch
        image.fetch_and_decode_single(&agent, &url, Some(16), None).expect("Small fetch failed");
        let fetch_time = now.elapsed();
        let now = std::time::Instant::now();
        assert!(image.image_opt.is_some()); // got image
        println!("Image stats: {:?}", image.get_image_stats());
        //  Second fetch, now that we have header info
        image.fetch_and_decode_single(&agent, &url, Some(max_size), None).expect("Full sized fetch failed");
        let img: DynamicImage = (&image.image_opt.unwrap())
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
        fetch_test_texture(&agent, line, TEXTURE_OUT_SIZE);
    }
}

#[test]
fn fetch_multiple_textures_parallel() {
    use image::DynamicImage;
    use image::GenericImageView;
    use std::io::BufRead;
    use core::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use anyhow::{anyhow, Error};
    use crate::{PvQueue, PvQueueLink};
    simple_logger::init_with_level(log::Level::Error).unwrap();    // so logging shows up
    log::error!("Errors go to standard output.");
    ////const TEST_UUIDS: &str = "samples/bugislanduuidlist.txt"; // test of UUIDs at Bug Island, some of which have problems.
    const TEST_UUIDS: &str = "samples/biguuidlist.txt"; // Larger 44K list of non-public UUIDs.
    const USER_AGENT: &str = "Test asset fetcher. Contact info@animats.com if problems.";
    fn fetch_test_texture(agent: &ureq::Agent, uuid: &str, max_size: u32, bottleneck: &PvQueueLink) -> Result<(), Error> {
        const TEXTURE_CAP: &str = "http://asset-cdn.glb.agni.lindenlab.com";
        let url = format!("{}/?texture_id={}", TEXTURE_CAP, uuid);
        let mut image = FetchedImage::default();
        let stat = image.fetch(&agent, &url, Some(max_size), Some(bottleneck));
        if let Err(e) = stat {
            println!("Fetch error for url {}: {:?}", uuid, e);
            match e {
                AssetError::Http(ureq::Error::Status(http_status,_)) => {                   
                    if http_status == 404 {
                        return Ok(())   // ignore file not found problem
                    }
                }
                _ => {}
            }
            return Err(anyhow!("Fetch error for url {}: {:?}", uuid, e));
        }
        let _lok = PvQueue::lock(bottleneck);   // bottleneck the saving part
        let now = std::time::Instant::now();
        let img: DynamicImage = (&image.image_opt.unwrap())
            .try_into()
            .expect("Conversion failed"); // convert
        let out_file = format!("/tmp/TEST-{}.png", uuid); // Linux only
        img.save(out_file).expect("File save failed"); // save as PNG file
        Ok(())
    }
    println!("---Fetch multiple textures parallel start---");
    //  Try all the files in the list
    use crossbeam_channel::unbounded;
    let basedir = env!["CARGO_MANIFEST_DIR"];           // where the manifest is
    let file = std::fs::File::open(format!("{}/{}", basedir, TEST_UUIDS)).expect("Unable to open file of test UUIDs");
    let reader = std::io::BufReader::new(file);
    const TEXTURE_OUT_SIZE: u32 = 512;
    const WORKERS: usize = 18;  // push hard here
    const BOTTLENECK_COUNT: u32 = 6;            // no more than this many at one time in compute-bound decode
    let bottleneck = PvQueue::new(BOTTLENECK_COUNT);
    let agent = build_agent(USER_AGENT, 1);
    let receiver = {
        let (sender,receiver) = unbounded();
        for read_result in reader.lines() { 
            let line = read_result.expect("Error reading UUID file");
            let line = line.trim().to_string();
            if line.is_empty() { continue }
            if line.starts_with('#') { continue }
            sender.send(line.clone());
        }
        receiver                                        // drop sender, for EOF
    };
    //  Start worker threads.
    let mut workers = Vec::new();
    
    
    let fail = Arc::new(AtomicBool::new(false));
    println!("Starting {} worker threads to decompress {} files.", WORKERS, receiver.len());
    for n in 0..WORKERS {
        let agent_clone = agent.clone();
        let receiver_clone = receiver.clone();
        let fail_clone = Arc::clone(&fail);
        let bottleneck_clone = Arc::clone(&bottleneck);
        let worker = std::thread::spawn(move || {
            println!("Thread {} starting.", n);
            let mut cnt: usize = 0;
            thread_priority::set_current_thread_priority(thread_priority::ThreadPriority::Min).unwrap();
            while let Ok(item) = receiver_clone.recv() {
                if fail_clone.load(Ordering::Relaxed) { break; }
                if let Err(e) = fetch_test_texture(&agent_clone, &item, TEXTURE_OUT_SIZE, &bottleneck_clone) {
                    println!("Thread {} error: {:?}", n, e);
                    fail_clone.store(true, Ordering::Relaxed); // note fail
                }
                cnt += 1;   // tally
            }
            println!("Thread {} done. {} images processed.", n, cnt);
        });
        workers.push(worker);
    }
    println!("Started {} worker threads.", workers.len());
    for worker in workers { 
        println!("Waiting for threads to finish.");
        worker.join();
    }
    if fail.load(std::sync::atomic::Ordering::Relaxed) { panic!("A decode or fetch failed."); }
    println!("Done.");
}
