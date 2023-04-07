//! # jpeg2000-decoder.lib - library for fetching assets in JPEG 2000 format.
//
//  Primarily for Second Life/Open Simulator assets.
//
mod decode;
mod fetch;
mod pvqueue;
//  Exported symbols
pub use fetch::{fetch_asset};
pub use decode::{AssetError, ImageStats, FetchedImage};
pub use decode::{estimate_initial_read_size};
pub use pvqueue::{PvQueue, PvQueueLink};
