//! # fetch.rs  -- asset fetching from asset server
//
//  Animats
//  February, 2022
//
//  Loader for mesh and sculpt assets.
//  Called from threads in the asset load thread pool.
//
use ureq::{Agent, AgentBuilder,};
use std::time::{Duration};

/// Something has gone wrong if idle for this long.
const NETWORK_TIMEOUT: Duration = Duration::from_secs(15);  

/// Fetch asset from asset server.
/// Returns ureq::Error, so we can distinguish retryable errors.
pub fn fetch_asset(
    agent: &mut Agent,
    url: &str,
    byte_range_opt: Option<(u32, u32)>)
    -> Result<Vec<u8>, ureq::Error> {
    //  Build query, which may have a byte range specified.
    let query = if let Some(byte_range) = byte_range_opt {
        agent
            .get(&url)
            .set("Range", format!("bytes={}-{}",byte_range.0, byte_range.1).as_str())
    } else {
        agent
            .get(&url)
    };
    //  HTTP/HTTPS read.
    let resp = 
        query
            .call()?;
    let mut buffer = Vec::new();
    resp.into_reader().read_to_end(&mut buffer)?;
    Ok(buffer)
}

/// Build user agent for queries.
pub fn build_agent(user_agent: &str, max_connections: usize) -> Agent {
    const NETWORK_TIMEOUT: Duration = Duration::from_secs(15);  // something has gone wrong if this slow
    AgentBuilder::new()
        .user_agent(user_agent)
        .max_idle_connections_per_host(max_connections)     // we mostly hit the same host, so we want more idle connections available
        .timeout_connect(NETWORK_TIMEOUT)
        .timeout_read(NETWORK_TIMEOUT)
        .timeout_write(NETWORK_TIMEOUT)
        .build()
}

#[test]
fn test_fetch_asset() {
    const USER_AGENT: &str = "Test asset fetcher. Contact info@animats.com if problems.";
    const MAX_CONNECTIONS: usize = 1;       // don't overdo
    const URL1: &str = "http://www.example.com";    // something to read
    let mut agent = build_agent(USER_AGENT, MAX_CONNECTIONS);
    let result = fetch_asset(&mut agent, URL1, Some((0,200))); // first 200 bytes only
    match result {
        Ok(text) => {
            println!("Fetched {:?}", text);
        }
        Err(e) => panic!("Error: {:?}", e)
    }
}
