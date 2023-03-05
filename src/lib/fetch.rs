//! # fetch.rs  -- asset fetching from asset server
//
//  Animats
//  February, 2022
//
//  Loader for mesh and sculpt assets.
//  Called from threads in the asset load thread pool.
//
use std::time::Duration;
use ureq::{Agent, AgentBuilder};

/// Something has gone wrong if idle for this long.
const NETWORK_TIMEOUT: Duration = Duration::from_secs(15);

/// Is this HTTP fetch error retryable?
pub fn err_is_retryable(e: &ureq::Error) -> bool {
    match e {
        ureq::Error::Transport(_) => true, // always retry network errors
        ureq::Error::Status(status_code, _) => {
            match status_code {
                400 => false, // bad request
                401 => false, // forbidden
                402 => false, // payment required
                403 => false, // forbidden
                404 => false, // File not found, do not retry.
                405 => false, // method not allowed
                _ => true,    // retry everything else
            }
        }
    }        
}

/// Fetch asset from asset server.
/// Returns ureq::Error, so we can distinguish retryable errors.
fn fetch_asset_once(
    agent: &Agent,
    url: &str,
    byte_range_opt: Option<(u32, u32)>,
) -> Result<Vec<u8>, ureq::Error> {
    //  Build query, which may have a byte range specified.
    let query = if let Some(byte_range) = byte_range_opt {
        agent.get(&url).set(
            "Range",
            format!("bytes={}-{}", byte_range.0, byte_range.1).as_str(),
        )
    } else {
        agent.get(&url)
    };
    //  HTTP/HTTPS read.
    let resp = query.call()?;
    let mut buffer = Vec::new();
    resp.into_reader().read_to_end(&mut buffer)?;
    Ok(buffer)
}

/// Fetch asset from asset server, with retries
/// Returns ureq::Error, so we can distinguish retryable errors.
//  This should log retries, but we currently have no way to report them.
pub fn fetch_asset(
    agent: &Agent,
    url: &str,
    byte_range_opt: Option<(u32, u32)>,
) -> Result<Vec<u8>, ureq::Error> {
    const FETCH_RETRIES: usize = 3; // try this many times
    const FETCH_RETRY_WAIT: std::time::Duration = std::time::Duration::from_secs(2);    // wait between tries
    let mut retries = FETCH_RETRIES;
    loop {              // until success, or fail
        match fetch_asset_once(agent, url, byte_range_opt) {
            Ok(v) => return Ok(v),
            Err(e) => {
                if err_is_retryable(&e) && retries > 0 {
                    std::thread::sleep(FETCH_RETRY_WAIT);   // wait before retry
                    retries -= 1;
                } else {
                    return Err(e);     // not retryable or out of retries, fails
                }
            }
        }
    }
}

/// Build user agent for queries.
pub fn build_agent(user_agent: &str, max_connections: usize) -> Agent {
    AgentBuilder::new()
        .user_agent(user_agent)
        .max_idle_connections_per_host(max_connections) // we mostly hit the same host, so we want more idle connections available
        .timeout_connect(NETWORK_TIMEOUT)
        .timeout_read(NETWORK_TIMEOUT)
        .timeout_write(NETWORK_TIMEOUT)
        .build()
}

#[test]
fn test_fetch_asset() {
    const USER_AGENT: &str = "Test asset fetcher. Contact info@animats.com if problems.";
    const MAX_CONNECTIONS: usize = 1; // don't overdo
    const URL1: &str = "http://www.example.com"; // something to read
    let agent = build_agent(USER_AGENT, MAX_CONNECTIONS);
    let result = fetch_asset(&agent, URL1, Some((0, 200))); // first 200 bytes only
    match result {
        Ok(text) => {
            println!("Fetched {:?}", text);
        }
        Err(e) => panic!("Error: {:?}", e),
    }
}
