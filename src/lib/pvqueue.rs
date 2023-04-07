//! # Classic Dijkstra P and V in Rust.
//! Uses parking_lot to be fair.
use std::sync::Arc;
use parking_lot::{Condvar, Mutex};

/// P/V queue
pub struct PvQueue {
    /// The lock
    condvar: Condvar,
    /// Slots currently available.
    available: Mutex<u32>,  // > 0, resources available. < 0, threads waiting, =0, idle.
}

/// Arc wrapper for sharing across threads.
pub type PvQueueLink = Arc<PvQueue>;

impl PvQueue {
    /// Initialize
    pub fn new(capacity: u32) -> PvQueueLink {
        Arc::new(Self {
            available: Mutex::new(capacity),
            condvar: Condvar::new(),       
        })
    }
    
    /// p - wait if at capacity
    fn p(&self) {
        let mut cnt = self.available.lock();       
        if *cnt == 0 {                      // if no resources available
            self.condvar.wait(&mut cnt);    // wait
        } else {
            *cnt -= 1;                      // otherwise take one resource and go
        }    
    }
    
    /// v - release one resource, wake up someone if waiting.
    fn v(&self) {
        let mut cnt = self.available.lock();
        if !self.condvar.notify_one() { // start one if queued
            *cnt += 1;  // nobody waiting, free resource.
        }
    }
    
    /// Scoped lock
    pub fn lock(pv_queue: &Arc<PvQueue>) -> PvGuard {
        let pv_queue = Arc::clone(pv_queue);
        pv_queue.p();       //  Lock
        PvGuard {            // return handle for later drop.
            pv_queue
        }
    }
}

/// The lock guard type for PvQueue. Not exported,
pub struct PvGuard {
    /// The queue.
    pv_queue: Arc<PvQueue>,
}

/// Drop for the lock guard
impl Drop for PvGuard {
    /// Drop the guard.
    fn drop(&mut self) {
        self.pv_queue.v();
    }
}

#[test]
fn testpvqueuebasic() {
    //  Very basic test. Need a threaded test.
    let queue = PvQueue::new(2);
    //  Scoped test
    {   let _locked = PvQueue::lock(&queue);
        println!("Locked");
    }
    println!("Unlocked");
    let _locked1 = PvQueue::lock(&queue);   // 1 left
    let _locked2 = PvQueue::lock(&queue);   // 0 left
    //  Should now stall
}

#[test]
/// Run N threads and make sure no more than BOTTLENECK_COUNT are in the bottleneck section.
fn testpvqueuethreaded() {
    const WORKER_COUNT: usize = 10;
    const BOTTLENECK_COUNT: u32 = 3;
    let queue = PvQueue::new(BOTTLENECK_COUNT);
    let mut workers = Vec::new();
    let active = Arc::new(std::sync::atomic::AtomicU32::new(0)); // number of active workers
    for n in 0..WORKER_COUNT {
        let queue_clone = Arc::clone(&queue);
        let active_clone = Arc::clone(&active);
        workers.push(std::thread::spawn(move || {
            {   let _locked = PvQueue::lock(&queue_clone); // limit the number of threads active simultaneously
                let cnt = active_clone.fetch_add(1, std::sync::atomic::Ordering::Relaxed);    // count of others active.
                println!("Thread #{} start, {} actives.", n, cnt + 1);
                assert!(cnt < BOTTLENECK_COUNT);    // must not have too many actives inside the bottleneck.
                std::thread::sleep_ms(100); // stall;
                let _ = active_clone.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
                println!("Thread #{} done.", n);
            }         
        }))
    }
    //  Wait for completion
    while let Some(worker) = workers.pop() {
        worker.join().expect("Join failed");
    }
}
