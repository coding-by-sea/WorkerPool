//! A high-throughput, bounded multi-producer single-consumer (MPSC) task
//! queue and execution worker pool in async Rust.
//!
//! The worker pool must execute incoming asynchronous tasks with:
//!
//! - dynamic concurrency limits,
//! - per-task execution timeouts, and
//! - zero dynamic heap allocation per submitted task on the hot path.
//!
//! # Requirements & Functionality
//!
//! 1. **Bounded Lock-Free Queue**
//!
//!    Use a pre-allocated ring buffer or lock-free queue abstraction to store
//!    task handles. Do not allocate memory when pushing tasks once the pool
//!    is initialized.
//!
//! 2. **Concurrency Control**
//!
//!    Maintain an active concurrency limit `N`. Provide a lock-free mechanism
//!    to dynamically resize `N` up or down at runtime without dropping queued
//!    tasks or breaking active executions.
//!
//! 3. **Task Cancellation & Timeouts**
//!
//!    Each submitted task must accept a [`Duration`] timeout constraint. If
//!    task execution exceeds the specified duration, the task handle must be
//!    cancelled immediately, returning a `TaskError::TimedOut` result to the
//!    caller without blocking the worker thread.
//!
//! 4. **Graceful Shutdown**
//!
//!    Implement an explicit `shutdown(self)` protocol that waits for
//!    currently executing tasks to complete, up to a specified grace period,
//!    before dropping remaining queued tasks.
//!
//! [`Duration`]: std::time::Duration

use std::any::Any;
use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::Relaxed;
use std::time::Duration;

use concurrent_queue::{ConcurrentQueue, PushError};
use futures_util::FutureExt;
use tokio::select;
use tokio::sync::{Semaphore, oneshot};
use tokio::task::JoinSet;
use tokio::time::timeout;

#[derive(Debug)]
pub enum TaskError {
    TimedOut,
    WorkerPoolClosed,
    ExecutionFailed(String),
}

pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// A type-erased job: the user's task, its timeout, and the result send,
/// all wrapped into one future that outputs `()`.
type Job = Pin<Box<dyn Future<Output = ()> + Send + 'static>>;

struct Worker {
    work_queue: Arc<ConcurrentQueue<Job>>,
    /// One permit per job sitting in `work_queue`.
    queued: Arc<Semaphore>,
    /// One permit per free slot in `work_queue`.
    slots: Arc<Semaphore>,
    /// One permit per worker that should exit to shrink the pool.
    retiring: Arc<Semaphore>,
}

impl Worker {
    fn new(
        work_queue: Arc<ConcurrentQueue<Job>>,
        queued: Arc<Semaphore>,
        slots: Arc<Semaphore>,
        retiring: Arc<Semaphore>,
    ) -> Self {
        Self {
            work_queue,
            queued,
            slots,
            retiring,
        }
    }

    /// Runs jobs until the worker is told to retire or the pool closes.
    async fn run(self) {
        loop {
            select! {
                // Check `retiring` first, so a worker told to exit never
                // takes another job.
                biased;

                // A `retiring` permit means the pool shrank: this worker exits.
                Ok(permit) = self.retiring.acquire() => {
                    permit.forget();
                    break;
                }

                // A job was queued, or the pool is shutting down.
                res = self.queued.acquire() => {
                    // `queued` was closed: the pool is shutting down.
                    let Ok(permit) = res else {
                        break;
                    };
                    // The permit stands for the job we're about to pop, so
                    // it's used up.
                    permit.forget();

                    // Normally a permit means a job is waiting. The queue can
                    // only be empty here if the pool closed and drained it
                    // after we got the permit, so stop instead of running
                    // anything else.
                    let Ok(job) = self.work_queue.pop() else {
                        break;
                    };
                    self.slots.add_permits(1);
                    job.await;
                }
            }
        }
    }
}

/// A cheap, cloneable handle for submitting tasks to a [`TaskPool`].
///
/// Give one to each producer task. The pool's owner keeps the `TaskPool`
/// itself, which is the only thing that can resize or shut down the pool.
/// After shutdown, `submit` on any handle returns `WorkerPoolClosed`.
#[derive(Clone)]
pub struct Submitter {
    work_queue: Arc<ConcurrentQueue<Job>>,
    /// One permit per job sitting in `work_queue`.
    queued: Arc<Semaphore>,
    /// One permit per free slot in `work_queue`.
    slots: Arc<Semaphore>,
    retiring: Arc<Semaphore>,
}

impl Submitter {
    fn new_worker(&self) -> Worker {
        Worker::new(
            Arc::clone(&self.work_queue),
            Arc::clone(&self.queued),
            Arc::clone(&self.slots),
            Arc::clone(&self.retiring),
        )
    }

    /// Submit a task to the pool. Returns a receiver channel for the result.
    ///
    /// Waits for a free queue slot when the queue is full. Pushing onto the
    /// queue never allocates; boxing the job and the result channel still do.
    pub async fn submit<F, T>(
        &self,
        task: F,
        timeout: Duration,
    ) -> Result<oneshot::Receiver<Result<T, TaskError>>, TaskError>
    where
        F: FnOnce() -> BoxFuture<'static, T> + Send + 'static,
        T: Send + 'static,
    {
        let permit = self
            .slots
            .acquire()
            .await
            .map_err(|_| TaskError::WorkerPoolClosed)?;
        let (tx, rx) = oneshot::channel::<Result<T, TaskError>>();

        // `T` is only known here, so do the typed work here and erase it to `()`.
        let job: Job = Box::pin(async move {
            // `task()` is called inside the async block, so a panic while
            // creating the future is caught too, not just one while polling it.
            let guarded = AssertUnwindSafe(async move { task().await }).catch_unwind();
            let result = match tokio::time::timeout(timeout, guarded).await {
                Ok(Ok(value)) => Ok(value),
                Ok(Err(panic)) => Err(TaskError::ExecutionFailed(panic_message(&*panic))),
                Err(_) => Err(TaskError::TimedOut),
            };
            let _ = tx.send(result); // receiver may have been dropped; that's fine
        });

        // TODO: add a deterministic test for a `submit` that races with
        // `close`. It needs a `#[cfg(test)]` pause point between acquiring
        // the slot and pushing: pause there, close the pool, resume, and check
        // the caller gets `WorkerPoolClosed` instead of waiting forever.
        match self.work_queue.push(job) {
            Ok(()) => {
                // The job now holds this slot until a worker pops it.
                permit.forget();
                self.queued.add_permits(1); // wakes one sleeping worker
                Ok(rx)
            }
            // The pool closed after we got a slot. A closed queue rejects
            // every push, so no job can land after `close` has drained it.
            Err(PushError::Closed(_)) => Err(TaskError::WorkerPoolClosed),
            Err(PushError::Full(_)) => {
                unreachable!("holding a slot permit guarantees space in the queue")
            }
        }
    }

    /// Drop every job still in the queue. Each one's caller gets a `RecvError`.
    fn drain(&self) {
        while self.work_queue.pop().is_ok() {}
    }
}

pub struct TaskPool {
    /// The pool keeps its own handle; `submitter()` hands out clones.
    submitter: Submitter,
    task_pool: JoinSet<()>,
    max_concurrency: Arc<AtomicUsize>,
}

impl TaskPool {
    /// Create a pool whose queue holds up to `capacity` jobs, with
    /// `max_concurrency` workers running jobs.
    pub fn new(capacity: usize, max_concurrency: usize) -> Self {
        let submitter = Submitter {
            work_queue: Arc::new(ConcurrentQueue::bounded(capacity)),
            queued: Arc::new(Semaphore::new(0)), // queue starts empty
            slots: Arc::new(Semaphore::new(capacity)),
            retiring: Arc::new(Semaphore::new(0)),
        };
        let mut task_pool = JoinSet::<()>::new();
        let max_concurrency = Arc::new(AtomicUsize::new(max_concurrency));
        for _ in 0..max_concurrency.load(Relaxed) {
            task_pool.spawn(submitter.new_worker().run());
        }

        Self {
            submitter,
            task_pool,
            max_concurrency,
        }
    }

    /// A handle that producer tasks can use to submit work concurrently.
    pub fn submitter(&self) -> Submitter {
        self.submitter.clone()
    }

    /// Submit a task to the pool. Same as [`Submitter::submit`].
    pub async fn submit<F, T>(
        &self,
        task: F,
        timeout: Duration,
    ) -> Result<oneshot::Receiver<Result<T, TaskError>>, TaskError>
    where
        F: FnOnce() -> BoxFuture<'static, T> + Send + 'static,
        T: Send + 'static,
    {
        self.submitter.submit(task, timeout).await
    }

    /// Dynamically adjust the concurrency limit at runtime.
    pub fn resize_concurrency(&mut self, new_concurrency: usize) {
        let current_concurrency = self.max_concurrency.swap(new_concurrency, Relaxed);
        if new_concurrency > current_concurrency {
            let added = new_concurrency - current_concurrency;
            // Workers still due to retire from an earlier shrink can simply
            // stay; take back their unused `retiring` permits first.
            let cancelled = self.submitter.retiring.forget_permits(added);
            for _ in 0..(added - cancelled) {
                self.task_pool.spawn(self.submitter.new_worker().run());
            }
        } else if new_concurrency < current_concurrency {
            self.submitter.retiring.add_permits(current_concurrency - new_concurrency);
        }
    }

    /// Trigger a graceful shutdown, draining active tasks up to `grace_period`.
    ///
    /// Workers finish the job they're running, then exit. Jobs still in the
    /// queue are dropped, so their callers get a `RecvError`. If the grace
    /// period runs out, dropping the `JoinSet` aborts the remaining workers.
    pub async fn shutdown(mut self, grace_period: Duration) -> Result<(), TaskError> {
        self.close();

        // `Drop` is implemented, so the `JoinSet` can't be moved out of
        // `self` directly; take it and leave an empty one behind.
        let workers = std::mem::take(&mut self.task_pool);

        // Jobs catch their own panics, so workers never panic and `join_all`
        // won't re-raise one here.
        timeout(grace_period, workers.join_all())
            .await
            .map(|_| ())
            .map_err(|_| TaskError::TimedOut)
    }

    /// Stop accepting work and drop queued jobs, without waiting for
    /// running jobs. Safe to call more than once.
    fn close(&self) {
        self.submitter.slots.close(); // new submits on any handle get WorkerPoolClosed
        self.submitter.queued.close(); // idle workers wake up and exit
        self.submitter.work_queue.close(); // pushes already past `slots` fail

        // Drop queued jobs now, even if producers still hold handles that
        // keep the queue itself alive. The queue is closed, so nothing can be
        // pushed after this drain.
        self.submitter.drain();
    }
}

/// Dropping the pool without calling `shutdown` closes it immediately:
/// queued jobs are dropped and running jobs are aborted (when the `JoinSet`
/// is dropped). Call `shutdown` instead to let running jobs finish.
impl Drop for TaskPool {
    fn drop(&mut self) {
        self.close();
    }
}

/// Turn a caught panic's payload into a readable message.
fn panic_message(payload: &(dyn Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        s.to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "task panicked".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering::SeqCst;

    /// Shrinking while every worker is busy leaves `retiring` permits unused.
    /// Growing back before anyone retires should take those permits back
    /// instead of spawning new workers that would just retire.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn grow_after_shrink_cancels_pending_retirements() {
        let mut pool = TaskPool::new(16, 4);

        // Keep all 4 workers busy so none of them can retire yet.
        let mut busy = Vec::new();
        for _ in 0..4 {
            let rx = pool
                .submit(
                    || Box::pin(tokio::time::sleep(Duration::from_millis(100))),
                    Duration::from_secs(1),
                )
                .await
                .unwrap();
            busy.push(rx);
        }
        tokio::time::sleep(Duration::from_millis(20)).await; // let them start

        pool.resize_concurrency(1);
        assert_eq!(pool.submitter.retiring.available_permits(), 3);

        pool.resize_concurrency(4);
        assert_eq!(
            pool.submitter.retiring.available_permits(),
            0,
            "pending retirements should be cancelled"
        );
        assert_eq!(pool.task_pool.len(), 4, "no new workers should be spawned");

        for rx in busy {
            rx.await.unwrap().unwrap();
        }

        // The pool still runs 4 jobs at once afterwards.
        let running = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let mut jobs = Vec::new();
        for _ in 0..8 {
            let (running, peak) = (running.clone(), peak.clone());
            let rx = pool
                .submit(
                    move || {
                        Box::pin(async move {
                            let now = running.fetch_add(1, SeqCst) + 1;
                            peak.fetch_max(now, SeqCst);
                            tokio::time::sleep(Duration::from_millis(30)).await;
                            running.fetch_sub(1, SeqCst);
                        })
                    },
                    Duration::from_secs(1),
                )
                .await
                .unwrap();
            jobs.push(rx);
        }
        for rx in jobs {
            rx.await.unwrap().unwrap();
        }
        assert_eq!(peak.load(SeqCst), 4);
    }
}
