//! Behavior tests for `TaskPool`, one group per requirement in the crate docs.
//!
//! Every test runs on a multi-threaded runtime and is wrapped in a 5 s limit,
//! so a hang shows up as a failure instead of blocking the test run.

use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use WorkerPool::{BoxFuture, TaskError, TaskPool};

const TEST_LIMIT: Duration = Duration::from_secs(5);
const LONG: Duration = Duration::from_secs(10);

/// Turn an async block into the `FnOnce() -> BoxFuture` shape `submit` expects.
fn task<T, Fut>(fut: Fut) -> impl FnOnce() -> BoxFuture<'static, T> + Send + 'static
where
    Fut: Future<Output = T> + Send + 'static,
{
    move || Box::pin(fut)
}

/// Run a test body on its own multi-threaded runtime, failing it if it takes
/// longer than `TEST_LIMIT`.
///
/// The runtime is shut down with `shutdown_background`, which doesn't wait
/// for the pool's workers. That way a worker stuck in a loop can't hang the
/// whole test run.
fn run<Fut: Future>(fut: Fut) -> Fut::Output {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(8)
        .enable_all()
        .build()
        .unwrap();
    let out = rt.block_on(async {
        tokio::time::timeout(TEST_LIMIT, fut)
            .await
            .expect("test exceeded time limit (pool hung?)")
    });
    rt.shutdown_background();
    out
}

/// Tracks how many tasks are running at once and the highest count seen.
#[derive(Default)]
struct ConcurrencyProbe {
    running: AtomicUsize,
    peak: AtomicUsize,
}

impl ConcurrencyProbe {
    async fn run_for(&self, d: Duration) {
        let now = self.running.fetch_add(1, Ordering::SeqCst) + 1;
        self.peak.fetch_max(now, Ordering::SeqCst);
        tokio::time::sleep(d).await;
        self.running.fetch_sub(1, Ordering::SeqCst);
    }

    fn peak(&self) -> usize {
        self.peak.load(Ordering::SeqCst)
    }
}

/// Sets a flag when dropped, so a test can tell a future was cancelled.
struct DropFlag(Arc<AtomicBool>);

impl Drop for DropFlag {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

// ---------------------------------------------------------------------------
// Basic execution
// ---------------------------------------------------------------------------

#[test]
fn returns_task_result() {
    run(async {
        let pool = TaskPool::new(16, 2);
        let rx = pool
            .submit(task(async { 40 + 2 }), Duration::from_secs(1))
            .await
            .expect("submit failed");

        let result = rx.await.expect("result channel closed");
        assert!(matches!(result, Ok(42)), "got {result:?}");
    });
}

#[test]
fn one_pool_handles_different_return_types() {
    run(async {
        let pool = TaskPool::new(16, 2);
        let num = pool.submit(task(async { 7u64 }), Duration::from_secs(1)).await.unwrap();
        let text = pool
            .submit(task(async { String::from("hi") }), Duration::from_secs(1))
            .await
            .unwrap();
        let list = pool.submit(task(async { vec![1, 2, 3] }), Duration::from_secs(1)).await.unwrap();

        assert_eq!(num.await.unwrap().unwrap(), 7);
        assert_eq!(text.await.unwrap().unwrap(), "hi");
        assert_eq!(list.await.unwrap().unwrap(), vec![1, 2, 3]);
    });
}

#[test]
fn runs_many_tasks() {
    run(async {
        let pool = TaskPool::new(128, 4);
        let mut receivers = Vec::new();
        for i in 0..100u32 {
            receivers.push(pool.submit(task(async move { i * 2 }), Duration::from_secs(1)).await.unwrap());
        }
        for (i, rx) in receivers.into_iter().enumerate() {
            assert_eq!(rx.await.unwrap().unwrap(), i as u32 * 2);
        }
    });
}

// ---------------------------------------------------------------------------
// Bounded queue
// ---------------------------------------------------------------------------

/// Submitting more tasks than `capacity` must not lose any of them: the
/// caller should wait for space (or get an error), never a silent drop.
#[test]
fn queue_overflow_does_not_drop_tasks() {
    run(async {
        let pool = TaskPool::new(2, 1);
        let mut receivers = Vec::new();
        for i in 0..10u32 {
            let rx = pool
                .submit(
                    task(async move {
                        tokio::time::sleep(Duration::from_millis(10)).await;
                        i
                    }),
                    Duration::from_secs(1),
                )
                .await;
            match rx {
                Ok(rx) => receivers.push((i, rx)),
                Err(e) => panic!("submit {i} returned {e:?}"),
            }
        }
        for (i, rx) in receivers {
            let result = rx.await.unwrap_or_else(|_| panic!("task {i} was dropped"));
            assert_eq!(result.unwrap(), i);
        }
    });
}

// ---------------------------------------------------------------------------
// Timeouts and cancellation
// ---------------------------------------------------------------------------

#[test]
fn slow_task_times_out() {
    run(async {
        let pool = TaskPool::new(16, 2);
        let started = Instant::now();
        let rx = pool
            .submit(task(async { tokio::time::sleep(LONG).await }), Duration::from_millis(50))
            .await
            .unwrap();

        let result = rx.await.unwrap();
        assert!(matches!(result, Err(TaskError::TimedOut)), "got {result:?}");
        assert!(started.elapsed() < Duration::from_secs(1), "timeout took {:?}", started.elapsed());
    });
}

#[test]
fn timed_out_task_is_cancelled() {
    run(async {
        let pool = TaskPool::new(16, 1);
        let dropped = Arc::new(AtomicBool::new(false));
        let flag = DropFlag(dropped.clone());

        let rx = pool
            .submit(
                task(async move {
                    let _flag = flag;
                    tokio::time::sleep(LONG).await;
                }),
                Duration::from_millis(50),
            )
            .await
            .unwrap();

        assert!(matches!(rx.await.unwrap(), Err(TaskError::TimedOut)));
        assert!(dropped.load(Ordering::SeqCst), "timed-out future was not dropped");
    });
}

/// With a single worker, a task that times out must free the worker so the
/// next task still runs.
#[test]
fn worker_continues_after_timeout() {
    run(async {
        let pool = TaskPool::new(16, 1);
        let slow = pool
            .submit(task(async { tokio::time::sleep(LONG).await }), Duration::from_millis(50))
            .await
            .unwrap();
        let fast = pool.submit(task(async { "done" }), Duration::from_secs(1)).await.unwrap();

        assert!(matches!(slow.await.unwrap(), Err(TaskError::TimedOut)));
        assert_eq!(fast.await.unwrap().unwrap(), "done");
    });
}

// ---------------------------------------------------------------------------
// Panics
// ---------------------------------------------------------------------------

/// A panicking task returns `ExecutionFailed` and doesn't kill its worker:
/// with a single worker, the next task must still run.
#[test]
fn panicking_task_returns_execution_failed() {
    run(async {
        let pool = TaskPool::new(16, 1);
        let bad = pool
            .submit(task(async { panic!("boom") }), Duration::from_secs(1))
            .await
            .unwrap();
        let good = pool.submit(task(async { 5 }), Duration::from_secs(1)).await.unwrap();

        let result: Result<(), TaskError> = bad.await.unwrap();
        assert!(
            matches!(&result, Err(TaskError::ExecutionFailed(msg)) if msg == "boom"),
            "got {result:?}"
        );
        assert_eq!(good.await.unwrap().unwrap(), 5);
    });
}

#[test]
fn shutdown_succeeds_after_task_panic() {
    run(async {
        let pool = TaskPool::new(16, 2);
        let rx = pool
            .submit(task(async { panic!("boom") }), Duration::from_secs(1))
            .await
            .unwrap();
        let _: Result<(), TaskError> = rx.await.unwrap();

        let result = pool.shutdown(Duration::from_secs(1)).await;
        assert!(result.is_ok(), "shutdown returned {result:?}");
    });
}

// ---------------------------------------------------------------------------
// Concurrency limit
// ---------------------------------------------------------------------------

#[test]
fn never_exceeds_concurrency_limit() {
    run(async {
        let pool = TaskPool::new(32, 2);
        let probe = Arc::new(ConcurrencyProbe::default());
        let mut receivers = Vec::new();
        for _ in 0..8 {
            let p = probe.clone();
            receivers.push(
                pool.submit(task(async move { p.run_for(Duration::from_millis(30)).await }), Duration::from_secs(2))
                    .await
                    .unwrap(),
            );
        }
        for rx in receivers {
            rx.await.unwrap().unwrap();
        }
        assert_eq!(probe.peak(), 2, "peak concurrency was {}", probe.peak());
    });
}

#[test]
fn tasks_run_in_parallel_up_to_limit() {
    run(async {
        let pool = TaskPool::new(16, 4);
        let started = Instant::now();
        let mut receivers = Vec::new();
        for _ in 0..4 {
            receivers.push(
                pool.submit(task(tokio::time::sleep(Duration::from_millis(200))), Duration::from_secs(2))
                    .await
                    .unwrap(),
            );
        }
        for rx in receivers {
            rx.await.unwrap().unwrap();
        }
        // Serially this would take 800 ms.
        assert!(started.elapsed() < Duration::from_millis(500), "took {:?}", started.elapsed());
    });
}

// ---------------------------------------------------------------------------
// Dynamic resizing
// ---------------------------------------------------------------------------

#[test]
fn resize_up_increases_parallelism() {
    run(async {
        let mut pool = TaskPool::new(32, 1);
        pool.resize_concurrency(4);

        let probe = Arc::new(ConcurrencyProbe::default());
        let mut receivers = Vec::new();
        for _ in 0..8 {
            let p = probe.clone();
            receivers.push(
                pool.submit(task(async move { p.run_for(Duration::from_millis(50)).await }), Duration::from_secs(2))
                    .await
                    .unwrap(),
            );
        }
        for rx in receivers {
            rx.await.unwrap().unwrap();
        }
        assert_eq!(probe.peak(), 4, "peak concurrency was {}", probe.peak());
    });
}

#[test]
fn resize_down_lowers_parallelism() {
    run(async {
        let mut pool = TaskPool::new(32, 4);
        pool.resize_concurrency(1);

        let probe = Arc::new(ConcurrencyProbe::default());
        let mut receivers = Vec::new();
        for _ in 0..6 {
            let p = probe.clone();
            receivers.push(
                pool.submit(task(async move { p.run_for(Duration::from_millis(20)).await }), Duration::from_secs(2))
                    .await
                    .unwrap(),
            );
        }
        for rx in receivers {
            rx.await.unwrap().unwrap();
        }
        assert_eq!(probe.peak(), 1, "peak concurrency was {}", probe.peak());
    });
}

/// Shrinking while tasks are running and queued must not interrupt or drop any.
#[test]
fn resize_mid_flight_keeps_all_tasks() {
    run(async {
        let mut pool = TaskPool::new(32, 4);
        let mut receivers = Vec::new();
        for i in 0..12u32 {
            receivers.push(
                pool.submit(
                    task(async move {
                        tokio::time::sleep(Duration::from_millis(50)).await;
                        i
                    }),
                    Duration::from_secs(2),
                )
                .await
                .unwrap(),
            );
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
        pool.resize_concurrency(1);

        for (i, rx) in receivers.into_iter().enumerate() {
            let result = rx.await.unwrap_or_else(|_| panic!("task {i} was dropped"));
            assert_eq!(result.unwrap(), i as u32);
        }
    });
}

// ---------------------------------------------------------------------------
// Submitter handles
// ---------------------------------------------------------------------------

/// Several producer tasks submit through their own handles at the same time.
#[test]
fn submitters_work_from_many_tasks() {
    run(async {
        let pool = TaskPool::new(8, 4);
        let mut producers = Vec::new();
        for p in 0..4u32 {
            let submitter = pool.submitter();
            producers.push(tokio::spawn(async move {
                let mut sum = 0;
                for i in 0..25u32 {
                    let rx = submitter
                        .submit(task(async move { p * 100 + i }), Duration::from_secs(1))
                        .await
                        .unwrap();
                    sum += rx.await.unwrap().unwrap();
                }
                sum
            }));
        }

        let mut total = 0;
        for producer in producers {
            total += producer.await.unwrap();
        }
        // Each producer p adds p*100*25 + (0+1+...+24) = 2500p + 300.
        assert_eq!(total, (0..4).map(|p| 2500 * p + 300).sum::<u32>());
    });
}

/// Resizing while other tasks hold submitter handles still works.
#[test]
fn resize_while_submitters_exist() {
    run(async {
        let mut pool = TaskPool::new(16, 1);
        let submitter = pool.submitter();
        pool.resize_concurrency(2);

        let rx = submitter.submit(task(async { 9 }), Duration::from_secs(1)).await.unwrap();
        assert_eq!(rx.await.unwrap().unwrap(), 9);
    });
}

#[test]
fn submit_after_shutdown_is_rejected() {
    run(async {
        let pool = TaskPool::new(16, 1);
        let submitter = pool.submitter();
        pool.shutdown(Duration::from_secs(1)).await.unwrap();

        let result = submitter.submit(task(async { 1 }), Duration::from_secs(1)).await;
        assert!(matches!(result, Err(TaskError::WorkerPoolClosed)));
    });
}

// ---------------------------------------------------------------------------
// Graceful shutdown
// ---------------------------------------------------------------------------

#[test]
fn shutdown_waits_for_running_task() {
    run(async {
        let pool = TaskPool::new(16, 2);
        let rx = pool
            .submit(
                task(async {
                    tokio::time::sleep(Duration::from_millis(100)).await;
                    "finished"
                }),
                Duration::from_secs(2),
            )
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(10)).await; // let it start

        let result = pool.shutdown(Duration::from_secs(2)).await;
        assert!(result.is_ok(), "shutdown returned {result:?}");
        assert_eq!(rx.await.expect("task cut off").unwrap(), "finished");
    });
}

#[test]
fn shutdown_stops_after_grace_period() {
    run(async {
        let pool = TaskPool::new(16, 1);
        let rx = pool
            .submit(task(async { tokio::time::sleep(LONG).await }), LONG)
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(10)).await;

        let started = Instant::now();
        let _ = pool.shutdown(Duration::from_millis(100)).await;
        assert!(started.elapsed() < Duration::from_secs(1), "shutdown took {:?}", started.elapsed());
        assert!(rx.await.is_err(), "task past the grace period should be dropped");
    });
}

#[test]
fn shutdown_drops_queued_tasks() {
    run(async {
        let pool = TaskPool::new(16, 1);
        let _blocker = pool
            .submit(task(async { tokio::time::sleep(LONG).await }), LONG)
            .await
            .unwrap();
        let ran = Arc::new(AtomicBool::new(false));
        let r = ran.clone();
        let queued = pool
            .submit(task(async move { r.store(true, Ordering::SeqCst) }), LONG)
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(10)).await;

        let _ = pool.shutdown(Duration::from_millis(50)).await;
        assert!(queued.await.is_err(), "queued task should be dropped");
        assert!(!ran.load(Ordering::SeqCst), "queued task should not run");
    });
}

/// Same as above, but a producer still holds a handle, which keeps the queue
/// itself alive. Shutdown must still drop the queued job right away.
#[test]
fn shutdown_drops_queued_tasks_while_handle_held() {
    run(async {
        let pool = TaskPool::new(16, 1);
        let handle = pool.submitter();
        let _blocker = handle
            .submit(task(async { tokio::time::sleep(LONG).await }), LONG)
            .await
            .unwrap();
        let queued = handle.submit(task(async { 1 }), LONG).await.unwrap();
        tokio::time::sleep(Duration::from_millis(10)).await;

        let _ = pool.shutdown(Duration::from_millis(50)).await;
        let result = tokio::time::timeout(Duration::from_secs(1), queued).await;
        assert!(
            matches!(result, Ok(Err(_))),
            "queued task should be dropped, not left waiting"
        );
        drop(handle);
    });
}

/// Dropping the pool without `shutdown` closes it: held handles are rejected
/// and queued jobs are dropped instead of hanging.
#[test]
fn dropping_pool_closes_held_handles() {
    run(async {
        let pool = TaskPool::new(16, 1);
        let handle = pool.submitter();
        let _blocker = handle
            .submit(task(async { tokio::time::sleep(LONG).await }), LONG)
            .await
            .unwrap();
        let queued = handle.submit(task(async { 1 }), LONG).await.unwrap();
        tokio::time::sleep(Duration::from_millis(10)).await;

        drop(pool);

        let result = tokio::time::timeout(Duration::from_secs(1), queued).await;
        assert!(
            matches!(result, Ok(Err(_))),
            "queued task should be dropped, not left waiting"
        );
        let late = handle.submit(task(async { 2 }), LONG).await;
        assert!(matches!(late, Err(TaskError::WorkerPoolClosed)));
    });
}
