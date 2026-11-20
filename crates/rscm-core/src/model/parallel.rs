//! Build and run many independent models at once on a rayon thread pool.
//!
//! Each model owns its state, so running models in parallel gives the same results as running
//! them one after another.
//! [`build_models`] and [`run_models`] work on the current rayon pool, so callers choose the thread
//! count with [`Pool::install`], `rayon::ThreadPool::install` or `RAYON_NUM_THREADS`.
//! [`run_each`] streams builders from an iterator onto a [`Pool`],
//! so runs start while later builders are still being made.

use crate::errors::{RSCMError, RSCMResult};
use crate::model::{Model, ModelBuilder};
use rayon::prelude::*;
use std::borrow::{Borrow, BorrowMut};
use std::sync::{mpsc, Arc};

/// A thread pool that runs models, built once and reused across calls.
///
/// The default is rayon's global pool, with one thread per core unless `RAYON_NUM_THREADS` says
/// otherwise.
/// Clones share the same threads.
#[derive(Clone, Debug, Default)]
pub struct Pool(Option<Arc<rayon::ThreadPool>>);

impl Pool {
    /// A pool of `n_threads` threads, or rayon's global pool for `None`.
    pub fn new(n_threads: Option<usize>) -> RSCMResult<Self> {
        match n_threads {
            None => Ok(Self(None)),
            n => Self::configured(n, |builder| builder),
        }
    }

    /// A new pool of `n_threads` threads, or one per core for `None`,
    /// with its rayon builder adjusted by `configure`, such as to add thread start handlers.
    pub fn configured(
        n_threads: Option<usize>,
        configure: impl FnOnce(rayon::ThreadPoolBuilder) -> rayon::ThreadPoolBuilder,
    ) -> RSCMResult<Self> {
        if n_threads == Some(0) {
            return Err(RSCMError::Error("n_threads must be at least 1".into()));
        }
        configure(rayon::ThreadPoolBuilder::new().num_threads(n_threads.unwrap_or(0)))
            .build()
            .map(|pool| Self(Some(Arc::new(pool))))
            .map_err(|e| RSCMError::Error(e.to_string()))
    }

    /// The number of threads the pool runs work on.
    pub fn n_threads(&self) -> usize {
        match &self.0 {
            None => rayon::current_num_threads(),
            Some(pool) => pool.current_num_threads(),
        }
    }

    /// Run `op` inside the pool, so rayon's parallel iterators in it use the pool's threads.
    pub fn install<R: Send>(&self, op: impl FnOnce() -> R + Send) -> R {
        match &self.0 {
            None => op(),
            Some(pool) => pool.install(op),
        }
    }

    /// Apply `work` to each item of `source` on the pool and return the results in input order.
    ///
    /// The calling thread pulls items from `source` and hands each to the pool as it arrives,
    /// so work on early items overlaps producing later ones.
    /// The call returns once every item is done.
    /// `source` runs only on the calling thread, so it need not be `Send`.
    pub fn map_each<I, T, F>(&self, source: impl IntoIterator<Item = I>, work: F) -> Vec<T>
    where
        I: Send,
        T: Send,
        F: Fn(usize, I) -> T + Sync,
    {
        let (sender, receiver) = mpsc::channel();
        match &self.0 {
            None => rayon::in_place_scope(|scope| spawn_each(scope, source, &sender, &work)),
            Some(pool) => pool.in_place_scope(|scope| spawn_each(scope, source, &sender, &work)),
        }
        drop(sender);

        let mut results: Vec<(usize, T)> = receiver.into_iter().collect();
        results.sort_unstable_by_key(|(index, _)| *index);
        results.into_iter().map(|(_, result)| result).collect()
    }
}

fn spawn_each<'scope, I, T, F>(
    scope: &rayon::Scope<'scope>,
    source: impl IntoIterator<Item = I>,
    sender: &mpsc::Sender<(usize, T)>,
    work: &'scope F,
) where
    I: Send + 'scope,
    T: Send + 'scope,
    F: Fn(usize, I) -> T + Sync,
{
    for (index, item) in source.into_iter().enumerate() {
        let sender = sender.clone();
        scope.spawn(move |_| {
            let _ = sender.send((index, work(index, item)));
        });
    }
}

/// Build and run each builder from `source` on `pool`,
/// returning `finish` applied to each model in input order.
///
/// Each item is built, run with [`Model::try_run`] and passed to `finish(index, model)` on a pool
/// thread, so `finish` can reduce the model to a smaller result and drop it there.
/// `finish` receives the source's own error, the build error or the run error for a failed item.
/// A failed item does not stop the others.
pub fn run_each<T, F>(
    pool: &Pool,
    source: impl IntoIterator<Item = RSCMResult<ModelBuilder>>,
    finish: F,
) -> Vec<T>
where
    T: Send,
    F: Fn(usize, RSCMResult<Model>) -> T + Sync,
{
    pool.map_each(source, |index, builder| {
        let model = builder
            .and_then(|b| build_and_run(&b))
            .and_then(|(model, warnings, run)| {
                for message in warnings {
                    tracing::warn!("{message}");
                }
                run.map(|()| model)
            });
        finish(index, model)
    })
}

/// Build `builder` and run the model, returning the build's warnings and the run's result.
///
/// A build error is the outer error, so callers can tell it from a failed run.
pub fn build_and_run(builder: &ModelBuilder) -> RSCMResult<(Model, Vec<String>, RSCMResult<()>)> {
    let (mut model, warnings) = builder.build_with_warnings()?;
    let run = model.try_run();
    Ok((model, warnings, run))
}

/// Build every builder, returning each model or its build error in input order.
pub fn build_models<B>(builders: &[B]) -> Vec<RSCMResult<Model>>
where
    B: Borrow<ModelBuilder> + Sync,
{
    builders.par_iter().map(|b| b.borrow().build()).collect()
}

/// Run every model to completion, returning each run's result in input order.
///
/// A failed model does not stop the others.
/// As with `Model::try_run`, discard a model whose run failed.
pub fn run_models<M>(models: &mut [M]) -> Vec<RSCMResult<()>>
where
    M: BorrowMut<Model> + Send,
{
    models
        .par_iter_mut()
        .map(|m| m.borrow_mut().try_run())
        .collect()
}
