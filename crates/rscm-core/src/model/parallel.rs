//! Build and run many independent models at once on the rayon thread pool.
//!
//! Each model owns its state, so running models in parallel gives the same results as running
//! them one after another.
//! The work runs on the current rayon pool, so callers choose the thread count with
//! `rayon::ThreadPool::install` or `RAYON_NUM_THREADS`.

use crate::errors::RSCMResult;
use crate::model::{Model, ModelBuilder};
use rayon::prelude::*;
use std::borrow::{Borrow, BorrowMut};

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
