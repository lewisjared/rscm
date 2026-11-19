//! Thread pools for building and running models from Python.

use crate::model::Pool;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::sync::OnceLock;

/// A thread pool for building and running models, built once and reused across calls.
#[pyclass(frozen)]
#[pyo3(name = "ThreadPool")]
pub struct PyThreadPool(pub Pool);

#[pymethods]
impl PyThreadPool {
    #[new]
    #[pyo3(signature = (n_threads = None))]
    fn new(n_threads: Option<usize>) -> PyResult<Self> {
        python_pool(n_threads).map(Self)
    }

    #[getter]
    fn n_threads(&self) -> usize {
        self.0.n_threads()
    }

    fn __repr__(&self) -> String {
        format!("ThreadPool(n_threads={})", self.0.n_threads())
    }
}

/// The pool to run on: `pool` if given, a new pool of `n_threads` threads, or the shared default.
pub(crate) fn resolve_pool(
    n_threads: Option<usize>,
    pool: Option<&PyThreadPool>,
) -> PyResult<Pool> {
    match (n_threads, pool) {
        (Some(_), Some(_)) => Err(PyValueError::new_err("pass n_threads or pool, not both")),
        (_, Some(pool)) => Ok(pool.0.clone()),
        (Some(n), None) => python_pool(Some(n)),
        (None, None) => {
            static DEFAULT: OnceLock<Result<Pool, String>> = OnceLock::new();
            DEFAULT
                .get_or_init(|| python_pool(None).map_err(|e| e.to_string()))
                .clone()
                .map_err(PyRuntimeError::new_err)
        }
    }
}

/// A pool of `n_threads` threads, or one per core, whose workers each keep one Python thread state.
///
/// Without this, every Python component solve on a worker creates and deletes a thread state,
/// which makes Python components several times slower on free-threaded builds.
fn python_pool(n_threads: Option<usize>) -> PyResult<Pool> {
    Pool::configured(n_threads, |builder| {
        builder
            .start_handler(|_| thread_state::keep())
            .exit_handler(|_| thread_state::release())
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// One Python thread state per pool worker, kept for the worker's lifetime.
mod thread_state {
    use pyo3::ffi;
    use std::cell::Cell;

    thread_local! {
        static KEPT: Cell<Option<(ffi::PyGILState_STATE, *mut ffi::PyThreadState)>> =
            const { Cell::new(None) };
    }

    fn interpreter_running() -> bool {
        // SAFETY: both only read interpreter status and are safe to call from any thread.
        unsafe { ffi::Py_IsInitialized() != 0 && ffi::Py_IsFinalizing() == 0 }
    }

    pub fn keep() {
        if !interpreter_running() {
            return;
        }
        // SAFETY: the interpreter is running, and the thread state is detached before returning,
        // so this thread holds no GIL between tasks.
        let kept = unsafe {
            let gil = ffi::PyGILState_Ensure();
            (gil, ffi::PyEval_SaveThread())
        };
        KEPT.with(|cell| cell.set(Some(kept)));
    }

    pub fn release() {
        let Some((gil, state)) = KEPT.with(Cell::take) else {
            return;
        };
        // Leaked at shutdown: finalisation frees the thread states itself.
        if !interpreter_running() {
            return;
        }
        // SAFETY: `state` is this thread's own state, saved by `keep` and not yet released.
        unsafe {
            ffi::PyEval_RestoreThread(state);
            ffi::PyGILState_Release(gil);
        }
    }
}
