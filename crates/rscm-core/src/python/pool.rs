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
    if n_threads == Some(0) {
        return Err(PyValueError::new_err("n_threads must be at least 1"));
    }
    Pool::configured(n_threads, |builder| {
        builder
            .start_handler(|_| thread_state::keep())
            .exit_handler(|_| thread_state::release())
    })
    .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// One Python thread state per pool worker, kept for the worker's lifetime.
/// Stop pool workers touching Python before finalisation, when taking the GIL would end their thread.
#[pyfunction]
fn close_thread_states(py: Python<'_>) {
    py.detach(thread_state::close);
}

/// Register `close_thread_states` to run at interpreter exit.
pub(crate) fn register_shutdown(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let close = wrap_pyfunction!(close_thread_states, m)?;
    m.py()
        .import("atexit")?
        .call_method1("register", (close,))?;
    Ok(())
}

mod thread_state {
    use pyo3::ffi;
    use std::cell::Cell;
    use std::sync::{Condvar, Mutex};

    thread_local! {
        static KEPT: Cell<Option<(ffi::PyGILState_STATE, *mut ffi::PyThreadState)>> =
            const { Cell::new(None) };
    }

    /// Whether workers may still take the GIL, and how many are doing so.
    static GATE: Mutex<(bool, usize)> = Mutex::new((false, 0));
    static IDLE: Condvar = Condvar::new();

    /// Run `op` unless the gate is closed, keeping `close` waiting until it finishes.
    fn through_gate(op: impl FnOnce()) {
        {
            let mut gate = GATE.lock().unwrap_or_else(|e| e.into_inner());
            if gate.0 {
                return;
            }
            gate.1 += 1;
        }
        op();
        let mut gate = GATE.lock().unwrap_or_else(|e| e.into_inner());
        gate.1 -= 1;
        IDLE.notify_all();
    }

    /// Close the gate and wait for workers inside it. Call without holding the GIL.
    pub fn close() {
        let mut gate = GATE.lock().unwrap_or_else(|e| e.into_inner());
        gate.0 = true;
        while gate.1 > 0 {
            gate = IDLE.wait(gate).unwrap_or_else(|e| e.into_inner());
        }
    }

    fn interpreter_running() -> bool {
        // SAFETY: both only read interpreter status and are safe to call from any thread.
        unsafe { ffi::Py_IsInitialized() != 0 && ffi::Py_IsFinalizing() == 0 }
    }

    pub fn keep() {
        through_gate(|| {
            if !interpreter_running() {
                return;
            }
            // SAFETY: the interpreter is running and cannot start finalising until the gate
            // is passed. The state is detached before returning, so no GIL is held between tasks.
            let kept = unsafe {
                let gil = ffi::PyGILState_Ensure();
                (gil, ffi::PyEval_SaveThread())
            };
            KEPT.with(|cell| cell.set(Some(kept)));
        });
    }

    /// Free the kept state, or leak it after shutdown starts, for finalisation to free.
    pub fn release() {
        let Some((gil, state)) = KEPT.with(Cell::take) else {
            return;
        };
        through_gate(|| {
            if !interpreter_running() {
                return;
            }
            // SAFETY: `state` is this thread's own state, saved by `keep` and not yet released,
            // and the interpreter cannot start finalising until the gate is passed.
            unsafe {
                ffi::PyEval_RestoreThread(state);
                ffi::PyGILState_Release(gil);
            }
        });
    }
}
