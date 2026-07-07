// SPDX-License-Identifier: Apache-2.0

use pyo3::types::{PyModule, PyModuleMethods};
use pyo3::{Bound, PyResult, pymodule, wrap_pyfunction};

pub mod ops;
pub mod sched;

#[pymodule]
fn ffi<'py>(m: &Bound<'py, PyModule>) -> PyResult<()> {
    pyo3_log::init();

    // This may bind `flatflow::sched::sched` and `flatflow::sched::sched_unstable` to
    // `flatflow.ffi.sched` and `flatflow.ffi.sched_unstable` in the Python frontend, respectively.
    m.add_function(wrap_pyfunction!(sched::sched, m)?)?;
    m.add_function(wrap_pyfunction!(sched::sched_unstable, m)?)?;

    Ok(())
}
