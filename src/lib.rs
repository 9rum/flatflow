// SPDX-License-Identifier: Apache-2.0

use pyo3::pymodule;

pub mod ops;
pub mod sched;

#[pymodule]
mod ffi {
    use pyo3::types::PyModule;
    use pyo3::{Bound, PyResult};

    // This may bind `flatflow::sched::sched` and `flatflow::sched::sched_unstable` to
    // `flatflow.ffi.sched` and `flatflow.ffi.sched_unstable` in the Python frontend, respectively.
    #[pymodule_export]
    use super::sched::sched;
    #[pymodule_export]
    use super::sched::sched_unstable;

    #[pymodule_export]
    #[allow(non_upper_case_globals)]
    const __version__: &str = env!("CARGO_PKG_VERSION");

    #[pymodule_init]
    #[allow(unused_variables)]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        pyo3_log::init();

        Ok(())
    }
}
