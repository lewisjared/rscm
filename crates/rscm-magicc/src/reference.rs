//! Fixed calendar-year references used by MAGICC7 feedbacks

use rscm_core::state::ScalarWindow;
use rscm_core::timeseries::{FloatValue, Time};

/// Value of an annual series at `year`, clamped to the stored time axis.
///
/// Feedback references in MAGICC7 are fixed calendar years, which may lie
/// before the run starts or after the current timestep.
pub(crate) fn value_at_year(window: &ScalarWindow, t_current: Time, year: Time) -> FloatValue {
    let index = window.index() as isize;
    let last = window.len() as isize - 1;
    let offset = ((year - t_current).round() as isize).clamp(-index, last - index);
    window
        .at_offset(offset)
        .unwrap_or_else(|| window.at_start())
}

/// Reference for a feedback starting at `start_year`.
///
/// MAGICC7 follows the current value until then and fixes the reference at the year before.
pub(crate) fn reference_value(
    window: &ScalarWindow,
    t_current: Time,
    start_year: Time,
) -> FloatValue {
    if t_current < start_year {
        window.at_start()
    } else {
        value_at_year(window, t_current, start_year - 1.0)
    }
}

/// Warming since `start_year`, zero before it and never negative.
///
/// This is MAGICC7's temperature feedback driver for the carbon cycle.
pub(crate) fn feedback_warming(
    temperature: &ScalarWindow,
    t_current: Time,
    start_year: Time,
) -> FloatValue {
    if t_current < start_year {
        return 0.0;
    }
    (temperature.at_start() - value_at_year(temperature, t_current, start_year)).max(0.0)
}
