use crate::errors::RSCMResult;
use crate::interpolate::strategies::{find_segment, Interp1DStrategy, SegmentOptions};
use num::Float;
use numpy::ndarray::{ArrayBase, Data};
use numpy::Ix1;
use std::cmp::min;

/// Next-value 1D interpolation
///
/// The interpolated value is always equal to the next value in the array
/// from which to interpolate.
///
/// This can be confusing to think about.
///
/// At the boundaries (i.e time(i)) we return values(i).
/// For other values of time_target between time(i) and time(i + 1),
/// we always take y(i + 1) (i.e. we always take the 'next' value).
/// As a result,
/// y_target = y(i + 1) for time(i) < time_target <= time(i + 1)
///
/// If helpful, we have drawn a picture of how this works below.
/// Symbols:
/// - time: y-value selected for this time-value
/// - i: closed (i.e. inclusive) boundary
/// - o: open (i.e. exclusive) boundary
///
/// y(4):                                    oxxxxxxxxxxxxxxxxxxxxxxxxxx
/// y(3):                        oxxxxxxxxxxxi
/// y(2):            oxxxxxxxxxxxi
/// y(1): xxxxxxxxxxxi
///       -----------|-----------|-----------|-----------|--------------
///               time(1)     time(2)     time(3)     time(4)
///
/// One other way to think about this is
/// that the y-values are shifted to the left compared to the time-values.
/// As a result, y(1) is only used for (backward) extrapolation,
/// it isn't actually used in the interpolation domain at all.
#[derive(Clone)]
pub struct NextStrategy {
    extrapolate: bool,
}

impl NextStrategy {
    pub fn new(extrapolate: bool) -> Self {
        Self { extrapolate }
    }
}

impl<At, Ay> Interp1DStrategy<At, Ay> for NextStrategy
where
    At: Data,
    At::Elem: Float,
    Ay: Data,
    Ay::Elem: Float,
{
    fn interpolate(
        &self,
        time: &ArrayBase<At, Ix1>,
        y: &ArrayBase<Ay, Ix1>,
        time_target: At::Elem,
    ) -> RSCMResult<Ay::Elem> {
        let segment_info = find_segment(time_target, time, self.extrapolate);

        let (segment_options, end_segment_idx) = match segment_info {
            Ok(info) => info,
            Err(e) => return Err(e),
        };
        // Clip the index to exclude the last bound
        let end_segment_idx = min(end_segment_idx, y.len() - 1);

        if segment_options == SegmentOptions::OnBoundary {
            // Fast return
            return Ok(y[end_segment_idx]);
        }

        let res = match segment_options {
            SegmentOptions::ExtrapolateBackward => y[0],
            SegmentOptions::ExtrapolateForward => y[y.len() - 1],
            SegmentOptions::InSegment | SegmentOptions::OnBoundary => y[end_segment_idx],
        };

        Ok(res)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use is_close::is_close;
    use numpy::array;
    use std::iter::zip;

    #[test]
    fn test_next() {
        let time = array![0.0, 0.5, 1.0, 1.5];
        let y = array![5.0, 8.0, 9.0];

        let target = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let exps = vec![5.0, 8.0, 8.0, 9.0, 9.0];

        let strategy = NextStrategy::new(false);

        zip(target.into_iter(), exps.into_iter()).for_each(|(t, e)| {
            println!("target={}, expected={}", t, e);
            assert!(is_close!(strategy.interpolate(&time, &y, t).unwrap(), e));
        })
    }

    #[test]
    fn test_next_extrapolation_error() {
        let time = array![0.0, 1.0];
        let y = array![5.0];

        let target = vec![-1.0, -0.01, 1.01, 1.2];

        let strategy = NextStrategy::new(false);

        target.into_iter().for_each(|t| {
            println!("target={t}");
            let res = strategy.interpolate(&time, &y, t);
            assert!(res.is_err());

            let err = res.err().unwrap();
            assert!(err.to_string().starts_with("Extrapolation is not allowed"))
        })
    }

    #[test]
    fn test_next_extrapolation() {
        let time = array![0.0, 0.5, 1.0, 1.5];
        let y = array![5.0, 8.0, 9.0];

        let target = vec![-1.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.2];
        let exps = vec![5.0, 5.0, 8.0, 8.0, 9.0, 9.0, 9.0];

        let strategy = NextStrategy::new(true);

        zip(target.into_iter(), exps.into_iter()).for_each(|(t, e)| {
            let value = strategy.interpolate(&time, &y, t).unwrap();
            println!("target={}, expected={} found={}", t, e, value);
            assert!(is_close!(value, e));
        })
    }
}
