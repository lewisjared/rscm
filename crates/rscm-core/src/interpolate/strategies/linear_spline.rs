use crate::errors::RSCMResult;
use crate::interpolate::strategies::{find_segment, Interp1DStrategy, SegmentOptions};
use num::Float;
use numpy::ndarray::{s, ArrayBase, Data};
use numpy::Ix1;
use std::cmp::min;

/// LinearSpline 1D interpolation
///
/// The interpolated value is
/// derived from a linear interpolation of the two points
/// to either side of time_target.
///
/// The resulting curve is therefore only zero-order continuous.
#[derive(Clone)]
pub struct LinearSplineStrategy {
    extrapolate: bool,
}

impl LinearSplineStrategy {
    pub fn new(extrapolate: bool) -> Self {
        Self { extrapolate }
    }
}

impl<At, Ay> Interp1DStrategy<At, Ay> for LinearSplineStrategy
where
    At: Data,
    At::Elem: Float,
    Ay: Data,
    Ay::Elem: Float + From<At::Elem>,
{
    fn interpolate(
        &self,
        time: &ArrayBase<At, Ix1>,
        y: &ArrayBase<Ay, Ix1>,
        time_target: At::Elem,
    ) -> RSCMResult<Ay::Elem> {
        let segment_info = find_segment(
            time_target,
            // Trim off the last bound as it isn't needed for linear extrapolation
            &time.slice(s![..time.len() - 1]),
            self.extrapolate,
        );

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

        let (time1, time2, y1, y2) = match segment_options {
            SegmentOptions::ExtrapolateBackward => {
                // Use first two points
                let time1 = time[0];
                let y1 = y[0];

                let time2 = time[1];
                let y2 = y[1];

                (time1, time2, y1, y2)
            }
            SegmentOptions::ExtrapolateForward => {
                assert!(y.len() >= 2);
                // Use last two points (excludes the influence of the bound of the last value
                let time1 = time[y.len() - 2];
                let y1 = y[y.len() - 2];

                let time2 = time[y.len() - 1];
                let y2 = y[y.len() - 1];

                (time1, time2, y1, y2)
            }
            SegmentOptions::InSegment | SegmentOptions::OnBoundary => {
                // Use points surrounding time_target
                let time1 = time[end_segment_idx - 1];
                let y1 = y[end_segment_idx - 1];

                let time2 = time[end_segment_idx];
                let y2 = y[end_segment_idx];

                (time1, time2, y1, y2)
            }
        };

        let time1: Ay::Elem = time1.into();
        let time2: Ay::Elem = time2.into();
        let time_target: Ay::Elem = time_target.into();

        let m = (y2 - y1) / (time2 - time1);

        Ok(m * (time_target - time1) + y1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use is_close::is_close;
    use numpy::array;
    use std::iter::zip;

    #[test]
    fn test_linear() {
        let time = array![0.0, 0.5, 1.0, 1.5];
        let y = array![5.0, 8.0, 9.0];

        let target = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let exps = vec![5.0, 6.5, 8.0, 8.5, 9.0];

        let strategy = LinearSplineStrategy::new(false);

        zip(target.into_iter(), exps.into_iter()).for_each(|(t, e)| {
            println!("target={}, expected={}", t, e);
            assert!(is_close!(strategy.interpolate(&time, &y, t).unwrap(), e));
        })
    }

    #[test]
    fn test_linear_extrapolation_error() {
        let time = array![0.0, 1.0];
        let y = array![5.0];

        let target = vec![-1.0, -0.01, 1.01, 1.2];

        let strategy = LinearSplineStrategy::new(false);

        target.into_iter().for_each(|t| {
            println!("target={t}");
            let res = strategy.interpolate(&time, &y, t);
            assert!(res.is_err());

            let err = res.err().unwrap();
            assert!(err.to_string().starts_with("Extrapolation is not allowed"))
        })
    }

    #[test]
    fn test_linear_extrapolation() {
        let time = array![0.0, 0.5, 1.0, 1.5];
        let y = array![5.0, 8.0, 9.0];

        let target = vec![1.5, 2.0];
        let exps = vec![10.0, 11.0];

        let strategy = LinearSplineStrategy::new(true);

        zip(target.into_iter(), exps.into_iter()).for_each(|(t, e)| {
            let res = strategy.interpolate(&time, &y, t).unwrap();
            println!("target={}, expected={}, found={}", t, e, res);
            assert!(is_close!(res, e));
        })
    }
}
