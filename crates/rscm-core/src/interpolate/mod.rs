/// 1d interpolation
///
///
/// # Technical implementation
/// Dynamic dispatch was difficult to implement because of the generics associated with
/// the `Interp1DStrategy` trait. These generics make the trait not object-safe which
/// doesn't allow the
///
/// Instead static dispatching was implemented using the enum `InterpolationStrategy` which
/// implements the `Interp1DStrategy` trait.
///
/// Static dispatching decreases the ability for consumers of this library to implement their
/// own custom interpolation strategies. This isn't intended as a generic implementation of
/// interpolation routines so that is a satisfactory tradeoff.
///
///
use crate::errors::RSCMResult;
use num::Float;
use numpy::ndarray::{ArrayBase, Data};
use numpy::Ix1;
use strategies::{Interp1DStrategy, InterpolationStrategy};

pub mod strategies;

/// Interpolator
pub struct Interp1d<At, Ay>
where
    At: Data,
    Ay: Data,
{
    time: ArrayBase<At, Ix1>,
    // TODO: Expand to support shape (t, ...)
    y: ArrayBase<Ay, Ix1>,
    strategy: InterpolationStrategy,
}

impl<At, Ay> Interp1d<At, Ay>
where
    At: Data,
    At::Elem: Float,
    Ay: Data,
    Ay::Elem: Float + From<At::Elem>,
{
    pub fn new(
        time: ArrayBase<At, Ix1>,
        y: ArrayBase<Ay, Ix1>,
        strategy: InterpolationStrategy,
    ) -> Self {
        Self { time, y, strategy }
    }
    pub fn with_strategy(&mut self, strategy: InterpolationStrategy) -> &mut Self {
        self.strategy = strategy;
        self
    }

    pub fn interpolate(&self, time_target: At::Elem) -> RSCMResult<Ay::Elem> {
        self.strategy.interpolate(&self.time, &self.y, time_target)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpolate::strategies::next::NextStrategy;
    use numpy::array;
    use numpy::ndarray::Array;

    #[test]
    fn exterpolate() {
        let data = array![1.0, 1.5, 2.0];
        let years = Array::range(2020.0, 2023.0, 1.0);
        let query = 2024.0;
        let expected = 2.0;

        let interpolator = Interp1d::new(
            years,
            data,
            InterpolationStrategy::from(NextStrategy::new(true)),
        );
        let result = interpolator.interpolate(query).unwrap();

        assert_eq!(result, expected);
    }

    #[test]
    fn interpolate_with_view() {
        let data = array![1.0, 1.5, 2.0];
        let years = Array::range(2020.0, 2023.0, 1.0);
        let query = 2024.0;
        let expected = 2.0;

        let interpolator = Interp1d::new(
            years.view(),
            data,
            InterpolationStrategy::from(NextStrategy::new(true)),
        );
        let result = interpolator.interpolate(query).unwrap();

        assert_eq!(result, expected);
    }
}
