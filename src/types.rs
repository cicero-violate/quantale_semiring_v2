//! Strong wrappers for quantale scalar primitives.

use std::cmp::Ordering;
use std::ops::{Add, AddAssign, Mul, MulAssign};

use serde::{Deserialize, Serialize};

use crate::algebra::{clamp_quantale_value, BOTTOM, Q_UNIT};

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct QuantaleWeight(pub f32);

impl QuantaleWeight {
    pub const BOTTOM: Self = Self(BOTTOM);
    pub const UNIT: Self = Self(Q_UNIT);

    pub const fn zero() -> Self {
        Self::BOTTOM
    }

    pub const fn one() -> Self {
        Self::UNIT
    }

    pub fn new(value: f32) -> Self {
        Self(clamp_quantale_value(value))
    }

    pub const fn inner(self) -> f32 {
        self.0
    }

    pub const fn raw(self) -> f32 {
        self.inner()
    }

    pub fn join(self, rhs: Self) -> Self {
        self + rhs
    }

    pub fn compose(self, rhs: Self) -> Self {
        self * rhs
    }
}

impl Add for QuantaleWeight {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.0.max(rhs.0))
    }
}

impl AddAssign for QuantaleWeight {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Mul for QuantaleWeight {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(self.0 * rhs.0)
    }
}

impl MulAssign for QuantaleWeight {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl PartialEq for QuantaleWeight {
    #[inline]
    fn eq(&self, rhs: &Self) -> bool {
        self.0 == rhs.0
    }
}

impl Eq for QuantaleWeight {}

impl PartialOrd for QuantaleWeight {
    #[inline]
    fn partial_cmp(&self, rhs: &Self) -> Option<Ordering> {
        Some(self.cmp(rhs))
    }
}

impl Ord for QuantaleWeight {
    #[inline]
    fn cmp(&self, rhs: &Self) -> Ordering {
        self.0.total_cmp(&rhs.0)
    }
}

impl From<f32> for QuantaleWeight {
    fn from(value: f32) -> Self {
        Self::new(value)
    }
}

impl From<QuantaleWeight> for f32 {
    fn from(value: QuantaleWeight) -> Self {
        value.0
    }
}
