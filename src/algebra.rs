//! Quantale scalar fragment.

pub const Q_BOTTOM: f32 = 0.0;
pub const Q_UNIT: f32 = 1.0;

/// Backwards-compatible aliases. In this version bottom/unit are not infinities.
pub const NEG_INF: f32 = Q_BOTTOM;
pub const POS_INF: f32 = Q_UNIT;

pub fn clamp_quantale_value(value: f32) -> f32 {
    if value.is_nan() || value <= Q_BOTTOM {
        Q_BOTTOM
    } else if value >= Q_UNIT {
        Q_UNIT
    } else {
        value
    }
}
