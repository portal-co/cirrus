//! Composed security budgeting for the no-ZK DirectSparse profile.
//!
//! Mirrors the pinned upstream `spartan-whir` `security.rs`
//! (`derive_direct_component_security` with `SpartanSoundnessMode::NoZk`):
//! the requested security level is split across three error components —
//! Spartan algebraic checks over the extension field, WHIR arguments, and
//! Poseidon commitment binding — and each component receives slack for the
//! number of events an adversary may combine. Only the exact integer
//! arithmetic of the upstream `BigUint` computation is reproduced (via fixed
//! 192-bit limbs over `alloc`-free `u128` math); the SPARK and full-ZK modes
//! are out of scope.

use core::fmt;

use crate::field::KOALABEAR_MODULUS;
use crate::whir::spartan::{MAX_SECURITY_BITS, MIN_SECURITY_BITS};

/// The component that bounds the attainable security level.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SecurityBoundComponent {
    /// The KoalaBear quintic extension field size.
    ExtensionField,
    /// The WHIR proximity arguments.
    WhirArguments,
    /// The Poseidon Merkle commitments.
    PoseidonCommitments,
}

impl fmt::Display for SecurityBoundComponent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ExtensionField => f.write_str("extension field"),
            Self::WhirArguments => f.write_str("whir arguments"),
            Self::PoseidonCommitments => f.write_str("poseidon commitments"),
        }
    }
}

/// Why the composed security budget cannot be derived.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SecurityBudgetError {
    /// A checked integer computation overflowed.
    BudgetOverflow,
    /// The requested level exceeds what the component bounds allow.
    ComposedSecurityUnavailable {
        /// Requested (effective) security level in bits.
        requested_bits: u32,
        /// Attainable security level in bits.
        attainable_bits: u32,
        /// The component that bounds the attainable level.
        dominant_component: SecurityBoundComponent,
    },
    /// The slack-raised component level exceeds the digest-supported maximum.
    ComponentSecurityAboveMaximum,
}

impl fmt::Display for SecurityBudgetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BudgetOverflow => f.write_str("security budget overflow"),
            Self::ComposedSecurityUnavailable {
                requested_bits,
                attainable_bits,
                dominant_component,
            } => write!(
                f,
                "requested security {requested_bits} bits exceeds attainable \
                 {attainable_bits} bits (bounded by {dominant_component})"
            ),
            Self::ComponentSecurityAboveMaximum => {
                f.write_str("slack-raised component security exceeds the maximum")
            }
        }
    }
}

/// The derived per-component security levels, mirroring upstream's
/// component `SecurityConfig` fields.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ComponentSecurity {
    /// Security level assigned to the WHIR arguments.
    pub security_level_bits: u32,
    /// Security level assigned to the Poseidon Merkle commitments.
    pub merkle_security_bits: u32,
}

/// The full composed budget, mirroring upstream's `ComposedSecurityBudget`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ComposedSecurityBudget {
    /// Requested effective security level.
    pub requested_bits: u32,
    /// Attainable effective security level.
    pub attainable_bits: u32,
    /// Number of Spartan algebraic error events.
    pub algebraic_error_terms: usize,
    /// Number of WHIR arguments (one for DirectSparse).
    pub whir_argument_count: usize,
    /// Number of Poseidon commitment binding events.
    pub commitment_binding_events: usize,
    /// Slack assigned to the WHIR component.
    pub whir_slack_bits: u32,
    /// Slack assigned to the Merkle component.
    pub merkle_slack_bits: u32,
    /// The component bounding the attainable level.
    pub dominant_component: SecurityBoundComponent,
}

/// Number of Spartan algebraic error events for the no-ZK profile,
/// mirroring upstream `spartan_algebraic_error_terms(SpartanSoundnessMode::NoZk)`:
/// `4 * num_outer_rounds + 2 * num_inner_rounds + 2`.
pub fn spartan_algebraic_error_terms_no_zk(
    num_outer_rounds: usize,
    num_inner_rounds: usize,
) -> Result<usize, SecurityBudgetError> {
    num_outer_rounds
        .checked_mul(4)
        .and_then(|terms| {
            num_inner_rounds
                .checked_mul(2)
                .and_then(|inner| terms.checked_add(inner))
        })
        .and_then(|terms| terms.checked_add(2))
        .ok_or(SecurityBudgetError::BudgetOverflow)
}

/// `ceil(log2(3 * events))`, mirroring upstream `three_way_budget_slack`
/// (computed with fixed 64-bit arithmetic so the result is
/// platform-independent).
fn three_way_budget_slack(events: usize) -> Result<u32, SecurityBudgetError> {
    let weighted = events
        .checked_mul(3)
        .ok_or(SecurityBudgetError::BudgetOverflow)? as u64;
    Ok(u64::BITS - weighted.saturating_sub(1).leading_zeros())
}

/// `bits(floor(p^5 / divisor)) - 1` where `p` is the KoalaBear prime and
/// `bits` is the upstream `BigUint::bits` convention (zero for zero,
/// otherwise `floor(log2(q)) + 1`). Computed with exact 192-bit limb
/// arithmetic; `p^5 < 2^156`.
fn extension_order_quotient_bits(divisor: u64) -> u32 {
    debug_assert!(divisor > 0);
    let p = u128::from(KOALABEAR_MODULUS);
    let p2 = p * p; // < 2^62
    let p4 = p2 * p2; // < 2^124
    // p^5 = p4 * p as three little-endian u64 limbs.
    let p4_lo = u128::from(p4 as u64);
    let p4_hi = p4 >> 64;
    let a = p4_lo * p; // < 2^95
    let b = p4_hi * p; // < 2^126
    let limb0 = a as u64;
    let mid = (a >> 64) + u128::from(b as u64);
    let limb1 = mid as u64;
    let limb2 = (mid >> 64) + (b >> 64);
    let mut limbs = [limb0, limb1, limb2 as u64];

    // Long division of the 192-bit numerator by the u64 divisor.
    let divisor = u128::from(divisor);
    let mut remainder = 0u128;
    let mut quotient = [0u64; 3];
    for index in (0..3).rev() {
        let current = (remainder << 64) | u128::from(limbs[index]);
        quotient[index] = (current / divisor) as u64;
        remainder = current % divisor;
    }
    limbs = quotient;

    for index in (0..3).rev() {
        if limbs[index] != 0 {
            return 64 * index as u32 + (u64::BITS - limbs[index].leading_zeros());
        }
    }
    0
}

/// Derive the per-component security levels for a no-ZK DirectSparse
/// Spartan proof, mirroring upstream `derive_direct_component_security` with
/// one WHIR argument and `witness_whir_rounds + 1` commitment events.
///
/// `witness_whir_rounds` is the number of intermediate WHIR folding rounds
/// for the witness polynomial (`compute_number_of_rounds(num_variables,
/// schedule)`), `num_outer_rounds = log2(num_cons)`, and
/// `num_inner_rounds = log2(num_vars) + 1`.
pub fn derive_direct_component_security(
    requested_security_level_bits: u32,
    requested_merkle_security_bits: u32,
    witness_whir_rounds: usize,
    num_outer_rounds: usize,
    num_inner_rounds: usize,
) -> Result<(ComponentSecurity, ComposedSecurityBudget), SecurityBudgetError> {
    let algebraic_error_terms =
        spartan_algebraic_error_terms_no_zk(num_outer_rounds, num_inner_rounds)?;
    let commitment_binding_events = witness_whir_rounds
        .checked_add(1)
        .ok_or(SecurityBudgetError::BudgetOverflow)?;
    let whir_argument_count = 1usize;

    let requested_bits = requested_security_level_bits.min(requested_merkle_security_bits);
    let whir_slack_bits = three_way_budget_slack(whir_argument_count)?;
    let merkle_slack_bits = three_way_budget_slack(commitment_binding_events)?;

    let divisor = (algebraic_error_terms as u64)
        .checked_mul(3)
        .ok_or(SecurityBudgetError::BudgetOverflow)?;
    let field_attainable = extension_order_quotient_bits(divisor).saturating_sub(1);
    let whir_attainable = MAX_SECURITY_BITS.saturating_sub(whir_slack_bits);
    let merkle_attainable = MAX_SECURITY_BITS.saturating_sub(merkle_slack_bits);
    let (attainable_bits, dominant_component) = [
        (field_attainable, SecurityBoundComponent::ExtensionField),
        (whir_attainable, SecurityBoundComponent::WhirArguments),
        (
            merkle_attainable,
            SecurityBoundComponent::PoseidonCommitments,
        ),
    ]
    .into_iter()
    .min_by_key(|(bits, _)| *bits)
    .expect("three security components");

    let budget = ComposedSecurityBudget {
        requested_bits,
        attainable_bits,
        algebraic_error_terms,
        whir_argument_count,
        commitment_binding_events,
        whir_slack_bits,
        merkle_slack_bits,
        dominant_component,
    };
    if requested_bits > attainable_bits {
        return Err(SecurityBudgetError::ComposedSecurityUnavailable {
            requested_bits,
            attainable_bits,
            dominant_component,
        });
    }

    let component = ComponentSecurity {
        security_level_bits: requested_bits
            .checked_add(whir_slack_bits)
            .ok_or(SecurityBudgetError::BudgetOverflow)?,
        merkle_security_bits: requested_bits
            .checked_add(merkle_slack_bits)
            .ok_or(SecurityBudgetError::BudgetOverflow)?,
    };
    if component.security_level_bits > MAX_SECURITY_BITS
        || component.merkle_security_bits > MAX_SECURITY_BITS
        || component.security_level_bits < MIN_SECURITY_BITS
        || component.merkle_security_bits < MIN_SECURITY_BITS
    {
        return Err(SecurityBudgetError::ComponentSecurityAboveMaximum);
    }
    Ok((component, budget))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slack_matches_ceil_log2_of_three_times_events() {
        assert_eq!(three_way_budget_slack(1).unwrap(), 2); // 3 -> 2
        assert_eq!(three_way_budget_slack(2).unwrap(), 3); // 6 -> 3
        assert_eq!(three_way_budget_slack(5).unwrap(), 4); // 15 -> 4
        assert_eq!(three_way_budget_slack(11).unwrap(), 6); // 33 -> 6
    }

    #[test]
    fn extension_order_quotient_bits_is_exact() {
        // p^5 / 3 has 154 bits of magnitude: log2(p^5) ~= 154.98, so
        // bits(p^5 / 3) = 154 and the attainable level is 153.
        assert_eq!(extension_order_quotient_bits(3), 154);
        // Dividing by larger algebraic budgets only removes a few bits.
        assert_eq!(extension_order_quotient_bits(3 * 1024), 144);
    }
}
