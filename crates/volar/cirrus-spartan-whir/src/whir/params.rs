//! WHIR protocol parameters and derived configuration.
//!
//! This mirrors the pinned upstream `p3-whir` parameter derivation
//! (`parameters/whir.rs` plus `p3-security`'s WHIR error budgets) for the
//! KoalaBear base field and its quintic extension. All floating-point
//! formulas use `libm` so the derivation is `no_std`-reproducible.

use alloc::vec::Vec;
use core::f64::consts::LOG2_10;
use core::fmt;

use crate::dft::KOALABEAR_TWO_ADICITY;

/// Bits of the KoalaBear quintic extension field (`5 * 31`).
pub const QUINTIC_FIELD_SIZE_BITS: usize = 155;

/// Variables left for the final direct-send sumcheck phase, matching
/// upstream's `MAX_NUM_VARIABLES_TO_SEND_COEFFS`.
pub const MAX_NUM_VARIABLES_TO_SEND_COEFFS: usize = 6;

/// Proximity regime selector for the RS proximity analysis, mirroring
/// upstream `p3_security::whir::SecurityAssumption`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SecurityAssumption {
    /// Unique decoding: each oracle is within the unique-decoding radius.
    /// No list-decoding conjectures.
    UniqueDecoding,
    /// Johnson bound at `delta = 1 - sqrt(rho) - eta` with `eta = sqrt(rho)/20`.
    JohnsonBound,
    /// Capacity bound at `delta = 1 - rho - eta` with `eta = rho/20`.
    CapacityBound,
}

impl SecurityAssumption {
    /// `log2(eta)` for the regime's safety gap below its distance.
    fn log_eta(self, log_inv_rate: usize) -> f64 {
        match self {
            Self::UniqueDecoding => unreachable!("log_eta is undefined for UniqueDecoding"),
            // eta = sqrt(rho) / 20
            Self::JohnsonBound => -(0.5 * log_inv_rate as f64 + LOG2_10 + 1.),
            // eta = rho / 20
            Self::CapacityBound => -(log_inv_rate as f64 + LOG2_10 + 1.),
        }
    }

    /// `log2(L)` for the regime's list size at its distance.
    fn list_size_bits(self, log_degree: usize, log_inv_rate: usize) -> f64 {
        match self {
            Self::UniqueDecoding => 0.,
            Self::JohnsonBound => {
                let log_eta = self.log_eta(log_inv_rate);
                let log_inv_sqrt_rate = log_inv_rate as f64 / 2.;
                log_inv_sqrt_rate - (1. + log_eta)
            }
            Self::CapacityBound => (log_degree + log_inv_rate) as f64 - self.log_eta(log_inv_rate),
        }
    }

    /// \[BCSS25\] Theorem 1.5 dominant term in bits at `m = 10`:
    /// `log2(2 * (m + 1/2)^5 / (3 * rho^{3/2}) * n)`.
    fn jb_prox_gaps_dominant_term_bits(log_degree: usize, log_inv_rate: usize) -> f64 {
        let log_n = (log_degree + log_inv_rate) as f64;
        let constant = libm::log2(2. * libm::pow(10.5_f64, 5.) / 3.);
        let log_rho_neg_3_2 = 1.5 * log_inv_rate as f64;
        log_n + constant + log_rho_neg_3_2
    }

    /// Proximity-gap error in bits for combining `num_functions` functions.
    fn prox_gaps_error(
        self,
        log_degree: usize,
        log_inv_rate: usize,
        field_size_bits: usize,
        num_functions: usize,
    ) -> f64 {
        debug_assert!(num_functions >= 2);
        let error = match self {
            Self::UniqueDecoding => (log_degree + log_inv_rate) as f64,
            Self::JohnsonBound => Self::jb_prox_gaps_dominant_term_bits(log_degree, log_inv_rate),
            Self::CapacityBound => {
                (log_degree + 2 * log_inv_rate) as f64 - self.log_eta(log_inv_rate)
            }
        };
        let num_functions_1_log = libm::log2(num_functions as f64 - 1.);
        field_size_bits as f64 - (error + num_functions_1_log)
    }

    /// `log2(1 - delta)` for the regime's proximity parameter.
    fn log_1_delta(self, log_inv_rate: usize) -> f64 {
        let rate = 1. / f64::from(1 << log_inv_rate);
        let delta = match self {
            Self::UniqueDecoding => 0.5 * (1. - rate),
            Self::JohnsonBound => 1. - libm::sqrt(rate) - libm::pow(2., self.log_eta(log_inv_rate)),
            Self::CapacityBound => 1. - rate - libm::pow(2., self.log_eta(log_inv_rate)),
        };
        libm::log2(1. - delta)
    }

    /// Number of queries needed to reach `protocol_security_level` bits.
    pub fn queries(self, protocol_security_level: usize, log_inv_rate: usize) -> usize {
        let num_queries_f = -(protocol_security_level as f64) / self.log_1_delta(log_inv_rate);
        libm::ceil(num_queries_f) as usize
    }

    /// Bits of security from `num_queries` queries.
    fn queries_error(self, log_inv_rate: usize, num_queries: usize) -> f64 {
        -(num_queries as f64) * self.log_1_delta(log_inv_rate)
    }

    /// OOD sampling error in bits (STIR Lemma 4.5).
    fn ood_error(
        self,
        log_degree: usize,
        log_inv_rate: usize,
        field_size_bits: usize,
        ood_samples: usize,
    ) -> f64 {
        if matches!(self, Self::UniqueDecoding) {
            return 0.;
        }
        let list_size_bits = self.list_size_bits(log_degree, log_inv_rate);
        let error = 2. * list_size_bits + (log_degree * ood_samples) as f64;
        (ood_samples * field_size_bits) as f64 + 1. - error
    }

    /// Smallest OOD sample count reaching `security_level` bits, or `None`
    /// when the field is too small. Unique decoding uses no OOD samples.
    fn determine_ood_samples(
        self,
        security_level: usize,
        log_degree: usize,
        log_inv_rate: usize,
        field_size_bits: usize,
    ) -> Option<usize> {
        if matches!(self, Self::UniqueDecoding) {
            return Some(0);
        }
        (1..64).find(|&ood_samples| {
            self.ood_error(log_degree, log_inv_rate, field_size_bits, ood_samples)
                >= security_level as f64
        })
    }

    /// Sumcheck soundness term for the fold step, in bits.
    fn fold_sumcheck_error(
        self,
        field_size_bits: usize,
        num_variables: usize,
        log_inv_rate: usize,
    ) -> f64 {
        let list_size = self.list_size_bits(num_variables, log_inv_rate);
        field_size_bits as f64 - (list_size + 1.)
    }

    /// Query-combination soundness in bits.
    fn queries_combination_error(
        self,
        field_size_bits: usize,
        num_variables: usize,
        log_inv_rate: usize,
        ood_samples: usize,
        num_queries: usize,
    ) -> f64 {
        let list_size = self.list_size_bits(num_variables, log_inv_rate);
        let log_combination = libm::log2((ood_samples + num_queries) as f64);
        field_size_bits as f64 - (log_combination + list_size + 1.)
    }

    /// PoW bits needed at the fold step to reach `security_level`.
    fn folding_pow_bits(
        self,
        security_level: usize,
        field_size_bits: usize,
        num_variables: usize,
        log_inv_rate: usize,
    ) -> f64 {
        let prox_gaps_error = self.prox_gaps_error(num_variables, log_inv_rate, field_size_bits, 2);
        let sumcheck_error = self.fold_sumcheck_error(field_size_bits, num_variables, log_inv_rate);
        let error = prox_gaps_error.min(sumcheck_error);
        0_f64.max(security_level as f64 - error)
    }
}

/// Folding-factor strategy, mirroring upstream `p3_whir::parameters::FoldingFactor`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FoldingFactor {
    /// A fixed folding factor used in all rounds.
    Constant(usize),
    /// A different factor for the first round and a fixed one for the rest.
    ConstantFromSecondRound(usize, usize),
    /// Explicit folding factors for each pre-direct-send folding phase.
    PerRound(Vec<usize>),
}

/// Why a folding schedule cannot be derived.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FoldingFactorError {
    /// A zero folding factor.
    ZeroFactor,
    /// The folding factor exceeds the (remaining) number of variables.
    TooLarge(usize, usize),
    /// An empty explicit schedule.
    EmptySchedule,
    /// An explicit schedule stops above the direct-send threshold.
    InsufficientFolding {
        /// Total variables.
        num_variables: usize,
        /// Variables left unfolded by the schedule.
        remaining: usize,
    },
}

impl fmt::Display for FoldingFactorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroFactor => f.write_str("folding factor must be positive"),
            Self::TooLarge(factor, variables) => {
                write!(f, "folding factor {factor} exceeds {variables} variables")
            }
            Self::EmptySchedule => f.write_str("explicit folding schedule must not be empty"),
            Self::InsufficientFolding {
                num_variables,
                remaining,
            } => write!(
                f,
                "folding schedule leaves {remaining} of {num_variables} variables above the direct-send threshold"
            ),
        }
    }
}

impl core::error::Error for FoldingFactorError {}

impl FoldingFactor {
    /// Derive the concrete per-phase folding factors, mirroring upstream's
    /// `compute_folding_schedule`.
    pub fn compute_folding_schedule(
        &self,
        num_variables: usize,
    ) -> Result<Vec<usize>, FoldingFactorError> {
        match self {
            Self::Constant(factor) => {
                if *factor == 0 {
                    return Err(FoldingFactorError::ZeroFactor);
                }
                if *factor > num_variables {
                    return Err(FoldingFactorError::TooLarge(*factor, num_variables));
                }
                let mut remaining = num_variables;
                let mut schedule = Vec::new();
                loop {
                    let round_factor = (*factor).min(remaining);
                    schedule.push(round_factor);
                    remaining -= round_factor;
                    if remaining <= MAX_NUM_VARIABLES_TO_SEND_COEFFS {
                        return Ok(schedule);
                    }
                }
            }
            Self::ConstantFromSecondRound(first_round_factor, factor) => {
                if *first_round_factor == 0 || *factor == 0 {
                    return Err(FoldingFactorError::ZeroFactor);
                }
                if *first_round_factor > num_variables {
                    return Err(FoldingFactorError::TooLarge(
                        *first_round_factor,
                        num_variables,
                    ));
                }
                let mut remaining = num_variables;
                let mut schedule = alloc::vec![*first_round_factor];
                remaining -= *first_round_factor;
                while remaining > MAX_NUM_VARIABLES_TO_SEND_COEFFS {
                    let round_factor = (*factor).min(remaining);
                    schedule.push(round_factor);
                    remaining -= round_factor;
                }
                Ok(schedule)
            }
            Self::PerRound(factors) => {
                if factors.is_empty() {
                    return Err(FoldingFactorError::EmptySchedule);
                }
                for &factor in factors {
                    if factor == 0 {
                        return Err(FoldingFactorError::ZeroFactor);
                    }
                    if factor > num_variables {
                        return Err(FoldingFactorError::TooLarge(factor, num_variables));
                    }
                }
                let mut remaining = num_variables;
                let mut schedule = Vec::new();
                for &factor in factors {
                    if factor > remaining {
                        return Err(FoldingFactorError::TooLarge(factor, remaining));
                    }
                    schedule.push(factor);
                    remaining -= factor;
                    if remaining <= MAX_NUM_VARIABLES_TO_SEND_COEFFS {
                        return Ok(schedule);
                    }
                }
                Err(FoldingFactorError::InsufficientFolding {
                    num_variables,
                    remaining,
                })
            }
        }
    }
}

/// User-facing WHIR protocol parameters, mirroring upstream
/// `ProtocolParameters`.
#[derive(Clone, Debug, PartialEq)]
pub struct ProtocolParameters {
    /// Initial logarithmic inverse rate for the first committed codeword.
    pub starting_log_inv_rate: usize,
    /// Log-inverse rates for the codewords committed after each intermediate
    /// round; empty derives them from the folding schedule.
    pub round_log_inv_rates: Vec<usize>,
    /// The folding-factor strategy.
    pub folding_factor: FoldingFactor,
    /// The proximity soundness regime.
    pub soundness_type: SecurityAssumption,
    /// Target security level in bits.
    pub security_level: usize,
    /// Grinding budget: the maximum derived PoW difficulty accepted.
    pub pow_bits: usize,
}

/// Derived configuration for a single intermediate WHIR round.
#[derive(Clone, Debug, PartialEq)]
pub struct RoundConfig {
    /// PoW difficulty (bits) for the STIR query phase.
    pub pow_bits: usize,
    /// PoW difficulty (bits) for the folding sumcheck phase.
    pub folding_pow_bits: usize,
    /// Number of STIR proximity queries in this round.
    pub num_queries: usize,
    /// Number of out-of-domain evaluation samples.
    pub ood_samples: usize,
    /// Variables remaining after folding in this round.
    pub num_variables: usize,
    /// Variables folded in this round.
    pub folding_factor: usize,
    /// Log-inverse rate of the codeword committed after this round.
    pub log_inv_rate: usize,
    /// Evaluation domain size before folding in this round.
    pub domain_size: usize,
    /// Generator of the folded evaluation domain after this round's fold.
    pub folded_domain_gen: crate::KoalaBear,
}

/// Why user-facing parameters cannot form a valid WHIR configuration.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum WhirConfigError {
    /// The folding factor is incompatible with the polynomial size.
    FoldingFactor(FoldingFactorError),
    /// The domain after the first fold exceeds the base-field two-adicity.
    FoldedDomainExceedsTwoAdicity {
        /// Log2 of the folded domain size.
        log_folded_domain_size: usize,
        /// Base-field two-adicity.
        two_adicity: usize,
    },
    /// The initial domain size cannot be represented as a `usize`.
    InitialDomainExceedsUsize {
        /// Number of multilinear variables.
        num_variables: usize,
        /// Starting log-inverse rate.
        starting_log_inv_rate: usize,
    },
    /// Explicit per-round codeword rates have the wrong length.
    RoundRateCountMismatch {
        /// Expected count.
        expected: usize,
        /// Supplied count.
        actual: usize,
    },
    /// A requested codeword rate would grow the RS domain.
    RateGrowsDomain {
        /// The offending round.
        round: usize,
    },
    /// The starting code rate is not redundant.
    NonRedundantStartingRate {
        /// The offending log-inverse rate.
        log_inv_rate: usize,
    },
    /// A per-round code rate is not redundant.
    NonRedundantRoundRate {
        /// The offending round.
        round: usize,
        /// The offending log-inverse rate.
        log_inv_rate: usize,
    },
    /// No OOD sample count reaches the requested security level.
    OodSamplesInfeasible {
        /// Target security level.
        security_level: usize,
        /// Extension-field size in bits.
        field_size_bits: usize,
    },
    /// A derived PoW difficulty exceeds the grinding budget.
    PowBitsExceedBudget {
        /// Required bits.
        required: usize,
        /// Configured budget.
        budget: usize,
    },
}

impl fmt::Display for WhirConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FoldingFactor(error) => write!(f, "{error}"),
            Self::FoldedDomainExceedsTwoAdicity {
                log_folded_domain_size,
                two_adicity,
            } => write!(
                f,
                "folded domain 2^{log_folded_domain_size} exceeds base-field two-adicity 2^{two_adicity}"
            ),
            Self::InitialDomainExceedsUsize {
                num_variables,
                starting_log_inv_rate,
            } => write!(
                f,
                "initial domain exponent {num_variables} + {starting_log_inv_rate} does not fit usize"
            ),
            Self::RoundRateCountMismatch { expected, actual } => write!(
                f,
                "expected {expected} explicit codeword rates, got {actual}"
            ),
            Self::RateGrowsDomain { round } => {
                write!(f, "round {round}: requested rate would grow the RS domain")
            }
            Self::NonRedundantStartingRate { log_inv_rate } => write!(
                f,
                "starting log-inv-rate 2^-{log_inv_rate} is a non-redundant code"
            ),
            Self::NonRedundantRoundRate {
                round,
                log_inv_rate,
            } => write!(
                f,
                "round {round}: log-inv-rate 2^-{log_inv_rate} is a non-redundant code"
            ),
            Self::OodSamplesInfeasible {
                security_level,
                field_size_bits,
            } => write!(
                f,
                "no OOD sample count reaches {security_level}-bit security with a {field_size_bits}-bit field"
            ),
            Self::PowBitsExceedBudget { required, budget } => write!(
                f,
                "derived proof-of-work of {required} bits exceeds the {budget}-bit grinding budget"
            ),
        }
    }
}

impl core::error::Error for WhirConfigError {}

/// Round a real-valued PoW gap up to whole grinding bits, matching
/// upstream's `ceil_pow_bits`.
fn ceil_pow_bits(gap: f64) -> usize {
    libm::ceil(gap) as usize
}

/// Fully derived WHIR protocol configuration, mirroring upstream
/// `WhirConfig::new` for the KoalaBear quintic profile.
#[derive(Clone, Debug, PartialEq)]
pub struct WhirConfig {
    /// Number of variables in the original multilinear polynomial.
    pub num_variables: usize,
    /// The user-facing parameters this config was derived from.
    pub params: ProtocolParameters,
    /// Per-round derived configuration for each intermediate STIR round.
    pub round_parameters: Vec<RoundConfig>,
    /// Concrete folding factors used before the final direct-send phase.
    pub folding_schedule: Vec<usize>,
    /// OOD samples during the commitment phase.
    pub commitment_ood_samples: usize,
    /// PoW bits for the initial folding sumcheck.
    pub starting_folding_pow_bits: usize,
    /// STIR queries in the final proximity test.
    pub final_queries: usize,
    /// PoW bits for the final STIR query phase.
    pub final_pow_bits: usize,
    /// Sumcheck rounds in the final phase.
    pub final_sumcheck_rounds: usize,
    /// PoW bits for the final folding sumcheck.
    pub final_folding_pow_bits: usize,
}

impl WhirConfig {
    /// Derive a full protocol configuration from user-facing parameters,
    /// mirroring upstream `WhirConfig::new` with
    /// `field_size_bits = QUINTIC_FIELD_SIZE_BITS`.
    pub fn new(
        num_variables: usize,
        whir_parameters: ProtocolParameters,
    ) -> Result<Self, WhirConfigError> {
        let initial_num_variables = num_variables;

        if whir_parameters.starting_log_inv_rate == 0 {
            return Err(WhirConfigError::NonRedundantStartingRate {
                log_inv_rate: whir_parameters.starting_log_inv_rate,
            });
        }

        let folding_schedule = whir_parameters
            .folding_factor
            .compute_folding_schedule(num_variables)
            .map_err(WhirConfigError::FoldingFactor)?;

        let protocol_security_level = whir_parameters
            .security_level
            .saturating_sub(whir_parameters.pow_bits);

        let field_size_bits = QUINTIC_FIELD_SIZE_BITS;

        let mut log_inv_rate = whir_parameters.starting_log_inv_rate;
        let mut num_variables = num_variables;

        let log_domain_size = num_variables
            .checked_add(log_inv_rate)
            .filter(|&log_domain_size| log_domain_size < usize::BITS as usize)
            .ok_or(WhirConfigError::InitialDomainExceedsUsize {
                num_variables,
                starting_log_inv_rate: log_inv_rate,
            })?;
        let mut domain_size: usize = 1 << log_domain_size;

        let log_folded_domain_size = log_domain_size - folding_schedule[0];
        if log_folded_domain_size > KOALABEAR_TWO_ADICITY {
            return Err(WhirConfigError::FoldedDomainExceedsTwoAdicity {
                log_folded_domain_size,
                two_adicity: KOALABEAR_TWO_ADICITY,
            });
        }

        let folded_variables: usize = folding_schedule.iter().sum();
        let num_rounds = folding_schedule.len() - 1;
        let final_sumcheck_rounds = num_variables - folded_variables;

        let round_log_inv_rates = if whir_parameters.round_log_inv_rates.is_empty() {
            let mut rates = Vec::with_capacity(num_rounds);
            let mut rate = whir_parameters.starting_log_inv_rate;
            for &folding_factor in folding_schedule.iter().take(num_rounds) {
                rate += folding_factor - 1;
                rates.push(rate);
            }
            rates
        } else {
            if whir_parameters.round_log_inv_rates.len() != num_rounds {
                return Err(WhirConfigError::RoundRateCountMismatch {
                    expected: num_rounds,
                    actual: whir_parameters.round_log_inv_rates.len(),
                });
            }
            whir_parameters.round_log_inv_rates.clone()
        };

        if let Some(round) = round_log_inv_rates.iter().position(|&rate| rate == 0) {
            return Err(WhirConfigError::NonRedundantRoundRate {
                round,
                log_inv_rate: round_log_inv_rates[round],
            });
        }

        if let FoldingFactor::PerRound(factors) = &whir_parameters.folding_factor
            && factors.len() != num_rounds + 1
        {
            return Err(WhirConfigError::RoundRateCountMismatch {
                expected: num_rounds + 1,
                actual: factors.len(),
            });
        }

        let soundness = whir_parameters.soundness_type;
        let security_level = whir_parameters.security_level;

        let commitment_ood_samples = soundness
            .determine_ood_samples(security_level, num_variables, log_inv_rate, field_size_bits)
            .ok_or(WhirConfigError::OodSamplesInfeasible {
                security_level,
                field_size_bits,
            })?;

        let starting_folding_pow_bits = soundness.folding_pow_bits(
            security_level,
            field_size_bits,
            num_variables,
            log_inv_rate,
        );

        let mut round_parameters = Vec::with_capacity(num_rounds);
        num_variables -= folding_schedule[0];

        for (round, &next_rate) in round_log_inv_rates.iter().enumerate() {
            let folding_factor = folding_schedule[round];
            if next_rate > log_inv_rate + folding_factor {
                return Err(WhirConfigError::RateGrowsDomain { round });
            }
            let rs_reduction_factor = log_inv_rate + folding_factor - next_rate;

            let num_queries = soundness.queries(protocol_security_level, log_inv_rate);

            let ood_samples = soundness
                .determine_ood_samples(security_level, num_variables, next_rate, field_size_bits)
                .ok_or(WhirConfigError::OodSamplesInfeasible {
                    security_level,
                    field_size_bits,
                })?;

            let query_error = soundness.queries_error(log_inv_rate, num_queries);
            let combination_error = soundness.queries_combination_error(
                field_size_bits,
                num_variables,
                next_rate,
                ood_samples,
                num_queries,
            );

            let pow_bits = 0_f64.max(security_level as f64 - query_error.min(combination_error));

            let folding_pow_bits = soundness.folding_pow_bits(
                security_level,
                field_size_bits,
                num_variables,
                next_rate,
            );

            let next_folding_factor = folding_schedule[round + 1];

            let folded_domain_gen =
                crate::dft::two_adic_generator(domain_size.ilog2() as usize - folding_factor);

            round_parameters.push(RoundConfig {
                pow_bits: ceil_pow_bits(pow_bits),
                folding_pow_bits: ceil_pow_bits(folding_pow_bits),
                num_queries,
                ood_samples,
                num_variables,
                folding_factor,
                log_inv_rate: next_rate,
                domain_size,
                folded_domain_gen,
            });

            num_variables -= next_folding_factor;
            log_inv_rate = next_rate;
            domain_size >>= rs_reduction_factor;
        }

        let final_queries = soundness.queries(protocol_security_level, log_inv_rate);

        let final_pow_bits =
            0_f64.max(security_level as f64 - soundness.queries_error(log_inv_rate, final_queries));

        let final_folding_pow_bits =
            0_f64.max(security_level as f64 - (field_size_bits - 1) as f64);

        debug_assert_eq!(
            initial_num_variables,
            folding_schedule.iter().sum::<usize>() + final_sumcheck_rounds
        );

        let config = Self {
            params: whir_parameters,
            commitment_ood_samples,
            num_variables: initial_num_variables,
            starting_folding_pow_bits: ceil_pow_bits(starting_folding_pow_bits),
            round_parameters,
            folding_schedule,
            final_queries,
            final_pow_bits: ceil_pow_bits(final_pow_bits),
            final_sumcheck_rounds,
            final_folding_pow_bits: ceil_pow_bits(final_folding_pow_bits),
        };

        debug_assert_eq!(
            config.final_round_config().num_variables,
            config.final_sumcheck_rounds
        );

        let required = config.max_pow_bits();
        if required > config.params.pow_bits {
            return Err(WhirConfigError::PowBitsExceedBudget {
                required,
                budget: config.params.pow_bits,
            });
        }

        Ok(config)
    }

    /// Size of the initial evaluation domain.
    pub fn starting_domain_size(&self) -> usize {
        1 << (self.num_variables + self.params.starting_log_inv_rate)
    }

    /// Number of intermediate STIR rounds (excludes the final round).
    pub fn n_rounds(&self) -> usize {
        self.round_parameters.len()
    }

    /// How many bits the RS domain shrinks by at the given round.
    pub fn rs_reduction_factor(&self, round: usize) -> usize {
        let previous_log_inv_rate = if round == 0 {
            self.params.starting_log_inv_rate
        } else {
            self.round_parameters[round - 1].log_inv_rate
        };
        previous_log_inv_rate + self.round_folding_factor(round)
            - self.round_parameters[round].log_inv_rate
    }

    /// Largest PoW difficulty demanded by any phase.
    pub fn max_pow_bits(&self) -> usize {
        let outer = self
            .starting_folding_pow_bits
            .max(self.final_pow_bits)
            .max(self.final_folding_pow_bits);
        self.round_parameters
            .iter()
            .map(|r| r.pow_bits.max(r.folding_pow_bits))
            .fold(outer, usize::max)
    }

    /// Concrete derived folding factor for a given round.
    pub fn round_folding_factor(&self, round: usize) -> usize {
        self.folding_schedule[round]
    }

    /// Total variables folded through the given round index, inclusive.
    pub fn total_folded_through(&self, round: usize) -> usize {
        self.folding_schedule.iter().take(round + 1).sum()
    }

    /// The synthetic `RoundConfig` for the final direct-send phase.
    pub fn final_round_config(&self) -> RoundConfig {
        if self.round_parameters.is_empty() {
            RoundConfig {
                num_variables: self.num_variables - self.round_folding_factor(0),
                folding_factor: self.round_folding_factor(self.n_rounds()),
                num_queries: self.final_queries,
                pow_bits: self.final_pow_bits,
                log_inv_rate: self.params.starting_log_inv_rate,
                domain_size: self.starting_domain_size(),
                folded_domain_gen: crate::dft::two_adic_generator(
                    self.starting_domain_size().ilog2() as usize - self.round_folding_factor(0),
                ),
                ood_samples: 0,
                folding_pow_bits: self.final_folding_pow_bits,
            }
        } else {
            let rs_reduction_factor = self.rs_reduction_factor(self.n_rounds() - 1);
            let folding_factor = self.round_folding_factor(self.n_rounds());
            let last = self.round_parameters.last().expect("nonempty rounds");
            let domain_size = last.domain_size >> rs_reduction_factor;
            let folded_domain_gen = crate::dft::two_adic_generator(
                domain_size.ilog2() as usize - self.round_folding_factor(self.n_rounds()),
            );
            RoundConfig {
                num_variables: last.num_variables - folding_factor,
                folding_factor,
                num_queries: self.final_queries,
                pow_bits: self.final_pow_bits,
                log_inv_rate: last.log_inv_rate,
                domain_size,
                folded_domain_gen,
                ood_samples: 0,
                folding_pow_bits: self.final_folding_pow_bits,
            }
        }
    }

    /// Inverse rate of the codeword committed after an intermediate round.
    pub fn inv_rate(&self, round: usize) -> usize {
        1 << self.round_parameters[round].log_inv_rate
    }
}
