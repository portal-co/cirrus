//! Plain (no-ZK) WHIR polynomial commitment scheme.
//!
//! Mirrors the pinned upstream `p3-whir` `WhirProver`/`WhirVerifier` for the
//! KoalaBear -> quintic-extension profile with the Poseidon Merkle backend
//! and prefix variable order: Reed-Solomon/MLE commitment, sumcheck-based
//! folding rounds, out-of-domain sampling, proof-of-work grinding, STIR
//! query derivation with pruned Merkle multiproofs, and the final
//! direct-send polynomial.

use alloc::vec::Vec;
use core::fmt;

use crate::merkle::{MerkleError, PoseidonMerkleTree, PrunedMerklePaths};
use crate::poly::evaluate_mle_table;
use crate::whir::params::{RoundConfig, WhirConfig};
use crate::whir::sumcheck::{
    Constraint, EqStatement, SelectStatement, Statements, SumcheckData, WhirSumcheckError,
    WhirSumcheckProver, eval_constraints_poly, verify_final_sumcheck_rounds,
};
use crate::{KoalaBear, PoseidonDigest, PoseidonTranscript, QuinticExtension};

/// WHIR PCS failure modes, mirroring upstream `VerifierError` plus the
/// structural prover-side checks.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum WhirError {
    /// The proof carries the wrong number of intermediate rounds.
    RoundCountMismatch {
        /// Expected round count.
        expected: usize,
        /// Supplied round count.
        actual: usize,
    },
    /// A round entry is present but its Merkle-root slot is empty.
    MissingRoundCommitment {
        /// The offending round.
        round: usize,
    },
    /// A round's OOD answer count differs from the derived configuration.
    RoundOodAnswerCountMismatch {
        /// The offending round.
        round: usize,
        /// Expected count.
        expected: usize,
        /// Supplied count.
        actual: usize,
    },
    /// A proof-of-work witness failed the grinding check.
    InvalidPowWitness,
    /// The initial commitment OOD answer count differs from the
    /// configuration.
    InitialOodAnswerCountMismatch {
        /// Expected count.
        expected: usize,
        /// Supplied count.
        actual: usize,
    },
    /// The opened-row count differs from the queried-index count.
    StirQueryCountMismatch {
        /// The offending round.
        round_index: usize,
        /// Expected count.
        expected: usize,
        /// Supplied count.
        actual: usize,
    },
    /// A Merkle multiproof failed verification or its openings field does
    /// not match the round.
    MerkleProofInvalid,
    /// The final polynomial is absent.
    MissingFinalPoly,
    /// The final polynomial has the wrong evaluation count.
    FinalPolyLengthMismatch {
        /// Expected count.
        expected: usize,
        /// Supplied count.
        actual: usize,
    },
    /// The final STIR constraint evaluation failed on the final polynomial.
    StirChallengeFailed,
    /// The final `claimed_eval == weights * f(r)` identity failed.
    SumcheckFailed,
    /// A sumcheck sub-protocol failed.
    Sumcheck(WhirSumcheckError),
    /// A Merkle tree operation failed.
    Merkle(MerkleError),
    /// A DFT domain request was invalid for the configuration.
    InvalidDftDomain,
    /// The committed polynomial has the wrong length for the configuration.
    InvalidPolynomialLength {
        /// Expected evaluation count.
        expected: usize,
        /// Supplied count.
        actual: usize,
    },
}

impl fmt::Display for WhirError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RoundCountMismatch { expected, actual } => write!(
                f,
                "WHIR proof round count mismatch: expected {expected}, got {actual}"
            ),
            Self::MissingRoundCommitment { round } => {
                write!(f, "WHIR round {round} is missing its Merkle commitment")
            }
            Self::RoundOodAnswerCountMismatch {
                round,
                expected,
                actual,
            } => write!(
                f,
                "WHIR round {round} OOD answer count mismatch: expected {expected}, got {actual}"
            ),
            Self::InvalidPowWitness => f.write_str("invalid WHIR proof-of-work witness"),
            Self::InitialOodAnswerCountMismatch { expected, actual } => write!(
                f,
                "WHIR initial OOD answer count mismatch: expected {expected}, got {actual}"
            ),
            Self::StirQueryCountMismatch {
                round_index,
                expected,
                actual,
            } => write!(
                f,
                "WHIR round {round_index} STIR query count mismatch: expected {expected}, got {actual}"
            ),
            Self::MerkleProofInvalid => f.write_str("WHIR Merkle multiproof verification failed"),
            Self::MissingFinalPoly => f.write_str("WHIR proof is missing the final polynomial"),
            Self::FinalPolyLengthMismatch { expected, actual } => write!(
                f,
                "WHIR final polynomial length mismatch: expected {expected}, got {actual}"
            ),
            Self::StirChallengeFailed => {
                f.write_str("WHIR STIR constraint verification failed on the final polynomial")
            }
            Self::SumcheckFailed => f.write_str("WHIR final sumcheck identity failed"),
            Self::Sumcheck(error) => write!(f, "{error}"),
            Self::Merkle(error) => write!(f, "{error}"),
            Self::InvalidDftDomain => f.write_str("invalid WHIR DFT domain"),
            Self::InvalidPolynomialLength { expected, actual } => write!(
                f,
                "committed polynomial length mismatch: expected {expected}, got {actual}"
            ),
        }
    }
}

impl core::error::Error for WhirError {}

impl From<WhirSumcheckError> for WhirError {
    fn from(error: WhirSumcheckError) -> Self {
        Self::Sumcheck(error)
    }
}

impl From<MerkleError> for WhirError {
    fn from(error: MerkleError) -> Self {
        Self::Merkle(error)
    }
}

/// Rows opened at many queried positions, plus one shared pruned multiproof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SharedProofOpening<T> {
    /// `rows[q]` is the opened leaf row at the `q`-th queried position.
    pub rows: Vec<Vec<T>>,
    /// Compact multiproof authenticating every row at once.
    pub proof: PrunedMerklePaths,
}

/// Field-tagged shared-proof opening for one queried oracle.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum QueryOpenings {
    /// Base-field rows (the initial commitment).
    Base(SharedProofOpening<KoalaBear>),
    /// Extension-field rows (every folded round commitment).
    Extension(SharedProofOpening<QuinticExtension>),
}

/// Per-round WHIR proof data.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WhirRoundProof {
    /// Round commitment (Merkle root).
    pub commitment: Option<PoseidonDigest>,
    /// OOD evaluations for this round.
    pub ood_answers: Vec<QuinticExtension>,
    /// PoW witness after commitment.
    pub pow_witness: KoalaBear,
    /// STIR query openings against the previous round's commitment.
    pub openings: QueryOpenings,
    /// Sumcheck data for this round.
    pub sumcheck: SumcheckData,
}

/// Complete plain WHIR proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WhirProof {
    /// Initial OOD evaluations.
    pub initial_ood_answers: Vec<QuinticExtension>,
    /// Initial sumcheck data.
    pub initial_sumcheck: SumcheckData,
    /// Per-round proofs.
    pub rounds: Vec<WhirRoundProof>,
    /// Final polynomial evaluations (sent in the clear).
    pub final_poly: Option<Vec<QuinticExtension>>,
    /// Final round PoW witness.
    pub final_pow_witness: KoalaBear,
    /// Final round STIR query openings.
    pub final_openings: QueryOpenings,
    /// Final sumcheck data (present iff `final_sumcheck_rounds > 0`).
    pub final_sumcheck: Option<SumcheckData>,
}

/// Prover-side data retained from a WHIR commitment.
pub struct WhirProverData {
    /// The committed base-field polynomial (MLE evaluation table).
    pub polynomial: Vec<KoalaBear>,
    /// Merkle tree over the encoded initial codeword matrix.
    pub tree: PoseidonMerkleTree,
    /// Encoded initial codeword matrix rows.
    pub encoded_rows: Vec<Vec<KoalaBear>>,
    /// OOD answers sampled at commit time (already in the transcript).
    pub initial_ood_answers: Vec<QuinticExtension>,
    /// `(point, eval)` pairs for the commitment-phase OOD samples.
    pub ood_pairs: Vec<(Vec<QuinticExtension>, QuinticExtension)>,
    /// Number of variables of the committed polynomial.
    pub num_variables: usize,
}

/// Expand a univariate challenge into a multilinear point via the
/// big-endian power map, mirroring upstream `Point::expand_from_univariate`.
pub fn expand_from_univariate(
    point: QuinticExtension,
    num_variables: usize,
) -> Vec<QuinticExtension> {
    let mut res = alloc::vec![QuinticExtension::ZERO; num_variables];
    let mut cur = point;
    for i in (0..num_variables).rev() {
        res[i] = cur;
        cur = cur * cur;
    }
    res
}

/// Evaluate a base-field MLE table at an extension point, matching
/// upstream's `Poly::eval_base`.
pub fn eval_mle_base_at_ext(table: &[KoalaBear], point: &[QuinticExtension]) -> QuinticExtension {
    assert_eq!(table.len(), 1 << point.len());
    let mut folded: Vec<QuinticExtension> = table.iter().map(|&v| v.into()).collect();
    for &r in point {
        crate::whir::sumcheck::fix_prefix_var(&mut folded, r);
    }
    debug_assert_eq!(folded.len(), 1);
    folded[0]
}

/// Sample `num_queries` distinct STIR query indices from the folded domain,
/// mirroring upstream `get_challenge_stir_queries`.
pub fn get_challenge_stir_queries(
    domain_size: usize,
    folding_factor: usize,
    num_queries: usize,
    transcript: &mut PoseidonTranscript,
) -> Vec<usize> {
    let folded_domain_size = domain_size >> folding_factor;
    debug_assert!(folded_domain_size.is_power_of_two());
    let domain_size_bits = folded_domain_size.ilog2() as usize;
    let target = num_queries.min(folded_domain_size);

    let mut queries: Vec<usize> = Vec::with_capacity(target);
    while queries.len() < target {
        let q = transcript
            .sample_uniform_bits(domain_size_bits)
            .expect("validated bit counts sample successfully");
        if !queries.contains(&q) {
            queries.push(q);
        }
    }
    queries.sort_unstable();
    queries
}

/// Encode and commit the initial base-field polynomial, prefix order.
/// Mirrors upstream `commit_base` with `VariableOrder::Prefix`.
pub fn commit_base_prefix(
    poly: &[KoalaBear],
    folding: usize,
    starting_log_inv_rate: usize,
) -> Result<(Vec<Vec<KoalaBear>>, PoseidonMerkleTree), WhirError> {
    let num_variables = poly.len().ilog2() as usize;
    let height = 1 << (num_variables + starting_log_inv_rate - folding);
    let width = 1 << folding;
    let src_width = 1 << (num_variables - folding);

    let mut values = alloc::vec![KoalaBear::ZERO; height * width];
    // Transpose folding blocks into the leading rows; trailing rows stay
    // zero and become the Reed-Solomon expansion.
    for r in 0..src_width {
        for c in 0..width {
            values[r * width + c] = poly[c * src_width + r];
        }
    }
    crate::dft::dft_batch_base(&mut values, height, width)
        .map_err(|_| WhirError::InvalidDftDomain)?;
    let rows: Vec<Vec<KoalaBear>> = values.chunks_exact(width).map(|row| row.to_vec()).collect();
    let tree = PoseidonMerkleTree::commit_rows(&rows)?;
    Ok((rows, tree))
}

/// Encode and commit a folded extension-field polynomial, prefix order.
/// Mirrors upstream `commit_extension` with `VariableOrder::Prefix`; the
/// committed Merkle rows are the coefficient-flattened base-field view,
/// exactly like upstream's `ExtensionMmcs`.
pub fn commit_extension_prefix(
    evals: &[QuinticExtension],
    folding: usize,
    inv_rate: usize,
) -> Result<(Vec<Vec<KoalaBear>>, PoseidonMerkleTree), WhirError> {
    let num_variables = evals.len().ilog2() as usize;
    let height = inv_rate * (1 << (num_variables - folding));
    let width = 1 << folding;
    let src_width = 1 << (num_variables - folding);

    let mut values = alloc::vec![QuinticExtension::ZERO; height * width];
    for r in 0..src_width {
        for c in 0..width {
            values[r * width + c] = evals[c * src_width + r];
        }
    }
    crate::dft::dft_batch_ext(&mut values, height, width)
        .map_err(|_| WhirError::InvalidDftDomain)?;
    let rows: Vec<Vec<KoalaBear>> = values
        .chunks_exact(width)
        .map(|row| {
            let mut flat = Vec::with_capacity(width * crate::QUINTIC_DEGREE);
            for element in row {
                flat.extend(
                    element
                        .canonical_coefficients()
                        .map(|c| KoalaBear::from_u64(u64::from(c))),
                );
            }
            flat
        })
        .collect();
    let tree = PoseidonMerkleTree::commit_rows(&rows)?;
    Ok((rows, tree))
}

/// Regroup a coefficient-flattened base row into extension elements.
fn unflatten_ext_row(row: &[KoalaBear]) -> Vec<QuinticExtension> {
    debug_assert_eq!(row.len() % crate::QUINTIC_DEGREE, 0);
    row.chunks_exact(crate::QUINTIC_DEGREE)
        .map(|coefficients| QuinticExtension::new(core::array::from_fn(|i| coefficients[i])))
        .collect()
}

/// Flatten an extension row into the committed base-field view.
fn flatten_ext_row(row: &[QuinticExtension]) -> Vec<KoalaBear> {
    let mut flat = Vec::with_capacity(row.len() * crate::QUINTIC_DEGREE);
    for element in row {
        flat.extend(
            element
                .canonical_coefficients()
                .map(|c| KoalaBear::from_u64(u64::from(c))),
        );
    }
    flat
}

/// Active Merkle data for the polynomial currently being queried.
enum RoundData {
    Base {
        tree: PoseidonMerkleTree,
        rows: Vec<Vec<KoalaBear>>,
    },
    Ext {
        tree: PoseidonMerkleTree,
        rows: Vec<Vec<KoalaBear>>,
    },
}

/// Execute the full WHIR proving protocol, mirroring upstream
/// `WhirProver::prove` for the prefix order.
#[allow(clippy::too_many_arguments)]
pub fn whir_prove(
    config: &WhirConfig,
    initial_ood_answers: Vec<QuinticExtension>,
    claims: Vec<(Vec<QuinticExtension>, QuinticExtension)>,
    polynomial: &[KoalaBear],
    base_tree: PoseidonMerkleTree,
    base_rows: Vec<Vec<KoalaBear>>,
    transcript: &mut PoseidonTranscript,
) -> Result<WhirProof, WhirError> {
    // Initial batched equality constraint over the statement claims plus
    // the commitment-phase OOD pairs.
    let mut eq_statement = EqStatement::initialize(config.num_variables);
    for (point, eval) in claims {
        eq_statement.add_evaluated_constraint(point, eval);
    }
    let alpha = transcript.sample_quintic();
    let initial_constraint = Constraint::new(
        alpha,
        config.num_variables,
        alloc::vec![Statements::Eq(eq_statement)],
    );
    let (weights, sum) = initial_constraint.combine_new();
    let evals: Vec<QuinticExtension> = polynomial.iter().map(|&v| v.into()).collect();
    let mut prover = WhirSumcheckProver::new(evals, weights, sum);

    let mut initial_sumcheck = SumcheckData::default();
    let mut folding_randomness = prover.compute_sumcheck_polynomials(
        &mut initial_sumcheck,
        transcript,
        config.round_folding_factor(0),
        config.starting_folding_pow_bits,
        None,
    );

    let mut round_data = RoundData::Base {
        tree: base_tree,
        rows: base_rows,
    };

    let mut rounds = Vec::with_capacity(config.n_rounds());
    for round_index in 0..config.n_rounds() {
        let num_variables = config.num_variables - config.total_folded_through(round_index);
        debug_assert_eq!(num_variables, prover.num_variables());

        let round_params = &config.round_parameters[round_index];
        let folding_factor_next = config.round_folding_factor(round_index + 1);
        let inv_rate = config.inv_rate(round_index);

        // Commit the current folded polynomial.
        let (rows, tree) = commit_extension_prefix(prover.evals(), folding_factor_next, inv_rate)?;
        let root = tree.root();
        transcript.observe_slice(&root);

        // OOD sampling.
        let mut ood_statement = EqStatement::initialize(num_variables);
        let mut ood_answers = Vec::with_capacity(round_params.ood_samples);
        for _ in 0..round_params.ood_samples {
            let point = expand_from_univariate(transcript.sample_quintic(), num_variables);
            let eval = prover.eval_at(&point);
            transcript.observe_quintic(eval);
            ood_answers.push(eval);
            ood_statement.add_evaluated_constraint(point, eval);
        }

        // PoW grinding before queries.
        let pow_witness = if round_params.pow_bits > 0 {
            transcript
                .grind(round_params.pow_bits)
                .map_err(|_| WhirError::InvalidPowWitness)?
        } else {
            KoalaBear::ZERO
        };

        // Transcript checkpoint between PoW and query generation.
        let _checkpoint = transcript.sample_base();

        // STIR query sampling.
        let stir_indices = get_challenge_stir_queries(
            round_params.domain_size,
            config.round_folding_factor(round_index),
            round_params.num_queries,
            transcript,
        );

        let query_randomness = folding_randomness.clone();
        let mut stir_statement = SelectStatement::initialize(num_variables);

        let openings = match &round_data {
            RoundData::Base { tree, rows } => {
                let (opened, proof) = tree.open_multi(&stir_indices, rows)?;
                for (row, &index) in opened.iter().zip(&stir_indices) {
                    let eval = eval_mle_base_at_ext(row, &query_randomness);
                    let var = round_params.folded_domain_gen.pow(index as u64);
                    stir_statement.add_constraint(var, eval);
                }
                QueryOpenings::Base(SharedProofOpening {
                    rows: opened,
                    proof,
                })
            }
            RoundData::Ext { tree, rows } => {
                let (opened, proof) = tree.open_multi(&stir_indices, rows)?;
                let mut ext_rows = Vec::with_capacity(opened.len());
                for (row, &index) in opened.iter().zip(&stir_indices) {
                    let ext_row = unflatten_ext_row(row);
                    let eval = evaluate_mle_table(&ext_row, &query_randomness)
                        .expect("row width matches the folding randomness");
                    let var = round_params.folded_domain_gen.pow(index as u64);
                    stir_statement.add_constraint(var, eval);
                    ext_rows.push(ext_row);
                }
                QueryOpenings::Extension(SharedProofOpening {
                    rows: ext_rows,
                    proof,
                })
            }
        };

        // Batch OOD and STIR statements under a fresh challenge.
        let constraint = Constraint::new(
            transcript.sample_quintic(),
            num_variables,
            alloc::vec![
                Statements::Eq(ood_statement),
                Statements::Select(stir_statement),
            ],
        );

        let mut sumcheck = SumcheckData::default();
        folding_randomness = prover.compute_sumcheck_polynomials(
            &mut sumcheck,
            transcript,
            folding_factor_next,
            round_params.folding_pow_bits,
            Some(&constraint),
        );

        round_data = RoundData::Ext { tree, rows };
        rounds.push(WhirRoundProof {
            commitment: Some(root),
            ood_answers,
            pow_witness,
            openings,
            sumcheck,
        });
    }

    // Final round: send the polynomial in the clear, open the last queries.
    let final_poly: Vec<QuinticExtension> = prover.evals().to_vec();
    transcript.observe_quintic_slice(&final_poly);

    let final_pow_witness = if config.final_pow_bits > 0 {
        transcript
            .grind(config.final_pow_bits)
            .map_err(|_| WhirError::InvalidPowWitness)?
    } else {
        KoalaBear::ZERO
    };

    let final_indices = get_challenge_stir_queries(
        config.final_round_config().domain_size,
        config.round_folding_factor(config.n_rounds()),
        config.final_queries,
        transcript,
    );

    let final_openings = match &round_data {
        RoundData::Base { tree, rows } => {
            let (opened, proof) = tree.open_multi(&final_indices, rows)?;
            QueryOpenings::Base(SharedProofOpening {
                rows: opened,
                proof,
            })
        }
        RoundData::Ext { tree, rows } => {
            let (opened, proof) = tree.open_multi(&final_indices, rows)?;
            QueryOpenings::Extension(SharedProofOpening {
                rows: opened.iter().map(|row| unflatten_ext_row(row)).collect(),
                proof,
            })
        }
    };

    let final_sumcheck = if config.final_sumcheck_rounds > 0 {
        let mut data = SumcheckData::default();
        prover.compute_sumcheck_polynomials(
            &mut data,
            transcript,
            config.final_sumcheck_rounds,
            config.final_folding_pow_bits,
            None,
        );
        Some(data)
    } else {
        None
    };

    Ok(WhirProof {
        initial_ood_answers,
        initial_sumcheck,
        rounds,
        final_poly: Some(final_poly),
        final_pow_witness,
        final_openings,
        final_sumcheck,
    })
}

/// A parsed commitment: Merkle root plus the commitment-phase OOD claims.
pub struct ParsedCommitment {
    /// Merkle root of the committed evaluation table.
    pub root: PoseidonDigest,
    /// OOD challenge points and their claimed evaluations.
    pub ood_statement: EqStatement,
}

/// Parse the initial commitment from a proof, mirroring the OOD portion of
/// upstream `parse_plain_initial_commitment`. The caller observes the PCS
/// domain separator and the root first.
pub fn parse_initial_ood(
    config: &WhirConfig,
    proof: &WhirProof,
    transcript: &mut PoseidonTranscript,
) -> Result<EqStatement, WhirError> {
    if proof.initial_ood_answers.len() != config.commitment_ood_samples {
        return Err(WhirError::InitialOodAnswerCountMismatch {
            expected: config.commitment_ood_samples,
            actual: proof.initial_ood_answers.len(),
        });
    }
    let mut ood_statement = EqStatement::initialize(config.num_variables);
    for &eval in &proof.initial_ood_answers {
        let point = expand_from_univariate(transcript.sample_quintic(), config.num_variables);
        transcript.observe_quintic(eval);
        ood_statement.add_evaluated_constraint(point, eval);
    }
    Ok(ood_statement)
}

/// Verify a WHIR proof against a parsed commitment and the batched initial
/// constraint, mirroring upstream `WhirVerifier::verify` for prefix order.
/// Returns the full folding randomness on success.
pub fn whir_verify(
    config: &WhirConfig,
    proof: &WhirProof,
    transcript: &mut PoseidonTranscript,
    parsed_root: &PoseidonDigest,
    initial_constraint: Constraint,
    mut claimed_eval: QuinticExtension,
) -> Result<Vec<QuinticExtension>, WhirError> {
    let expected_rounds = config.n_rounds();
    if proof.rounds.len() != expected_rounds {
        return Err(WhirError::RoundCountMismatch {
            expected: expected_rounds,
            actual: proof.rounds.len(),
        });
    }

    let mut constraints = alloc::vec![initial_constraint];
    let mut round_folding_randomness: Vec<Vec<QuinticExtension>> = Vec::new();
    let mut prev_commitment = *parsed_root;

    let folding_randomness = proof.initial_sumcheck.verify_rounds(
        transcript,
        &mut claimed_eval,
        config.round_folding_factor(0),
        config.starting_folding_pow_bits,
    )?;
    round_folding_randomness.push(folding_randomness);

    for round_index in 0..config.n_rounds() {
        let round_params: &RoundConfig = &config.round_parameters[round_index];
        let round_proof = &proof.rounds[round_index];

        // Parse the round commitment: observe the root, replay OOD sampling.
        let root = round_proof
            .commitment
            .ok_or(WhirError::MissingRoundCommitment { round: round_index })?;
        if round_proof.ood_answers.len() != round_params.ood_samples {
            return Err(WhirError::RoundOodAnswerCountMismatch {
                round: round_index,
                expected: round_params.ood_samples,
                actual: round_proof.ood_answers.len(),
            });
        }
        transcript.observe_slice(&root);
        let mut ood_statement = EqStatement::initialize(round_params.num_variables);
        for &eval in &round_proof.ood_answers {
            let point =
                expand_from_univariate(transcript.sample_quintic(), round_params.num_variables);
            transcript.observe_quintic(eval);
            ood_statement.add_evaluated_constraint(point, eval);
        }

        // Verify STIR in-domain challenges against the previous commitment.
        let current_folding_randomness = round_folding_randomness
            .last()
            .expect("initial folding randomness is present");
        let stir_statement = verify_stir_challenges(
            &round_proof.openings,
            round_proof.pow_witness,
            false,
            transcript,
            round_params,
            &prev_commitment,
            current_folding_randomness,
            round_index,
        )?;

        // Rebuild the batched constraint the prover formed for this round.
        let constraint = Constraint::new(
            transcript.sample_quintic(),
            ood_statement.num_variables(),
            alloc::vec![
                Statements::Eq(ood_statement),
                Statements::Select(stir_statement),
            ],
        );
        constraint.combine_evals(&mut claimed_eval);
        constraints.push(constraint);

        let folding_randomness = round_proof.sumcheck.verify_rounds(
            transcript,
            &mut claimed_eval,
            config.round_folding_factor(round_index + 1),
            round_params.folding_pow_bits,
        )?;
        round_folding_randomness.push(folding_randomness);

        prev_commitment = root;
    }

    // Final round: receive the polynomial in the clear.
    let final_evaluations = proof
        .final_poly
        .as_ref()
        .ok_or(WhirError::MissingFinalPoly)?;
    let final_round_config = config.final_round_config();
    let expected_final_poly_len = 1 << final_round_config.num_variables;
    if final_evaluations.len() != expected_final_poly_len {
        return Err(WhirError::FinalPolyLengthMismatch {
            expected: expected_final_poly_len,
            actual: final_evaluations.len(),
        });
    }
    transcript.observe_quintic_slice(final_evaluations);

    let final_round_folding_randomness = round_folding_randomness
        .last()
        .expect("at least the initial folding randomness is present");
    let stir_statement = verify_stir_challenges(
        &proof.final_openings,
        proof.final_pow_witness,
        true,
        transcript,
        &final_round_config,
        &prev_commitment,
        final_round_folding_randomness,
        config.n_rounds(),
    )?;

    if !stir_statement.verify(final_evaluations) {
        return Err(WhirError::StirChallengeFailed);
    }

    let final_sumcheck_randomness = verify_final_sumcheck_rounds(
        proof.final_sumcheck.as_ref(),
        transcript,
        &mut claimed_eval,
        config.final_sumcheck_rounds,
        config.final_folding_pow_bits,
    )?;
    round_folding_randomness.push(final_sumcheck_randomness.clone());

    // Compute the full folding randomness across all rounds.
    let folding_randomness: Vec<QuinticExtension> =
        round_folding_randomness.into_iter().flatten().collect();

    // Evaluate the constraint polynomial at the folding point.
    let evaluation_of_weights = eval_constraints_poly(&constraints, &folding_randomness);

    // Final consistency check: claimed_eval == weights * f(r).
    let final_value = evaluate_mle_table(final_evaluations, &final_sumcheck_randomness)
        .map_err(|_| WhirError::SumcheckFailed)?;
    if claimed_eval != evaluation_of_weights * final_value {
        return Err(WhirError::SumcheckFailed);
    }

    Ok(folding_randomness)
}

/// Verify STIR in-domain queries and produce the associated selection
/// statement, mirroring upstream `WhirVerifier::verify_stir_challenges`.
#[allow(clippy::too_many_arguments)]
fn verify_stir_challenges(
    openings: &QueryOpenings,
    pow_witness: KoalaBear,
    is_final: bool,
    transcript: &mut PoseidonTranscript,
    params: &RoundConfig,
    commitment: &PoseidonDigest,
    folding_randomness: &[QuinticExtension],
    round_index: usize,
) -> Result<SelectStatement, WhirError> {
    // Verify the PoW witness before generating challenges.
    if params.pow_bits > 0 {
        let ok = transcript
            .check_witness(params.pow_bits, pow_witness)
            .map_err(|_| WhirError::InvalidPowWitness)?;
        if !ok {
            return Err(WhirError::InvalidPowWitness);
        }
    }

    // Transcript checkpoint after PoW (non-final rounds only).
    if !is_final {
        let _checkpoint = transcript.sample_base();
    }

    // Sample STIR query positions.
    let stir_indices = get_challenge_stir_queries(
        params.domain_size,
        params.folding_factor,
        params.num_queries,
        transcript,
    );

    let height = params.domain_size >> params.folding_factor;
    let width = 1 << params.folding_factor;

    let expect_base = round_index == 0;
    let answers: Vec<Vec<QuinticExtension>> = match (openings, expect_base) {
        (QueryOpenings::Base(opening), true) => {
            if opening.rows.len() != stir_indices.len() {
                return Err(WhirError::StirQueryCountMismatch {
                    round_index,
                    expected: stir_indices.len(),
                    actual: opening.rows.len(),
                });
            }
            PoseidonMerkleTree::verify_multi(
                commitment,
                height,
                width,
                &stir_indices,
                &opening.rows,
                &opening.proof,
            )
            .map_err(|_| WhirError::MerkleProofInvalid)?;
            opening
                .rows
                .iter()
                .map(|row| row.iter().map(|&f| f.into()).collect())
                .collect()
        }
        (QueryOpenings::Extension(opening), false) => {
            if opening.rows.len() != stir_indices.len() {
                return Err(WhirError::StirQueryCountMismatch {
                    round_index,
                    expected: stir_indices.len(),
                    actual: opening.rows.len(),
                });
            }
            let flat_rows: Vec<Vec<KoalaBear>> = opening
                .rows
                .iter()
                .map(|row| flatten_ext_row(row))
                .collect();
            PoseidonMerkleTree::verify_multi(
                commitment,
                height,
                width * crate::QUINTIC_DEGREE,
                &stir_indices,
                &flat_rows,
                &opening.proof,
            )
            .map_err(|_| WhirError::MerkleProofInvalid)?;
            opening.rows.clone()
        }
        _ => return Err(WhirError::MerkleProofInvalid),
    };

    // Evaluate the folded polynomial at each queried position.
    let folds: Vec<QuinticExtension> = answers
        .iter()
        .map(|row| {
            evaluate_mle_table(row, folding_randomness)
                .expect("row width matches the folding randomness")
        })
        .collect();

    Ok(SelectStatement::new(
        params.num_variables,
        stir_indices
            .iter()
            .map(|&index| params.folded_domain_gen.pow(index as u64))
            .collect(),
        folds,
    ))
}
