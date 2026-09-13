//! Sparse R1CS shape, padding, matrix-vector, and satisfaction substrate.
//!
//! The column convention matches the pinned upstream implementation:
//! `[ private witness | constant one | public inputs ]`.

use alloc::vec;
use alloc::vec::Vec;
use core::cmp::max;
use core::fmt;

use crate::FieldElement;

/// One sparse matrix entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SparseMatEntry<F> {
    /// Zero-based row.
    pub row: usize,
    /// Zero-based column.
    pub col: usize,
    /// Field coefficient.
    pub val: F,
}

/// A row/column-dimensioned sparse matrix.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SparseMatrix<F> {
    /// Number of rows.
    pub num_rows: usize,
    /// Number of columns.
    pub num_cols: usize,
    /// Sparse entries; entries are accumulated in input order.
    pub entries: Vec<SparseMatEntry<F>>,
}

impl<F> SparseMatrix<F> {
    /// Number of sparse entries.
    pub fn nnz(&self) -> usize {
        self.entries.len()
    }

    /// Whether the matrix has no sparse entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// A Spartan R1CS shape.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct R1csShape<F> {
    /// Number of constraints.
    pub num_cons: usize,
    /// Number of private witness variables.
    pub num_vars: usize,
    /// Number of public input values.
    pub num_io: usize,
    /// Left sparse matrix.
    pub a: SparseMatrix<F>,
    /// Right sparse matrix.
    pub b: SparseMatrix<F>,
    /// Result sparse matrix.
    pub c: SparseMatrix<F>,
}

/// A private witness assignment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct R1csWitness<F> {
    /// Private witness columns.
    pub w: Vec<F>,
}

/// Why R1CS substrate validation or evaluation failed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum R1csError {
    /// Matrix dimensions or entries do not match the declared shape.
    InvalidShape,
    /// A witness or assignment vector has the wrong length.
    InvalidWitnessLength {
        /// Expected element count.
        expected: usize,
        /// Actual element count.
        found: usize,
    },
    /// Regular power-of-two padding cannot be represented.
    Padding,
    /// A constraint row is unsatisfied.
    UnsatisfiedConstraint {
        /// Zero-based failing row.
        row: usize,
    },
}

impl fmt::Display for R1csError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape => f.write_str("invalid R1CS shape"),
            Self::InvalidWitnessLength { expected, found } => {
                write!(
                    f,
                    "invalid R1CS witness length {found}, expected {expected}"
                )
            }
            Self::Padding => f.write_str("cannot pad R1CS shape regularly"),
            Self::UnsatisfiedConstraint { row } => {
                write!(f, "R1CS constraint row {row} is unsatisfied")
            }
        }
    }
}

impl core::error::Error for R1csError {}

impl<F> R1csShape<F> {
    /// Total assignment columns, including the constant-one column.
    pub fn assignment_len(&self) -> Result<usize, R1csError> {
        self.num_vars
            .checked_add(self.num_io)
            .and_then(|value| value.checked_add(1))
            .ok_or(R1csError::InvalidShape)
    }

    /// Validate matrix dimensions and sparse-entry bounds.
    pub fn validate(&self) -> Result<(), R1csError> {
        if self.num_cons == 0 {
            return Err(R1csError::InvalidShape);
        }
        let expected_cols = self.assignment_len()?;
        validate_matrix_dimensions(&self.a, self.num_cons, expected_cols)?;
        validate_matrix_dimensions(&self.b, self.num_cons, expected_cols)?;
        validate_matrix_dimensions(&self.c, self.num_cons, expected_cols)?;
        validate_matrix_entries(&self.a)?;
        validate_matrix_entries(&self.b)?;
        validate_matrix_entries(&self.c)?;
        Ok(())
    }

    /// Upstream-compatible regular padding. Witness columns are inserted
    /// before the constant-one/public columns so `[W | 1 | X]` is preserved.
    pub fn pad_regular(&self) -> Result<Self, R1csError>
    where
        F: Clone,
    {
        self.validate()?;

        let num_vars_target = max(
            self.num_vars,
            self.num_io.checked_add(1).ok_or(R1csError::Padding)?,
        );
        let num_vars_padded = num_vars_target
            .checked_next_power_of_two()
            .ok_or(R1csError::Padding)?;
        let num_cons_padded = self
            .num_cons
            .checked_next_power_of_two()
            .ok_or(R1csError::Padding)?;
        if self.num_io >= num_vars_padded {
            return Err(R1csError::Padding);
        }

        let vars_delta = num_vars_padded.saturating_sub(self.num_vars);
        let num_cols_padded = num_vars_padded
            .checked_add(self.num_io)
            .and_then(|value| value.checked_add(1))
            .ok_or(R1csError::Padding)?;

        let pad_matrix = |matrix: &SparseMatrix<F>| -> SparseMatrix<F> {
            let mut entries = matrix.entries.clone();
            if vars_delta > 0 {
                for entry in &mut entries {
                    if entry.col >= self.num_vars {
                        entry.col += vars_delta;
                    }
                }
            }
            SparseMatrix {
                num_rows: num_cons_padded,
                num_cols: num_cols_padded,
                entries,
            }
        };

        Ok(Self {
            num_cons: num_cons_padded,
            num_vars: num_vars_padded,
            num_io: self.num_io,
            a: pad_matrix(&self.a),
            b: pad_matrix(&self.b),
            c: pad_matrix(&self.c),
        })
    }
}

impl<F: FieldElement> R1csShape<F> {
    /// Multiply all three matrices by a complete `[W | 1 | X]` assignment.
    pub fn multiply_vec(&self, z: &[F]) -> Result<(Vec<F>, Vec<F>, Vec<F>), R1csError> {
        self.validate()?;
        self.multiply_vec_unchecked(z)
    }

    /// Matrix-vector multiplication after caller-side shape validation.
    pub fn multiply_vec_unchecked(&self, z: &[F]) -> Result<(Vec<F>, Vec<F>, Vec<F>), R1csError> {
        self.validate_matrix_vector_input_len(z)?;
        Ok((
            multiply_sparse_matrix_vector(&self.a, z)?,
            multiply_sparse_matrix_vector(&self.b, z)?,
            multiply_sparse_matrix_vector(&self.c, z)?,
        ))
    }

    /// Zero-pad a private witness to `num_vars`, producing the dense MLE table.
    pub fn witness_to_mle(&self, witness: &[F]) -> Result<Vec<F>, R1csError> {
        self.validate()?;
        if witness.len() > self.num_vars {
            return Err(R1csError::InvalidWitnessLength {
                expected: self.num_vars,
                found: witness.len(),
            });
        }
        let mut out = witness.to_vec();
        out.resize(self.num_vars, F::ZERO);
        Ok(out)
    }

    /// Validate a private witness and public values against every constraint.
    pub fn validate_satisfaction(
        &self,
        witness: &R1csWitness<F>,
        public_inputs: &[F],
    ) -> Result<(), R1csError> {
        let mut z = self.witness_to_mle(&witness.w)?;
        if public_inputs.len() != self.num_io {
            return Err(R1csError::InvalidWitnessLength {
                expected: self.num_io,
                found: public_inputs.len(),
            });
        }
        z.push(F::ONE);
        z.extend_from_slice(public_inputs);
        let (az, bz, cz) = self.multiply_vec(&z)?;
        for row in 0..self.num_cons {
            if az[row] * bz[row] != cz[row] {
                return Err(R1csError::UnsatisfiedConstraint { row });
            }
        }
        Ok(())
    }

    fn validate_matrix_vector_input_len(&self, z: &[F]) -> Result<(), R1csError> {
        let expected = self
            .assignment_len()
            .map_err(|_| R1csError::InvalidWitnessLength {
                expected: usize::MAX,
                found: z.len(),
            })?;
        if z.len() != expected {
            return Err(R1csError::InvalidWitnessLength {
                expected,
                found: z.len(),
            });
        }
        Ok(())
    }
}

fn validate_matrix_dimensions<F>(
    matrix: &SparseMatrix<F>,
    expected_rows: usize,
    expected_cols: usize,
) -> Result<(), R1csError> {
    if matrix.num_rows != expected_rows || matrix.num_cols != expected_cols {
        return Err(R1csError::InvalidShape);
    }
    Ok(())
}

fn validate_matrix_entries<F>(matrix: &SparseMatrix<F>) -> Result<(), R1csError> {
    for entry in &matrix.entries {
        if entry.row >= matrix.num_rows || entry.col >= matrix.num_cols {
            return Err(R1csError::InvalidShape);
        }
    }
    Ok(())
}

fn multiply_sparse_matrix_vector<F: FieldElement>(
    matrix: &SparseMatrix<F>,
    z: &[F],
) -> Result<Vec<F>, R1csError> {
    if z.len() != matrix.num_cols {
        return Err(R1csError::InvalidWitnessLength {
            expected: matrix.num_cols,
            found: z.len(),
        });
    }
    let mut out = vec![F::ZERO; matrix.num_rows];
    for entry in &matrix.entries {
        out[entry.row] = out[entry.row] + entry.val * z[entry.col];
    }
    Ok(out)
}
