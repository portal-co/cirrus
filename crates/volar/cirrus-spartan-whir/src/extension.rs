//! KoalaBear quintic extension arithmetic.
//!
//! Elements are polynomials `a0 + a1*X + ... + a4*X^4` reduced by
//! `X^5 + X^2 - 1`, hence `X^5 = 1 - X^2`. The reduction schedule here is
//! intentionally simple and independent of Plonky3's packed kernels; oracle
//! tests compare the full algebra against upstream.

use core::fmt;

use crate::KoalaBear;

/// Degree of the KoalaBear quintic extension.
pub const QUINTIC_DEGREE: usize = 5;

/// Why a quintic-extension operation cannot be performed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QuinticExtensionError {
    /// A coefficient was not canonical.
    NoncanonicalCoefficient {
        /// Coefficient index.
        index: usize,
        /// Offending value.
        value: u32,
    },
    /// Zero has no multiplicative inverse.
    ZeroInverse,
}

impl fmt::Display for QuinticExtensionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoncanonicalCoefficient { index, value } => {
                write!(
                    f,
                    "quintic coefficient {index} value {value} is not canonical"
                )
            }
            Self::ZeroInverse => f.write_str("zero has no quintic-extension inverse"),
        }
    }
}

impl core::error::Error for QuinticExtensionError {}

/// A canonical element of `KoalaBear[X]/(X^5 + X^2 - 1)`.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct QuinticExtension(pub [KoalaBear; QUINTIC_DEGREE]);

impl QuinticExtension {
    /// The additive identity.
    pub const ZERO: Self = Self([KoalaBear::ZERO; QUINTIC_DEGREE]);
    /// The multiplicative identity.
    pub const ONE: Self = Self([
        KoalaBear::ONE,
        KoalaBear::ZERO,
        KoalaBear::ZERO,
        KoalaBear::ZERO,
        KoalaBear::ZERO,
    ]);

    /// Construct from already-canonical coefficients.
    pub const fn new(coefficients: [KoalaBear; QUINTIC_DEGREE]) -> Self {
        Self(coefficients)
    }

    /// Construct from canonical `u32` coefficients.
    pub fn from_canonical_coefficients(
        coefficients: [u32; QUINTIC_DEGREE],
    ) -> Result<Self, QuinticExtensionError> {
        let mut out = [KoalaBear::ZERO; QUINTIC_DEGREE];
        let mut index = 0;
        while index < QUINTIC_DEGREE {
            out[index] = KoalaBear::from_canonical(coefficients[index]).map_err(|_| {
                QuinticExtensionError::NoncanonicalCoefficient {
                    index,
                    value: coefficients[index],
                }
            })?;
            index += 1;
        }
        Ok(Self(out))
    }

    /// Canonical basis coefficients.
    pub const fn coefficients(self) -> [KoalaBear; QUINTIC_DEGREE] {
        self.0
    }

    /// Canonical `u32` basis coefficients.
    pub const fn canonical_coefficients(self) -> [u32; QUINTIC_DEGREE] {
        [
            self.0[0].canonical(),
            self.0[1].canonical(),
            self.0[2].canonical(),
            self.0[3].canonical(),
            self.0[4].canonical(),
        ]
    }

    /// Whether this is zero.
    pub fn is_zero(self) -> bool {
        self.0.iter().all(|value| value.is_zero())
    }

    /// Whether this element lies in the base field.
    pub fn is_base(self) -> bool {
        self.0[1..].iter().all(|value| value.is_zero())
    }

    /// Extension addition.
    pub fn add(self, rhs: Self) -> Self {
        Self(core::array::from_fn(|index| self.0[index] + rhs.0[index]))
    }

    /// Extension subtraction.
    pub fn sub(self, rhs: Self) -> Self {
        Self(core::array::from_fn(|index| self.0[index] - rhs.0[index]))
    }

    /// Additive inverse.
    pub fn neg(self) -> Self {
        Self(core::array::from_fn(|index| -self.0[index]))
    }

    /// Multiplication by a base-field scalar.
    pub fn mul_base(self, rhs: KoalaBear) -> Self {
        Self(core::array::from_fn(|index| self.0[index] * rhs))
    }

    /// Extension multiplication reduced by `X^5 = 1 - X^2`.
    pub fn mul(self, rhs: Self) -> Self {
        let mut convolution = [KoalaBear::ZERO; 9];
        for (i, left) in self.0.iter().enumerate() {
            for (j, right) in rhs.0.iter().enumerate() {
                convolution[i + j] = convolution[i + j] + (*left * *right);
            }
        }
        // Reduction identities derived from X^5 = 1 - X^2:
        // X^5 = 1 - X^2
        // X^6 = X - X^3
        // X^7 = X^2 - X^4
        // X^8 = X^3 + X^2 - 1
        Self([
            convolution[0] + convolution[5] - convolution[8],
            convolution[1] + convolution[6],
            convolution[2] - convolution[5] + convolution[7] + convolution[8],
            convolution[3] - convolution[6] + convolution[8],
            convolution[4] - convolution[7],
        ])
    }

    /// Extension squaring.
    pub fn square(self) -> Self {
        self.mul(self)
    }

    /// Raise to an unsigned exponent given as a 5-limb little-endian integer
    /// over base `2^64` (320 bits total, sufficient for `p^5 - 2`).
    pub fn pow_wide(self, mut exponent: [u64; 5]) -> Self {
        let mut result = Self::ONE;
        let mut base = self;
        for limb in &mut exponent {
            let mut bits = *limb;
            while bits != 0 {
                if bits & 1 != 0 {
                    result = result * base;
                }
                base = base.square();
                bits >>= 1;
            }
        }
        result
    }

    /// Multiplicative inverse.
    ///
    /// This uses the independent linear-algebra method also used by the Mode-B
    /// RAM witness materializer: multiplication by `self` is a 5x5
    /// KoalaBear-linear map, and its inverse is read from the reduced matrix.
    pub fn inverse(self) -> Result<Self, QuinticExtensionError> {
        if self.is_zero() {
            return Err(QuinticExtensionError::ZeroInverse);
        }
        let modulus = i64::from(crate::KOALABEAR_MODULUS);
        let mut matrix = [[0_i64; QUINTIC_DEGREE + 1]; QUINTIC_DEGREE];
        for column in 0..QUINTIC_DEGREE {
            let basis = core::array::from_fn(|index| {
                if index == column {
                    KoalaBear::ONE
                } else {
                    KoalaBear::ZERO
                }
            });
            let product = self * QuinticExtension::new(basis);
            for row in 0..QUINTIC_DEGREE {
                matrix[row][column] = i64::from(product.0[row].canonical());
            }
        }
        for row in 0..QUINTIC_DEGREE {
            matrix[row][QUINTIC_DEGREE] = i64::from(row == 0);
        }
        for pivot in 0..QUINTIC_DEGREE {
            let swap = (pivot..QUINTIC_DEGREE)
                .find(|row| matrix[*row][pivot] != 0)
                .ok_or(QuinticExtensionError::ZeroInverse)?;
            matrix.swap(pivot, swap);
            let inverse = kb_inverse(matrix[pivot][pivot], modulus);
            for entry in &mut matrix[pivot] {
                *entry = (*entry * inverse).rem_euclid(modulus);
            }
            for row in 0..QUINTIC_DEGREE {
                if row != pivot {
                    let factor = matrix[row][pivot];
                    for column in pivot..=QUINTIC_DEGREE {
                        matrix[row][column] = (matrix[row][column]
                            - factor * matrix[pivot][column])
                            .rem_euclid(modulus);
                    }
                }
            }
        }
        let coefficients = core::array::from_fn(|row| {
            KoalaBear::from_canonical(matrix[row][QUINTIC_DEGREE] as u32)
                .expect("Gauss-Jordan result is canonical")
        });
        Ok(Self::new(coefficients))
    }
}

impl core::ops::Add for QuinticExtension {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self::add(self, rhs)
    }
}

impl core::ops::Sub for QuinticExtension {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self::sub(self, rhs)
    }
}

impl core::ops::Neg for QuinticExtension {
    type Output = Self;

    fn neg(self) -> Self {
        Self::neg(self)
    }
}

impl core::ops::Mul for QuinticExtension {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self::mul(self, rhs)
    }
}

fn kb_inverse(value: i64, modulus: i64) -> i64 {
    let (mut old_r, mut r) = (value.rem_euclid(modulus), modulus);
    let (mut old_s, mut s) = (1_i64, 0_i64);
    while r != 0 {
        let quotient = old_r / r;
        (old_r, r) = (r, old_r - quotient * r);
        (old_s, s) = (s, old_s - quotient * s);
    }
    debug_assert_eq!(old_r, 1);
    old_s.rem_euclid(modulus)
}
