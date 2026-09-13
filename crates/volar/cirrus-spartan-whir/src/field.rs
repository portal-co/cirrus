//! Canonical KoalaBear base-field arithmetic.
//!
//! The modulus is `2^31 - 2^24 + 1 = 2,130,706,433`. Values are always kept
//! as canonical representatives in `0..p`; parsing rejects noncanonical
//! encodings rather than reducing them silently.

use core::fmt;

/// KoalaBear base-field modulus.
pub const KOALABEAR_MODULUS: u32 = 2_130_706_433;
const P: u64 = KOALABEAR_MODULUS as u64;
const P_MINUS_2: u64 = P - 2;

/// Why a KoalaBear operation cannot be performed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KoalaBearError {
    /// A serialized/base value was not a canonical representative.
    Noncanonical {
        /// Offending value.
        value: u32,
    },
    /// Zero has no multiplicative inverse.
    ZeroInverse,
}

impl fmt::Display for KoalaBearError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Noncanonical { value } => {
                write!(f, "KoalaBear value {value} is not canonical")
            }
            Self::ZeroInverse => f.write_str("zero has no KoalaBear inverse"),
        }
    }
}

impl core::error::Error for KoalaBearError {}

/// A canonical KoalaBear base-field element.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct KoalaBear(u32);

impl KoalaBear {
    /// The additive identity.
    pub const ZERO: Self = Self(0);
    /// The multiplicative identity.
    pub const ONE: Self = Self(1);

    /// Construct from a canonical representative.
    pub const fn from_canonical(value: u32) -> Result<Self, KoalaBearError> {
        if value < KOALABEAR_MODULUS {
            Ok(Self(value))
        } else {
            Err(KoalaBearError::Noncanonical { value })
        }
    }

    /// Construct a checked constant array. This is intended for embedded
    /// round constants; it panics at compile time on a noncanonical value.
    pub(crate) const fn new_array<const N: usize>(input: [u32; N]) -> [Self; N] {
        let mut out = [Self::ZERO; N];
        let mut index = 0;
        while index < N {
            assert!(input[index] < KOALABEAR_MODULUS);
            out[index] = Self(input[index]);
            index += 1;
        }
        out
    }

    /// Construct a checked constant two-dimensional array.
    pub(crate) const fn new_2d_array<const N: usize, const M: usize>(
        input: [[u32; N]; M],
    ) -> [[Self; N]; M] {
        let mut out = [[Self::ZERO; N]; M];
        let mut row = 0;
        while row < M {
            out[row] = Self::new_array(input[row]);
            row += 1;
        }
        out
    }

    /// Reduce a `u64` modulo the field modulus.
    pub const fn from_u64(value: u64) -> Self {
        Self((value % P) as u32)
    }

    /// Reduce a signed integer modulo the field modulus.
    pub const fn from_i64(value: i64) -> Self {
        let reduced = value % (KOALABEAR_MODULUS as i64);
        if reduced < 0 {
            Self((reduced + KOALABEAR_MODULUS as i64) as u32)
        } else {
            Self(reduced as u32)
        }
    }

    /// Canonical representative in `0..p`.
    pub const fn canonical(self) -> u32 {
        self.0
    }

    /// Canonical little-endian four-byte encoding.
    pub const fn to_le_bytes(self) -> [u8; 4] {
        self.0.to_le_bytes()
    }

    /// Parse a canonical little-endian four-byte encoding.
    pub const fn from_le_bytes(bytes: [u8; 4]) -> Result<Self, KoalaBearError> {
        Self::from_canonical(u32::from_le_bytes(bytes))
    }

    /// Whether this is zero.
    pub const fn is_zero(self) -> bool {
        self.0 == 0
    }

    /// Field addition.
    pub const fn add(self, rhs: Self) -> Self {
        let sum = self.0 as u64 + rhs.0 as u64;
        Self(if sum >= P {
            (sum - P) as u32
        } else {
            sum as u32
        })
    }

    /// Field subtraction.
    pub const fn sub(self, rhs: Self) -> Self {
        if self.0 >= rhs.0 {
            Self(self.0 - rhs.0)
        } else {
            Self((self.0 as u64 + P - rhs.0 as u64) as u32)
        }
    }

    /// Additive inverse.
    pub const fn neg(self) -> Self {
        if self.0 == 0 {
            Self::ZERO
        } else {
            Self(KOALABEAR_MODULUS - self.0)
        }
    }

    /// Field multiplication.
    pub const fn mul(self, rhs: Self) -> Self {
        Self(((self.0 as u64 * rhs.0 as u64) % P) as u32)
    }

    /// Field squaring.
    pub const fn square(self) -> Self {
        self.mul(self)
    }

    /// Double this value.
    pub const fn double(self) -> Self {
        self.add(self)
    }

    /// Multiply by a small unsigned constant.
    pub const fn mul_u32(self, rhs: u32) -> Self {
        self.mul(Self::from_u64(rhs as u64))
    }

    /// Divide by two modulo the odd field modulus.
    pub const fn halve(self) -> Self {
        if self.0 & 1 == 0 {
            Self(self.0 >> 1)
        } else {
            Self(((self.0 as u64 + P) >> 1) as u32)
        }
    }

    /// Divide by `2^exponent` modulo the field modulus.
    pub const fn div_2exp_u64(self, exponent: u64) -> Self {
        match Self::from_u64(1_u64 << exponent).inverse() {
            Ok(inverse) => self.mul(inverse),
            Err(_) => Self::ZERO,
        }
    }

    /// Raise to an unsigned exponent.
    pub const fn pow(self, mut exponent: u64) -> Self {
        let mut result = Self::ONE;
        let mut base = self;
        while exponent != 0 {
            if exponent & 1 != 0 {
                result = result.mul(base);
            }
            base = base.square();
            exponent >>= 1;
        }
        result
    }

    /// Multiplicative inverse, computed as `self^(p-2)`.
    pub const fn inverse(self) -> Result<Self, KoalaBearError> {
        if self.is_zero() {
            Err(KoalaBearError::ZeroInverse)
        } else {
            Ok(self.pow(P_MINUS_2))
        }
    }
}

impl core::ops::Add for KoalaBear {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self::add(self, rhs)
    }
}

impl core::ops::Sub for KoalaBear {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self::sub(self, rhs)
    }
}

impl core::ops::Neg for KoalaBear {
    type Output = Self;

    fn neg(self) -> Self {
        Self::neg(self)
    }
}

impl core::ops::Mul for KoalaBear {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self::mul(self, rhs)
    }
}
