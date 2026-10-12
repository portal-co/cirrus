//! Thin Cirrus adapter over `volar_mpc::bitgc_session`.
//!
//! This module is an explicit `bitgc-session` preview seam. It adapts Cirrus's
//! owned `Pusher<Vec<u8>>`/iterator record shape to the shared bounded frame
//! and OT transport traits without importing Cirrus types into `volar-mpc`.
//! It does not route the legacy `bitgc-reference` backend through the session,
//! expose BitGC Δ/keys, or enable protected-session admission.

extern crate alloc;

use alloc::vec::Vec;

use cirrus_core::Pusher;
use volar_mpc::bitgc_session::{
    AuthenticatedBitGcOtTransport, BITGC_FRAME_MAX_BYTES, BITGC_OT_MAX_INPUT_BYTES,
    BitGcFrameTransport, BitGcIoError, BitGcOtTransport, BitGcRole, BitGcSession,
    BitGcSessionBinding, BitGcSessionError, BitGcSessionState,
};

const OT_ENVELOPE_BOUND: usize = 128;

/// Bounded channel error for Cirrus's record-backed session transport.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CirrusSessionChannelError {
    /// The channel was cancelled and cannot resume.
    Cancelled,
    /// The source contained no next record.
    Exhausted,
    /// A source record exceeded the caller's receive bound.
    Oversized {
        /// Actual record length.
        actual: usize,
        /// Maximum accepted record length.
        maximum: usize,
    },
}

/// A transport-neutral session channel backed by a Cirrus sink and ordered
/// owned records.
///
/// Frame and OT channels must use separate instances. The source is consumed
/// in order; malformed, duplicate, or out-of-order protocol records are
/// rejected by `volar_mpc::bitgc_session` after this bounded transport check.
pub struct CirrusSessionChannel<'a, I>
where
    I: Iterator<Item = Vec<u8>>,
{
    sink: &'a mut dyn Pusher<Vec<u8>>,
    source: I,
    cancelled: bool,
}

impl<'a, I> CirrusSessionChannel<'a, I>
where
    I: Iterator<Item = Vec<u8>>,
{
    /// Construct a channel from a record sink and ordered receive source.
    pub fn new(sink: &'a mut dyn Pusher<Vec<u8>>, source: I) -> Self {
        Self {
            sink,
            source,
            cancelled: false,
        }
    }

    /// Whether cancellation has made this channel terminal.
    pub const fn is_cancelled(&self) -> bool {
        self.cancelled
    }
}

impl<I> BitGcFrameTransport for CirrusSessionChannel<'_, I>
where
    I: Iterator<Item = Vec<u8>>,
{
    type Error = CirrusSessionChannelError;

    fn send_frame(&mut self, frame: &[u8]) -> Result<(), Self::Error> {
        if self.cancelled {
            return Err(CirrusSessionChannelError::Cancelled);
        }
        if frame.len() > BITGC_FRAME_MAX_BYTES {
            return Err(CirrusSessionChannelError::Oversized {
                actual: frame.len(),
                maximum: BITGC_FRAME_MAX_BYTES,
            });
        }
        self.sink.push(frame.to_vec());
        Ok(())
    }

    fn receive_frame(&mut self, max_bytes: usize) -> Result<Vec<u8>, Self::Error> {
        if self.cancelled {
            return Err(CirrusSessionChannelError::Cancelled);
        }
        let record = self
            .source
            .next()
            .ok_or(CirrusSessionChannelError::Exhausted)?;
        let maximum = core::cmp::min(max_bytes, BITGC_FRAME_MAX_BYTES);
        if record.len() > maximum {
            return Err(CirrusSessionChannelError::Oversized {
                actual: record.len(),
                maximum,
            });
        }
        Ok(record)
    }

    fn cancel(&mut self) -> Result<(), Self::Error> {
        self.cancelled = true;
        Ok(())
    }
}

impl<I> BitGcOtTransport for CirrusSessionChannel<'_, I>
where
    I: Iterator<Item = Vec<u8>>,
{
    type Error = CirrusSessionChannelError;

    fn send_ot_message(&mut self, message: &[u8]) -> Result<(), Self::Error> {
        if self.cancelled {
            return Err(CirrusSessionChannelError::Cancelled);
        }
        let maximum = BITGC_OT_MAX_INPUT_BYTES.saturating_add(OT_ENVELOPE_BOUND);
        if message.len() > maximum {
            return Err(CirrusSessionChannelError::Oversized {
                actual: message.len(),
                maximum,
            });
        }
        self.sink.push(message.to_vec());
        Ok(())
    }

    fn receive_ot_message(&mut self, max_bytes: usize) -> Result<Vec<u8>, Self::Error> {
        if self.cancelled {
            return Err(CirrusSessionChannelError::Cancelled);
        }
        let record = self
            .source
            .next()
            .ok_or(CirrusSessionChannelError::Exhausted)?;
        let maximum = core::cmp::min(
            max_bytes,
            BITGC_OT_MAX_INPUT_BYTES.saturating_add(OT_ENVELOPE_BOUND),
        );
        if record.len() > maximum {
            return Err(CirrusSessionChannelError::Oversized {
                actual: record.len(),
                maximum,
            });
        }
        Ok(record)
    }

    fn cancel(&mut self) -> Result<(), Self::Error> {
        self.cancelled = true;
        Ok(())
    }
}

/// Thin Cirrus owner of the shared BitGC session state machine.
///
/// Cryptographic material remains in the caller's role-specific Volar state;
/// this wrapper only forwards binding/state and transport operations.
pub struct CirrusBitGcSession {
    inner: BitGcSession,
}

impl CirrusBitGcSession {
    /// Construct an unauthenticated structural session.
    pub fn new(role: BitGcRole, binding: BitGcSessionBinding) -> Result<Self, BitGcSessionError> {
        Ok(Self {
            inner: BitGcSession::new(role, binding)?,
        })
    }

    /// Construct a session from an externally established symmetric key.
    pub fn new_authenticated(
        role: BitGcRole,
        binding: BitGcSessionBinding,
        key: [u8; 32],
    ) -> Result<Self, BitGcSessionError> {
        Ok(Self {
            inner: BitGcSession::new_authenticated(role, binding, key)?,
        })
    }

    /// Construct a session from the shared PSK nonce handshake.
    pub fn new_psk_authenticated(
        role: BitGcRole,
        binding: BitGcSessionBinding,
        psk: [u8; 32],
        nonce: [u8; 32],
    ) -> Result<Self, BitGcSessionError> {
        Ok(Self {
            inner: BitGcSession::new_psk_authenticated(role, binding, psk, nonce)?,
        })
    }

    /// Access the shared session state for frame/input/output operations.
    pub const fn inner(&self) -> &BitGcSession {
        &self.inner
    }

    /// Mutably access the shared session state for frame/input/output
    /// operations.
    pub fn inner_mut(&mut self) -> &mut BitGcSession {
        &mut self.inner
    }

    /// Consume the adapter and return the shared session.
    pub fn into_inner(self) -> BitGcSession {
        self.inner
    }

    /// Current shared protocol state.
    pub const fn state(&self) -> BitGcSessionState {
        self.inner.state()
    }

    /// Forward the shared hello frame constructor.
    pub fn hello(&mut self) -> Result<Vec<u8>, BitGcSessionError> {
        self.inner.hello()
    }

    /// Forward one bounded data frame constructor.
    pub fn send_data(&mut self, payload: &[u8]) -> Result<Vec<u8>, BitGcSessionError> {
        self.inner.send_data(payload)
    }

    /// Construct an authenticated OT transport bound to this session.
    pub fn authenticated_ot_transport<'a, I>(
        &self,
        channel: CirrusSessionChannel<'a, I>,
    ) -> Result<AuthenticatedBitGcOtTransport<CirrusSessionChannel<'a, I>>, BitGcSessionError>
    where
        I: Iterator<Item = Vec<u8>>,
    {
        self.inner.authenticated_ot_transport(channel)
    }

    /// Return the inner frame session for transport-driven operations.
    pub fn frame_session(&mut self) -> &mut BitGcSession {
        &mut self.inner
    }
}

/// Shared transport error alias used by adapter callers.
pub type CirrusSessionIoError = BitGcIoError<CirrusSessionChannelError>;

/// Shared frame kind re-export for the adapter surface.
pub use volar_mpc::bitgc_session::BitGcFrameKind as SessionFrameKind;
