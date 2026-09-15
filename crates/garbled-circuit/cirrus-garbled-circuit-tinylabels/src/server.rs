//! Explicit server-only TinyLabels manifest admission and bounded staging.
//!
//! This module exists only with the `tinylabels-server` feature. Embedded
//! builds retain the allocation-free shared construction types but cannot name
//! this server admission surface. The module deliberately stops at manifest,
//! resource, and bounded-stream framing: a protected deployment still needs
//! the shared core's reviewed sampler and canonical polynomial codec.

#[cfg(any(target_os = "none", target_arch = "arm", target_arch = "riscv32"))]
compile_error!(
    "tinylabels-server is a server-class profile and is unavailable on embedded Cirrus targets"
);

use volar_spec::tinylabels::frame::FrameBinding;

/// Ordered external labels for one fixed Cirrus interpreter trace.
///
/// The trace producer (ERT or LLVM) derives `topology_digest` from its
/// canonical fixed topology. TinyLabels does not make dynamic control flow or
/// unknown topology acceptable.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InputLabelManifest {
    /// Session binding chosen by the enclosing authenticated transport.
    pub session_id: [u8; 32],
    /// Digest of the fixed interpreter topology/program.
    pub topology_digest: [u8; 32],
    /// Ordered external input positions; table/intermediate labels are absent.
    pub positions: alloc::vec::Vec<u32>,
    /// Current construction requires complete 16-byte wire labels.
    pub label_bytes: usize,
}

/// Explicit server-only resource budget.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResourceBudget {
    /// Largest stage-frame payload admitted by the transport adapter.
    pub max_frame_payload: usize,
    /// Maximum reusable/offline public storage accepted for the selected mode.
    pub max_reusable_bytes: u64,
    /// Maximum public storage accepted per online use.
    pub max_per_use_bytes: u64,
}

/// Public resource declaration for one selected TinyLabels profile.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResourceRequest {
    /// Required reusable/offline bytes.
    pub reusable_bytes: u64,
    /// Required per-use bytes.
    pub per_use_bytes: u64,
}

/// Server profile admission failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdmissionError {
    /// Labels are not complete 16-byte labels.
    WrongLabelWidth,
    /// A position occurs more than once in the public order.
    DuplicatePosition,
    /// Public resource request exceeds the explicit profile budget.
    ResourceBudgetExceeded,
    /// A use counter was replayed or skipped.
    UnexpectedUseCounter,
}

/// An admitted fixed-trace server profile.
#[derive(Clone, Debug)]
pub struct ServerAdmission {
    manifest: InputLabelManifest,
    parameter_fingerprint: [u8; 32],
    budget: ResourceBudget,
    next_use_counter: u64,
}

impl ServerAdmission {
    /// Admit an explicitly server-class TinyLabels profile.
    pub fn admit(
        manifest: InputLabelManifest,
        parameter_fingerprint: [u8; 32],
        budget: ResourceBudget,
        request: ResourceRequest,
    ) -> Result<Self, AdmissionError> {
        if manifest.label_bytes != 16 {
            return Err(AdmissionError::WrongLabelWidth);
        }
        let mut seen = alloc::collections::BTreeSet::new();
        if !manifest
            .positions
            .iter()
            .all(|position| seen.insert(*position))
        {
            return Err(AdmissionError::DuplicatePosition);
        }
        if request.reusable_bytes > budget.max_reusable_bytes
            || request.per_use_bytes > budget.max_per_use_bytes
        {
            return Err(AdmissionError::ResourceBudgetExceeded);
        }
        Ok(Self {
            manifest,
            parameter_fingerprint,
            budget,
            next_use_counter: 0,
        })
    }

    /// Number of delivered external labels, in public manifest order.
    pub fn label_count(&self) -> usize {
        self.manifest.positions.len()
    }

    /// Bounded inbound payload limit for this admission.
    pub fn max_frame_payload(&self) -> usize {
        self.budget.max_frame_payload
    }

    /// Begin exactly the next online use.
    pub fn begin_use(
        &mut self,
        manifest_digest: [u8; 32],
        use_counter: u64,
    ) -> Result<FrameBinding, AdmissionError> {
        if use_counter != self.next_use_counter {
            return Err(AdmissionError::UnexpectedUseCounter);
        }
        self.next_use_counter = self.next_use_counter.saturating_add(1);
        Ok(FrameBinding {
            parameter_fingerprint: self.parameter_fingerprint,
            session_id: self.manifest.session_id,
            manifest_digest,
            use_counter,
        })
    }
}

/// Fixed-capacity byte frame used at the coroutine seam.
///
/// Callers choose `MAX` from the admitted resource budget. `push` returns the
/// original frame when it cannot fit, so a transport can apply backpressure
/// without truncating a peer-controlled message.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BoundedFrame<const MAX: usize> {
    bytes: [u8; MAX],
    len: usize,
}

impl<const MAX: usize> BoundedFrame<MAX> {
    /// Copy one frame into fixed storage if it fits.
    pub fn new(frame: &[u8]) -> Result<Self, &[u8]> {
        if frame.len() > MAX {
            return Err(frame);
        }
        let mut bytes = [0u8; MAX];
        bytes[..frame.len()].copy_from_slice(frame);
        Ok(Self {
            bytes,
            len: frame.len(),
        })
    }

    /// Borrow the exact encoded frame bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes[..self.len]
    }
}

#[cfg(test)]
mod tests {
    use core::mem::ManuallyDrop;
    use core::pin::Pin;

    use cirrus_coroutine::{Coroutine, Pusher};

    use super::{
        AdmissionError, BoundedFrame, InputLabelManifest, ResourceBudget, ResourceRequest,
        ServerAdmission,
    };

    fn stream_frames(pusher: &mut Pusher<'_, BoundedFrame<16>, 2, 128>) {
        loop {
            pusher.push(BoundedFrame::new(&[0xA1]).unwrap());
            pusher.push(BoundedFrame::new(&[0xB2, 0xC3]).unwrap());
        }
    }

    #[test]
    fn bounded_coroutine_stream_preserves_frame_order_without_a_vec() {
        let mut coroutine = ManuallyDrop::new(Coroutine::new(stream_frames));
        // SAFETY: `ManuallyDrop` keeps the started producer stack allocated for
        // this short replay; this is the same lifecycle pattern the coroutine
        // crate's own synchronous streaming tests use.
        let pinned = unsafe { Pin::new_unchecked(&mut *coroutine) };
        let mut puller = pinned.puller();
        assert_eq!(puller.take_or_refill().as_bytes(), &[0xA1]);
        assert_eq!(puller.take_or_refill().as_bytes(), &[0xB2, 0xC3]);
        assert_eq!(puller.take_or_refill().as_bytes(), &[0xA1]);
    }

    #[test]
    fn server_admission_is_explicit_bounded_and_monotonic() {
        let mut admission = ServerAdmission::admit(
            InputLabelManifest {
                session_id: [1; 32],
                topology_digest: [2; 32],
                positions: alloc::vec![4, 9],
                label_bytes: 16,
            },
            [3; 32],
            ResourceBudget {
                max_frame_payload: 1024,
                max_reusable_bytes: 4096,
                max_per_use_bytes: 512,
            },
            ResourceRequest {
                reusable_bytes: 4096,
                per_use_bytes: 512,
            },
        )
        .unwrap();
        assert_eq!(admission.label_count(), 2);
        assert_eq!(admission.max_frame_payload(), 1024);
        assert_eq!(admission.begin_use([4; 32], 0).unwrap().use_counter, 0);
        assert_eq!(
            admission.begin_use([4; 32], 0),
            Err(AdmissionError::UnexpectedUseCounter)
        );
    }
}
