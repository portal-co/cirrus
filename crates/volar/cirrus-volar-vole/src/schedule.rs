//! VOLE adapter seam for canonical shared-storage schedule tickets.
//!
//! This module does not acquire locks or alter the existing ideal-functionality
//! contexts. It binds an adapter's logical VOLE work item to one manifest
//! ordinal and gives verifiers a fail-closed replay cursor. The later VOLE
//! worker adapter can use the ticket to select its correlation and trace
//! ranges without deriving an order from host completion timing.

use alloc::vec::Vec;
use core::fmt;

use cirrus_core::PusherByRef;
use volar_ir_common::{
    CommunicationGroup, EpochId, IterationId, ManifestError, OrdinalRange, ScheduleHasher,
    ScheduleLimits, ScheduleManifest, WorkerLane,
};

use crate::locked::PullerByRef;

/// One manifest-bound VOLE work item.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VoleScheduleTicket {
    /// Adapter-selected digest of the canonical schedule bytes.
    pub schedule_digest: [u8; 32],
    /// Canonical consistency epoch.
    pub epoch: EpochId,
    /// Canonical logical iteration.
    pub iteration: IterationId,
    /// Deterministic virtual worker lane.
    pub worker_lane: WorkerLane,
    /// Epoch transport group.
    pub communication_group: CommunicationGroup,
    /// Reserved correlation/hat ordinal range.
    pub correlation_range: OrdinalRange,
    /// Reserved logical communication range.
    pub communication_range: OrdinalRange,
}

impl VoleScheduleTicket {
    /// Resolve one ticket from a validated manifest and externally computed
    /// schedule digest.
    pub fn from_manifest(
        manifest: &ScheduleManifest,
        schedule_digest: [u8; 32],
        epoch: EpochId,
        iteration: IterationId,
    ) -> Result<Self, VoleScheduleError> {
        manifest.validate(ScheduleLimits::default())?;
        let epoch_manifest = manifest
            .epochs
            .iter()
            .find(|candidate| candidate.ordinal == epoch)
            .ok_or(VoleScheduleError::UnknownEpoch { epoch })?;
        let descriptor = epoch_manifest
            .iterations
            .iter()
            .find(|candidate| candidate.ordinal == iteration)
            .ok_or(VoleScheduleError::UnknownIteration { iteration })?;
        Ok(Self {
            schedule_digest,
            epoch,
            iteration,
            worker_lane: descriptor.worker_lane,
            communication_group: epoch_manifest.communication_group,
            correlation_range: descriptor.correlation_range,
            communication_range: descriptor.communication_range,
        })
    }

    /// Resolve a ticket after computing the digest with the adapter's hasher.
    pub fn from_manifest_with<H: ScheduleHasher>(
        manifest: &ScheduleManifest,
        limits: ScheduleLimits,
        hasher: &H,
        epoch: EpochId,
        iteration: IterationId,
    ) -> Result<Self, VoleScheduleError> {
        let digest = manifest.digest_with(limits, hasher)?;
        Self::from_manifest(manifest, digest, epoch, iteration)
    }
}

/// Fail-closed errors for ticket construction and replay.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VoleScheduleError {
    /// The shared manifest is malformed.
    Manifest(ManifestError),
    /// The requested epoch is not present.
    UnknownEpoch {
        /// Missing epoch identity.
        epoch: EpochId,
    },
    /// The requested iteration is not present in the requested epoch.
    UnknownIteration {
        /// Missing iteration identity.
        iteration: IterationId,
    },
    /// The next replay item did not match the expected canonical ticket.
    OutOfOrder {
        /// Ticket required by the manifest cursor.
        expected: IterationId,
        /// Ticket supplied by the caller.
        found: IterationId,
    },
    /// Replay ended before all manifest items were consumed.
    Incomplete {
        /// First ticket still required by the cursor.
        expected: IterationId,
    },
    /// A replay cursor was used after completion.
    Complete,
    /// The worker emitted a different number of hats than its reserved range.
    CorrelationCountMismatch {
        /// Iteration whose correlation range was violated.
        iteration: IterationId,
        /// Number of hats reserved by the manifest.
        expected: u64,
        /// Number of hats supplied by the worker.
        actual: usize,
    },
    /// The reserved hat count cannot be represented by this host's `usize`.
    CorrelationRangeTooLarge {
        /// Iteration whose range is too large for this adapter.
        iteration: IterationId,
        /// Number of hats reserved by the manifest.
        len: u64,
    },
}

impl fmt::Display for VoleScheduleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Manifest(error) => write!(f, "invalid VOLE schedule manifest: {error}"),
            Self::UnknownEpoch { epoch } => write!(f, "unknown VOLE schedule epoch {epoch:?}"),
            Self::UnknownIteration { iteration } => {
                write!(f, "unknown VOLE schedule iteration {iteration:?}")
            }
            Self::OutOfOrder { expected, found } => write!(
                f,
                "VOLE schedule replay expected {expected:?}, found {found:?}"
            ),
            Self::Incomplete { expected } => {
                write!(f, "VOLE schedule replay ended before {expected:?}")
            }
            Self::Complete => f.write_str("VOLE schedule replay is already complete"),
            Self::CorrelationCountMismatch {
                iteration,
                expected,
                actual,
            } => write!(
                f,
                "VOLE iteration {iteration:?} emitted {actual} hats; expected {expected}"
            ),
            Self::CorrelationRangeTooLarge { iteration, len } => write!(
                f,
                "VOLE iteration {iteration:?} reserves {len} hats, which exceeds host capacity"
            ),
        }
    }
}

impl From<ManifestError> for VoleScheduleError {
    fn from(error: ManifestError) -> Self {
        Self::Manifest(error)
    }
}

fn correlation_len(ticket: VoleScheduleTicket) -> Result<usize, VoleScheduleError> {
    usize::try_from(ticket.correlation_range.len).map_err(|_| {
        VoleScheduleError::CorrelationRangeTooLarge {
            iteration: ticket.iteration,
            len: ticket.correlation_range.len,
        }
    })
}

struct HatSinkState<H> {
    hats: Vec<H>,
    expected: usize,
    pushed: usize,
    overflowed: bool,
}

/// Thread-safe per-iteration hat buffer.
///
/// A worker can use this as the `PusherByRef` for an existing locked VOLE
/// context. Hats remain private to the ticket until [`Self::finish`] validates
/// the reserved correlation range; callers can then merge completed buffers
/// by manifest ordinal rather than by host completion order.
pub struct VoleScheduleHatSink<H> {
    ticket: VoleScheduleTicket,
    state: spin::Mutex<HatSinkState<H>>,
}

impl<H> VoleScheduleHatSink<H> {
    /// Create an empty hat buffer for one manifest ticket.
    pub fn new(ticket: VoleScheduleTicket) -> Result<Self, VoleScheduleError> {
        let expected = correlation_len(ticket)?;
        Ok(Self {
            ticket,
            state: spin::Mutex::new(HatSinkState {
                hats: Vec::with_capacity(expected),
                expected,
                pushed: 0,
                overflowed: false,
            }),
        })
    }

    /// The manifest ticket governing this buffer.
    pub const fn ticket(&self) -> VoleScheduleTicket {
        self.ticket
    }

    /// Validate and take the worker's hats for canonical merge.
    pub fn finish(self) -> Result<Vec<H>, VoleScheduleError> {
        let state = self.state.into_inner();
        if state.overflowed || state.hats.len() != state.expected {
            return Err(VoleScheduleError::CorrelationCountMismatch {
                iteration: self.ticket.iteration,
                expected: self.ticket.correlation_range.len,
                actual: state.pushed,
            });
        }
        Ok(state.hats)
    }
}

impl<H> cirrus_core::Pusher<H> for VoleScheduleHatSink<H> {
    fn push(&mut self, hat: H) {
        self.push_by_ref(hat);
    }
}

impl<H> PusherByRef<H> for VoleScheduleHatSink<H> {
    fn push_by_ref(&self, hat: H) {
        let mut state = self.state.lock();
        state.pushed = state.pushed.saturating_add(1);
        if state.hats.len() < state.expected {
            state.hats.push(hat);
        } else {
            state.overflowed = true;
        }
    }
}

/// Thread-safe verifier-side source for one ticket's ordered hats.
///
/// Construction checks the reserved count, while [`Self::finish`] checks that
/// the verifier consumed every supplied hat.
pub struct VoleScheduleHatSource<H> {
    ticket: VoleScheduleTicket,
    hats: Vec<H>,
    next: spin::Mutex<usize>,
}

impl<H> VoleScheduleHatSource<H> {
    /// Bind a complete hat buffer to one manifest ticket.
    pub fn new(ticket: VoleScheduleTicket, hats: Vec<H>) -> Result<Self, VoleScheduleError> {
        let expected = correlation_len(ticket)?;
        if hats.len() != expected {
            return Err(VoleScheduleError::CorrelationCountMismatch {
                iteration: ticket.iteration,
                expected: ticket.correlation_range.len,
                actual: hats.len(),
            });
        }
        Ok(Self {
            ticket,
            hats,
            next: spin::Mutex::new(0),
        })
    }

    /// The manifest ticket governing this source.
    pub const fn ticket(&self) -> VoleScheduleTicket {
        self.ticket
    }

    /// Validate that the verifier consumed the complete reserved range.
    pub fn finish(self) -> Result<(), VoleScheduleError> {
        let next = *self.next.lock();
        if next != self.hats.len() {
            return Err(VoleScheduleError::CorrelationCountMismatch {
                iteration: self.ticket.iteration,
                expected: self.ticket.correlation_range.len,
                actual: next,
            });
        }
        Ok(())
    }
}

impl<H: Clone> PullerByRef<H> for VoleScheduleHatSource<H> {
    fn next_by_ref(&self) -> Option<H> {
        let mut next = self.next.lock();
        let hat = self.hats.get(*next)?.clone();
        *next += 1;
        Some(hat)
    }
}

/// Canonical verifier-side replay cursor.
#[derive(Clone, Debug)]
pub struct VoleScheduleCursor {
    tickets: Vec<VoleScheduleTicket>,
    next: usize,
}

impl VoleScheduleCursor {
    /// Construct a cursor containing every manifest ticket in replay order.
    pub fn new(
        manifest: &ScheduleManifest,
        schedule_digest: [u8; 32],
    ) -> Result<Self, VoleScheduleError> {
        manifest.validate(ScheduleLimits::default())?;
        let mut tickets = Vec::new();
        for epoch in &manifest.epochs {
            for iteration in &epoch.iterations {
                tickets.push(VoleScheduleTicket {
                    schedule_digest,
                    epoch: epoch.ordinal,
                    iteration: iteration.ordinal,
                    worker_lane: iteration.worker_lane,
                    communication_group: epoch.communication_group,
                    correlation_range: iteration.correlation_range,
                    communication_range: iteration.communication_range,
                });
            }
        }
        Ok(Self { tickets, next: 0 })
    }

    /// Accept exactly the next canonical ticket.
    pub fn accept(&mut self, ticket: VoleScheduleTicket) -> Result<(), VoleScheduleError> {
        let Some(expected) = self.tickets.get(self.next).copied() else {
            return Err(VoleScheduleError::Complete);
        };
        if expected != ticket {
            return Err(VoleScheduleError::OutOfOrder {
                expected: expected.iteration,
                found: ticket.iteration,
            });
        }
        self.next += 1;
        Ok(())
    }

    /// Require that every canonical ticket has been replayed.
    pub fn finish(&self) -> Result<(), VoleScheduleError> {
        if let Some(expected) = self.tickets.get(self.next) {
            Err(VoleScheduleError::Incomplete {
                expected: expected.iteration,
            })
        } else {
            Ok(())
        }
    }

    /// Whether every manifest item has been accepted.
    pub fn is_complete(&self) -> bool {
        self.next == self.tickets.len()
    }

    /// Number of accepted tickets.
    pub const fn accepted(&self) -> usize {
        self.next
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;
    use volar_ir_common::{EpochManifest, IterationManifest, ScheduleManifest};

    struct SumHasher;

    impl ScheduleHasher for SumHasher {
        fn digest(&self, bytes: &[u8]) -> [u8; 32] {
            let mut digest = [0u8; 32];
            for (index, byte) in bytes.iter().copied().enumerate() {
                digest[index % digest.len()] = digest[index % digest.len()].wrapping_add(byte);
            }
            digest
        }
    }

    fn manifest() -> ScheduleManifest {
        let mut epoch = EpochManifest::new(EpochId(0), CommunicationGroup(0));
        epoch
            .iterations
            .push(IterationManifest::new(IterationId(0), 10, WorkerLane(0)));
        epoch
            .iterations
            .push(IterationManifest::new(IterationId(1), 11, WorkerLane(1)));
        let mut manifest = ScheduleManifest::new([1; 32], [2; 32], 7, 2, 2);
        manifest.epochs.push(epoch);
        manifest
    }

    #[test]
    fn cursor_accepts_only_canonical_ticket_order() {
        let manifest = manifest();
        let digest = [9; 32];
        let mut cursor = VoleScheduleCursor::new(&manifest, digest).unwrap();
        let second =
            VoleScheduleTicket::from_manifest(&manifest, digest, EpochId(0), IterationId(1))
                .unwrap();
        assert!(matches!(
            cursor.accept(second),
            Err(VoleScheduleError::OutOfOrder { .. })
        ));
        let first =
            VoleScheduleTicket::from_manifest(&manifest, digest, EpochId(0), IterationId(0))
                .unwrap();
        cursor.accept(first).unwrap();
        cursor.accept(second).unwrap();
        assert!(cursor.finish().is_ok());
        assert!(cursor.is_complete());
    }

    #[test]
    fn ticket_digest_is_bound_to_adapter_hash_input() {
        let manifest = manifest();
        let ticket = VoleScheduleTicket::from_manifest_with(
            &manifest,
            ScheduleLimits::default(),
            &SumHasher,
            EpochId(0),
            IterationId(0),
        )
        .unwrap();
        assert_eq!(
            ticket.schedule_digest,
            SumHasher.digest(&manifest.canonical_bytes(ScheduleLimits::default()).unwrap())
        );
    }

    #[test]
    fn hat_sink_validates_reserved_correlation_count() {
        let mut manifest = manifest();
        manifest.epochs[0].iterations[0].correlation_range = OrdinalRange::new(4, 2);
        let ticket =
            VoleScheduleTicket::from_manifest(&manifest, [9; 32], EpochId(0), IterationId(0))
                .unwrap();
        let sink = VoleScheduleHatSink::new(ticket).unwrap();
        sink.push_by_ref(10u8);
        sink.push_by_ref(11u8);
        assert_eq!(sink.finish().unwrap(), vec![10, 11]);

        let sink = VoleScheduleHatSink::new(ticket).unwrap();
        sink.push_by_ref(10u8);
        sink.push_by_ref(11u8);
        sink.push_by_ref(12u8);
        assert!(matches!(
            sink.finish(),
            Err(VoleScheduleError::CorrelationCountMismatch {
                expected: 2,
                actual: 3,
                ..
            })
        ));
    }

    #[test]
    fn hat_source_requires_complete_consumption() {
        let mut manifest = manifest();
        manifest.epochs[0].iterations[0].correlation_range = OrdinalRange::new(4, 2);
        let ticket =
            VoleScheduleTicket::from_manifest(&manifest, [9; 32], EpochId(0), IterationId(0))
                .unwrap();
        let source = VoleScheduleHatSource::new(ticket, vec![10u8, 11]).unwrap();
        assert_eq!(source.next_by_ref(), Some(10));
        assert!(matches!(
            source.finish(),
            Err(VoleScheduleError::CorrelationCountMismatch {
                expected: 2,
                actual: 1,
                ..
            })
        ));

        let source = VoleScheduleHatSource::new(ticket, vec![10u8, 11]).unwrap();
        assert_eq!(source.next_by_ref(), Some(10));
        assert_eq!(source.next_by_ref(), Some(11));
        assert_eq!(source.next_by_ref(), None);
        assert!(source.finish().is_ok());
    }
}
