//! VOLE adapter seam for canonical shared-storage schedule tickets.
//!
//! This module does not acquire locks or alter the existing ideal-functionality
//! contexts. It binds an adapter's logical VOLE work item to one manifest
//! ordinal and gives verifiers a fail-closed replay cursor. The later VOLE
//! worker adapter can use the ticket to select its correlation and trace
//! ranges without deriving an order from host completion timing.

use alloc::vec::Vec;
use core::fmt;

use volar_ir_common::{
    CommunicationGroup, EpochId, IterationId, ManifestError, OrdinalRange, ScheduleHasher,
    ScheduleLimits, ScheduleManifest, WorkerLane,
};

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
        }
    }
}

impl From<ManifestError> for VoleScheduleError {
    fn from(error: ManifestError) -> Self {
        Self::Manifest(error)
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
}
