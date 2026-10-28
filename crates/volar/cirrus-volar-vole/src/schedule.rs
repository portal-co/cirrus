//! VOLE adapter seam for canonical shared-storage schedule tickets.
//!
//! This module does not acquire locks or alter the existing ideal-functionality
//! contexts. It binds an adapter's logical VOLE work item to one manifest
//! ordinal and gives verifiers a fail-closed replay cursor. The later VOLE
//! worker adapter can use the ticket to select its correlation and trace
//! ranges without deriving an order from host completion timing.

use alloc::{collections::BTreeSet, vec::Vec};
use core::fmt;

use cirrus_core::PusherByRef;
use hybrid_array::{Array, ArraySize};
use volar_ir_common::{
    CommunicationGroup, EpochId, IterationId, ManifestError, OrdinalRange, ScheduleHasher,
    ScheduleLimits, ScheduleManifest, StorageId, TypeId, WorkerLane,
};

use crate::locked::PullerByRef;
use volar_spec::vole::{Delta, VoleArray};

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
        Self::from_manifest_with_limits(
            manifest,
            ScheduleLimits::default(),
            schedule_digest,
            epoch,
            iteration,
        )
    }

    /// Resolve a ticket with explicit validation limits and digest.
    pub fn from_manifest_with_limits(
        manifest: &ScheduleManifest,
        limits: ScheduleLimits,
        schedule_digest: [u8; 32],
        epoch: EpochId,
        iteration: IterationId,
    ) -> Result<Self, VoleScheduleError> {
        manifest.validate(limits)?;
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
        Self::from_manifest_with_limits(manifest, limits, digest, epoch, iteration)
    }

    /// Verify that public ticket fields still match the canonical manifest.
    ///
    /// Ticket fields are public so they can cross an adapter boundary, but a
    /// caller must not be able to relabel a ticket by changing its lane,
    /// group, or reserved ranges after construction.
    pub fn validate_against(&self, manifest: &ScheduleManifest) -> Result<(), VoleScheduleError> {
        self.validate_against_with_limits(manifest, ScheduleLimits::default())
    }

    /// Verify a ticket against a manifest using explicit resource limits.
    pub fn validate_against_with_limits(
        &self,
        manifest: &ScheduleManifest,
        limits: ScheduleLimits,
    ) -> Result<(), VoleScheduleError> {
        let expected = Self::from_manifest_with_limits(
            manifest,
            limits,
            self.schedule_digest,
            self.epoch,
            self.iteration,
        )?;
        if expected != *self {
            return Err(VoleScheduleError::TicketMismatch {
                epoch: self.epoch,
                iteration: self.iteration,
            });
        }
        Ok(())
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
    /// A ticket identity exists, but its public fields do not match the manifest.
    TicketMismatch {
        /// Epoch named by the malformed ticket.
        epoch: EpochId,
        /// Iteration named by the malformed ticket.
        iteration: IterationId,
    },
    /// Replay ended before all manifest items were consumed.
    Incomplete {
        /// First ticket still required by the cursor.
        expected: IterationId,
    },
    /// A replay cursor was used after completion.
    Complete,
    /// Two fragments were supplied for one logical iteration.
    DuplicateIteration {
        /// Repeated iteration identity.
        iteration: IterationId,
    },
    /// Fragments were created under different schedule digests.
    ScheduleDigestMismatch,
    /// Two fragments reserve overlapping correlation ordinals.
    CorrelationRangeOverlap {
        /// Iteration whose range overlaps a prior fragment.
        iteration: IterationId,
    },
    /// A ticket's correlation range overflows the ordinal space.
    CorrelationRangeOverflow {
        /// Iteration whose range is malformed.
        iteration: IterationId,
    },
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
    /// The verifier requested a hat after the ticket's range was exhausted.
    CorrelationOverConsumed {
        /// Iteration whose source was over-consumed.
        iteration: IterationId,
        /// Number of hats reserved by the manifest.
        expected: u64,
    },
    /// A trace fragment did not preserve strictly increasing logical time.
    TraceOrderMismatch {
        /// Iteration whose trace was malformed.
        iteration: IterationId,
        /// Previous logical-time ordinal.
        previous: u64,
        /// Current logical-time ordinal.
        found: u64,
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
            Self::TicketMismatch { epoch, iteration } => write!(
                f,
                "VOLE schedule ticket {epoch:?}/{iteration:?} does not match its manifest"
            ),
            Self::Incomplete { expected } => {
                write!(f, "VOLE schedule replay ended before {expected:?}")
            }
            Self::Complete => f.write_str("VOLE schedule replay is already complete"),
            Self::DuplicateIteration { iteration } => {
                write!(f, "VOLE schedule repeated iteration {iteration:?}")
            }
            Self::ScheduleDigestMismatch => {
                f.write_str("VOLE schedule fragments use different manifest digests")
            }
            Self::CorrelationRangeOverlap { iteration } => write!(
                f,
                "VOLE iteration {iteration:?} overlaps a prior correlation range"
            ),
            Self::CorrelationRangeOverflow { iteration } => write!(
                f,
                "VOLE iteration {iteration:?} has an overflowing correlation range"
            ),
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
            Self::CorrelationOverConsumed {
                iteration,
                expected,
            } => write!(
                f,
                "VOLE iteration {iteration:?} consumed more than its {expected} reserved hats"
            ),
            Self::TraceOrderMismatch {
                iteration,
                previous,
                found,
            } => write!(
                f,
                "VOLE iteration {iteration:?} trace time {found} follows {previous}"
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
    ticket
        .correlation_range
        .start
        .checked_add(ticket.correlation_range.len)
        .ok_or(VoleScheduleError::CorrelationRangeOverflow {
            iteration: ticket.iteration,
        })?;
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

/// A completed, ticket-bound worker hat buffer.
#[derive(Debug)]
pub struct VoleScheduleHatFragment<H> {
    /// Manifest ticket governing the hats.
    pub ticket: VoleScheduleTicket,
    /// Hats emitted by the worker in intra-iteration order.
    pub hats: Vec<H>,
}

/// Kind of authenticated storage operation in a schedule-bound trace.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VoleScheduleStorageAccessKind {
    /// Read the authenticated value at an address.
    Read,
    /// Write an authenticated value at an address.
    Write,
}

/// One schedule-bound storage trace entry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VoleScheduleStorageAccess {
    /// Epoch containing the logical operation.
    pub epoch: EpochId,
    /// Iteration containing the logical operation.
    pub iteration: IterationId,
    /// Strictly increasing logical time within the iteration.
    pub logical_time: u64,
    /// Typed storage namespace.
    pub storage: StorageId,
    /// Typed lane within the namespace.
    pub type_id: TypeId,
    /// Public address identity supplied by the storage adapter.
    pub address: u64,
    /// Operation kind.
    pub kind: VoleScheduleStorageAccessKind,
}

/// Completed, ticket-bound storage trace emitted by one VOLE worker.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VoleScheduleTraceFragment {
    /// Manifest ticket governing the entries.
    pub ticket: VoleScheduleTicket,
    /// Entries in logical execution order, never host completion order.
    pub entries: Vec<VoleScheduleStorageAccess>,
}

impl VoleScheduleTraceFragment {
    /// Bind a worker trace to a ticket and validate its logical order.
    pub fn new(
        ticket: VoleScheduleTicket,
        entries: Vec<VoleScheduleStorageAccess>,
    ) -> Result<Self, VoleScheduleError> {
        validate_trace_entries(ticket, &entries)?;
        Ok(Self { ticket, entries })
    }
}

fn validate_trace_entries(
    ticket: VoleScheduleTicket,
    entries: &[VoleScheduleStorageAccess],
) -> Result<(), VoleScheduleError> {
    let mut previous = None;
    for entry in entries {
        if entry.epoch != ticket.epoch || entry.iteration != ticket.iteration {
            return Err(VoleScheduleError::TicketMismatch {
                epoch: ticket.epoch,
                iteration: ticket.iteration,
            });
        }
        if let Some(previous) = previous
            && entry.logical_time <= previous
        {
            return Err(VoleScheduleError::TraceOrderMismatch {
                iteration: ticket.iteration,
                previous,
                found: entry.logical_time,
            });
        }
        previous = Some(entry.logical_time);
    }
    Ok(())
}

/// Merge completed storage traces by manifest ordinal.
///
/// The returned entries carry structured logical time and typed storage
/// identity. The merge rejects wrong-digest, relabelled, duplicate, missing,
/// and locally out-of-order fragments before exposing a trace to a verifier.
pub fn merge_trace_fragments_for_manifest(
    manifest: &ScheduleManifest,
    schedule_digest: [u8; 32],
    fragments: Vec<VoleScheduleTraceFragment>,
) -> Result<Vec<VoleScheduleStorageAccess>, VoleScheduleError> {
    merge_trace_fragments_for_manifest_with_limits(
        manifest,
        ScheduleLimits::default(),
        schedule_digest,
        fragments,
    )
}

/// Merge storage traces with explicit manifest resource limits.
pub fn merge_trace_fragments_for_manifest_with_limits(
    manifest: &ScheduleManifest,
    limits: ScheduleLimits,
    schedule_digest: [u8; 32],
    mut fragments: Vec<VoleScheduleTraceFragment>,
) -> Result<Vec<VoleScheduleStorageAccess>, VoleScheduleError> {
    manifest.validate(limits)?;
    let mut seen = BTreeSet::new();
    for fragment in &fragments {
        if fragment.ticket.schedule_digest != schedule_digest {
            return Err(VoleScheduleError::ScheduleDigestMismatch);
        }
        fragment
            .ticket
            .validate_against_with_limits(manifest, limits)?;
        validate_trace_entries(fragment.ticket, &fragment.entries)?;
        if !seen.insert((fragment.ticket.epoch, fragment.ticket.iteration)) {
            return Err(VoleScheduleError::DuplicateIteration {
                iteration: fragment.ticket.iteration,
            });
        }
    }
    for epoch in &manifest.epochs {
        for iteration in &epoch.iterations {
            if !seen.contains(&(epoch.ordinal, iteration.ordinal)) {
                return Err(VoleScheduleError::Incomplete {
                    expected: iteration.ordinal,
                });
            }
        }
    }
    fragments.sort_unstable_by_key(|fragment| (fragment.ticket.epoch, fragment.ticket.iteration));
    let mut output = Vec::new();
    for fragment in fragments {
        output.extend(fragment.entries);
    }
    Ok(output)
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
    pub fn finish(self) -> Result<VoleScheduleHatFragment<H>, VoleScheduleError> {
        let state = self.state.into_inner();
        if state.overflowed || state.hats.len() != state.expected {
            return Err(VoleScheduleError::CorrelationCountMismatch {
                iteration: self.ticket.iteration,
                expected: self.ticket.correlation_range.len,
                actual: state.pushed,
            });
        }
        Ok(VoleScheduleHatFragment {
            ticket: self.ticket,
            hats: state.hats,
        })
    }
}

/// Merge completed worker buffers in canonical iteration order.
///
/// Physical completion order is deliberately ignored. The function validates
/// digest identity, iteration uniqueness, reserved hat counts, and correlation
/// range disjointness before exposing one ordered stream to the verifier.
pub fn merge_hat_fragments<H>(
    mut fragments: Vec<VoleScheduleHatFragment<H>>,
) -> Result<Vec<H>, VoleScheduleError> {
    fragments.sort_unstable_by_key(|fragment| {
        (
            fragment.ticket.epoch,
            fragment.ticket.iteration,
            fragment.ticket.correlation_range.start,
        )
    });
    let digest = fragments
        .first()
        .map(|fragment| fragment.ticket.schedule_digest);
    let mut seen_iterations = BTreeSet::new();
    let mut ranges: Vec<OrdinalRange> = Vec::new();
    let mut output = Vec::new();
    for fragment in fragments {
        if Some(fragment.ticket.schedule_digest) != digest {
            return Err(VoleScheduleError::ScheduleDigestMismatch);
        }
        if !seen_iterations.insert(fragment.ticket.iteration) {
            return Err(VoleScheduleError::DuplicateIteration {
                iteration: fragment.ticket.iteration,
            });
        }
        let expected = correlation_len(fragment.ticket)?;
        if fragment.hats.len() != expected {
            return Err(VoleScheduleError::CorrelationCountMismatch {
                iteration: fragment.ticket.iteration,
                expected: fragment.ticket.correlation_range.len,
                actual: fragment.hats.len(),
            });
        }
        let range = fragment.ticket.correlation_range;
        if ranges.iter().copied().any(|prior| {
            prior.start < range.start + range.len && range.start < prior.start + prior.len
        }) {
            return Err(VoleScheduleError::CorrelationRangeOverlap {
                iteration: fragment.ticket.iteration,
            });
        }
        ranges.push(range);
        output.extend(fragment.hats);
    }
    Ok(output)
}

/// Validate ticket labels against a manifest before merging worker buffers.
///
/// [`merge_hat_fragments`] checks the internal consistency of a fragment set;
/// this entry point additionally prevents a caller from mutating a public
/// ticket's lane, group, or reserved ranges while retaining the same digest.
pub fn merge_hat_fragments_for_manifest<H>(
    manifest: &ScheduleManifest,
    schedule_digest: [u8; 32],
    fragments: Vec<VoleScheduleHatFragment<H>>,
) -> Result<Vec<H>, VoleScheduleError> {
    merge_hat_fragments_for_manifest_with_limits(
        manifest,
        ScheduleLimits::default(),
        schedule_digest,
        fragments,
    )
}

/// Validate and merge worker buffers with explicit manifest resource limits.
pub fn merge_hat_fragments_for_manifest_with_limits<H>(
    manifest: &ScheduleManifest,
    limits: ScheduleLimits,
    schedule_digest: [u8; 32],
    fragments: Vec<VoleScheduleHatFragment<H>>,
) -> Result<Vec<H>, VoleScheduleError> {
    manifest.validate(limits)?;
    let mut seen = BTreeSet::new();
    for fragment in &fragments {
        if fragment.ticket.schedule_digest != schedule_digest {
            return Err(VoleScheduleError::ScheduleDigestMismatch);
        }
        fragment
            .ticket
            .validate_against_with_limits(manifest, limits)?;
        if !seen.insert((fragment.ticket.epoch, fragment.ticket.iteration)) {
            return Err(VoleScheduleError::DuplicateIteration {
                iteration: fragment.ticket.iteration,
            });
        }
    }
    for epoch in &manifest.epochs {
        for iteration in &epoch.iterations {
            if !seen.contains(&(epoch.ordinal, iteration.ordinal)) {
                return Err(VoleScheduleError::Incomplete {
                    expected: iteration.ordinal,
                });
            }
        }
    }
    merge_hat_fragments(fragments)
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

struct HatSourceState {
    next: usize,
    over_consumed: bool,
}

/// Thread-safe verifier-side source for one ticket's ordered hats.
///
/// Construction checks the reserved count, while [`Self::finish`] checks that
/// the verifier consumed every supplied hat and did not over-consume it.
pub struct VoleScheduleHatSource<H> {
    ticket: VoleScheduleTicket,
    hats: Vec<H>,
    state: spin::Mutex<HatSourceState>,
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
            state: spin::Mutex::new(HatSourceState {
                next: 0,
                over_consumed: false,
            }),
        })
    }

    /// The manifest ticket governing this source.
    pub const fn ticket(&self) -> VoleScheduleTicket {
        self.ticket
    }

    /// Take the next hat, rejecting an exhausted ticket explicitly.
    pub fn try_next_by_ref(&self) -> Result<H, VoleScheduleError>
    where
        H: Clone,
    {
        let mut state = self.state.lock();
        let Some(hat) = self.hats.get(state.next).cloned() else {
            state.over_consumed = true;
            return Err(VoleScheduleError::CorrelationOverConsumed {
                iteration: self.ticket.iteration,
                expected: self.ticket.correlation_range.len,
            });
        };
        state.next += 1;
        Ok(hat)
    }

    /// Validate that the verifier consumed the complete reserved range.
    pub fn finish(self) -> Result<(), VoleScheduleError> {
        let state = self.state.into_inner();
        if state.over_consumed {
            return Err(VoleScheduleError::CorrelationOverConsumed {
                iteration: self.ticket.iteration,
                expected: self.ticket.correlation_range.len,
            });
        }
        if state.next != self.hats.len() {
            return Err(VoleScheduleError::CorrelationCountMismatch {
                iteration: self.ticket.iteration,
                expected: self.ticket.correlation_range.len,
                actual: state.next,
            });
        }
        Ok(())
    }
}

impl<H: Clone> PullerByRef<H> for VoleScheduleHatSource<H> {
    fn next_by_ref(&self) -> Option<H> {
        self.try_next_by_ref().ok()
    }
}

impl<H: Clone> Iterator for VoleScheduleHatSource<H> {
    type Item = H;

    fn next(&mut self) -> Option<Self::Item> {
        self.try_next_by_ref().ok()
    }
}

/// Bind one ticket's prover hat sink directly to the existing Cirrus VOLE
/// context. The sink still enforces the ticket's reserved correlation count;
/// callers must call [`VoleScheduleHatSink::finish`] after the circuit run.
impl<N: VoleArray<T>, T> VoleScheduleHatSink<Array<T, N>> {
    /// Create a prover context whose AND hats are recorded in this ticket sink.
    pub fn prover_context<'a>(
        &'a mut self,
        bit_to_t: fn(bool) -> T,
    ) -> crate::VoleProverContext<'a, 'a, N, T> {
        crate::VoleProverContext {
            hats: self,
            bit_to_t,
        }
    }
}

impl<N: VoleArray<T>, T: Send + Sync> VoleScheduleHatSink<Array<T, N>> {
    /// Create the thread-safe locked context used by real role-separated
    /// VOLE execution.
    pub fn locked_prover_context<'a>(
        &'a self,
        bit_to_t: fn(bool) -> T,
    ) -> crate::LockedVoleProverContext<'a, 'a, N, T> {
        crate::LockedVoleProverContext {
            hats: self,
            bit_to_t,
        }
    }
}

/// Bind one ticket's verifier hat source directly to the existing Cirrus VOLE
/// context. The returned context owns the source iterator; after execution,
/// move out `context.hats` and call [`VoleScheduleHatSource::finish`] to
/// enforce complete consumption.
impl<N: ArraySize, T: Clone> VoleScheduleHatSource<Array<T, N>> {
    /// Create a verifier context that consumes this ticket's hats in order.
    pub fn verifier_context<H>(
        self,
        delta: Delta<N, T>,
        hook: H,
    ) -> crate::VoleVerifierContext<N, T, Self, H> {
        crate::VoleVerifierContext {
            delta,
            hats: self,
            hook,
            gate_index: 0,
        }
    }
}

impl<N: ArraySize, T: Clone + Send + Sync> VoleScheduleHatSource<Array<T, N>> {
    /// Create the thread-safe locked context used by real role-separated
    /// VOLE execution.
    pub fn locked_verifier_context<'a>(
        &'a self,
        delta: Delta<N, T>,
    ) -> crate::LockedVoleVerifierContext<'a, 'a, N, T> {
        crate::LockedVoleVerifierContext { delta, hats: self }
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
        Self::new_with_limits(manifest, ScheduleLimits::default(), schedule_digest)
    }

    /// Construct a cursor with explicit manifest resource limits.
    pub fn new_with_limits(
        manifest: &ScheduleManifest,
        limits: ScheduleLimits,
        schedule_digest: [u8; 32],
    ) -> Result<Self, VoleScheduleError> {
        manifest.validate(limits)?;
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

    /// Return the next canonical ticket required by this cursor.
    pub fn next_ticket(&self) -> Option<VoleScheduleTicket> {
        self.tickets.get(self.next).copied()
    }

    /// Accept exactly the next canonical ticket.
    pub fn accept(&mut self, ticket: VoleScheduleTicket) -> Result<(), VoleScheduleError> {
        let Some(expected) = self.next_ticket() else {
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
        let first =
            VoleScheduleTicket::from_manifest(&manifest, digest, EpochId(0), IterationId(0))
                .unwrap();
        assert_eq!(cursor.next_ticket(), Some(first));
        let second =
            VoleScheduleTicket::from_manifest(&manifest, digest, EpochId(0), IterationId(1))
                .unwrap();
        assert!(matches!(
            cursor.accept(second),
            Err(VoleScheduleError::OutOfOrder { .. })
        ));
        cursor.accept(first).unwrap();
        assert_eq!(cursor.next_ticket(), Some(second));
        cursor.accept(second).unwrap();
        assert!(cursor.finish().is_ok());
        assert!(cursor.is_complete());
        assert_eq!(cursor.next_ticket(), None);
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
        let fragment = sink.finish().unwrap();
        assert_eq!(fragment.ticket, ticket);
        assert_eq!(fragment.hats, vec![10, 11]);

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
    fn fragments_merge_by_manifest_order_not_completion_order() {
        let mut manifest = manifest();
        manifest.epochs[0].iterations[0].correlation_range = OrdinalRange::new(4, 2);
        manifest.epochs[0].iterations[1].correlation_range = OrdinalRange::new(6, 1);
        let first =
            VoleScheduleTicket::from_manifest(&manifest, [9; 32], EpochId(0), IterationId(0))
                .unwrap();
        let second =
            VoleScheduleTicket::from_manifest(&manifest, [9; 32], EpochId(0), IterationId(1))
                .unwrap();
        let fragments = vec![
            VoleScheduleHatFragment {
                ticket: second,
                hats: vec![20u8],
            },
            VoleScheduleHatFragment {
                ticket: first,
                hats: vec![10u8, 11],
            },
        ];
        assert_eq!(merge_hat_fragments(fragments).unwrap(), vec![10, 11, 20]);
    }

    #[test]
    fn fragment_merge_rejects_duplicate_digest_and_overlapping_ranges() {
        let mut manifest = manifest();
        manifest.epochs[0].iterations[0].correlation_range = OrdinalRange::new(4, 2);
        manifest.epochs[0].iterations[1].correlation_range = OrdinalRange::new(6, 1);
        let first =
            VoleScheduleTicket::from_manifest(&manifest, [9; 32], EpochId(0), IterationId(0))
                .unwrap();
        let second =
            VoleScheduleTicket::from_manifest(&manifest, [9; 32], EpochId(0), IterationId(1))
                .unwrap();
        assert!(matches!(
            merge_hat_fragments(vec![
                VoleScheduleHatFragment {
                    ticket: first,
                    hats: vec![10u8, 11],
                },
                VoleScheduleHatFragment {
                    ticket: first,
                    hats: vec![12, 13],
                },
            ]),
            Err(VoleScheduleError::DuplicateIteration { .. })
        ));

        let wrong_digest = VoleScheduleTicket {
            schedule_digest: [8; 32],
            ..second
        };
        assert!(matches!(
            merge_hat_fragments(vec![
                VoleScheduleHatFragment {
                    ticket: first,
                    hats: vec![10u8, 11],
                },
                VoleScheduleHatFragment {
                    ticket: wrong_digest,
                    hats: vec![12],
                },
            ]),
            Err(VoleScheduleError::ScheduleDigestMismatch)
        ));

        let overlapping = VoleScheduleTicket {
            correlation_range: OrdinalRange::new(5, 1),
            ..second
        };
        assert!(matches!(
            merge_hat_fragments(vec![
                VoleScheduleHatFragment {
                    ticket: first,
                    hats: vec![10u8, 11],
                },
                VoleScheduleHatFragment {
                    ticket: overlapping,
                    hats: vec![12],
                },
            ]),
            Err(VoleScheduleError::CorrelationRangeOverlap { .. })
        ));
    }

    #[test]
    fn explicit_limits_apply_to_ticket_cursor_and_merge() {
        let manifest = manifest();
        let limits = ScheduleLimits {
            max_iterations: 1,
            ..ScheduleLimits::default()
        };
        assert!(matches!(
            VoleScheduleTicket::from_manifest_with_limits(
                &manifest,
                limits,
                [9; 32],
                EpochId(0),
                IterationId(0),
            ),
            Err(VoleScheduleError::Manifest(
                ManifestError::TooManyIterations { .. }
            ))
        ));
        assert!(matches!(
            VoleScheduleCursor::new_with_limits(&manifest, limits, [9; 32]),
            Err(VoleScheduleError::Manifest(
                ManifestError::TooManyIterations { .. }
            ))
        ));
    }

    #[test]
    fn checked_merge_rejects_relabelled_ticket() {
        let mut manifest = manifest();
        manifest.epochs[0].iterations.truncate(1);
        manifest.epochs[0].iterations[0].correlation_range = OrdinalRange::new(4, 1);
        let digest = [9; 32];
        let ticket =
            VoleScheduleTicket::from_manifest(&manifest, digest, EpochId(0), IterationId(0))
                .unwrap();
        let relabelled = VoleScheduleTicket {
            worker_lane: WorkerLane(99),
            ..ticket
        };
        assert!(matches!(
            merge_hat_fragments_for_manifest(
                &manifest,
                digest,
                vec![VoleScheduleHatFragment {
                    ticket: relabelled,
                    hats: vec![10u8],
                }]
            ),
            Err(VoleScheduleError::TicketMismatch {
                epoch: EpochId(0),
                iteration: IterationId(0),
            })
        ));
        assert_eq!(
            merge_hat_fragments_for_manifest(
                &manifest,
                digest,
                vec![VoleScheduleHatFragment {
                    ticket,
                    hats: vec![10u8],
                }]
            )
            .unwrap(),
            vec![10]
        );
    }

    #[test]
    fn checked_merge_rejects_missing_fragment() {
        let mut manifest = manifest();
        manifest.epochs[0].iterations[0].correlation_range = OrdinalRange::new(4, 1);
        manifest.epochs[0].iterations[1].correlation_range = OrdinalRange::new(5, 1);
        let digest = [9; 32];
        let first =
            VoleScheduleTicket::from_manifest(&manifest, digest, EpochId(0), IterationId(0))
                .unwrap();
        assert!(matches!(
            merge_hat_fragments_for_manifest(
                &manifest,
                digest,
                vec![VoleScheduleHatFragment {
                    ticket: first,
                    hats: vec![10u8],
                }]
            ),
            Err(VoleScheduleError::Incomplete {
                expected: IterationId(1),
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
        assert!(matches!(
            source.finish(),
            Err(VoleScheduleError::CorrelationOverConsumed { expected: 2, .. })
        ));
    }

    #[test]
    fn ticket_hat_buffers_wire_into_role_contexts() {
        use cipher::consts::U1;
        use volar_spec::{field::Galois128, vole::Delta};

        let mut manifest = manifest();
        manifest.epochs[0].iterations[0].correlation_range = OrdinalRange::new(0, 2);
        let ticket =
            VoleScheduleTicket::from_manifest(&manifest, [9; 32], EpochId(0), IterationId(0))
                .unwrap();
        let mut sink = VoleScheduleHatSink::new(ticket).unwrap();
        let _prover = sink.prover_context(|bit| Galois128(u128::from(bit)));
        drop(_prover);
        let _locked_prover = sink.locked_prover_context(|bit| Galois128(u128::from(bit)));
        drop(_locked_prover);
        sink.push_by_ref(Array::from_fn(|_| Galois128(1)));
        sink.push_by_ref(Array::from_fn(|_| Galois128(2)));
        let fragment = sink.finish().unwrap();

        let source = VoleScheduleHatSource::new(ticket, fragment.hats).unwrap();
        let delta = Delta {
            delta: Array::<Galois128, U1>::from_fn(|_| Galois128(3)),
        };
        let context = source.verifier_context(delta.clone(), crate::NoopVoleVerifierHook);
        let mut source = context.hats;
        assert!(source.next().is_some());
        assert!(source.next().is_some());
        assert!(source.finish().is_ok());

        let source = VoleScheduleHatSource::new(
            ticket,
            vec![
                Array::from_fn(|_| Galois128(1)),
                Array::from_fn(|_| Galois128(2)),
            ],
        )
        .unwrap();
        let _locked_verifier = source.locked_verifier_context(delta);
        drop(_locked_verifier);
    }

    #[test]
    fn trace_fragments_merge_by_ticket_and_preserve_typed_time() {
        let mut manifest = manifest();
        manifest.epochs[0].iterations.truncate(2);
        let digest = [9; 32];
        let first =
            VoleScheduleTicket::from_manifest(&manifest, digest, EpochId(0), IterationId(0))
                .unwrap();
        let second =
            VoleScheduleTicket::from_manifest(&manifest, digest, EpochId(0), IterationId(1))
                .unwrap();
        let first_entry = VoleScheduleStorageAccess {
            epoch: EpochId(0),
            iteration: IterationId(0),
            logical_time: 3,
            storage: StorageId(4),
            type_id: TypeId(7),
            address: 11,
            kind: VoleScheduleStorageAccessKind::Read,
        };
        let second_entry = VoleScheduleStorageAccess {
            epoch: EpochId(0),
            iteration: IterationId(1),
            logical_time: 1,
            storage: StorageId(4),
            type_id: TypeId(8),
            address: 12,
            kind: VoleScheduleStorageAccessKind::Write,
        };
        let merged = merge_trace_fragments_for_manifest(
            &manifest,
            digest,
            vec![
                VoleScheduleTraceFragment::new(second, vec![second_entry]).unwrap(),
                VoleScheduleTraceFragment::new(first, vec![first_entry]).unwrap(),
            ],
        )
        .unwrap();
        assert_eq!(merged, vec![first_entry, second_entry]);
    }

    #[test]
    fn trace_fragment_rejects_noncanonical_time_and_missing_iteration() {
        let mut manifest = manifest();
        manifest.epochs[0].iterations.truncate(1);
        let digest = [9; 32];
        let ticket =
            VoleScheduleTicket::from_manifest(&manifest, digest, EpochId(0), IterationId(0))
                .unwrap();
        let entry = |logical_time| VoleScheduleStorageAccess {
            epoch: EpochId(0),
            iteration: IterationId(0),
            logical_time,
            storage: StorageId(1),
            type_id: TypeId(2),
            address: 0,
            kind: VoleScheduleStorageAccessKind::Read,
        };
        assert!(matches!(
            VoleScheduleTraceFragment::new(ticket, vec![entry(2), entry(2)]),
            Err(VoleScheduleError::TraceOrderMismatch { .. })
        ));
        assert!(matches!(
            merge_trace_fragments_for_manifest(&manifest, [8; 32], vec![]),
            Err(VoleScheduleError::Incomplete { .. })
        ));
    }
}
