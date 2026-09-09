// @pinnedness: unpinned
// @stability: very-unstable
//! @ai: assisted
//!
//! OT transport binding for [`miniot`]: run the ML-KEM 1-of-N OT over a
//! framed byte transport, so the two parties can live in separate processes
//! or on an in-memory channel.
//!
//! `miniot`'s three functions are pure value transformers — the harness moves
//! the `EncapsulationKey1024` / `(Ciphertext, SharedKey)` arrays itself. This
//! module adds that movement:
//!
//! - [`FrameTransport`]: the minimal no_std framed-byte channel the OT rides
//!   on. Two implementations are provided: [`DuplexTransport`] (in-memory
//!   rendezvous, for lockstep tests) and — behind the `std` feature —
//!   [`TcpTransport`] (length-prefixed blocking TCP, for cross-process).
//! - [`ot_send`] / [`ot_receive`]: drive one full 1-of-N OT over a
//!   `FrameTransport`, the sender offering `N` payload bytes and the receiver
//!   recovering the one at its choice index.
//!
//! # Wire format
//!
//! A frame is `u32-LE length ‖ payload`. An OT exchange is two frames:
//!
//! 1. **Receiver → sender:** the `N` encapsulation keys, each fixed-width
//!    ([`EK_BYTES`]), concatenated, chosen index first by convention of the
//!    receiver's placement (the sender does not learn which).
//! 2. **Sender → receiver:** for each index, the ML-KEM ciphertext
//!    ([`CT_BYTES`]) followed by the KDF-masked payload (payload length bytes).
//!
//! The sender learns nothing about the choice index (it encapsulates against
//! all `N` keys); the receiver can decapsulate only its choice index's
//! ciphertext, so it recovers only that payload.

#![no_std]

extern crate alloc;

use alloc::vec::Vec;

use digest::{Digest, array::Array};
use ml_kem::kem::common::rand_core::CryptoRng;
use ml_kem::ml_kem_1024::Ciphertext;
use ml_kem::{Encapsulate, EncapsulationKey1024, KeyExport};

/// Byte length of an ML-KEM-1024 encapsulation key.
pub const EK_BYTES: usize = 1568;
/// Byte length of an ML-KEM-1024 ciphertext.
pub const CT_BYTES: usize = 1568;

// ============================================================================
// Transport trait
// ============================================================================

/// A framed byte channel: the minimal transport the OT exchange needs.
///
/// Each `send` moves one length-delimited frame; `recv` returns exactly one
/// frame's payload. This is transport-agnostic — in-memory queues, TCP, or any
/// reliable ordered byte stream can back it.
pub trait FrameTransport {
    /// Send one frame.
    fn send(&mut self, frame: &[u8]);
    /// Receive one frame's payload.
    fn recv(&mut self) -> Vec<u8>;
}

// ============================================================================
// In-memory duplex transport (lockstep tests)
// ============================================================================

/// A pair of connected in-memory transports over shared rendezvous queues.
///
/// Create with [`duplex`]; each half implements [`FrameTransport`]. `send`
/// pushes onto the peer's queue, `recv` pops from the caller's own queue, so a
/// single thread can interleave both roles by scripting the sends before the
/// receives (the same lockstep discipline as `volar-mpc`'s `run_local`).
pub struct DuplexTransport {
    incoming: alloc::rc::Rc<core::cell::RefCell<alloc::collections::VecDeque<Vec<u8>>>>,
    outgoing: alloc::rc::Rc<core::cell::RefCell<alloc::collections::VecDeque<Vec<u8>>>>,
}

/// Create a connected pair of in-memory transports; `a.send` is `b.recv` and
/// vice versa.
pub fn duplex() -> (DuplexTransport, DuplexTransport) {
    use alloc::rc::Rc;
    use alloc::collections::VecDeque;
    use core::cell::RefCell;
    let ab = Rc::new(RefCell::new(VecDeque::new()));
    let ba = Rc::new(RefCell::new(VecDeque::new()));
    (
        DuplexTransport { incoming: ba.clone(), outgoing: ab.clone() },
        DuplexTransport { incoming: ab, outgoing: ba },
    )
}

impl FrameTransport for DuplexTransport {
    fn send(&mut self, frame: &[u8]) {
        self.outgoing.borrow_mut().push_back(frame.to_vec());
    }
    fn recv(&mut self) -> Vec<u8> {
        self.incoming
            .borrow_mut()
            .pop_front()
            .expect("duplex transport: no frame available (lockstep mis-sequencing)")
    }
}

// ============================================================================
// OT over a transport
// ============================================================================

/// KDF-expand a 32-byte ML-KEM shared key to `n` bytes via the digest in
/// counter mode: `H(key ‖ u32-LE counter)` repeated.
fn kdf_expand<D: Digest>(key: &[u8], n: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n);
    let mut ctr: u32 = 0;
    while out.len() < n {
        let mut h = D::new();
        h.update(key);
        h.update(ctr.to_le_bytes());
        out.extend_from_slice(&h.finalize());
        ctr += 1;
    }
    out.truncate(n);
    out
}

fn ek_from_bytes(b: &[u8]) -> Option<EncapsulationKey1024> {
    if b.len() != EK_BYTES {
        return None;
    }
    let arr: &Array<u8, _> = b.try_into().ok()?;
    EncapsulationKey1024::new(arr).ok()
}

/// Receiver half, const-generic arity `N`: send the `N` encapsulation keys
/// (real key planted at `choice`), then read the sender's frame and recover
/// the chosen payload of `payload_len` bytes.
pub fn ot_receive_n<D: Digest, T: FrameTransport, const N: usize>(
    transport: &mut T,
    rng: &mut (dyn CryptoRng + '_),
    choice: usize,
    payload_len: usize,
) -> Vec<u8> {
    assert!(choice < N, "choice index out of range");
    let (eks, decaps) = crate::miniot_start_recv::<N>(rng, choice);
    // Frame 1: the N encapsulation keys.
    let mut f1 = Vec::with_capacity(N * EK_BYTES);
    for ek in &eks {
        f1.extend_from_slice(ek.to_bytes().as_slice());
    }
    transport.send(&f1);

    // Frame 2: N × (ciphertext ‖ masked payload).
    let f2 = transport.recv();
    let half = CT_BYTES + payload_len;
    assert_eq!(f2.len(), N * half, "sender frame length mismatch");
    let seg = &f2[choice * half..(choice + 1) * half];
    let ct: Ciphertext = seg[..CT_BYTES].try_into().expect("ciphertext length");
    let masked = &seg[CT_BYTES..];
    let shared = ml_kem::Decapsulate::decapsulate(&decaps, &ct);
    let keystream = kdf_expand::<D>(shared.as_slice(), payload_len);
    (0..payload_len).map(|i| masked[i] ^ keystream[i]).collect()
}

/// Sender half, const-generic arity `N`: read the receiver's encapsulation
/// keys, encapsulate a fresh shared key against each, and send each payload
/// masked with that key. All payloads must be the same length.
pub fn ot_send_n<D: Digest, T: FrameTransport, const N: usize>(
    transport: &mut T,
    rng: &mut (dyn CryptoRng + '_),
    payloads: &[&[u8]; N],
) {
    let payload_len = payloads[0].len();
    assert!(
        payloads.iter().all(|p| p.len() == payload_len),
        "all OT payloads must be the same length"
    );
    let f1 = transport.recv();
    assert_eq!(f1.len(), N * EK_BYTES, "receiver frame length mismatch");

    let mut f2 = Vec::with_capacity(N * (CT_BYTES + payload_len));
    for i in 0..N {
        let ek = ek_from_bytes(&f1[i * EK_BYTES..(i + 1) * EK_BYTES]).expect("bad encapsulation key");
        let (ct, shared) = ek.encapsulate_with_rng(rng);
        let keystream = kdf_expand::<D>(shared.as_slice(), payload_len);
        f2.extend_from_slice(ct.as_slice());
        for j in 0..payload_len {
            f2.push(payloads[i][j] ^ keystream[j]);
        }
    }
    transport.send(&f2);
}

// ============================================================================
// Thread-safe channel transport (std)
// ============================================================================

/// A connected pair of thread-safe transports over `std::sync::mpsc` channels,
/// behind the `std` feature. Each half is `Send`, so the two OT roles can run
/// on separate OS threads (real concurrency, no lockstep scripting).
#[cfg(feature = "std")]
pub struct ChannelTransport {
    incoming: std::sync::mpsc::Receiver<Vec<u8>>,
    outgoing: std::sync::mpsc::Sender<Vec<u8>>,
}

/// Create a connected, thread-safe pair; `a.send` is `b.recv` and vice versa.
#[cfg(feature = "std")]
pub fn channel_pair() -> (ChannelTransport, ChannelTransport) {
    let (ab_tx, ab_rx) = std::sync::mpsc::channel::<Vec<u8>>();
    let (ba_tx, ba_rx) = std::sync::mpsc::channel::<Vec<u8>>();
    (
        ChannelTransport { incoming: ba_rx, outgoing: ab_tx },
        ChannelTransport { incoming: ab_rx, outgoing: ba_tx },
    )
}

#[cfg(feature = "std")]
impl FrameTransport for ChannelTransport {
    fn send(&mut self, frame: &[u8]) {
        self.outgoing.send(frame.to_vec()).expect("channel send");
    }
    fn recv(&mut self) -> Vec<u8> {
        self.incoming.recv().expect("channel recv")
    }
}

// ============================================================================
// TCP transport (std)
// ============================================================================

/// A framed blocking-TCP [`FrameTransport`], behind the `std` feature, using
/// the same `u32-LE length ‖ payload` framing as the rest of the workspace.
#[cfg(feature = "std")]
pub struct TcpTransport {
    stream: std::net::TcpStream,
}

#[cfg(feature = "std")]
impl TcpTransport {
    /// Wrap an existing stream.
    pub fn new(stream: std::net::TcpStream) -> Self {
        Self { stream }
    }
    /// Connect to `addr`.
    pub fn connect(addr: &str) -> std::io::Result<Self> {
        Ok(Self::new(std::net::TcpStream::connect(addr)?))
    }
    /// Accept one connection from `listener`.
    pub fn accept(listener: &std::net::TcpListener) -> std::io::Result<Self> {
        let (stream, _) = listener.accept()?;
        Ok(Self::new(stream))
    }
    /// Independent handle to the same connection.
    pub fn try_clone(&self) -> std::io::Result<Self> {
        Ok(Self::new(self.stream.try_clone()?))
    }
}

#[cfg(feature = "std")]
impl FrameTransport for TcpTransport {
    fn send(&mut self, frame: &[u8]) {
        use std::io::Write;
        let len = frame.len() as u32;
        self.stream.write_all(&len.to_le_bytes()).expect("write len");
        self.stream.write_all(frame).expect("write frame");
        self.stream.flush().expect("flush");
    }
    fn recv(&mut self) -> Vec<u8> {
        use std::io::Read;
        let mut lenb = [0u8; 4];
        self.stream.read_exact(&mut lenb).expect("read len");
        let len = u32::from_le_bytes(lenb) as usize;
        let mut buf = alloc::vec![0u8; len];
        self.stream.read_exact(&mut buf).expect("read frame");
        buf
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use rand::rand_core::UnwrapErr;
    use rand::rngs::SysRng;
    use sha2::Sha256;

    /// In-memory duplex 1-of-2 OT: the receiver recovers exactly its chosen
    /// payload, for both choices.
    #[test]
    fn duplex_ot_1_of_2() {
        let p0 = b"payload-zero-16B!".to_vec();
        let p1 = b"payload-one--16B!".to_vec();
        assert_eq!(p0.len(), p1.len());

        for choice in 0..2usize {
            let (mut a, mut b) = duplex();
            let mut send_rng = UnwrapErr(SysRng);
            let mut recv_rng = UnwrapErr(SysRng);

            // Receiver posts its encapsulation keys first.
            // Lockstep: we must run receiver setup before sender recv, so
            // interleave by hand using the const-generic helpers' halves.
            // Simplest: run receiver fully against `b` after sender against
            // `a` has produced its frame — but the sender needs the receiver's
            // keys first. So: receiver sends keys, sender runs, receiver reads.
            let mut recv_state = {
                // Drive the receiver's first half inline (send keys), then the
                // sender, then the receiver's second half (read+recover).
                // ot_receive_n does both halves in one call, so instead call
                // it on a separate "thread" of control via the duplex queues:
                // pre-run is not possible single-threaded without splitting.
                // We therefore split manually here.
                ()
            };
            let _ = recv_state;

            // Manual interleave: replicate ot_receive_n's two phases.
            let (eks, decaps) = crate::miniot_start_recv::<2>(&mut recv_rng, choice);
            let mut f1 = Vec::with_capacity(2 * EK_BYTES);
            for ek in &eks {
                f1.extend_from_slice(ml_kem::KeyExport::to_bytes(ek).as_slice());
            }
            b.send(&f1); // receiver -> sender

            ot_send_n::<Sha256, _, 2>(&mut a, &mut send_rng, &[&p0, &p1]);

            // Receiver second half.
            let f2 = b.recv();
            let half = CT_BYTES + p0.len();
            let seg = &f2[choice * half..(choice + 1) * half];
            let ct: Ciphertext = seg[..CT_BYTES].try_into().unwrap();
            let masked = &seg[CT_BYTES..];
            let shared = ml_kem::Decapsulate::decapsulate(&decaps, &ct);
            let keystream = kdf_expand::<Sha256>(shared.as_slice(), p0.len());
            let got: Vec<u8> = (0..p0.len()).map(|i| masked[i] ^ keystream[i]).collect();

            let want = if choice == 0 { &p0 } else { &p1 };
            assert_eq!(&got, want, "choice {} recovers its payload", choice);
        }
    }

    /// The full const-generic `ot_receive_n`/`ot_send_n` pair over a duplex
    /// transport, run on two OS threads (real concurrency, no manual
    /// interleave).
    #[cfg(feature = "std")]
    #[test]
    fn duplex_ot_threaded() {
        let p0 = std::vec![0xAAu8; 32];
        let p1 = std::vec![0xBBu8; 32];
        let (mut a, mut b) = channel_pair();

        let p0c = p0.clone();
        let p1c = p1.clone();
        let sender = std::thread::spawn(move || {
            let mut rng = UnwrapErr(SysRng);
            ot_send_n::<Sha256, _, 2>(&mut a, &mut rng, &[&p0c, &p1c]);
        });

        let mut recv_rng = UnwrapErr(SysRng);
        let got = ot_receive_n::<Sha256, _, 2>(&mut b, &mut recv_rng, 1, 32);
        sender.join().unwrap();
        assert_eq!(got, p1);
    }

    /// 1-of-N OT with N=4 over the threaded duplex transport.
    #[cfg(feature = "std")]
    #[test]
    fn duplex_ot_1_of_n() {
        let payloads: [std::vec::Vec<u8>; 4] = [
            std::vec![1u8; 24],
            std::vec![2u8; 24],
            std::vec![3u8; 24],
            std::vec![4u8; 24],
        ];
        for choice in 0..4usize {
            let (mut a, mut b) = channel_pair();
            let pc = payloads.clone();
            let sender = std::thread::spawn(move || {
                let mut rng = UnwrapErr(SysRng);
                ot_send_n::<Sha256, _, 4>(&mut a, &mut rng, &[&pc[0], &pc[1], &pc[2], &pc[3]]);
            });
            let mut recv_rng = UnwrapErr(SysRng);
            let got = ot_receive_n::<Sha256, _, 4>(&mut b, &mut recv_rng, choice, 24);
            sender.join().unwrap();
            assert_eq!(got, payloads[choice], "choice {}", choice);
        }
    }

    /// Cross-process 1-of-2 OT over a real loopback TCP socket.
    #[cfg(feature = "std")]
    #[test]
    fn tcp_ot_1_of_2() {
        let p0 = std::vec![0x11u8; 16];
        let p1 = std::vec![0x22u8; 16];

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = std::format!("{}", listener.local_addr().unwrap());

        let p0c = p0.clone();
        let p1c = p1.clone();
        let sender = std::thread::spawn(move || {
            let mut t = TcpTransport::accept(&listener).unwrap();
            let mut rng = UnwrapErr(SysRng);
            ot_send_n::<Sha256, _, 2>(&mut t, &mut rng, &[&p0c, &p1c]);
        });

        let mut t = TcpTransport::connect(&addr).unwrap();
        let mut recv_rng = UnwrapErr(SysRng);
        let got = ot_receive_n::<Sha256, _, 2>(&mut t, &mut recv_rng, 0, 16);
        sender.join().unwrap();
        assert_eq!(got, p0);
    }
}
