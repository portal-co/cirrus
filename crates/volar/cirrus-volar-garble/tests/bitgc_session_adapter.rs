//! Explicit Cirrus adapter coverage for the shared BitGC session transport.
#![cfg(all(test, feature = "bitgc-session"))]

use cirrus_core::Pusher;
use cirrus_volar_garble::session::{
    CirrusBitGcSession, CirrusSessionChannel, CirrusSessionChannelError, SessionFrameKind,
};
use volar_mpc::bitgc_session::{
    BITGC_FRAME_MAX_BYTES, BitGcFrame, BitGcFrameTransport, BitGcOtTransport, BitGcRole,
    BitGcSessionBinding,
};
use volar_spec::bitgc::Variant;

struct VecPusher(Vec<Vec<u8>>);

impl Pusher<Vec<u8>> for VecPusher {
    fn push(&mut self, value: Vec<u8>) {
        self.0.push(value);
    }
}

fn binding() -> BitGcSessionBinding {
    BitGcSessionBinding {
        session_id: [7; 32],
        variant: Variant::Small,
        parameter_fingerprint: [8; 32],
        circuit_digest: [9; 32],
    }
}

fn authenticated_pair() -> (CirrusBitGcSession, CirrusBitGcSession) {
    (
        CirrusBitGcSession::new_authenticated(BitGcRole::Garbler, binding(), [0x41; 32]).unwrap(),
        CirrusBitGcSession::new_authenticated(BitGcRole::Evaluator, binding(), [0x41; 32]).unwrap(),
    )
}

#[test]
fn cirrus_channel_drives_authenticated_shared_frames() {
    let (mut garbler, mut evaluator) = authenticated_pair();
    let garbler_hello = garbler.hello().unwrap();
    let evaluator_hello = evaluator.hello().unwrap();
    garbler.inner_mut().accept(&evaluator_hello).unwrap();
    evaluator.inner_mut().accept(&garbler_hello).unwrap();

    let mut to_evaluator = VecPusher(Vec::new());
    {
        let mut channel = CirrusSessionChannel::new(&mut to_evaluator, core::iter::empty());
        garbler
            .frame_session()
            .send_over(&mut channel, SessionFrameKind::Data, &[1, 2, 3])
            .unwrap();
    }

    let mut evaluator_sink = VecPusher(Vec::new());
    let mut channel = CirrusSessionChannel::new(&mut evaluator_sink, to_evaluator.0.into_iter());
    let encoded = evaluator
        .frame_session()
        .receive_over(&mut channel)
        .unwrap();
    let frame = BitGcFrame::decode(&encoded).unwrap();
    assert_eq!(
        evaluator
            .frame_session()
            .payload_without_auth(&frame)
            .unwrap(),
        &[1, 2, 3]
    );
}

#[test]
fn cirrus_channel_drives_authenticated_ot_envelope() {
    let (mut garbler, mut evaluator) = authenticated_pair();
    let garbler_hello = garbler.hello().unwrap();
    let evaluator_hello = evaluator.hello().unwrap();
    garbler.inner_mut().accept(&evaluator_hello).unwrap();
    evaluator.inner_mut().accept(&garbler_hello).unwrap();

    let mut to_evaluator = VecPusher(Vec::new());
    {
        let channel = CirrusSessionChannel::new(&mut to_evaluator, core::iter::empty());
        let mut ot = garbler.authenticated_ot_transport(channel).unwrap();
        ot.send_ot_message(&[9, 8, 7]).unwrap();
    }

    let mut evaluator_sink = VecPusher(Vec::new());
    let channel = CirrusSessionChannel::new(&mut evaluator_sink, to_evaluator.0.into_iter());
    let mut ot = evaluator.authenticated_ot_transport(channel).unwrap();
    assert_eq!(ot.receive_ot_message(3).unwrap(), vec![9, 8, 7]);
}

#[test]
fn cirrus_channel_rejects_oversized_and_cancelled_records() {
    let mut sink = VecPusher(Vec::new());
    let mut channel = CirrusSessionChannel::new(&mut sink, vec![vec![1, 2, 3]].into_iter());
    assert_eq!(
        channel.receive_frame(2),
        Err(CirrusSessionChannelError::Oversized {
            actual: 3,
            maximum: 2,
        })
    );
    assert!(channel.is_cancelled());

    let mut sink = VecPusher(Vec::new());
    let oversized = vec![0; BITGC_FRAME_MAX_BYTES + 1];
    let mut channel = CirrusSessionChannel::new(&mut sink, vec![oversized].into_iter());
    assert_eq!(
        channel.receive_frame(usize::MAX),
        Err(CirrusSessionChannelError::Oversized {
            actual: BITGC_FRAME_MAX_BYTES + 1,
            maximum: BITGC_FRAME_MAX_BYTES,
        })
    );
    assert!(channel.is_cancelled());

    let mut sink = VecPusher(Vec::new());
    let mut channel = CirrusSessionChannel::new(&mut sink, core::iter::empty());
    BitGcFrameTransport::cancel(&mut channel).unwrap();
    assert_eq!(
        channel.receive_frame(8),
        Err(CirrusSessionChannelError::Cancelled)
    );
}
