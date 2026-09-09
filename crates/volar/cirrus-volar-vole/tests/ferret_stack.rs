//! Framed TCP loopback: LWE + SoftSpoken + Ferret-Reg pool driving a
//! Cirrus VOLE circuit in a loop. Hats share the same tagged stream as
//! the OT messages.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::thread;

use cipher::consts::U1;
use cirrus_core::{ContextWithBitAnd, ContextWithBitXor, ContextWithCreate, Pusher};
use cirrus_recompile_core::Recorder;
use cirrus_volar_vole::{NoopVoleVerifierHook, VoleProverContext, VoleVerifierContext};
use hybrid_array::Array;
use volar_spec::{
    SpecRng,
    field::Galois128,
    ot::{
        ferret::FERRET_REG_TOY,
        two_party::{
            StackIo, stack_bea95_receiver, stack_bea95_sender, stack_setup_receiver,
            stack_setup_sender,
        },
        wire::TAG_HAT,
    },
    vole::{Delta, Q, Vope, setup::vole_commit_bit_shares},
};

struct TestRng(u64);
impl SpecRng for TestRng {
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        (z ^ (z >> 31)) as u32
    }
}

struct TcpIo {
    stream: TcpStream,
}

impl StackIo for TcpIo {
    fn send(&mut self, tag: u8, payload: &[u8]) {
        let len = (1 + payload.len()) as u32;
        self.stream.write_all(&len.to_le_bytes()).unwrap();
        self.stream.write_all(&[tag]).unwrap();
        self.stream.write_all(payload).unwrap();
        self.stream.flush().unwrap();
    }

    fn recv(&mut self, expected_tag: u8) -> Vec<u8> {
        let mut lenb = [0u8; 4];
        self.stream.read_exact(&mut lenb).unwrap();
        let len = u32::from_le_bytes(lenb) as usize;
        let mut buf = vec![0u8; len];
        self.stream.read_exact(&mut buf).unwrap();
        assert_eq!(buf[0], expected_tag, "framed tag mismatch");
        buf[1..].to_vec()
    }
}

struct HatPush<'a>(&'a mut TcpIo);
impl Pusher<Array<Galois128, U1>> for HatPush<'_> {
    fn push(&mut self, hat: Array<Galois128, U1>) {
        self.0.send(TAG_HAT, &hat[0].0.to_le_bytes());
    }
}

struct HatIter<'a>(&'a mut TcpIo);
impl Iterator for HatIter<'_> {
    type Item = Array<Galois128, U1>;
    fn next(&mut self) -> Option<Self::Item> {
        let bytes = self.0.recv(TAG_HAT);
        let mut block = [0u8; 16];
        block.copy_from_slice(&bytes);
        Some(Array::from_fn(|_| Galois128(u128::from_le_bytes(block))))
    }
}

fn bit_to_t(b: bool) -> Galois128 {
    Galois128(b as u128)
}

/// XOR then AND then XOR — two committed inputs, one hat.
fn xor_and_program() -> cirrus_recompile_core::Program {
    let mut recorder = Recorder::new();
    let a = recorder.create(false).unwrap();
    let b = recorder.create(false).unwrap();
    let x = recorder.bitxor(a, b).unwrap();
    let y = recorder.bitand(a, b).unwrap();
    let out = recorder.bitxor(x, y).unwrap();
    recorder.finish(vec![a, b], vec![out])
}

fn shares(r0: [u8; 16], z: [u8; 16], bit: bool) -> (Vope<U1, Galois128, U1>, Q<U1, Galois128>) {
    vole_commit_bit_shares(
        Array::<Galois128, U1>::from_fn(|_| Galois128(u128::from_le_bytes(r0))),
        Array::<Galois128, U1>::from_fn(|_| Galois128(u128::from_le_bytes(z))),
        bit_to_t,
        bit,
    )
}

#[test]
fn ferret_stack_tcp_circuit_loop() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let params = FERRET_REG_TOY;
    const ITERS: usize = 32;

    let verifier = thread::spawn(move || {
        let program = xor_and_program();
        let (stream, _) = listener.accept().unwrap();
        stream.set_nodelay(true).unwrap();
        let mut io = TcpIo { stream };
        let mut rng = TestRng(0x1111_2222_3333_4444);
        let mut sender = stack_setup_sender(&mut rng, params, &mut io);
        let delta = Delta {
            delta: Array::from_fn(|_| Galois128(u128::from_le_bytes(sender.seed.delta))),
        };
        for it in 0..ITERS {
            let a = it % 2 == 0;
            let b = it % 3 != 0;
            let r0_a = stack_bea95_sender(&mut rng, &mut sender, &mut io);
            let r0_b = stack_bea95_sender(&mut rng, &mut sender, &mut io);
            let q_a = shares(r0_a, [0u8; 16], a).1;
            let q_b = shares(r0_b, [0u8; 16], b).1;
            let mut verifier = VoleVerifierContext {
                delta: delta.clone(),
                hats: HatIter(&mut io),
                hook: NoopVoleVerifierHook,
                gate_index: 0,
            };
            let outs = cirrus_recompile_rt::execute(&mut verifier, &program, &[q_a, q_b]).unwrap();
            let vope_v = {
                let raw = io.recv(TAG_HAT);
                let mut b = [0u8; 16];
                b.copy_from_slice(&raw);
                b
            };
            let vope: Vope<U1, Galois128, U1> = Vope {
                u: Array::from_fn(|_| Array::from_fn(|_| bit_to_t((a ^ b) ^ (a && b)))),
                v: Array::from_fn(|_| Galois128(u128::from_le_bytes(vope_v))),
            };
            assert!(
                vope.clone() * delta.clone() == outs[0],
                "share mismatch iter {it}"
            );
        }
    });

    let prover = thread::spawn(move || {
        let stream = TcpStream::connect(addr).unwrap();
        stream.set_nodelay(true).unwrap();
        let mut io = TcpIo { stream };
        let mut rng = TestRng(0xAAAA_BBBB_CCCC_DDDD);
        let mut receiver = stack_setup_receiver(&mut rng, params, &mut io);
        let program = xor_and_program();
        for it in 0..ITERS {
            let a = it % 2 == 0;
            let b = it % 3 != 0;
            let z_a = stack_bea95_receiver(&mut rng, &mut receiver, &mut io, a);
            let z_b = stack_bea95_receiver(&mut rng, &mut receiver, &mut io, b);
            let (vope_a, _) = shares([0u8; 16], z_a, a);
            let (vope_b, _) = shares([0u8; 16], z_b, b);
            let mut hats = HatPush(&mut io);
            let mut prover = VoleProverContext {
                hats: &mut hats,
                bit_to_t,
            };
            let outs =
                cirrus_recompile_rt::execute(&mut prover, &program, &[vope_a, vope_b]).unwrap();
            io.send(TAG_HAT, &outs[0].v[0].0.to_le_bytes());
        }
    });

    prover.join().unwrap();
    verifier.join().unwrap();
}
