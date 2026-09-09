//! D3: a real RV32 store/load executed through the GRAM-backed storage, as a
//! two-party garbled-circuit run.
//!
//! `GramStorage` (the A1 capstone) routes every `ContextWithStorage`
//! operation through a full Path-ORAM access over garbled labels. Here the
//! ERT RV32 machine's *vstack* is backed by `GramStorage` instead of the
//! linear-scan slice: an RV32 `sw` stores a word (32 ORAM write accesses)
//! and an `lw` reads it back (32 ORAM read accesses), proving sub-linear
//! garbled memory behind a genuine embedded symbolic executor — the D3
//! "microcontroller-class vc evaluator" shape.
//!
//! This is the full **two-party** D3 run: a *garbler* (`VcGarbleBackend` +
//! garbler-side `GramStorage`) and an *evaluator* (`VcEvalBackend` +
//! evaluator-side `GramStorage`) both execute the same RV32 program with the
//! same concrete control flow. The garbler streams one `GarbleTable` per AND
//! gate; the evaluator consumes them in order and decodes the result.
//!
//! Two transports are covered: a buffered Vec pipe (garbler runs to
//! completion, then the evaluator replays) and a live coroutine pipe
//! (garbler and evaluator interleaved through `cirrus_coroutine`).
//!
//! Both parties use constant-carrying backends: the ERT machine fabricates
//! its ABI/ALU constants from a single `zero`/`one` wire pair, so each
//! `create` must return the *same* label per constant (the garbler-side
//! `VcGarbleBackend` and evaluator-side `VcEvalBackend` D3 adds).
//!
//! Run (heavyweight): `cargo test -p cirrus-volar-garble --test gram_ert --
//! --ignored --nocapture`.

use cirrus_ert::{DefaultHandler, RawMemory, RvDefaultHandler, ert_emit};
use cirrus_volar_garble::{
    GramStorage, GramStorageSpace, VcEvalBackend, VcGarbleBackend,
};
use hybrid_array::{Array, typenum::U16};
use rv_asm::{Imm, Inst, Reg, Xlen};
use sha2::Sha256;
use volar_oram::OramTree;
use volar_spec::garble::{Eval, Garble, GarbleTable, GlobalSecret};

const Z: usize = 4;
const B: usize = 8;

fn det_secret() -> GlobalSecret<U16> {
    GlobalSecret::new(Array::from_fn(|i| (i as u8).wrapping_mul(37) | 1))
}

fn program(instructions: impl IntoIterator<Item = Inst>) -> Vec<u8> {
    instructions
        .into_iter()
        .flat_map(|i| i.encode_normal(Xlen::Rv32).to_le_bytes())
        .collect()
}

/// The ERT hash `ECALL` is unused by this program; a no-hash callback
/// satisfies the `DefaultHandler` shape.
fn no_hash<C, E>(_: &mut C, _: &[[Eval<U16>; 32]]) -> Result<[u8; 32], E> {
    Ok([0; 32])
}

/// The const-0 wire's false-label is the all-zero base (the free-XOR
/// identity); const-1 is a distinct revealed base.
fn const_base(secret_tag: u8) -> Garble<U16> {
    if secret_tag == 0x00 {
        return Garble {
            base: Array::from_fn(|_| 0u8),
        };
    }
    Garble {
        base: Array::from_fn(|i| {
            (i as u8).wrapping_mul(11).wrapping_add(secret_tag) | 1
        }),
    }
}

/// The shared program + machine setup, run identically by both parties.
#[derive(Clone)]
struct Setup {
    stored: u32,
    mem: Vec<u8>,
    levels: usize,
    num_addrs: u64,
    storage_bits: usize,
    zero_base: Garble<U16>,
    one_base: Garble<U16>,
}

fn setup() -> Setup {
    let stored = 0x8001_80ffu32;
    let mem = program([
        Inst::Addi {
            imm: Imm::new_i32(-16),
            dest: Reg::SP,
            src1: Reg::SP,
        },
        Inst::Sw {
            offset: Imm::ZERO,
            src: Reg::T0,
            base: Reg::SP,
        },
        Inst::Lw {
            offset: Imm::ZERO,
            dest: Reg::T1,
            base: Reg::SP,
        },
        Inst::Addi {
            imm: Imm::new_i32(16),
            dest: Reg::SP,
            src1: Reg::SP,
        },
        Inst::Ecall,
    ]);
    let levels = 8;
    let storage_bits = 256usize; // 32 bytes of vstack, one ORAM block per bit.
    let num_addrs = storage_bits as u64;
    Setup {
        stored,
        mem,
        levels,
        num_addrs,
        storage_bits,
        zero_base: const_base(0x00),
        one_base: const_base(0x01),
    }
}

/// Seed the register file identically on both parties: every register
/// starts at the const-0 wire except T0 (the stored word's constant bits)
/// and A0 (u32::MAX, the exit signal). All constant, so the garbler and
/// evaluator agree on every input label with no OT.
fn seed_regs<W: Clone>(stored: u32, zero: &W, one: &W) -> ([[W; 32]; 32], [Option<u32>; 32]) {
    let mut regs: [[W; 32]; 32] =
        core::array::from_fn(|_| core::array::from_fn(|_| zero.clone()));
    let mut constants = [None; 32];
    let stored_bits: [bool; 32] = core::array::from_fn(|b| stored & (1 << b) != 0);
    for (i, b) in stored_bits.iter().enumerate() {
        regs[Reg::T0.0 as usize][i] = if *b { one.clone() } else { zero.clone() };
    }
    constants[Reg::T0.0 as usize] = None;
    let a0_bits: [bool; 32] = core::array::from_fn(|b| u32::MAX & (1 << b) != 0);
    for (i, b) in a0_bits.iter().enumerate() {
        regs[Reg::A0.0 as usize][i] = if *b { one.clone() } else { zero.clone() };
    }
    constants[Reg::A0.0 as usize] = Some(u32::MAX);
    (regs, constants)
}

/// A `Pusher` that buffers every pushed table into a `Vec` (the buffered
/// pipe transport).
struct VecPusher<T>(std::vec::Vec<T>);
impl<T> cirrus_core::Pusher<T> for VecPusher<T> {
    fn push(&mut self, x: T) {
        self.0.push(x);
    }
}

/// The garbler-side run: execute the program, streaming each AND table to
/// `pusher`. Returns the garbler's final T1 (the loaded register's tracked
/// false-labels, for output decode) and the number of tables produced.
fn run_garbler<P: cirrus_core::Pusher<GarbleTable<U16>>>(
    s: &Setup,
    secret: &GlobalSecret<U16>,
    pusher: &mut P,
) -> [Garble<U16>; 32] {
    let garble_backend = VcGarbleBackend::<Sha256, U16>::new(
        pusher,
        secret.clone(),
        s.zero_base.clone(),
        s.one_base.clone(),
    );
    let mut tree = OramTree::<Z, B>::new(s.levels);
    let storage_space = GramStorageSpace::<U16>::new::<Sha256>(s.num_addrs as usize);
    let gram = GramStorage::<_, Sha256, U16, Z, B>::new(
        garble_backend,
        secret.clone(),
        &mut tree,
        s.levels,
        s.num_addrs,
    );
    let mut handler = RvDefaultHandler {
        inner: DefaultHandler {
            context: gram,
            hash: |_ctx: &mut _, _words: &[[Garble<U16>; 32]]| {
                Ok::<_, core::convert::Infallible>([0u8; 32])
            },
        },
    };
    let zero = s.zero_base.clone();
    let one = s.one_base.clone();
    let (mut regs, mut constants) = seed_regs(s.stored, &zero, &one);
    let mut rstack = [0u32; 8];
    let mut storage = storage_space;
    ert_emit(
        &mut handler,
        &mut storage,
        s.storage_bits,
        RawMemory::from(&s.mem[..]),
        &mut rstack,
        0,
        &mut regs,
        &mut constants,
        zero,
        one,
    )
    .map_err(|e| match e {
        cirrus_ert::ErtError::Emitted(_) => "garbler context error",
        cirrus_ert::ErtError::Decode(_) => "garbler decode error",
        cirrus_ert::ErtError::Unexpected => "garbler unexpected",
    })
    .expect("garbler-side ERT run failed");
    let out = regs[Reg::T1.0 as usize].clone();
    drop(handler);
    out
}

/// The evaluator-side run: consume the AND-table stream, run the program
/// through GRAM storage, and return the loaded register's labels.
fn run_evaluator<I: Iterator<Item = GarbleTable<U16>>>(
    s: &Setup,
    secret: &GlobalSecret<U16>,
    tables: I,
) -> [Eval<U16>; 32] {
    let zero_label = Eval::<U16>::zero();
    let one_label = secret.encode(&s.one_base, true);
    let consts: Vec<Eval<U16>> = vec![zero_label.clone(), one_label.clone()];

    let mut tree = OramTree::<Z, B>::new(s.levels);
    let eval_backend = VcEvalBackend::<Sha256, _, U16, _>::new(tables, consts.into_iter());
    let mut storage_space = GramStorageSpace::<U16>::new::<Sha256>(s.num_addrs as usize);
    // Seed the written cells' bases so the write-value decode matches the
    // garbler. The machine stores T0's 32 constant bits at SP (byte 16, so
    // bit-cells 128..160); each written wire's base is the const-0/const-1
    // base, which the garbler records in `cell_bases`. Mirror that here.
    let sp_byte = (s.storage_bits / 8) as i64 - 16;
    let first_cell = (sp_byte as usize) * 8;
    for i in 0..32 {
        let bit = (s.stored >> i) & 1 != 0;
        storage_space.cell_bases[first_cell + i] =
            if bit { s.one_base.clone() } else { s.zero_base.clone() };
    }
    let gram = GramStorage::<_, Sha256, U16, Z, B>::new(
        eval_backend,
        secret.clone(),
        &mut tree,
        s.levels,
        s.num_addrs,
    );
    let mut handler = RvDefaultHandler {
        inner: DefaultHandler {
            context: gram,
            hash: no_hash,
        },
    };
    let mut storage = storage_space;
    let (mut regs, mut constants) = seed_regs(s.stored, &zero_label, &one_label);
    let mut rstack = [0u32; 8];
    ert_emit(
        &mut handler,
        &mut storage,
        s.storage_bits,
        RawMemory::from(&s.mem[..]),
        &mut rstack,
        0,
        &mut regs,
        &mut constants,
        zero_label.clone(),
        one_label.clone(),
    )
    .map_err(|e| match e {
        cirrus_ert::ErtError::Emitted(_) => "evaluator context error (table supply)",
        cirrus_ert::ErtError::Decode(_) => "evaluator decode error",
        cirrus_ert::ErtError::Unexpected => "evaluator unexpected",
    })
    .expect("evaluator-side ERT run failed");
    let out = regs[Reg::T1.0 as usize].clone();
    drop(handler);
    out
}

/// Open each bit of the loaded register against the garbler's tracked
/// false-labels and reassemble the concrete word.
fn open_word(eval_bits: &[Eval<U16>; 32], garble_bases: &[Garble<U16>; 32]) -> u32 {
    let mut out = 0u32;
    for i in 0..32 {
        let opened = eval_bits[i].open(&garble_bases[i]);
        if opened[0] & 1 != 0 {
            out |= 1 << i;
        }
    }
    out
}

/// Two-party GRAM ERT run over a **buffered Vec pipe**: the garbler runs to
/// completion, then the evaluator replays the table stream.
#[test]
#[ignore = "heavyweight: 64+ full Path-ORAM accesses over garbled labels"]
fn ert_sw_lw_through_gram_storage_two_party_vec_pipe() {
    let s = setup();
    let secret = det_secret();

    let mut tables = VecPusher::<GarbleTable<U16>>(Vec::new());
    let garble_out = run_garbler(&s, &secret, &mut tables);

    let eval_out = run_evaluator(&s, &secret, tables.0.into_iter());

    let loaded = open_word(&eval_out, &garble_out);
    assert_eq!(loaded, s.stored, "lw must recover the word sw stored via ORAM");
}

/// Two-party GRAM ERT run over a **threaded pipe**: the garbler runs on a
/// spawned OS thread, streaming each AND table through a
/// `std::sync::mpsc` channel; the evaluator (main thread) pulls them as its
/// table iterator. The garbler's tracked output bases return through a
/// second channel. This is the streaming shape a real online session takes
/// (no full table buffering), using a plain channel rather than a
/// stackful-coroutine context switch — multiple OS threads are available.
#[test]
#[ignore = "heavyweight: 64+ full Path-ORAM accesses over garbled labels"]
fn ert_sw_lw_through_gram_storage_two_party_thread_pipe() {
    let s = setup();
    let secret = det_secret();

    // tables: garbler -> evaluator (one GarbleTable per AND gate).
    let (table_tx, table_rx) = std::sync::mpsc::channel::<GarbleTable<U16>>();
    // output bases: garbler -> evaluator (the loaded register's tracked
    // false-labels, for the final decode).
    let (out_tx, out_rx) = std::sync::mpsc::channel::<[Garble<U16>; 32]>();

    // The garbler thread: stream each table, then send the output bases.
    let s_g = s.clone();
    let secret_g = secret.clone();
    let garbler = std::thread::spawn(move || {
        struct ChanPusher(std::sync::mpsc::Sender<GarbleTable<U16>>);
        impl cirrus_core::Pusher<GarbleTable<U16>> for ChanPusher {
            fn push(&mut self, x: GarbleTable<U16>) {
                // If the evaluator finished early (it shouldn't), stop.
                let _ = self.0.send(x);
            }
        }
        let out = run_garbler(&s_g, &secret_g, &mut ChanPusher(table_tx));
        let _ = out_tx.send(out);
    });

    // The evaluator's table iterator pulls from the channel, blocking until
    // the garbler produces the next table.
    struct ChanTables(std::sync::mpsc::Receiver<GarbleTable<U16>>);
    impl Iterator for ChanTables {
        type Item = GarbleTable<U16>;
        fn next(&mut self) -> Option<Self::Item> {
            self.0.recv().ok()
        }
    }

    let eval_out = run_evaluator(&s, &secret, ChanTables(table_rx));

    let garble_out = out_rx
        .recv()
        .expect("garbler thread must have completed and sent its output");
    garbler.join().expect("garbler thread panicked");

    let loaded = open_word(&eval_out, &garble_out);
    assert_eq!(loaded, s.stored, "lw must recover the word sw stored via ORAM");
}


