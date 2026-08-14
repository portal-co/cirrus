extern crate std;

use core::{array, convert::Infallible};

use crate::{ErtError, RawMemory, ert_emit, simple_add};

fn word(value: u32) -> [bool; 32] {
    array::from_fn(|bit| value & (1 << bit) != 0)
}

fn value(word: &[bool; 32]) -> u32 {
    word.iter()
        .enumerate()
        .fold(0, |value, (bit, set)| value | ((*set as u32) << bit))
}

fn image(words: &[u16]) -> std::vec::Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

fn no_hash(_: &[[bool; 32]]) -> Result<[u8; 32], Infallible> {
    Ok([0; 32])
}

fn run(
    code: &[u16],
    regs: &mut [[bool; 32]; 16],
    constants: &mut [Option<u32>; 16],
) -> Result<(), ErtError<Infallible>> {
    let image = image(code);
    let mut context = ();
    let mut hash = no_hash;
    let mut rstack = [0; 16];
    let mut vstack = [false; 128];
    ert_emit(
        &mut context,
        &mut hash,
        RawMemory::from(&image[..]),
        &mut rstack,
        &mut vstack,
        1,
        regs,
        constants,
        false,
        true,
    )
}

fn exit(regs: &mut [[bool; 32]; 16], constants: &mut [Option<u32>; 16]) {
    regs[0] = word(u32::MAX);
    constants[0] = Some(u32::MAX);
}

#[test]
fn simple_add_matches_the_riscv_compatibility_helper() {
    assert_eq!(
        value(&simple_add(&mut (), &word(u32::MAX), &word(2), false, false, true).unwrap()),
        1
    );
}

#[test]
fn thumb16_arithmetic_flags_and_conditional_branch_execute() {
    let mut regs = [[false; 32]; 16];
    let mut constants = [None; 16];
    exit(&mut regs, &mut constants);
    // movs r1, #3; subs r1, #3; bne +2; movs r2, #9; svc #0
    assert!(
        run(
            &[0x2103, 0x3903, 0xd100, 0x2209, 0xdf00],
            &mut regs,
            &mut constants
        )
        .is_ok()
    );
    assert_eq!(constants[1], Some(0));
    assert_eq!(constants[2], Some(9));
}

#[test]
fn runtime_shift_and_multiply_keep_symbolic_metadata() {
    let mut regs = [[false; 32]; 16];
    let mut constants = [None; 16];
    regs[1] = word(3);
    regs[2] = word(5);
    exit(&mut regs, &mut constants);
    // lsls r1, r2; muls r1, r2; svc #0
    assert!(run(&[0x4091, 0x4351, 0xdf00], &mut regs, &mut constants).is_ok());
    assert_eq!(constants[1], None);
    assert_eq!(value(&regs[1]), 480);
}

#[test]
fn symbolic_stack_push_and_pop_preserve_a_word() {
    let mut regs = [[false; 32]; 16];
    let mut constants = [None; 16];
    regs[1] = word(0x1234_5678);
    exit(&mut regs, &mut constants);
    // push {r1}; movs r1,#0; pop {r2}; svc #0
    assert!(run(&[0xb402, 0x2100, 0xbc04, 0xdf00], &mut regs, &mut constants).is_ok());
    assert_eq!(constants[2], None);
    assert_eq!(regs[2], word(0x1234_5678));
}

#[test]
fn thumb2_constants_shifted_logic_and_long_multiply_decode() {
    let mut regs = [[false; 32]; 16];
    let mut constants = [None; 16];
    exit(&mut regs, &mut constants);
    // movw/movt r1,#0x12345678; add.w r2,r1,#4; ror.w r3,r2,#8;
    // eor.w r4,r3,r2,ror #4; umull r5,r6,r1,r2; svc #0.
    assert!(
        run(
            &[
                0xf245, 0x6178, 0xf2c1, 0x2134, 0xf101, 0x0204, 0xea4f, 0x2332, 0xea83, 0x1432,
                0xfba1, 0x5602, 0xdf00,
            ],
            &mut regs,
            &mut constants
        )
        .is_ok()
    );
    assert_eq!(constants[1], Some(0x1234_5678));
    assert_eq!(constants[2], Some(0x1234_567c));
    assert_eq!(constants[3], Some(0x7c12_3456));
    assert_eq!(constants[4], Some(0x7c12_3456 ^ 0xc123_4567));
    let product = 0x1234_5678u64 * 0x1234_567cu64;
    assert_eq!(constants[5], Some(product as u32));
    assert_eq!(constants[6], Some((product >> 32) as u32));
}

#[test]
fn non_thumb_entry_and_invalid_encoding_are_rejected() {
    let mut regs = [[false; 32]; 16];
    let mut constants = [None; 16];
    let bytes = [0u8; 2];
    let mut context = ();
    let mut hash = no_hash;
    let mut rstack = [0; 2];
    let mut vstack = [false; 128];
    assert!(matches!(
        ert_emit(
            &mut context,
            &mut hash,
            RawMemory::from(&bytes[..]),
            &mut rstack,
            &mut vstack,
            0,
            &mut regs,
            &mut constants,
            false,
            true
        ),
        Err(ErtError::Unexpected)
    ));
    exit(&mut regs, &mut constants);
    assert!(matches!(
        run(&[0xbe00], &mut regs, &mut constants),
        Err(ErtError::Decode(_))
    ));
}

#[test]
fn sha_helper_encodings_preserve_their_bitwise_meaning() {
    // The production workload calls these helpers with a BL/POP-PC pair. Keep
    // the exact emitted encodings here so a decoder regression is localized.
    let call = [0xf000, 0xf804, 0x4601, 0x2000, 0x3801, 0xdf00];
    let mut regs = [[false; 32]; 16];
    let mut constants = [None; 16];
    let e = 0x510e_527f;
    regs[0] = word(e);
    exit(&mut regs, &mut constants);
    regs[0] = word(e);
    constants[0] = None;
    let mut code = call.to_vec();
    code.extend([
        0xb580, 0x466f, 0xea4f, 0x11b0, 0xea81, 0x21f0, 0xea81, 0x6070, 0xbd80,
    ]);
    assert!(run(&code, &mut regs, &mut constants).is_ok());
    assert_eq!(
        value(&regs[1]),
        e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25)
    );

    let mut regs = [[false; 32]; 16];
    let mut constants = [None; 16];
    let (x, y, z) = (0x510e_527f, 0x9b05_688c, 0x1f83_d9ab);
    regs[0] = word(x);
    regs[1] = word(y);
    regs[2] = word(z);
    exit(&mut regs, &mut constants);
    regs[0] = word(x);
    regs[1] = word(y);
    regs[2] = word(z);
    constants[..3].fill(None);
    let mut code = call.to_vec();
    code.extend([0xb580, 0x466f, 0x4001, 0xea22, 0x0000, 0x4408, 0xbd80]);
    assert!(run(&code, &mut regs, &mut constants).is_ok());
    assert_eq!(value(&regs[1]), (x & y) ^ (!x & z));

    let mut regs = [[false; 32]; 16];
    let mut constants = [None; 16];
    let a = 0x6a09_e667;
    regs[0] = word(a);
    exit(&mut regs, &mut constants);
    regs[0] = word(a);
    constants[0] = None;
    let mut code = call.to_vec();
    code.extend([
        0xb580, 0x466f, 0xea4f, 0x01b0, 0xea81, 0x3170, 0xea81, 0x50b0, 0xbd80,
    ]);
    assert!(run(&code, &mut regs, &mut constants).is_ok());
    assert_eq!(
        value(&regs[1]),
        a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22)
    );

    let mut regs = [[false; 32]; 16];
    let mut constants = [None; 16];
    let (x, y, z) = (0x6a09_e667, 0xbb67_ae85, 0x3c6e_f372);
    regs[0] = word(x);
    regs[1] = word(y);
    regs[2] = word(z);
    exit(&mut regs, &mut constants);
    regs[0] = word(x);
    regs[1] = word(y);
    regs[2] = word(z);
    constants[..3].fill(None);
    let mut code = call.to_vec();
    code.extend([
        0xb580, 0x466f, 0xea02, 0x0301, 0x4051, 0x4008, 0x4058, 0xbd80,
    ]);
    assert!(run(&code, &mut regs, &mut constants).is_ok());
    assert_eq!(value(&regs[1]), (x & y) ^ (x & z) ^ (y & z));
}

#[test]
fn high_register_moves_keep_the_sha_state_registers_distinct() {
    let mut regs = [[false; 32]; 16];
    let mut constants = [None; 16];
    exit(&mut regs, &mut constants);
    // movw r11,#0xe667; movt r11,#0x6a09; mov r0,r11; mov r1,r0;
    // movs r0,#0; subs r0,#1; svc #0.
    assert!(
        run(
            &[
                0xf24e, 0x6b67, 0xf6c6, 0x2b09, 0x4658, 0x4601, 0x2000, 0x3801, 0xdf00,
            ],
            &mut regs,
            &mut constants
        )
        .is_ok()
    );
    assert_eq!(value(&regs[1]), 0x6a09_e667);
}

#[test]
fn sha_frame_spills_preserve_the_selected_high_registers() {
    let mut regs = [[false; 32]; 16];
    let mut constants = [None; 16];
    let (b, c, d, h) = (0xbb67_ae85, 0x3c6e_f372, 0xa54f_f53a, 0x5be0_cd19);
    regs[0] = word(b);
    regs[3] = word(h);
    regs[9] = word(d);
    regs[10] = word(c);
    // sub sp,#16; strd r3,r9,[sp,#4]; str.w r10,[sp]; str r0,[sp,#12];
    // ldr.w r1,[sp,#12]; ldr.w r2,[sp]; add sp,#16; exit.
    assert!(
        run(
            &[
                0xb084, 0xe9cd, 0x3901, 0xf8cd, 0xa000, 0x9003, 0xf8dd, 0x100c, 0xf8dd, 0x2000,
                0xb004, 0x2000, 0x3801, 0xdf00,
            ],
            &mut regs,
            &mut constants
        )
        .is_ok()
    );
    assert_eq!(value(&regs[1]), b);
    assert_eq!(value(&regs[2]), c);
}

#[test]
fn register_indexed_stack_load_store_keeps_a_concrete_index() {
    let mut regs = [[false; 32]; 16];
    let mut constants = [None; 16];
    regs[2] = word(0x1234_5678);
    // sub sp,#16; mov r3,sp; movs r1,#4; str r2,[r3,r1];
    // ldr r4,[r3,r1]; add sp,#16; exit.
    assert!(
        run(
            &[
                0xb084, 0x466b, 0x2104, 0x505a, 0x585c, 0xb004, 0x2000, 0x3801, 0xdf00,
            ],
            &mut regs,
            &mut constants,
        )
        .is_ok()
    );
    assert_eq!(value(&regs[4]), 0x1234_5678);
}

#[test]
fn thumb2_mla_mls_and_long_products_decode() {
    let mut regs = [[false; 32]; 16];
    let mut constants = [None; 16];
    regs[1] = word(7);
    regs[2] = word(9);
    regs[4] = word(11);
    regs[6] = word(100);
    exit(&mut regs, &mut constants);
    constants[1] = None;
    constants[2] = None;
    constants[4] = None;
    constants[6] = None;
    // mla r3,r1,r2,r4; mls r5,r1,r2,r6; umull r7,r8,r1,r2;
    // smull r9,r10,r1,r2; svc #0.
    assert!(
        run(
            &[
                0xfb01, 0x4302, 0xfb01, 0x6512, 0xfba1, 0x7802, 0xfb81, 0x9a02, 0xdf00,
            ],
            &mut regs,
            &mut constants,
        )
        .is_ok()
    );
    assert_eq!(value(&regs[3]), 74);
    assert_eq!(value(&regs[5]), 37);
    assert_eq!(value(&regs[7]), 63);
    assert_eq!(value(&regs[8]), 0);
    assert_eq!(value(&regs[9]), 63);
    assert_eq!(value(&regs[10]), 0);
}
