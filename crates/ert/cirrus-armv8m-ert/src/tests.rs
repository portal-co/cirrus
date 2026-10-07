extern crate std;

use core::{array, convert::Infallible};

use cirrus_core::{
    ContextWithBitAnd, ContextWithBitOr, ContextWithBitXor, ContextWithCreate, ContextWithStorage,
    ContextWithValue, HasError, StorageAddressBit,
};

use crate::{
    ArmDefaultHandler, DefaultHandler, ErtError, RawMemory, SecurityAttribute, SecurityState,
    ert_emit, simple_add,
};

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

fn no_hash<C>(_: &mut C, _: &[[bool; 32]]) -> Result<[u8; 32], Infallible> {
    Ok([0; 32])
}

fn permit_all<H>(_: &mut H, _: SecurityState) -> bool {
    true
}

fn always_secure<H>(_: &mut H, _: u32) -> SecurityAttribute {
    SecurityAttribute::Secure
}

fn all_non_secure<H>(_: &mut H, _: u32) -> SecurityAttribute {
    SecurityAttribute::NonSecure
}

fn secure_only<H>(_: &mut H, state: SecurityState) -> bool {
    state == SecurityState::Secure
}

#[allow(clippy::type_complexity)]
fn run_with<G, A>(
    code: &[u16],
    regs: &mut [[bool; 32]; 16],
    constants: &mut [Option<u32>; 16],
    svc_permitted: G,
    security_attribute: A,
) -> Result<(), ErtError<Infallible>>
where
    G: FnMut(
        &mut DefaultHandler<(), fn(&mut (), &[[bool; 32]]) -> Result<[u8; 32], Infallible>>,
        SecurityState,
    ) -> bool,
    A: FnMut(
        &mut DefaultHandler<(), fn(&mut (), &[[bool; 32]]) -> Result<[u8; 32], Infallible>>,
        u32,
    ) -> SecurityAttribute,
{
    let image = image(code);
    let mut handler = ArmDefaultHandler {
        inner: DefaultHandler {
            // Run instruction semantics against the identity Boolean backend;
            // Volar and garbled backends are covered in their own suites.
            context: (),
            hash: no_hash as fn(&mut (), &[[bool; 32]]) -> Result<[u8; 32], Infallible>,
        },
        svc_permitted,
        security_attribute,
    };
    let mut rstack = [0; 16];
    let mut vstack = [false; 128];
    let storage_bits = vstack.len();
    ert_emit(
        &mut handler,
        &mut vstack,
        storage_bits,
        RawMemory::from(&image[..]),
        &mut rstack,
        1,
        regs,
        constants,
        false,
        true,
    )
}

fn run(
    code: &[u16],
    regs: &mut [[bool; 32]; 16],
    constants: &mut [Option<u32>; 16],
) -> Result<(), ErtError<Infallible>> {
    let image = image(code);
    let mut handler = ArmDefaultHandler {
        inner: DefaultHandler {
            context: (),
            hash: no_hash,
        },
        svc_permitted: permit_all,
        security_attribute: always_secure,
    };
    let mut rstack = [0; 16];
    let mut vstack = [false; 128];
    let storage_bits = vstack.len();
    ert_emit(
        &mut handler,
        &mut vstack,
        storage_bits,
        RawMemory::from(&image[..]),
        &mut rstack,
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

#[derive(Default)]
struct CountingContext {
    bitand: usize,
    bitor: usize,
    bitxor: usize,
}

impl HasError for CountingContext {
    type Error = Infallible;
}

impl ContextWithValue<bool> for CountingContext {
    type Wrapped = bool;
}

impl ContextWithCreate<bool> for CountingContext {
    fn create(&mut self, value: bool) -> Result<bool, Self::Error> {
        Ok(value)
    }
}

impl ContextWithBitAnd<bool> for CountingContext {
    fn bitand(&mut self, left: bool, right: bool) -> Result<bool, Self::Error> {
        self.bitand += 1;
        Ok(left & right)
    }

    fn bitand_assign(&mut self, left: &mut bool, right: bool) -> Result<(), Self::Error> {
        self.bitand += 1;
        *left &= right;
        Ok(())
    }
}

impl ContextWithBitOr<bool> for CountingContext {
    fn bitor(&mut self, left: bool, right: bool) -> Result<bool, Self::Error> {
        self.bitor += 1;
        Ok(left | right)
    }

    fn bitor_assign(&mut self, left: &mut bool, right: bool) -> Result<(), Self::Error> {
        self.bitor += 1;
        *left |= right;
        Ok(())
    }
}

impl ContextWithBitXor<bool> for CountingContext {
    fn bitxor(&mut self, left: bool, right: bool) -> Result<bool, Self::Error> {
        self.bitxor += 1;
        Ok(left ^ right)
    }

    fn bitxor_assign(&mut self, left: &mut bool, right: bool) -> Result<(), Self::Error> {
        self.bitxor += 1;
        *left ^= right;
        Ok(())
    }
}

impl ContextWithStorage<bool> for CountingContext {
    type Storage = [bool];

    fn storage_read(
        &mut self,
        storage: &mut Self::Storage,
        address: &[StorageAddressBit<bool>],
    ) -> Result<bool, Self::Error> {
        Ok(storage[storage_index(address)])
    }

    fn storage_write(
        &mut self,
        storage: &mut Self::Storage,
        address: &[StorageAddressBit<bool>],
        value: bool,
    ) -> Result<(), Self::Error> {
        storage[storage_index(address)] = value;
        Ok(())
    }
}

fn storage_index(address: &[StorageAddressBit<bool>]) -> usize {
    address
        .iter()
        .enumerate()
        .fold(0usize, |index, (bit, address)| {
            index | ((address.wire as usize) << bit)
        })
}

fn run_counting(
    code: &[u16],
    regs: &mut [[bool; 32]; 16],
    constants: &mut [Option<u32>; 16],
) -> CountingContext {
    let image = image(code);
    let mut handler = ArmDefaultHandler {
        inner: DefaultHandler {
            context: CountingContext::default(),
            hash: no_hash,
        },
        svc_permitted: permit_all,
        security_attribute: always_secure,
    };
    let mut rstack = [0; 16];
    let mut vstack = [false; 128];
    let storage_bits = vstack.len();
    assert!(
        ert_emit(
            &mut handler,
            &mut vstack,
            storage_bits,
            RawMemory::from(&image[..]),
            &mut rstack,
            1,
            regs,
            constants,
            false,
            true,
        )
        .is_ok()
    );
    handler.inner.context
}

#[test]
fn a_constant_and_or_operand_avoids_bitwise_gates() {
    let mut regs = [[false; 32]; 16];
    let mut constants = [None; 16];
    let symbolic = 0b1010_1100_1111_0000_0000_1111_0011_0101u32;
    regs[1] = word(symbolic);
    regs[2] = word(0x0f0f_0f0f);
    constants[2] = Some(0x0f0f_0f0f);
    regs[3] = word(symbolic);
    regs[4] = word(0x0f0f_0f0f);
    constants[4] = Some(0x0f0f_0f0f);
    exit(&mut regs, &mut constants);

    // ands r1, r2; orrs r3, r4; svc 0
    let counts = run_counting(&[0x4011, 0x4323, 0xdf00], &mut regs, &mut constants);

    assert_eq!(counts.bitand, 0);
    assert_eq!(counts.bitor, 0);
    assert_eq!(value(&regs[1]), symbolic & 0x0f0f_0f0f);
    assert_eq!(constants[1], None);
    assert_eq!(value(&regs[3]), symbolic | 0x0f0f_0f0f);
    assert_eq!(constants[3], None);
}

#[test]
fn a_degenerate_and_or_mask_folds_to_a_full_constant() {
    let mut regs = [[false; 32]; 16];
    let mut constants = [None; 16];
    regs[1] = word(0x1234_5678);
    regs[2] = word(0);
    constants[2] = Some(0);
    regs[3] = word(0x1234_5678);
    regs[4] = word(u32::MAX);
    constants[4] = Some(u32::MAX);
    exit(&mut regs, &mut constants);

    // ands r1, r2; orrs r3, r4; svc 0
    let counts = run_counting(&[0x4011, 0x4323, 0xdf00], &mut regs, &mut constants);

    assert_eq!(counts.bitand, 0);
    assert_eq!(counts.bitor, 0);
    assert_eq!(value(&regs[1]), 0);
    assert_eq!(constants[1], Some(0));
    assert_eq!(value(&regs[3]), u32::MAX);
    assert_eq!(constants[3], Some(u32::MAX));
}

#[test]
fn bic_with_a_concrete_left_operand_avoids_bitand_gates() {
    let mut regs = [[false; 32]; 16];
    let mut constants = [None; 16];
    let symbolic = 0b1010_1100_1111_0000_0000_1111_0011_0101u32;
    regs[1] = word(0x0f0f_0f0f);
    constants[1] = Some(0x0f0f_0f0f);
    regs[2] = word(symbolic);
    exit(&mut regs, &mut constants);

    // bics r1, r2; svc 0
    let counts = run_counting(&[0x4391, 0xdf00], &mut regs, &mut constants);

    assert_eq!(counts.bitand, 0);
    assert_eq!(counts.bitxor, 0x0f0f_0f0fu32.count_ones() as usize);
    assert_eq!(value(&regs[1]), 0x0f0f_0f0f & !symbolic);
    assert_eq!(constants[1], None);
}

#[test]
fn bxns_transitions_to_non_secure_and_execution_continues() {
    let mut regs = [[false; 32]; 16];
    let mut constants = [None; 16];
    // movs r0, #4; bxns r0; movs r1, #9; movs r0, #0; subs r0, #1; svc 0.
    // Every address is treated as Non-secure, so the fetch gate never fires
    // after the Secure -> Non-secure transition.
    let code = [0x2004, 0x4704, 0x2109, 0x2000, 0x3801, 0xdf00];

    assert!(run_with(&code, &mut regs, &mut constants, permit_all, all_non_secure).is_ok());

    assert_eq!(constants[1], Some(9));
}

#[test]
fn fetching_secure_attributed_memory_while_non_secure_faults() {
    let mut regs = [[false; 32]; 16];
    let mut constants = [None; 16];
    // movs r0, #4; bxns r0; movs r1, #9; movs r0, #0; subs r0, #1; svc 0.
    // Every address is treated as Secure, so the instruction right after the
    // Secure -> Non-secure transition (address 4, not `SG`) is rejected.
    let code = [0x2004, 0x4704, 0x2109, 0x2000, 0x3801, 0xdf00];

    assert!(matches!(
        run_with(&code, &mut regs, &mut constants, permit_all, always_secure),
        Err(ErtError::Unexpected)
    ));
}

#[test]
fn secure_gateway_re_enters_secure_state_and_permits_a_gated_svc() {
    fn non_secure_callable_at_four<H>(_: &mut H, address: u32) -> SecurityAttribute {
        if address == 4 {
            SecurityAttribute::NonSecureCallable
        } else {
            SecurityAttribute::Secure
        }
    }

    // Without a gateway: movs r0, #4; bxns r0; movs r0, #0; subs r0, #1;
    // svc 0. `svc_permitted` only allows the call while Secure, and no `SG`
    // is executed, so the CPU is still Non-secure when it is reached.
    let mut regs = [[false; 32]; 16];
    let mut constants = [None; 16];
    let no_gateway = [0x2004, 0x4704, 0x2000, 0x3801, 0xdf00];
    assert!(matches!(
        run_with(
            &no_gateway,
            &mut regs,
            &mut constants,
            secure_only,
            all_non_secure
        ),
        Err(ErtError::Unexpected)
    ));

    // With a gateway: movs r0, #4; bxns r0; sg; movs r0, #0; subs r0, #1;
    // svc 0. `SG` sits at address 4, attributed Non-secure-callable, so it
    // re-enters Secure state before the gated `svc` is reached.
    let mut regs = [[false; 32]; 16];
    let mut constants = [None; 16];
    let with_gateway = [0x2004, 0x4704, 0xe97f, 0xe97f, 0x2000, 0x3801, 0xdf00];
    assert!(
        run_with(
            &with_gateway,
            &mut regs,
            &mut constants,
            secure_only,
            non_secure_callable_at_four,
        )
        .is_ok()
    );
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
fn apsr_nzcvq_round_trips_only_its_architectural_bits() {
    let mut regs = [[false; 32]; 16];
    let mut constants = [None; 16];
    // msr APSR_nzcvq, r1; mrs r2, APSR; svc #0
    regs[1] = word(0xabff_ffff);
    exit(&mut regs, &mut constants);
    assert!(
        run(
            &[0xf381, 0x8800, 0xf3ef, 0x8200, 0xdf00],
            &mut regs,
            &mut constants,
        )
        .is_ok()
    );
    assert_eq!(value(&regs[2]), 0xa800_0000);
    assert_eq!(constants[2], None);

    let mut concrete_regs = [[false; 32]; 16];
    let mut concrete_constants = [None; 16];
    concrete_regs[1] = word(0xabff_ffff);
    concrete_constants[1] = Some(0xabff_ffff);
    exit(&mut concrete_regs, &mut concrete_constants);
    assert!(
        run(
            &[0xf381, 0x8800, 0xf3ef, 0x8200, 0xdf00],
            &mut concrete_regs,
            &mut concrete_constants,
        )
        .is_ok()
    );
    assert_eq!(concrete_constants[2], Some(0xa800_0000));
}

#[test]
fn apsr_rejects_other_special_register_views_and_invalid_registers() {
    let mut regs = [[false; 32]; 16];
    let mut constants = [None; 16];
    exit(&mut regs, &mut constants);
    // MRS r2, PRIMASK; not the APSR_nzcvq view implemented by this facade.
    assert!(matches!(
        run(&[0xf3ef, 0x8210], &mut regs, &mut constants),
        Err(ErtError::Decode(_))
    ));

    let mut regs = [[false; 32]; 16];
    let mut constants = [None; 16];
    exit(&mut regs, &mut constants);
    // MRS sp, APSR is architecturally invalid for this register-transfer form.
    assert!(matches!(
        run(&[0xf3ef, 0x8d00], &mut regs, &mut constants),
        Err(ErtError::Decode(_))
    ));

    let mut regs = [[false; 32]; 16];
    let mut constants = [None; 16];
    exit(&mut regs, &mut constants);
    // MSR APSR_nzcvq, sp is rejected too.
    assert!(matches!(
        run(&[0xf38d, 0x8800], &mut regs, &mut constants),
        Err(ErtError::Decode(_))
    ));
}

#[test]
fn symbolic_carry_flows_through_adc_and_sbc() {
    let mut regs = [[false; 32]; 16];
    let mut constants = [None; 16];
    // Seed C from an unknown APSR word, then adcs r1, r2; sbcs r1, r2;
    // mrs r4, APSR; svc #0. The actual witness has C=1, while all metadata
    // remains symbolic through both arithmetic operations.
    regs[1] = word(u32::MAX);
    regs[2] = word(0);
    regs[3] = word(0x2000_0000);
    exit(&mut regs, &mut constants);
    assert!(
        run(
            &[0xf383, 0x8800, 0x4151, 0x4191, 0xf3ef, 0x8400, 0xdf00,],
            &mut regs,
            &mut constants,
        )
        .is_ok()
    );
    assert_eq!(value(&regs[1]), 0);
    assert_eq!(constants[1], None);
    assert_eq!(value(&regs[4]), 0x6000_0000);
    assert_eq!(constants[4], None);
}

#[test]
fn logical_move_and_shift_flag_writers_preserve_or_update_nzcvq() {
    // Seed C, V, and Q. MOVS and ANDS update only N/Z; the immediate LSL
    // below derives C from the shifted-out source bit while retaining V/Q.
    let mut regs = [[false; 32]; 16];
    let mut constants = [None; 16];
    regs[1] = word(0x8000_0000);
    regs[3] = word(0x3800_0000);
    regs[4] = word(0);
    constants[1] = Some(0x8000_0000);
    constants[3] = Some(0x3800_0000);
    constants[4] = Some(0);
    exit(&mut regs, &mut constants);
    // msr APSR_nzcvq,r3; lsls r1,r1,#1; movs r4,#0; ands r1,r4;
    // mrs r2,APSR; svc #0.
    assert!(
        run(
            &[
                0xf383, 0x8800, 0x0049, 0x2400, 0x4021, 0xf3ef, 0x8200, 0xdf00,
            ],
            &mut regs,
            &mut constants,
        )
        .is_ok()
    );
    // The shift produces zero and C=1; ANDS keeps C/V/Q while retaining the
    // zero result, so NZCVQ is 0b01111.
    assert_eq!(constants[2], Some(0x7800_0000));
}

#[test]
fn symbolic_it_materializes_all_predication_conditions_without_branching() {
    // Seed N=1, Z=0, C=1, V=0 as symbolic APSR bits. Every valid IT
    // condition then selects either the #1 candidate or the preexisting #0
    // in r2, without using symbolic control flow.
    let expected = [
        false, true, true, false, true, false, false, true, true, false, false, true, false, true,
    ];
    for (condition, expected) in expected.into_iter().enumerate() {
        let mut regs = [[false; 32]; 16];
        let mut constants = [None; 16];
        regs[3] = word(0xa000_0000);
        exit(&mut regs, &mut constants);
        assert!(
            run(
                &[
                    0xf383,
                    0x8800,
                    0xbf08 | ((condition as u16) << 4),
                    0x2201,
                    0xdf00,
                ],
                &mut regs,
                &mut constants,
            )
            .is_ok(),
            "condition {condition}"
        );
        assert_eq!(value(&regs[2]), expected as u32, "condition {condition}");
        assert_eq!(constants[2], None, "condition {condition}");
    }
}

#[test]
fn symbolic_cmp_and_single_instruction_it_materialize_compiler_booleans() {
    let cases = [
        // Equality.
        (7, 7, 0, true),
        // Unsigned higher-or-same.
        (0x8000_0000, 0, 2, true),
        // Signed less-than across the overflow boundary: MIN - 1 has N=0,V=1.
        (0x8000_0000, 1, 11, true),
    ];
    for (left, right, condition, expected) in cases {
        let mut regs = [[false; 32]; 16];
        let mut constants = [None; 16];
        regs[1] = word(left);
        regs[2] = word(right);
        exit(&mut regs, &mut constants);
        // cmp r1,r2; mov.w r3,#0; it <condition>; movs r3,#1; svc #0.
        // The non-flag-setting wide move preserves CMP's flags while the
        // predicated MOVS materializes the 0/1 value and its selected NZ
        // side effects.
        assert!(
            run(
                &[
                    0x4291,
                    0xf04f,
                    0x0300,
                    0xbf08 | ((condition as u16) << 4),
                    0x2301,
                    0xdf00,
                ],
                &mut regs,
                &mut constants,
            )
            .is_ok()
        );
        assert_eq!(value(&regs[3]), expected as u32);
        assert_eq!(constants[3], None);
    }
}

#[test]
fn symbolic_flags_remain_invalid_for_branches_and_general_it_blocks() {
    let mut regs = [[false; 32]; 16];
    let mut constants = [None; 16];
    regs[1] = word(1);
    regs[2] = word(1);
    exit(&mut regs, &mut constants);
    // cmp r1,r2; beq +0. The comparison flags are symbolic even though this
    // identity witness happens to make the branch true.
    assert!(matches!(
        run(&[0x4291, 0xd000], &mut regs, &mut constants),
        Err(ErtError::Unexpected)
    ));

    let mut regs = [[false; 32]; 16];
    let mut constants = [None; 16];
    regs[3] = word(0xa000_0000);
    exit(&mut regs, &mut constants);
    // msr APSR_nzcvq,r3; itt eq; moveq r2,#1; moveq r2,#2; svc #0.
    assert!(matches!(
        run(
            &[0xf383, 0x8800, 0xbf04, 0x2201, 0x2202, 0xdf00],
            &mut regs,
            &mut constants,
        ),
        Err(ErtError::Unexpected)
    ));
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
    let mut handler = ArmDefaultHandler {
        inner: DefaultHandler {
            context: (),
            hash: no_hash,
        },
        svc_permitted: permit_all,
        security_attribute: always_secure,
    };
    let mut rstack = [0; 2];
    let mut vstack = [false; 128];
    let storage_bits = vstack.len();
    assert!(matches!(
        ert_emit(
            &mut handler,
            &mut vstack,
            storage_bits,
            RawMemory::from(&bytes[..]),
            &mut rstack,
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
fn a_concrete_load_at_the_detect_address_returns_the_overridden_word() {
    // movs r0, #0x40; ldr r1, [r0]; movs r0, #0; subs r0, #1; svc 0.
    let code = image(&[0x2040, 0x6801, 0x2000, 0x3801, 0xdf00]);
    let memory = RawMemory::from(&code[..]).with_ert_detect(0x40, 0xdead_beef);
    let mut regs = [[false; 32]; 16];
    let mut constants = [None; 16];
    let mut handler = ArmDefaultHandler {
        inner: DefaultHandler {
            context: (),
            hash: no_hash,
        },
        svc_permitted: permit_all,
        security_attribute: always_secure,
    };
    let mut rstack = [0; 16];
    let mut vstack = [false; 128];
    let storage_bits = vstack.len();

    assert!(
        ert_emit(
            &mut handler,
            &mut vstack,
            storage_bits,
            memory,
            &mut rstack,
            1,
            &mut regs,
            &mut constants,
            false,
            true,
        )
        .is_ok()
    );

    assert_eq!(constants[1], Some(0xdead_beef));
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
