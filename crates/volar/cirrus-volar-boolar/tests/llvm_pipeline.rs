use core::convert::Infallible;
use std::{collections::BTreeMap, fs, io::Write};

use cirrus_core::{
    ContextWithBitAnd, ContextWithBitOr, ContextWithBitXor, ContextWithCreate, ContextWithStorage,
    ContextWithValue, HasError, StorageAddressBit,
};
use cirrus_volar_boolar::{StorageBank, execute, storage_requirements};
use volar_ir_build::{
    BoolarStage, Pipeline,
    volar_ir_passes::{self, LoweringMode},
};

/// A concrete Cirrus context for executing the fused circuit. Unlike the
/// symbolic sparse MUX context, this can evaluate the fixture's actual stack
/// addresses without materializing every address that a symbolic write might
/// select.
#[derive(Default)]
struct ConcreteContext;

impl HasError for ConcreteContext {
    type Error = Infallible;
}

impl ContextWithValue<bool> for ConcreteContext {
    type Wrapped = bool;
}

impl ContextWithCreate<bool> for ConcreteContext {
    fn create(&mut self, value: bool) -> Result<bool, Self::Error> {
        Ok(value)
    }
}

impl ContextWithBitAnd<bool> for ConcreteContext {
    fn bitand(&mut self, left: bool, right: bool) -> Result<bool, Self::Error> {
        Ok(left & right)
    }

    fn bitand_assign(&mut self, left: &mut bool, right: bool) -> Result<(), Self::Error> {
        *left &= right;
        Ok(())
    }
}

impl ContextWithBitOr<bool> for ConcreteContext {
    fn bitor(&mut self, left: bool, right: bool) -> Result<bool, Self::Error> {
        Ok(left | right)
    }

    fn bitor_assign(&mut self, left: &mut bool, right: bool) -> Result<(), Self::Error> {
        *left |= right;
        Ok(())
    }
}

impl ContextWithBitXor<bool> for ConcreteContext {
    fn bitxor(&mut self, left: bool, right: bool) -> Result<bool, Self::Error> {
        Ok(left ^ right)
    }

    fn bitxor_assign(&mut self, left: &mut bool, right: bool) -> Result<(), Self::Error> {
        *left ^= right;
        Ok(())
    }
}

impl ContextWithStorage<bool> for ConcreteContext {
    type Storage = BTreeMap<Vec<bool>, bool>;

    fn storage_read(
        &mut self,
        storage: &mut Self::Storage,
        address: &[StorageAddressBit<bool>],
    ) -> Result<bool, Self::Error> {
        Ok(*storage
            .get(&address.iter().map(|bit| bit.wire).collect::<Vec<_>>())
            .unwrap_or(&false))
    }

    fn storage_write(
        &mut self,
        storage: &mut Self::Storage,
        address: &[StorageAddressBit<bool>],
        value: bool,
    ) -> Result<(), Self::Error> {
        storage.insert(address.iter().map(|bit| bit.wire).collect(), value);
        Ok(())
    }
}

fn write_fixture(source: &str) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!(
        "cirrus-volar-boolar-llvm-pipeline-{}.ll",
        std::process::id()
    ));
    let mut file = fs::File::create(&path).expect("create LLVM fixture");
    file.write_all(source.as_bytes())
        .expect("write LLVM fixture");
    path
}

fn bits(value: u8) -> Vec<bool> {
    (0..u8::BITS).map(|bit| value & (1 << bit) != 0).collect()
}

fn byte(bits: &[bool]) -> u8 {
    bits.iter()
        .enumerate()
        .fold(0, |value, (bit, set)| value | (u8::from(*set) << bit))
}

#[test]
fn llvm_symbolic_branch_executes_through_boolar_fusion() {
    let path = write_fixture(
        r#"
define i8 @pick(i1 %cond, i8 %then_value, i8 %else_value) {
entry:
  br i1 %cond, label %then, label %otherwise
then:
  ret i8 %then_value
otherwise:
  ret i8 %else_value
}
"#,
    );
    let (blocks, mut types) = Pipeline::from_llvm(&path, &["pick"])
        .and_then(|pipeline| pipeline.lower_to_volar_ir())
        .expect("LLVM fixture lowers to Volar IR")
        .to_volar_ir();
    let (movfuscated, _, _, watched_params) =
        volar_ir_passes::movfuscate::movfuscate_ir_with_boundary_and_watch(
            &blocks,
            &mut types,
            &[(0, 0)],
        );
    let (boolar, lowered) =
        volar_ir_passes::lower_ir_to_boolar::lower_ir_to_boolar_with_tables(&movfuscated, &types);
    let circuit = Pipeline::<BoolarStage>::from_data(boolar)
        .fuse(64, LoweringMode::Unconditional)
        .expect("symbolic LLVM branch lowers through Boolar fusion")
        .to_boolar_circuit();
    let _ = fs::remove_file(&path);

    // The LLVM frontend ABI packs this fixture's arguments into one 64-bit
    // entry word: cond at bit 0, then the two i8s. Movfuscation assigns that
    // word to a physical state slot, so resolve its Boolar bit positions from
    // public watch metadata rather than assuming a slot layout.
    let [(_, _, combined)] = watched_params.as_slice() else {
        panic!("exactly one packed entry parameter is watched");
    };
    let entry_bits = lowered
        .var_bits
        .bits(0, *combined)
        .expect("watched entry parameter is lowered to Boolar bits")
        .iter()
        .map(|bit| bit.0 as usize)
        .collect::<Vec<_>>();
    assert_eq!(entry_bits.len(), 64);

    let requirements = storage_requirements(&circuit).expect("fused storage layout is valid");
    assert_eq!(
        requirements.len(),
        3,
        "fixture needs three fused stack lanes"
    );
    let run = |cond, then_value, else_value| {
        let mut inputs = vec![false; circuit.params as usize];
        inputs[entry_bits[0]] = cond;
        for (input, value) in entry_bits[1..9].iter().zip(bits(then_value)) {
            inputs[*input] = value;
        }
        for (input, value) in entry_bits[9..17].iter().zip(bits(else_value)) {
            inputs[*input] = value;
        }
        let mut storage0 = BTreeMap::new();
        let mut storage1 = BTreeMap::new();
        let mut storage2 = BTreeMap::new();
        let mut banks = [
            StorageBank {
                storage: requirements[0].storage,
                lane: requirements[0].lane,
                address_bits: requirements[0].address_bits,
                value: &mut storage0,
            },
            StorageBank {
                storage: requirements[1].storage,
                lane: requirements[1].lane,
                address_bits: requirements[1].address_bits,
                value: &mut storage1,
            },
            StorageBank {
                storage: requirements[2].storage,
                lane: requirements[2].lane,
                address_bits: requirements[2].address_bits,
                value: &mut storage2,
            },
        ];
        let mut context = ConcreteContext;
        byte(
            &execute(&mut context, &circuit, &inputs, &mut banks)
                .expect("fused circuit executes in Cirrus"),
        )
    };

    assert_eq!(run(true, 0x35, 0xCA), 0x35);
    assert_eq!(run(false, 0x35, 0xCA), 0xCA);
}
