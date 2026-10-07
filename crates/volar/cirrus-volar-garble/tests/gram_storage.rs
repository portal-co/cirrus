//! The A1 capstone: a GRAM-backed `ContextWithStorage` drives an ORAM access
//! through the garbled-circuit contexts (interpreter-side), replacing the
//! linear-scan slice with a sublinear ORAM. Write a bit, read it back, and
//! decode it against the garbler's deterministic base.

use cipher::consts::U16;
use cirrus_core::{ContextWithStorage, StorageAddressBit};
use cirrus_volar_garble::{GramStorage, GramStorageSpace, VolarEvalBackend};
use hybrid_array::Array;
use sha2::{Digest, Sha256};
use volar_oram::OramTree;
use volar_spec::garble::{Eval, Garble, GarbleTable, GlobalSecret, gram_decode_label};

const Z: usize = 4;
const B: usize = 8;

fn det_secret() -> GlobalSecret<U16> {
    GlobalSecret::new(Array::<u8, U16>::from_fn(|i| {
        (i as u8).wrapping_mul(37) | 1
    }))
}

/// Re-derive the deterministic base the GramStorage impl uses:
/// H(0xDA || a || b).
fn base(a: u64, b: u64) -> Garble<U16> {
    let mut d = Sha256::new();
    d.update(&[0xDAu8]);
    d.update(&a.to_le_bytes());
    d.update(&b.to_le_bytes());
    let hash = d.finalize();
    Garble {
        base: Array::<u8, U16>::from_fn(|j| hash[j]),
    }
}

#[test]
fn gram_storage_write_then_read_roundtrip() {
    let secret = det_secret();
    let levels = 4;
    let num_addrs = 8u64;

    let mut tree = OramTree::<Z, B>::new(levels);
    let tables: Vec<GarbleTable<U16>> = Vec::new();
    let eval_backend = VolarEvalBackend::<Sha256, _, U16>::new(tables.into_iter());
    let mut storage_space = GramStorageSpace::<U16>::new::<Sha256>(num_addrs as usize);
    let mut ctx = GramStorage::<_, Sha256, U16, Z, B>::new(
        eval_backend,
        secret.clone(),
        &mut tree,
        levels,
        num_addrs,
    );

    // WRITE cell 0 = true through the ContextWithStorage (capstone path).
    // The write value is decoded against the cell's current base
    // (`cell_bases[0]`, initially `gram_data_base(0, 0)`), so the value
    // label is encoded against that base.
    let value_base = base(0, 0);
    let value_label = secret.encode(&value_base, true);
    let addr0: Vec<StorageAddressBit<Eval<U16>>> = (0..3)
        .map(|i| StorageAddressBit {
            wire: Eval::zero(),
            known: Some(false),
        })
        .map(|mut b| {
            let _ = &mut b;
            b
        })
        .collect();
    ctx.storage_write(&mut storage_space, &addr0, value_label)
        .expect("storage_write");

    // READ cell 0 through the ContextWithStorage. This is access #2; the
    // read-data bit is re-garbled to gram_data_base(2, 0).
    let read_label = ctx
        .storage_read(&mut storage_space, &addr0)
        .expect("storage_read");
    let bit = gram_decode_label(&read_label, &base(2, 0));
    assert!(bit, "read must return the written bit (true)");

    // WRITE cell 0 = false, read it back — exercises a second round. The
    // cell's base is still the write's value base (the tree now holds the
    // bit under the value's base), so encode the second write against it.
    let value_base2 = base(0, 0);
    let value_label2 = secret.encode(&value_base2, false);
    ctx.storage_write(&mut storage_space, &addr0, value_label2)
        .expect("storage_write 2");
    let read_label2 = ctx
        .storage_read(&mut storage_space, &addr0)
        .expect("storage_read 2");
    let bit2 = gram_decode_label(&read_label2, &base(4, 0));
    assert!(!bit2, "second read must return the overwritten bit (false)");
}

#[test]
fn gram_storage_multiple_cells() {
    let secret = det_secret();
    let levels = 4;
    let num_addrs = 8u64;

    let mut tree = OramTree::<Z, B>::new(levels);
    let tables: Vec<GarbleTable<U16>> = Vec::new();
    let eval_backend = VolarEvalBackend::<Sha256, _, U16>::new(tables.into_iter());
    let mut storage_space = GramStorageSpace::<U16>::new::<Sha256>(num_addrs as usize);
    let mut ctx = GramStorage::<_, Sha256, U16, Z, B>::new(
        eval_backend,
        secret.clone(),
        &mut tree,
        levels,
        num_addrs,
    );

    // Helper: address bits for a concrete cell index (3-bit addresses).
    let addr_of = |cell: u64| -> Vec<StorageAddressBit<Eval<U16>>> {
        (0..3)
            .map(|i| StorageAddressBit {
                wire: Eval::zero(),
                known: Some((cell >> i) & 1 == 1),
            })
            .collect()
    };

    // Write distinct bits to cells 0, 3, 5 (accesses 1, 2, 3).
    let writes = [(0u64, true), (3u64, true), (5u64, false)];
    for (k, (cell, bit)) in writes.iter().enumerate() {
        let access = (k + 1) as u64;
        let vb = base(access, 1);
        let vl = secret.encode(&vb, *bit);
        ctx.storage_write(&mut storage_space, &addr_of(*cell), vl)
            .expect("write");
    }

    // Read them back (accesses 4, 5, 6); each read re-garbles to
    // gram_data_base(access, 0).
    for (k, (cell, want)) in writes.iter().enumerate() {
        let access = (writes.len() + k + 1) as u64;
        let rl = ctx
            .storage_read(&mut storage_space, &addr_of(*cell))
            .expect("read");
        let got = gram_decode_label(&rl, &base(access, 0));
        assert_eq!(got, *want, "cell {cell}");
    }
}
