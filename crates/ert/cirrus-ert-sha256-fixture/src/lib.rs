#![no_std]

//! Fixed SHA-256 compression workload shared by the bare-metal interpreter gates.

/// Compress one fixed-size SHA-256 block and return its first state word.
///
/// This intentionally ordinary scalar implementation is the locked workload
/// used to keep both architecture facades compatible with LLVM-generated
/// fixed-loop, stack-resident code.
#[unsafe(no_mangle)]
#[inline(never)]
pub extern "C" fn sha256_compress(
    w0: u32,
    w1: u32,
    w2: u32,
    w3: u32,
    w4: u32,
    w5: u32,
    w6: u32,
    w7: u32,
    w8: u32,
    w9: u32,
    w10: u32,
    w11: u32,
    w12: u32,
    w13: u32,
    w14: u32,
    w15: u32,
) -> u32 {
    let mut schedule = [
        w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15,
    ];
    let mut a = 0x6a09_e667u32;
    let mut b = 0xbb67_ae85u32;
    let mut c = 0x3c6e_f372u32;
    let mut d = 0xa54f_f53au32;
    let mut e = 0x510e_527fu32;
    let mut f = 0x9b05_688cu32;
    let mut g = 0x1f83_d9abu32;
    let mut h = 0x5be0_cd19u32;
    let mut round = 0u32;
    while round < 64 {
        let slot = (round & 15) as usize;
        let word = if round < 16 {
            schedule[slot]
        } else {
            let value = small_sigma_1(schedule[(round.wrapping_sub(2) & 15) as usize])
                .wrapping_add(schedule[(round.wrapping_sub(7) & 15) as usize])
                .wrapping_add(small_sigma_0(
                    schedule[(round.wrapping_sub(15) & 15) as usize],
                ))
                .wrapping_add(schedule[slot]);
            schedule[slot] = value;
            value
        };
        let temp1 = h
            .wrapping_add(big_sigma_1(e))
            .wrapping_add(choice(e, f, g))
            .wrapping_add(K[round as usize])
            .wrapping_add(word);
        let temp2 = big_sigma_0(a).wrapping_add(majority(a, b, c));
        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(temp1);
        d = c;
        c = b;
        b = a;
        a = temp1.wrapping_add(temp2);
        round += 1;
    }
    a.wrapping_add(0x6a09_e667)
}

#[inline(never)]
fn choice(x: u32, y: u32, z: u32) -> u32 {
    (x & y) ^ (!x & z)
}
#[inline(never)]
fn majority(x: u32, y: u32, z: u32) -> u32 {
    (x & y) ^ (x & z) ^ (y & z)
}
#[inline(never)]
fn big_sigma_0(x: u32) -> u32 {
    x.rotate_right(2) ^ x.rotate_right(13) ^ x.rotate_right(22)
}
#[inline(never)]
fn big_sigma_1(x: u32) -> u32 {
    x.rotate_right(6) ^ x.rotate_right(11) ^ x.rotate_right(25)
}
#[inline(never)]
fn small_sigma_0(x: u32) -> u32 {
    x.rotate_right(7) ^ x.rotate_right(18) ^ (x >> 3)
}
#[inline(never)]
fn small_sigma_1(x: u32) -> u32 {
    x.rotate_right(17) ^ x.rotate_right(19) ^ (x >> 10)
}

static K: [u32; 64] = [
    0x428a_2f98,
    0x7137_4491,
    0xb5c0_fbcf,
    0xe9b5_dba5,
    0x3956_c25b,
    0x59f1_11f1,
    0x923f_82a4,
    0xab1c_5ed5,
    0xd807_aa98,
    0x1283_5b01,
    0x2431_85be,
    0x550c_7dc3,
    0x72be_5d74,
    0x80de_b1fe,
    0x9bdc_06a7,
    0xc19b_f174,
    0xe49b_69c1,
    0xefbe_4786,
    0x0fc1_9dc6,
    0x240c_a1cc,
    0x2de9_2c6f,
    0x4a74_84aa,
    0x5cb0_a9dc,
    0x76f9_88da,
    0x983e_5152,
    0xa831_c66d,
    0xb003_27c8,
    0xbf59_7fc7,
    0xc6e0_0bf3,
    0xd5a7_9147,
    0x06ca_6351,
    0x1429_2967,
    0x27b7_0a85,
    0x2e1b_2138,
    0x4d2c_6dfc,
    0x5338_0d13,
    0x650a_7354,
    0x766a_0abb,
    0x81c2_c92e,
    0x9272_2c85,
    0xa2bf_e8a1,
    0xa81a_664b,
    0xc24b_8b70,
    0xc76c_51a3,
    0xd192_e819,
    0xd699_0624,
    0xf40e_3585,
    0x106a_a070,
    0x19a4_c116,
    0x1e376c08,
    0x2748774c,
    0x34b0bcb5,
    0x391c0cb3,
    0x4ed8aa4a,
    0x5b9cca4f,
    0x682e6ff3,
    0x748f82ee,
    0x78a5636f,
    0x84c87814,
    0x8cc70208,
    0x90befffa,
    0xa4506ceb,
    0xbef9a3f7,
    0xc67178f2,
];
