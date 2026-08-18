use std::sync::OnceLock;

pub(crate) fn crc32c(bytes: &[u8]) -> u32 {
    crc32c_append(0, bytes)
}

pub(super) fn crc32c_append(previous: u32, bytes: &[u8]) -> u32 {
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("crc") {
        // SAFETY: the feature is checked at runtime and the implementation
        // only performs unaligned-safe integer loads from `bytes`.
        return unsafe { crc32c_aarch64(previous, bytes) };
    }
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("sse4.2") {
        // SAFETY: SSE4.2 is checked at runtime. The CRC instructions operate
        // on integer registers; no aligned loads are required.
        return unsafe { crc32c_x86_64(previous, bytes) };
    }
    crc32c_software(previous, bytes)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "crc")]
unsafe fn crc32c_aarch64(previous: u32, mut bytes: &[u8]) -> u32 {
    use core::arch::aarch64::{__crc32cb, __crc32cd};

    let mut crc = !previous;
    while bytes.len() >= 8 {
        let word = u64::from_le_bytes(bytes[..8].try_into().unwrap());
        crc = __crc32cd(crc, word);
        bytes = &bytes[8..];
    }
    for &byte in bytes {
        crc = __crc32cb(crc, byte);
    }
    !crc
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
unsafe fn crc32c_x86_64(previous: u32, mut bytes: &[u8]) -> u32 {
    use core::arch::x86_64::{_mm_crc32_u8, _mm_crc32_u64};

    let mut crc = (!previous) as u64;
    while bytes.len() >= 8 {
        let word = u64::from_le_bytes(bytes[..8].try_into().unwrap());
        crc = _mm_crc32_u64(crc, word);
        bytes = &bytes[8..];
    }
    let mut tail = crc as u32;
    for &byte in bytes {
        tail = _mm_crc32_u8(tail, byte);
    }
    !tail
}

fn crc32c_software(previous: u32, bytes: &[u8]) -> u32 {
    static TABLE: OnceLock<[u32; 256]> = OnceLock::new();
    let table = TABLE.get_or_init(|| {
        let mut table = [0u32; 256];
        for (index, slot) in table.iter_mut().enumerate() {
            let mut value = index as u32;
            for _ in 0..8 {
                value = if value & 1 != 0 {
                    0x82f6_3b78 ^ (value >> 1)
                } else {
                    value >> 1
                };
            }
            *slot = value;
        }
        table
    });
    let mut crc = !previous;
    for byte in bytes {
        crc = table[((crc as u8) ^ byte) as usize] ^ (crc >> 8);
    }
    !crc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_the_castagnoli_check_value_and_fallback() {
        let input = b"123456789";
        assert_eq!(crc32c(input), 0xe306_9283);
        assert_eq!(crc32c(input), crc32c_software(0, input));

        let unaligned = &b"xvecgra-checksum"[..][1..];
        assert_eq!(crc32c(unaligned), crc32c_software(0, unaligned));
        let split = crc32c_append(crc32c(&input[..4]), &input[4..]);
        assert_eq!(split, crc32c(input));
    }
}
