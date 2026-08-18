use super::*;

#[derive(Clone, Debug)]
pub(crate) struct SnapshotVectorSection {
    pub byte_offset: usize,
    pub byte_len: usize,
    pub checksum: u32,
    pub block_checksums: Option<Arc<[u32]>>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct SnapshotCsrSection {
    pub byte_offset: usize,
    pub value_count: usize,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct SnapshotCsrSections {
    pub out_offsets: SnapshotCsrSection,
    pub out_ids: SnapshotCsrSection,
    pub in_offsets: SnapshotCsrSection,
    pub in_ids: SnapshotCsrSection,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct SnapshotSketchSection {
    pub byte_offset: usize,
    pub entry_count: usize,
    pub words_per_signature: usize,
    pub owner_columns: Option<SnapshotSketchOwnerColumns>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct SnapshotSketchOwnerColumns {
    pub owner_byte_offset: usize,
    pub owner_kind_byte_offset: usize,
    pub label_byte_offset: usize,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct SnapshotPropertyIndexSection {
    pub byte_offset: usize,
    pub entry_count: usize,
    pub entry_width: usize,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct SnapshotNumericPropertyIndexSection {
    pub byte_offset: usize,
    pub entry_count: usize,
    pub entry_width: usize,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct SnapshotRecordSections {
    pub node_byte_offset: usize,
    pub node_count: usize,
    pub node_slots: usize,
    pub edge_byte_offset: usize,
    pub edge_count: usize,
    pub edge_slots: usize,
    pub property_byte_offset: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct SnapshotSections {
    pub vectors: SnapshotVectorSection,
    pub csr: Option<SnapshotCsrSections>,
    pub sketches: Option<SnapshotSketchSection>,
    pub property_index: Option<SnapshotPropertyIndexSection>,
    pub numeric_property_index: Option<SnapshotNumericPropertyIndexSection>,
    pub records: Option<SnapshotRecordSections>,
}

pub(crate) fn read_snapshot(
    file_bytes: &[u8],
    header: Header,
    mut apply: impl FnMut(SnapshotOperation) -> Result<()>,
) -> Result<SnapshotSections> {
    if !header.has_snapshot() {
        return Err(Error::Corrupt("database has no checkpoint snapshot".into()));
    }
    let metadata_len = usize::try_from(header.snapshot_metadata_len)
        .map_err(|_| Error::Corrupt("checkpoint metadata is too large".into()))?;
    let metadata_start = HEADER_LEN as usize;
    let metadata_end = metadata_start
        .checked_add(metadata_len)
        .ok_or_else(|| Error::Corrupt("checkpoint metadata range overflow".into()))?;
    let metadata = file_bytes
        .get(metadata_start..metadata_end)
        .ok_or_else(|| Error::Corrupt("checkpoint metadata exceeds the file".into()))?;
    let content_len = metadata_len - 4;
    let expected = u32::from_le_bytes(metadata[content_len..].try_into().unwrap());
    if crc32c(&metadata[..content_len]) != expected {
        return Err(Error::Corrupt(
            "checkpoint metadata checksum mismatch".into(),
        ));
    }
    let (csr, block_checksums, sketches, property_index, numeric_property_index, records) =
        match &metadata[..8] {
            magic if magic == SNAPSHOT_MAGIC => {
                let operation_count =
                    usize::try_from(u64::from_le_bytes(metadata[8..16].try_into().unwrap()))
                        .map_err(|_| {
                            Error::Corrupt("checkpoint operation count exceeds usize".into())
                        })?;
                decode_snapshot_operations(
                    &metadata[16..content_len],
                    operation_count,
                    metadata_start + 16,
                    &mut apply,
                )?;
                (None, None, None, None, None, None)
            }
            magic if magic == COLUMNAR_SNAPSHOT_MAGIC => {
                let decoded = decode_columnar_snapshot(
                    &metadata[..content_len],
                    metadata_start,
                    false,
                    false,
                    false,
                    header.dimension,
                    &mut apply,
                )?;
                (
                    Some(decoded.csr),
                    decoded.block_checksums,
                    None,
                    None,
                    None,
                    Some(decoded.records),
                )
            }
            magic if magic == LEGACY_INDEXED_SNAPSHOT_MAGIC => {
                let decoded = decode_columnar_snapshot(
                    &metadata[..content_len],
                    metadata_start,
                    true,
                    false,
                    false,
                    header.dimension,
                    &mut apply,
                )?;
                (
                    Some(decoded.csr),
                    decoded.block_checksums,
                    decoded.sketches,
                    None,
                    None,
                    Some(decoded.records),
                )
            }
            magic if magic == INDEXED_SNAPSHOT_MAGIC => {
                let decoded = decode_columnar_snapshot(
                    &metadata[..content_len],
                    metadata_start,
                    true,
                    true,
                    false,
                    header.dimension,
                    &mut apply,
                )?;
                (
                    Some(decoded.csr),
                    decoded.block_checksums,
                    decoded.sketches,
                    decoded.property_index,
                    None,
                    Some(decoded.records),
                )
            }
            magic if magic == RANGE_INDEXED_SNAPSHOT_MAGIC => {
                let decoded = decode_columnar_snapshot(
                    &metadata[..content_len],
                    metadata_start,
                    true,
                    true,
                    true,
                    header.dimension,
                    &mut apply,
                )?;
                (
                    Some(decoded.csr),
                    decoded.block_checksums,
                    decoded.sketches,
                    decoded.property_index,
                    decoded.numeric_property_index,
                    Some(decoded.records),
                )
            }
            _ => return Err(Error::Corrupt("invalid checkpoint metadata magic".into())),
        };

    let vector_len = usize::try_from(header.snapshot_vector_len)
        .map_err(|_| Error::Corrupt("checkpoint vector section is too large".into()))?;
    let vector_offset = usize::try_from(header.snapshot_vector_offset)
        .map_err(|_| Error::Corrupt("checkpoint vector offset exceeds usize".into()))?;
    if let Some(checksums) = &block_checksums {
        let data_len = vector_len - 4;
        let expected_blocks = data_len.div_ceil(VECTOR_CHECKSUM_BLOCK_SIZE);
        if checksums.len() != expected_blocks {
            return Err(Error::Corrupt(
                "vector block checksum count does not match vector section".into(),
            ));
        }
    }
    let checksum_offset = vector_offset
        .checked_add(vector_len - 4)
        .ok_or_else(|| Error::Corrupt("checkpoint vector range overflow".into()))?;
    let checksum_end = checksum_offset
        .checked_add(4)
        .ok_or_else(|| Error::Corrupt("checkpoint checksum range overflow".into()))?;
    let checksum = file_bytes
        .get(checksum_offset..checksum_end)
        .ok_or_else(|| Error::Corrupt("checkpoint vector section exceeds the file".into()))?;
    Ok(SnapshotSections {
        vectors: SnapshotVectorSection {
            byte_offset: vector_offset,
            byte_len: vector_len - 4,
            checksum: u32::from_le_bytes(checksum.try_into().unwrap()),
            block_checksums,
        },
        csr,
        sketches,
        property_index,
        numeric_property_index,
        records,
    })
}

struct DecodedColumnarSnapshot {
    csr: SnapshotCsrSections,
    block_checksums: Option<Arc<[u32]>>,
    sketches: Option<SnapshotSketchSection>,
    property_index: Option<SnapshotPropertyIndexSection>,
    numeric_property_index: Option<SnapshotNumericPropertyIndexSection>,
    records: SnapshotRecordSections,
}

fn decode_columnar_snapshot(
    metadata: &[u8],
    metadata_file_offset: usize,
    indexed: bool,
    property_indexed: bool,
    numeric_property_indexed: bool,
    dimension: usize,
    apply: &mut impl FnMut(SnapshotOperation) -> Result<()>,
) -> Result<DecodedColumnarSnapshot> {
    let header_len = if numeric_property_indexed {
        RANGE_INDEXED_COLUMNAR_HEADER_LEN
    } else if property_indexed {
        INDEXED_COLUMNAR_HEADER_LEN
    } else if indexed {
        LEGACY_INDEXED_COLUMNAR_HEADER_LEN
    } else {
        COLUMNAR_HEADER_LEN
    };
    let section_count = if numeric_property_indexed {
        RANGE_INDEXED_COLUMNAR_SECTION_COUNT
    } else if property_indexed {
        INDEXED_COLUMNAR_SECTION_COUNT
    } else if indexed {
        LEGACY_INDEXED_COLUMNAR_SECTION_COUNT
    } else {
        COLUMNAR_SECTION_COUNT
    };
    if metadata.len() < header_len {
        return Err(Error::Corrupt(
            "columnar checkpoint header is truncated".into(),
        ));
    }
    let node_count = columnar_u64(metadata, 8)?;
    let edge_count = columnar_u64(metadata, 16)?;
    let symbol_count = columnar_u64(metadata, 24)?;
    let node_slots = columnar_u64(metadata, 32)?;
    let edge_slots = columnar_u64(metadata, 40)?;
    let indexed_vectors = columnar_u64(metadata, 48)?;
    let vector_checksum_count = usize::try_from(columnar_u64(metadata, 56)?)
        .map_err(|_| Error::Corrupt("vector block checksum count exceeds usize".into()))?;
    let mut descriptors = vec![(0usize, 0usize); section_count];
    let mut previous_end = header_len;
    for (index, descriptor) in descriptors.iter_mut().enumerate() {
        let start = 64 + index * 16;
        let offset = usize::try_from(columnar_u64(metadata, start)?)
            .map_err(|_| Error::Corrupt("columnar section offset exceeds usize".into()))?;
        let len = usize::try_from(columnar_u64(metadata, start + 8)?)
            .map_err(|_| Error::Corrupt("columnar section length exceeds usize".into()))?;
        let end = offset
            .checked_add(len)
            .ok_or_else(|| Error::Corrupt("columnar section range overflow".into()))?;
        if offset < header_len
            || offset < previous_end
            || !offset.is_multiple_of(8)
            || end > metadata.len()
        {
            return Err(Error::Corrupt(format!(
                "invalid columnar section {index} range"
            )));
        }
        *descriptor = (offset, len);
        previous_end = end;
    }
    let section = |index: usize| {
        let (offset, len) = descriptors[index];
        &metadata[offset..offset + len]
    };

    let mut symbols = Decoder::new(section(0));
    for expected in 0..symbol_count {
        let id = symbols.u32()?;
        if id as u64 != expected {
            return Err(Error::Corrupt(format!(
                "columnar symbol id {id} is not contiguous"
            )));
        }
        apply(SnapshotOperation::InternSymbol {
            id,
            value: symbols.string()?.into(),
        })?;
    }
    let mut block_checksums = Vec::with_capacity(vector_checksum_count);
    for _ in 0..vector_checksum_count {
        block_checksums.push(symbols.u32()?);
    }
    if !symbols.is_empty() {
        return Err(Error::Corrupt(
            "columnar symbol/checksum section contains trailing bytes".into(),
        ));
    }

    let node_bytes = section(1);
    let expected_node_bytes = usize::try_from(node_count)
        .ok()
        .and_then(|count| count.checked_mul(48))
        .ok_or_else(|| Error::Corrupt("columnar node section size overflow".into()))?;
    if node_bytes.len() != expected_node_bytes {
        return Err(Error::Corrupt(
            "columnar node record count does not match its section".into(),
        ));
    }
    let (_, property_section_len) = descriptors[3];
    let mut counted_vectors = 0u64;
    let mut previous_node = None;
    for record in node_bytes.chunks_exact(48) {
        let id = columnar_u64(record, 0)?;
        if previous_node.is_some_and(|previous| id <= previous) || id >= node_slots {
            return Err(Error::Corrupt(
                "columnar node ids are not strictly ordered within their slots".into(),
            ));
        }
        previous_node = Some(id);
        let property_offset = columnar_u64(record, 24)?;
        let property_len = columnar_u32(record, 36)?;
        validate_property_range(property_offset, property_len, property_section_len)?;
        if columnar_u64(record, 8)? == 0 || columnar_u32(record, 32)? as u64 >= symbol_count {
            return Err(Error::Corrupt(
                "columnar node generation or label is invalid".into(),
            ));
        }
        let vector_count = columnar_u32(record, 40)?;
        counted_vectors = counted_vectors
            .checked_add(vector_count as u64)
            .ok_or_else(|| Error::Corrupt("indexed vector count overflow".into()))?;
        usize::try_from(columnar_u64(record, 16)?)
            .map_err(|_| Error::Corrupt("node vector offset exceeds usize".into()))?;
    }

    let edge_bytes = section(2);
    let expected_edge_bytes = usize::try_from(edge_count)
        .ok()
        .and_then(|count| count.checked_mul(64))
        .ok_or_else(|| Error::Corrupt("columnar edge section size overflow".into()))?;
    if edge_bytes.len() != expected_edge_bytes {
        return Err(Error::Corrupt(
            "columnar edge record count does not match its section".into(),
        ));
    }
    let mut previous_edge = None;
    for record in edge_bytes.chunks_exact(64) {
        let id = columnar_u64(record, 0)?;
        if previous_edge.is_some_and(|previous| id <= previous) || id >= edge_slots {
            return Err(Error::Corrupt(
                "columnar edge ids are not strictly ordered within their slots".into(),
            ));
        }
        previous_edge = Some(id);
        let source = columnar_u64(record, 16)?;
        let target = columnar_u64(record, 24)?;
        if source >= node_slots
            || target >= node_slots
            || !columnar_node_exists(node_bytes, node_count, node_slots, source)
            || !columnar_node_exists(node_bytes, node_count, node_slots, target)
        {
            return Err(Error::Corrupt(
                "columnar edge endpoint exceeds node slots".into(),
            ));
        }
        let property_offset = columnar_u64(record, 40)?;
        let property_len = columnar_u32(record, 52)?;
        validate_property_range(property_offset, property_len, property_section_len)?;
        if columnar_u64(record, 8)? == 0 || columnar_u32(record, 48)? as u64 >= symbol_count {
            return Err(Error::Corrupt(
                "columnar edge generation or label is invalid".into(),
            ));
        }
        let vector_count = columnar_u32(record, 56)?;
        counted_vectors = counted_vectors
            .checked_add(vector_count as u64)
            .ok_or_else(|| Error::Corrupt("indexed vector count overflow".into()))?;
        usize::try_from(columnar_u64(record, 32)?)
            .map_err(|_| Error::Corrupt("edge vector offset exceeds usize".into()))?;
    }
    if counted_vectors != indexed_vectors {
        return Err(Error::Corrupt(
            "columnar indexed vector count does not match records".into(),
        ));
    }

    let expected_offsets = usize::try_from(node_slots)
        .ok()
        .and_then(|slots| slots.checked_add(1))
        .and_then(|count| count.checked_mul(8))
        .ok_or_else(|| Error::Corrupt("columnar CSR offset size overflow".into()))?;
    let expected_ids = usize::try_from(edge_count)
        .ok()
        .and_then(|count| count.checked_mul(8))
        .ok_or_else(|| Error::Corrupt("columnar CSR id size overflow".into()))?;
    if section(4).len() != expected_offsets
        || section(6).len() != expected_offsets
        || section(5).len() != expected_ids
        || section(7).len() != expected_ids
    {
        return Err(Error::Corrupt(
            "columnar CSR section sizes do not match graph counts".into(),
        ));
    }
    validate_csr_sections(section(4), section(5), edge_count, edge_slots)?;
    validate_csr_sections(section(6), section(7), edge_count, edge_slots)?;
    let mapped_section = |index: usize| SnapshotCsrSection {
        byte_offset: metadata_file_offset + descriptors[index].0,
        value_count: descriptors[index].1 / 8,
    };
    let sketches = if indexed {
        let (section_offset, section_len) = descriptors[8];
        let section = section(8);
        if section.len() < 24
            || (&section[..8] != SKETCH_MAGIC && &section[..8] != SKETCH_COLUMNS_MAGIC)
        {
            return Err(Error::Corrupt(
                "indexed checkpoint sketch section is truncated or invalid".into(),
            ));
        }
        let entry_count = usize::try_from(columnar_u64(section, 8)?)
            .map_err(|_| Error::Corrupt("sketch entry count exceeds usize".into()))?;
        let words_per_signature = columnar_u32(section, 16)? as usize;
        let maximum_words = crate::ann::signature_word_count(dimension);
        if entry_count != indexed_vectors as usize
            || words_per_signature == 0
            || words_per_signature > maximum_words
        {
            return Err(Error::Corrupt(
                "sketch header does not match checkpoint vector count".into(),
            ));
        }
        let word_count = entry_count
            .checked_mul(words_per_signature)
            .ok_or_else(|| Error::Corrupt("sketch word count overflow".into()))?;
        let (signature_offset, owner_columns) = if &section[..8] == SKETCH_MAGIC {
            (24usize, None)
        } else {
            let owner_offset = 24usize;
            let owner_kind_offset =
                owner_offset
                    .checked_add(entry_count.checked_mul(8).ok_or_else(|| {
                        Error::Corrupt("sketch owner column length overflow".into())
                    })?)
                    .ok_or_else(|| Error::Corrupt("sketch owner offset overflow".into()))?;
            let label_offset = align_up(
                owner_kind_offset
                    .checked_add(entry_count)
                    .ok_or_else(|| Error::Corrupt("sketch owner-kind offset overflow".into()))?,
                4,
            )
            .map_err(|_| Error::Corrupt("sketch label offset overflow".into()))?;
            let signature_offset = align_up(
                label_offset
                    .checked_add(entry_count.checked_mul(4).ok_or_else(|| {
                        Error::Corrupt("sketch label column length overflow".into())
                    })?)
                    .ok_or_else(|| Error::Corrupt("sketch label offset overflow".into()))?,
                8,
            )
            .map_err(|_| Error::Corrupt("sketch signature offset overflow".into()))?;
            validate_sketch_owner_columns(
                section,
                owner_offset,
                owner_kind_offset,
                label_offset,
                node_bytes,
                edge_bytes,
                dimension,
                entry_count,
                symbol_count,
            )?;
            (
                signature_offset,
                Some(SnapshotSketchOwnerColumns {
                    owner_byte_offset: metadata_file_offset + section_offset + owner_offset,
                    owner_kind_byte_offset: metadata_file_offset
                        + section_offset
                        + owner_kind_offset,
                    label_byte_offset: metadata_file_offset + section_offset + label_offset,
                }),
            )
        };
        let expected_len = signature_offset
            .checked_add(
                word_count
                    .checked_mul(8)
                    .ok_or_else(|| Error::Corrupt("sketch word bytes overflow".into()))?,
            )
            .ok_or_else(|| Error::Corrupt("sketch section length overflow".into()))?;
        if section_len != expected_len {
            return Err(Error::Corrupt(
                "sketch section length does not match its header".into(),
            ));
        }
        let byte_offset = metadata_file_offset + section_offset + signature_offset;
        if !byte_offset.is_multiple_of(8) {
            return Err(Error::Corrupt("sketch word section is not aligned".into()));
        }
        Some(SnapshotSketchSection {
            byte_offset,
            entry_count,
            words_per_signature,
            owner_columns,
        })
    } else {
        None
    };
    let property_index = if property_indexed {
        let (section_offset, section_len) = descriptors[9];
        let property_section = section(9);
        if property_section.len() < 24
            || &property_section[..8] != PROPERTY_INDEX_MAGIC
            || property_section[17..24] != [0; 7]
        {
            return Err(Error::Corrupt(
                "property index header is truncated or invalid".into(),
            ));
        }
        let entry_count = usize::try_from(columnar_u64(property_section, 8)?)
            .map_err(|_| Error::Corrupt("property index count exceeds usize".into()))?;
        let packed_width = match property_section[16] {
            4 => 4usize,
            8 => 8usize,
            _ => return Err(Error::Corrupt("property index ID width is invalid".into())),
        };
        let entry_width = 8 + packed_width;
        let expected_len = 24usize
            .checked_add(
                entry_count
                    .checked_mul(entry_width)
                    .ok_or_else(|| Error::Corrupt("property index length overflows".into()))?,
            )
            .ok_or_else(|| Error::Corrupt("property index section overflows".into()))?;
        if section_len != expected_len {
            return Err(Error::Corrupt(
                "property index length does not match its header".into(),
            ));
        }
        let entries = &property_section[24..];
        validate_property_index(
            entries,
            packed_width,
            node_bytes,
            edge_bytes,
            node_count,
            node_slots,
            edge_count,
            edge_slots,
            symbol_count,
        )?;
        Some(SnapshotPropertyIndexSection {
            byte_offset: metadata_file_offset
                .checked_add(section_offset)
                .and_then(|offset| offset.checked_add(24))
                .ok_or_else(|| Error::Corrupt("property index offset overflow".into()))?,
            entry_count,
            entry_width,
        })
    } else {
        None
    };
    let numeric_property_index = if numeric_property_indexed {
        let (section_offset, section_len) = descriptors[10];
        let numeric_section = section(10);
        if numeric_section.len() < 24
            || &numeric_section[..8] != NUMERIC_PROPERTY_INDEX_MAGIC
            || numeric_section[17..24] != [0; 7]
        {
            return Err(Error::Corrupt(
                "numeric property index header is truncated or invalid".into(),
            ));
        }
        let entry_count = usize::try_from(columnar_u64(numeric_section, 8)?)
            .map_err(|_| Error::Corrupt("numeric property index count exceeds usize".into()))?;
        let packed_width = match numeric_section[16] {
            4 => 4usize,
            8 => 8usize,
            _ => {
                return Err(Error::Corrupt(
                    "numeric property index ID width is invalid".into(),
                ));
            }
        };
        let entry_width = 16 + packed_width;
        let expected_len =
            24usize
                .checked_add(entry_count.checked_mul(entry_width).ok_or_else(|| {
                    Error::Corrupt("numeric property index length overflows".into())
                })?)
                .ok_or_else(|| Error::Corrupt("numeric property index section overflows".into()))?;
        if section_len != expected_len {
            return Err(Error::Corrupt(
                "numeric property index length does not match its header".into(),
            ));
        }
        validate_numeric_property_index(
            &numeric_section[24..],
            packed_width,
            node_bytes,
            edge_bytes,
            node_count,
            node_slots,
            edge_count,
            edge_slots,
            symbol_count,
        )?;
        Some(SnapshotNumericPropertyIndexSection {
            byte_offset: metadata_file_offset
                .checked_add(section_offset)
                .and_then(|offset| offset.checked_add(24))
                .ok_or_else(|| Error::Corrupt("numeric property index offset overflow".into()))?,
            entry_count,
            entry_width,
        })
    } else {
        None
    };
    Ok(DecodedColumnarSnapshot {
        csr: SnapshotCsrSections {
            out_offsets: mapped_section(4),
            out_ids: mapped_section(5),
            in_offsets: mapped_section(6),
            in_ids: mapped_section(7),
        },
        block_checksums: (!block_checksums.is_empty()).then(|| block_checksums.into()),
        sketches,
        property_index,
        numeric_property_index,
        records: SnapshotRecordSections {
            node_byte_offset: metadata_file_offset + descriptors[1].0,
            node_count: usize::try_from(node_count)
                .map_err(|_| Error::Corrupt("node count exceeds usize".into()))?,
            node_slots: usize::try_from(node_slots)
                .map_err(|_| Error::Corrupt("node slots exceed usize".into()))?,
            edge_byte_offset: metadata_file_offset + descriptors[2].0,
            edge_count: usize::try_from(edge_count)
                .map_err(|_| Error::Corrupt("edge count exceeds usize".into()))?,
            edge_slots: usize::try_from(edge_slots)
                .map_err(|_| Error::Corrupt("edge slots exceed usize".into()))?,
            property_byte_offset: metadata_file_offset + descriptors[3].0,
        },
    })
}

#[expect(
    clippy::too_many_arguments,
    reason = "validation cross-checks the property index against every record domain"
)]
fn validate_property_index(
    bytes: &[u8],
    packed_width: usize,
    node_records: &[u8],
    edge_records: &[u8],
    node_count: u64,
    node_slots: u64,
    edge_count: u64,
    edge_slots: u64,
    symbol_count: u64,
) -> Result<()> {
    let entry_width = 8 + packed_width;
    if !bytes.len().is_multiple_of(entry_width) {
        return Err(Error::Corrupt(
            "property index length is not a whole number of entries".into(),
        ));
    }
    let mut previous = None;
    for entry in bytes.chunks_exact(entry_width) {
        let key = columnar_u32(entry, 0)?;
        if key as u64 >= symbol_count {
            return Err(Error::Corrupt(
                "property index entry has invalid key".into(),
            ));
        }
        let fingerprint = columnar_u32(entry, 4)?;
        let packed_element = if packed_width == 4 {
            columnar_u32(entry, 8)? as u64
        } else {
            columnar_u64(entry, 8)?
        };
        let kind = (packed_element & 1) as u8;
        let id = packed_element >> 1;
        let ordering_key = (key, fingerprint, packed_element);
        if previous.is_some_and(|previous| previous >= ordering_key) {
            return Err(Error::Corrupt(
                "property index entries are not strictly ordered".into(),
            ));
        }
        let exists = if kind == 0 {
            columnar_record_exists(node_records, 48, node_count, node_slots, id)
        } else {
            columnar_record_exists(edge_records, 64, edge_count, edge_slots, id)
        };
        if !exists {
            return Err(Error::Corrupt(
                "property index references a missing element".into(),
            ));
        }
        previous = Some(ordering_key);
    }
    Ok(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "validation cross-checks the numeric index against every record domain"
)]
fn validate_numeric_property_index(
    bytes: &[u8],
    packed_width: usize,
    node_records: &[u8],
    edge_records: &[u8],
    node_count: u64,
    node_slots: u64,
    edge_count: u64,
    edge_slots: u64,
    symbol_count: u64,
) -> Result<()> {
    let entry_width = 16 + packed_width;
    if !bytes.len().is_multiple_of(entry_width) {
        return Err(Error::Corrupt(
            "numeric property index length is not a whole number of entries".into(),
        ));
    }
    let mut previous = None;
    for entry in bytes.chunks_exact(entry_width) {
        let key = columnar_u32(entry, 0)?;
        if key as u64 >= symbol_count {
            return Err(Error::Corrupt(
                "numeric property index entry has invalid key".into(),
            ));
        }
        let tag = entry[4];
        if !matches!(tag, 3 | 4) || entry[5..8] != [0; 3] {
            return Err(Error::Corrupt(
                "numeric property index entry has an invalid type".into(),
            ));
        }
        let sortable = columnar_u64(entry, 8)?;
        let packed_element = if packed_width == 4 {
            columnar_u32(entry, 16)? as u64
        } else {
            columnar_u64(entry, 16)?
        };
        let kind = (packed_element & 1) as u8;
        let id = packed_element >> 1;
        let ordering_key = (key, tag, sortable, packed_element);
        if previous.is_some_and(|previous| previous >= ordering_key) {
            return Err(Error::Corrupt(
                "numeric property index entries are not strictly ordered".into(),
            ));
        }
        let exists = if kind == 0 {
            columnar_record_exists(node_records, 48, node_count, node_slots, id)
        } else {
            columnar_record_exists(edge_records, 64, edge_count, edge_slots, id)
        };
        if !exists {
            return Err(Error::Corrupt(
                "numeric property index references a missing element".into(),
            ));
        }
        previous = Some(ordering_key);
    }
    Ok(())
}

fn columnar_record_exists(
    bytes: &[u8],
    record_size: usize,
    count: u64,
    slots: u64,
    id: u64,
) -> bool {
    if id >= slots {
        return false;
    }
    if count == slots {
        return true;
    }
    let mut left = 0usize;
    let mut right = bytes.len() / record_size;
    while left < right {
        let middle = left + (right - left) / 2;
        let start = middle * record_size;
        let candidate = u64::from_le_bytes(bytes[start..start + 8].try_into().unwrap());
        match candidate.cmp(&id) {
            std::cmp::Ordering::Less => left = middle + 1,
            std::cmp::Ordering::Greater => right = middle,
            std::cmp::Ordering::Equal => return true,
        }
    }
    false
}

fn validate_csr_sections(
    offsets: &[u8],
    ids: &[u8],
    edge_count: u64,
    edge_slots: u64,
) -> Result<()> {
    let mut previous = 0u64;
    for (index, chunk) in offsets.chunks_exact(8).enumerate() {
        let offset = u64::from_le_bytes(chunk.try_into().unwrap());
        if (index == 0 && offset != 0) || offset < previous || offset > edge_count {
            return Err(Error::Corrupt(
                "columnar CSR offsets are not monotonic".into(),
            ));
        }
        previous = offset;
    }
    if previous != edge_count {
        return Err(Error::Corrupt(
            "columnar CSR final offset does not equal edge count".into(),
        ));
    }
    if ids
        .chunks_exact(8)
        .any(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()) >= edge_slots)
    {
        return Err(Error::Corrupt(
            "columnar CSR contains an edge id outside edge slots".into(),
        ));
    }
    Ok(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "validation compares parallel sketch columns with both record tables"
)]
pub(super) fn validate_sketch_owner_columns(
    section: &[u8],
    owner_offset: usize,
    owner_kind_offset: usize,
    label_offset: usize,
    node_records: &[u8],
    edge_records: &[u8],
    dimension: usize,
    entry_count: usize,
    symbol_count: u64,
) -> Result<()> {
    let owners_len = entry_count
        .checked_mul(8)
        .ok_or_else(|| Error::Corrupt("sketch owner byte length overflow".into()))?;
    let labels_len = entry_count
        .checked_mul(4)
        .ok_or_else(|| Error::Corrupt("sketch label byte length overflow".into()))?;
    let owners_end = owner_offset
        .checked_add(owners_len)
        .ok_or_else(|| Error::Corrupt("sketch owner range overflow".into()))?;
    let owner_kinds_end = owner_kind_offset
        .checked_add(entry_count)
        .ok_or_else(|| Error::Corrupt("sketch owner-kind range overflow".into()))?;
    let labels_end = label_offset
        .checked_add(labels_len)
        .ok_or_else(|| Error::Corrupt("sketch label range overflow".into()))?;
    let owners = section
        .get(owner_offset..owners_end)
        .ok_or_else(|| Error::Corrupt("sketch owner column is truncated".into()))?;
    let owner_kinds = section
        .get(owner_kind_offset..owner_kinds_end)
        .ok_or_else(|| Error::Corrupt("sketch owner-kind column is truncated".into()))?;
    let labels = section
        .get(label_offset..labels_end)
        .ok_or_else(|| Error::Corrupt("sketch label column is truncated".into()))?;
    let mut populated = vec![false; entry_count];
    let mut validate_record =
        |id: u64, label: u32, vector_offset: u64, vector_count: u32, kind: u8| -> Result<()> {
            if label as u64 >= symbol_count || !vector_offset.is_multiple_of(dimension as u64) {
                return Err(Error::Corrupt(
                    "sketch owner record has invalid label or vector offset".into(),
                ));
            }
            let first = usize::try_from(vector_offset / dimension as u64)
                .map_err(|_| Error::Corrupt("sketch owner ordinal exceeds usize".into()))?;
            for vector_index in 0..vector_count as usize {
                let ordinal = first
                    .checked_add(vector_index)
                    .ok_or_else(|| Error::Corrupt("sketch owner ordinal overflow".into()))?;
                let slot = populated
                    .get_mut(ordinal)
                    .ok_or_else(|| Error::Corrupt("sketch owner ordinal exceeds entries".into()))?;
                let owner_start = ordinal * 8;
                let label_start = ordinal * 4;
                if *slot
                    || u64::from_le_bytes(owners[owner_start..owner_start + 8].try_into().unwrap())
                        != id
                    || owner_kinds[ordinal] != kind
                    || u32::from_le_bytes(labels[label_start..label_start + 4].try_into().unwrap())
                        != label
                {
                    return Err(Error::Corrupt(
                        "sketch owner columns disagree with element records".into(),
                    ));
                }
                *slot = true;
            }
            Ok(())
        };
    for record in node_records.chunks_exact(48) {
        validate_record(
            columnar_u64(record, 0)?,
            columnar_u32(record, 32)?,
            columnar_u64(record, 16)?,
            columnar_u32(record, 40)?,
            0,
        )?;
    }
    for record in edge_records.chunks_exact(64) {
        validate_record(
            columnar_u64(record, 0)?,
            columnar_u32(record, 48)?,
            columnar_u64(record, 32)?,
            columnar_u32(record, 56)?,
            1,
        )?;
    }
    if populated.iter().any(|populated| !populated) {
        return Err(Error::Corrupt(
            "sketch owner columns do not densely cover entries".into(),
        ));
    }
    Ok(())
}

fn columnar_node_exists(bytes: &[u8], count: u64, slots: u64, id: u64) -> bool {
    if id >= slots {
        return false;
    }
    if count == slots {
        return true;
    }
    let mut left = 0usize;
    let mut right = bytes.len() / 48;
    while left < right {
        let middle = left + (right - left) / 2;
        let start = middle * 48;
        let candidate = u64::from_le_bytes(bytes[start..start + 8].try_into().unwrap());
        match candidate.cmp(&id) {
            std::cmp::Ordering::Less => left = middle + 1,
            std::cmp::Ordering::Greater => right = middle,
            std::cmp::Ordering::Equal => return true,
        }
    }
    false
}

fn validate_property_range(offset: u64, len: u32, section_len: usize) -> Result<()> {
    let start = usize::try_from(offset)
        .map_err(|_| Error::Corrupt("property offset exceeds usize".into()))?;
    let end = start
        .checked_add(len as usize)
        .ok_or_else(|| Error::Corrupt("property range overflow".into()))?;
    if len < 4 || end > section_len {
        return Err(Error::Corrupt(
            "property record exceeds columnar property section".into(),
        ));
    }
    Ok(())
}

fn columnar_u32(bytes: &[u8], offset: usize) -> Result<u32> {
    let value = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| Error::Corrupt("truncated columnar u32".into()))?;
    Ok(u32::from_le_bytes(value.try_into().unwrap()))
}

fn columnar_u64(bytes: &[u8], offset: usize) -> Result<u64> {
    let value = bytes
        .get(offset..offset + 8)
        .ok_or_else(|| Error::Corrupt("truncated columnar u64".into()))?;
    Ok(u64::from_le_bytes(value.try_into().unwrap()))
}
