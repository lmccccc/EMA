"""Read saved-index metadata without loading, migrating or rewriting its graph."""

from pathlib import Path
import struct

from .figure4a_work_control import header_prefix


def read_layout(path, *, include_mappings=False):
    path = Path(path)
    size = path.stat().st_size
    header = header_prefix(path)
    with path.open("rb") as stream:
        def read(fmt):
            length = struct.calcsize("<" + fmt)
            data = stream.read(length)
            if len(data) != length:
                raise ValueError("Truncated index header")
            return struct.unpack("<" + fmt, data)[0]

        def skip(length):
            if length < 0 or stream.tell() + length > size:
                raise ValueError("Invalid serialized index array length")
            stream.seek(length, 1)

        def vector(fmt, retain=False, limit=32):
            count = read("Q")
            unit = struct.calcsize("<" + fmt)
            if count > size // unit or retain and count > limit:
                raise ValueError("Invalid serialized index vector length")
            if retain:
                return [read(fmt) for _ in range(count)]
            skip(count * unit)

        stream.seek(struct.calcsize("<8QiI5QdQQ"))
        header["cate_int_words"], header["max_cate_size"] = read("i"), read("i")
        header["attr_type"], header["attr_positions"] = vector("i", True), vector("i", True)
        header["attr_words"] = read("i")
        vector("I")
        buckets = read("i")
        for _ in range(3):
            vector("I")
        table_size = read("i")
        skip(4 * table_size * buckets * len(header["attr_type"]))
        mappings = read("Q")
        if mappings > 32:
            raise ValueError("Invalid counting-table mapping count")
        if include_mappings:
            header["counting_hash_table_mapping"] = [
                vector("i", True, max(table_size, header["max_cate_size"]))
                for _ in range(mappings)]
        else:
            for _ in range(mappings):
                vector("i")
        header["predicate_words"] = read("i")
        header["predicate_offsets"] = vector("i", True)
        header["ft_offset"], header["attr_offset"] = read("Q"), read("Q")
        header["neighbor_ft_offset"], header["ft_bytes_per_record"] = read("i"), read("i")
        position = stream.tell()
        version = read("i")
        header["format"] = version if 2 <= version <= 12 else 1
        header["records_offset"] = stream.tell() if header["format"] != 1 else position
    if (header["count"] <= 0 or header["count"] > header["max_elements"]
            or header["records_offset"] + header["count"] * header["record_bytes"] > size
            or header["label_offset"] + 8 > header["record_bytes"]
            or header["attr_offset"] + 4 * header["attr_words"] > header["record_bytes"]):
        raise ValueError("Index record geometry exceeds the saved file")
    header["dimension"] = (header["label_offset"] - header["vector_offset"]) // 4
    return header


def record_sample(path, header, rows):
    result = []
    with Path(path).open("rb") as stream:
        for row in rows:
            if not 0 <= row < header["count"]:
                raise ValueError("Requested index row is out of range")
            offset = header["records_offset"] + row * header["record_bytes"]
            stream.seek(offset + header["label_offset"])
            label = struct.unpack("<Q", stream.read(8))[0]
            stream.seek(offset + header["attr_offset"])
            words = struct.unpack("<" + "I" * header["attr_words"],
                                  stream.read(4 * header["attr_words"]))
            attributes = []
            for kind, position in zip(header["attr_type"], header["attr_positions"]):
                if kind == 0:
                    value = words[position]
                    attributes.append([value if value < 2**31 else value - 2**32])
                elif kind == 1:
                    attributes.append([
                        bit + 32 * word for word in range(header["cate_int_words"])
                        for bit in range(32) if words[position + word] & (1 << bit)])
                else:
                    raise ValueError("Unknown saved attribute type")
            result.append({"internal_row": row, "label": label, "attributes": attributes})
    return result
