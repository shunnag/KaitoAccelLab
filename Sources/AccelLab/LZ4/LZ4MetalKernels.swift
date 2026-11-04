enum LZ4MetalKernels {
    static let source = """
    #include <metal_stdlib>
    using namespace metal;

    struct BlockDesc {
        uint srcOffset;
        uint srcLength;
        uint dstOffset;
        uint dstLength;
        uint isStored;
        uint pad[3];
    };

    constant uint status_ok = 0;
    constant uint status_truncated = 1;
    constant uint status_output_bounds = 2;
    constant uint status_offset = 3;
    constant uint status_size = 4;
    constant uint extended_length = 15;
    constant uint extension_continuation = 255;
    constant uint minimum_match = 4;
    constant uint simd_width = 32;

    inline uint read_length(const device uchar* src, uint size, thread uint& cursor,
                            uint initial, uint limit, thread uint& length) {
        length = initial;
        if (length > limit) return status_output_bounds;
        if (initial == extended_length) {
            while (true) {
                if (cursor >= size) return status_truncated;
                uint byte = src[cursor++];
                if (byte > limit - length) return status_output_bounds;
                length += byte;
                if (byte != extension_continuation) break;
            }
        }
        return status_ok;
    }

    inline uint decode_thread(const device uchar* src, device uchar* dst, BlockDesc block) {
        if (block.isStored) {
            if (block.srcLength != block.dstLength) return status_size;
            for (uint i = 0; i < block.srcLength; ++i) dst[i] = src[i];
            return status_ok;
        }
        uint cursor = 0, written = 0;
        while (true) {
            if (cursor >= block.srcLength) return status_truncated;
            uint token = src[cursor++], literals = 0;
            uint error = read_length(src, block.srcLength, cursor, token >> 4,
                                     block.dstLength - written, literals);
            if (error) return error;
            if (literals > block.srcLength - cursor) return status_truncated;
            for (uint i = 0; i < literals; ++i) dst[written + i] = src[cursor + i];
            cursor += literals;
            written += literals;
            if (cursor == block.srcLength) return written == block.dstLength ? status_ok : status_size;
            if (block.srcLength - cursor < 2) return status_truncated;
            uint offset = uint(src[cursor]) | (uint(src[cursor + 1]) << 8);
            cursor += 2;
            if (offset == 0 || offset > written) return status_offset;
            if (block.dstLength - written < minimum_match) return status_output_bounds;
            uint length = 0;
            error = read_length(src, block.srcLength, cursor, token & extended_length,
                                 block.dstLength - written - minimum_match, length);
            if (error) return error;
            length += minimum_match;
            for (uint i = 0; i < length; ++i) dst[written + i] = dst[written + i - offset];
            written += length;
        }
    }

    kernel void lz4_decode_thread_per_block(
        const device uchar* src [[buffer(0)]], const device BlockDesc* blocks [[buffer(1)]],
        device uchar* dst [[buffer(2)]], device uint* status [[buffer(3)]],
        constant uint& blockCount [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
        if (gid >= blockCount) return;
        BlockDesc block = blocks[gid];
        status[gid] = decode_thread(src + block.srcOffset, dst + block.dstOffset, block);
    }

    // 全レーンが同じ入力を解析し、SIMD グループ内の分岐とバリアを揃える。
    inline uint decode_simd(const device uchar* src, device uchar* dst, BlockDesc block, uint lane) {
        if (block.isStored) {
            if (block.srcLength != block.dstLength) return status_size;
            for (uint i = lane; i < block.srcLength; i += simd_width) dst[i] = src[i];
            return status_ok;
        }
        uint cursor = 0, written = 0;
        while (true) {
            if (cursor >= block.srcLength) return status_truncated;
            uint token = src[cursor++], literals = 0;
            uint error = read_length(src, block.srcLength, cursor, token >> 4,
                                     block.dstLength - written, literals);
            if (error) return error;
            if (literals > block.srcLength - cursor) return status_truncated;
            for (uint i = lane; i < literals; i += simd_width) dst[written + i] = src[cursor + i];
            simdgroup_barrier(mem_flags::mem_device);
            cursor += literals;
            written += literals;
            if (cursor == block.srcLength) return written == block.dstLength ? status_ok : status_size;
            if (block.srcLength - cursor < 2) return status_truncated;
            uint offset = uint(src[cursor]) | (uint(src[cursor + 1]) << 8);
            cursor += 2;
            if (offset == 0 || offset > written) return status_offset;
            if (block.dstLength - written < minimum_match) return status_output_bounds;
            uint length = 0;
            error = read_length(src, block.srcLength, cursor, token & extended_length,
                                 block.dstLength - written - minimum_match, length);
            if (error) return error;
            length += minimum_match;
            // 参照距離以下の単位で同期し、重なる一致列の先行書き込みを可視化する。
            while (length > 0) {
                uint chunk = min(min(offset, simd_width), length);
                if (lane < chunk) dst[written + lane] = dst[written + lane - offset];
                simdgroup_barrier(mem_flags::mem_device);
                written += chunk;
                length -= chunk;
            }
        }
    }

    kernel void lz4_decode_simd_per_block(
        const device uchar* src [[buffer(0)]], const device BlockDesc* blocks [[buffer(1)]],
        device uchar* dst [[buffer(2)]], device uint* status [[buffer(3)]],
        constant uint& blockCount [[buffer(4)]], constant uint& simdgroupsPerThreadgroup [[buffer(5)]],
        uint sgid [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
        uint tgid [[threadgroup_position_in_grid]]) {
        uint index = tgid * simdgroupsPerThreadgroup + sgid;
        if (index >= blockCount) return;
        BlockDesc block = blocks[index];
        uint result = decode_simd(src + block.srcOffset, dst + block.dstOffset, block, lane);
        if (lane == 0) status[index] = result;
    }

    inline void copy_bytes16(const device uchar* src, device uchar* dst) {
        metal::uchar4 a = metal::uchar4(src[0], src[1], src[2], src[3]);
        metal::uchar4 b = metal::uchar4(src[4], src[5], src[6], src[7]);
        metal::uchar4 c = metal::uchar4(src[8], src[9], src[10], src[11]);
        metal::uchar4 d = metal::uchar4(src[12], src[13], src[14], src[15]);
        dst[0] = a[0]; dst[1] = a[1]; dst[2] = a[2]; dst[3] = a[3];
        dst[4] = b[0]; dst[5] = b[1]; dst[6] = b[2]; dst[7] = b[3];
        dst[8] = c[0]; dst[9] = c[1]; dst[10] = c[2]; dst[11] = c[3];
        dst[12] = d[0]; dst[13] = d[1]; dst[14] = d[2]; dst[15] = d[3];
    }

    inline void copy_simd16(const device uchar* src, device uchar* dst, uint length, uint lane) {
        for (uint base = 0; base < length; base += 512) {
            uint index = base + lane * 16;
            if (index + 16 <= length) {
                copy_bytes16(src + index, dst + index);
            } else if (index < length) {
                uint end = min(base + 512, length);
                while (index < end) {
                    dst[index] = src[index];
                    ++index;
                }
            }
        }
    }

    inline uint decode_simd16(const device uchar* src, device uchar* dst, BlockDesc block, uint lane) {
        if (block.isStored) {
            if (block.srcLength != block.dstLength) return status_size;
            copy_simd16(src, dst, block.srcLength, lane);
            return status_ok;
        }
        uint cursor = 0, written = 0;
        while (true) {
            if (cursor >= block.srcLength) return status_truncated;
            uint token = src[cursor++], literals = 0;
            uint error = read_length(src, block.srcLength, cursor, token >> 4,
                                     block.dstLength - written, literals);
            if (error) return error;
            if (literals > block.srcLength - cursor) return status_truncated;
            copy_simd16(src + cursor, dst + written, literals, lane);
            simdgroup_barrier(mem_flags::mem_device);
            cursor += literals;
            written += literals;
            if (cursor == block.srcLength) return written == block.dstLength ? status_ok : status_size;
            if (block.srcLength - cursor < 2) return status_truncated;
            uint offset = uint(src[cursor]) | (uint(src[cursor + 1]) << 8);
            cursor += 2;
            if (offset == 0 || offset > written) return status_offset;
            if (block.dstLength - written < minimum_match) return status_output_bounds;
            uint length = 0;
            error = read_length(src, block.srcLength, cursor, token & extended_length,
                                 block.dstLength - written - minimum_match, length);
            if (error) return error;
            length += minimum_match;
            while (length > 0) {
                uint chunk = min(min(offset, simd_width * 16), length);
                uint index = lane * 16;
                if (index + 16 <= chunk) {
                    copy_bytes16(dst + written + index - offset, dst + written + index);
                } else if (index < chunk) {
                    while (index < chunk) {
                        dst[written + index] = dst[written + index - offset];
                        ++index;
                    }
                }
                simdgroup_barrier(mem_flags::mem_device);
                written += chunk;
                length -= chunk;
            }
        }
    }

    kernel void lz4_decode_simd16_per_block(
        const device uchar* src [[buffer(0)]], const device BlockDesc* blocks [[buffer(1)]],
        device uchar* dst [[buffer(2)]], device uint* status [[buffer(3)]],
        constant uint& blockCount [[buffer(4)]], constant uint& simdgroupsPerThreadgroup [[buffer(5)]],
        uint sgid [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
        uint tgid [[threadgroup_position_in_grid]]) {
        uint index = tgid * simdgroupsPerThreadgroup + sgid;
        if (index >= blockCount) return;
        BlockDesc block = blocks[index];
        uint result = decode_simd16(src + block.srcOffset, dst + block.dstOffset, block, lane);
        if (lane == 0) status[index] = result;
    }

    inline void copy_thread16(const device uchar* src, device uchar* dst, uint length) {
        uint index = 0;
        while (index + 16 <= length) {
            copy_bytes16(src + index, dst + index);
            index += 16;
        }
        while (index < length) {
            dst[index] = src[index];
            ++index;
        }
    }

    inline uint decode_thread16(const device uchar* src, device uchar* dst, BlockDesc block) {
        if (block.isStored) {
            if (block.srcLength != block.dstLength) return status_size;
            copy_thread16(src, dst, block.srcLength);
            return status_ok;
        }
        uint cursor = 0, written = 0;
        while (true) {
            if (cursor >= block.srcLength) return status_truncated;
            uint token = src[cursor++], literals = 0;
            uint error = read_length(src, block.srcLength, cursor, token >> 4,
                                     block.dstLength - written, literals);
            if (error) return error;
            if (literals > block.srcLength - cursor) return status_truncated;
            copy_thread16(src + cursor, dst + written, literals);
            cursor += literals;
            written += literals;
            if (cursor == block.srcLength) return written == block.dstLength ? status_ok : status_size;
            if (block.srcLength - cursor < 2) return status_truncated;
            uint offset = uint(src[cursor]) | (uint(src[cursor + 1]) << 8);
            cursor += 2;
            if (offset == 0 || offset > written) return status_offset;
            if (block.dstLength - written < minimum_match) return status_output_bounds;
            uint length = 0;
            error = read_length(src, block.srcLength, cursor, token & extended_length,
                                 block.dstLength - written - minimum_match, length);
            if (error) return error;
            length += minimum_match;
            if (offset >= 16) {
                while (length >= 16) {
                    copy_bytes16(dst + written - offset, dst + written);
                    written += 16;
                    length -= 16;
                }
            }
            while (length > 0) {
                dst[written] = dst[written - offset];
                ++written;
                --length;
            }
        }
    }

    kernel void lz4_decode_thread16_per_block(
        const device uchar* src [[buffer(0)]], const device BlockDesc* blocks [[buffer(1)]],
        device uchar* dst [[buffer(2)]], device uint* status [[buffer(3)]],
        constant uint& blockCount [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
        if (gid >= blockCount) return;
        BlockDesc block = blocks[gid];
        status[gid] = decode_thread16(src + block.srcOffset, dst + block.dstOffset, block);
    }
    """
}
