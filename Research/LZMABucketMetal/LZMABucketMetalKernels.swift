enum LZMABucketMetalKernels {
    static let source = """
    #include <metal_stdlib>
    using namespace metal;

    struct BucketTask {
        uint start;
        uint count;
        uint candidate;
        uint pad;
    };

    struct BucketConstants {
        uint dictionarySize;
        uint cyclicSize;
        uint niceLen;
        uint cutValue;
        uint inputCount;
        uint taskCount;
        uint positionLimit;
        uint waveStart;
        uint waveParity;
    };

    struct BucketMatch {
        ushort length;
        ushort pad;
        uint distance;
    };

    inline uint matchLength(const device uchar* a, const device uchar* b, uint start, uint limit) {
        uint length = start;
        while (length < limit && a[length] == b[length]) { ++length; }
        return length;
    }

    kernel void lzma_bucket_decode(
        const device uchar* input [[buffer(0)]],
        const device uint* positions [[buffer(1)]],
        const device BucketTask* tasks [[buffer(2)]],
        device uint* son [[buffer(3)]],
        device BucketMatch* matches [[buffer(4)]],
        device uint* counts [[buffer(5)]],
        constant BucketConstants& c [[buffer(6)]],
        device uint* candidates [[buffer(7)]],
        const device uint* shortHeads [[buffer(8)]],
        uint gid [[thread_position_in_grid]]) {
        if (gid >= c.taskCount) return;
        const BucketTask task = tasks[gid];
        uint candidate = task.candidate;
        for (uint taskIndex = 0; taskIndex < task.count; ++taskIndex) {
            const uint position = positions[task.start + taskIndex];
            const uint cyclic = (position - 1) % c.cyclicSize;
            const device uchar* cur = input + position - 1;
            const uint available = c.inputCount - (position - 1);
            if (position - 1 >= c.positionLimit || available < 4) {
                counts[position - 1] = 0;
                continue;
            }
            const uint limit = min(available, c.niceLen);
            const uint treeSpan = c.cyclicSize * 2;
            device uint* currentSon = son + c.waveParity * treeSpan;
            device uint* previousSon = son + (1 - c.waveParity) * treeSpan;
            device uint* ptr0 = currentSon + cyclic * 2 + 1;
            device uint* ptr1 = currentSon + cyclic * 2;
            uint len0 = 0, len1 = 0, depth = c.cutValue, best = 3, count = 2;
            bool fullMatch = false;
            const uint m2 = shortHeads[(position - 1) * 2];
            const uint m3 = shortHeads[(position - 1) * 2 + 1];
            const uint d2 = m2 == 0 || position - m2 > c.dictionarySize ? 0 : position - m2;
            const uint d3 = m3 == 0 || position - m3 > c.dictionarySize ? 0 : position - m3;
            matches[(position - 1) * c.cutValue] = BucketMatch{0, 0, d2};
            matches[(position - 1) * c.cutValue + 1] = BucketMatch{0, 0, d3};
            candidates[position - 1] = candidate;
            while (candidate != 0 && depth > 0) {
                const uint delta = position - candidate;
                if (delta > c.dictionarySize) break;
                const int offset = int(cyclic) - int(delta);
                const uint index = offset < 0 ? offset + int(c.cyclicSize) : offset;
                const device uchar* candidateData = cur - delta;
                device uint* pair = (candidate > c.waveStart ? currentSon : previousSon) + index * 2;
                const uint c0 = pair[0], c1 = pair[1];
                const uint d0 = position - c0, d1 = position - c1;
                const int o0 = int(cyclic) - int(d0), o1 = int(cyclic) - int(d1);
                const uint i0 = d0 <= c.dictionarySize
                    ? o0 + ((uint(o0) >> 31) & c.cyclicSize) : cyclic;
                const uint i1 = d1 <= c.dictionarySize
                    ? o1 + ((uint(o1) >> 31) & c.cyclicSize) : cyclic;
                const uint len = matchLength(cur, candidateData, min(len0, len1), limit);
                if (len > best) {
                    best = len;
                    matches[(position - 1) * c.cutValue + count] = BucketMatch{ushort(len), 0, delta};
                    ++count;
                }
                if (len == limit) {
                    ptr1[0] = c0;
                    ptr0[0] = c1;
                    fullMatch = true;
                    break;
                }
                if (candidateData[len] < cur[len]) {
                    ptr1[0] = candidate;
                    ptr1 = pair + 1;
                    candidate = c1;
                    len1 = len;
                } else {
                    ptr0[0] = candidate;
                    ptr0 = pair;
                    candidate = c0;
                    len0 = len;
                }
                --depth;
            }
            if (!fullMatch) {
                ptr0[0] = 0;
                ptr1[0] = 0;
            }
            counts[position - 1] = count;
            candidate = position;
        }
    }
    """
}
