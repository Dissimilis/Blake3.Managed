using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Blake3.Managed.Internal;

/// <summary>
/// SSE 4-way parallel multi-chunk hashing. Used for inputs with 4-7 whole chunks,
/// which are too small for the AVX2 8-way path. Port of HashManyNeon's 4-way kernel.
/// </summary>
internal static class HashManySse41
{
    public static bool IsSupported => Sse2.IsSupported && Ssse3.IsSupported;

    private static readonly Vector128<byte> Rot16Mask128 = Vector128.Create(
        (byte)2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13);

    private static readonly Vector128<byte> Rot8Mask128 = Vector128.Create(
        (byte)1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<uint> RotateRight16(Vector128<uint> v)
    {
#if NET8_0_OR_GREATER
        if (Avx512F.VL.IsSupported)
            return Avx512F.VL.RotateRight(v, 16);
#endif
        return Ssse3.Shuffle(v.AsByte(), Rot16Mask128).AsUInt32();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<uint> RotateRight12(Vector128<uint> v)
    {
#if NET8_0_OR_GREATER
        if (Avx512F.VL.IsSupported)
            return Avx512F.VL.RotateRight(v, 12);
#endif
        return Sse2.Or(Sse2.ShiftRightLogical(v, 12), Sse2.ShiftLeftLogical(v, 20));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<uint> RotateRight8(Vector128<uint> v)
    {
#if NET8_0_OR_GREATER
        if (Avx512F.VL.IsSupported)
            return Avx512F.VL.RotateRight(v, 8);
#endif
        return Ssse3.Shuffle(v.AsByte(), Rot8Mask128).AsUInt32();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<uint> RotateRight7(Vector128<uint> v)
    {
#if NET8_0_OR_GREATER
        if (Avx512F.VL.IsSupported)
            return Avx512F.VL.RotateRight(v, 7);
#endif
        return Sse2.Or(Sse2.ShiftRightLogical(v, 7), Sse2.ShiftLeftLogical(v, 25));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void G128(ref Vector128<uint> a, ref Vector128<uint> b,
                             ref Vector128<uint> c, ref Vector128<uint> d,
                             Vector128<uint> mx, Vector128<uint> my)
    {
        a = Sse2.Add(Sse2.Add(a, b), mx);
        d = RotateRight16(Sse2.Xor(d, a));
        c = Sse2.Add(c, d);
        b = RotateRight12(Sse2.Xor(b, c));
        a = Sse2.Add(Sse2.Add(a, b), my);
        d = RotateRight8(Sse2.Xor(d, a));
        c = Sse2.Add(c, d);
        b = RotateRight7(Sse2.Xor(b, c));
    }

    /// <summary>
    /// Four independent G functions, interleaved phase by phase. See the AVX2 kernel's G256x4 for
    /// why: a single G is a serial chain, and RyuJIT does not reschedule across four of them.
    /// The four G functions of a round act on disjoint state, so this cannot change the result.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void G128x4(
        ref Vector128<uint> a0, ref Vector128<uint> b0, ref Vector128<uint> c0, ref Vector128<uint> d0,
        ref Vector128<uint> a1, ref Vector128<uint> b1, ref Vector128<uint> c1, ref Vector128<uint> d1,
        ref Vector128<uint> a2, ref Vector128<uint> b2, ref Vector128<uint> c2, ref Vector128<uint> d2,
        ref Vector128<uint> a3, ref Vector128<uint> b3, ref Vector128<uint> c3, ref Vector128<uint> d3,
        Vector128<uint> m0x, Vector128<uint> m0y,
        Vector128<uint> m1x, Vector128<uint> m1y,
        Vector128<uint> m2x, Vector128<uint> m2y,
        Vector128<uint> m3x, Vector128<uint> m3y)
    {
        a0 = Sse2.Add(Sse2.Add(a0, b0), m0x);
        a1 = Sse2.Add(Sse2.Add(a1, b1), m1x);
        a2 = Sse2.Add(Sse2.Add(a2, b2), m2x);
        a3 = Sse2.Add(Sse2.Add(a3, b3), m3x);

        d0 = RotateRight16(Sse2.Xor(d0, a0));
        d1 = RotateRight16(Sse2.Xor(d1, a1));
        d2 = RotateRight16(Sse2.Xor(d2, a2));
        d3 = RotateRight16(Sse2.Xor(d3, a3));

        c0 = Sse2.Add(c0, d0);
        c1 = Sse2.Add(c1, d1);
        c2 = Sse2.Add(c2, d2);
        c3 = Sse2.Add(c3, d3);

        b0 = RotateRight12(Sse2.Xor(b0, c0));
        b1 = RotateRight12(Sse2.Xor(b1, c1));
        b2 = RotateRight12(Sse2.Xor(b2, c2));
        b3 = RotateRight12(Sse2.Xor(b3, c3));

        a0 = Sse2.Add(Sse2.Add(a0, b0), m0y);
        a1 = Sse2.Add(Sse2.Add(a1, b1), m1y);
        a2 = Sse2.Add(Sse2.Add(a2, b2), m2y);
        a3 = Sse2.Add(Sse2.Add(a3, b3), m3y);

        d0 = RotateRight8(Sse2.Xor(d0, a0));
        d1 = RotateRight8(Sse2.Xor(d1, a1));
        d2 = RotateRight8(Sse2.Xor(d2, a2));
        d3 = RotateRight8(Sse2.Xor(d3, a3));

        c0 = Sse2.Add(c0, d0);
        c1 = Sse2.Add(c1, d1);
        c2 = Sse2.Add(c2, d2);
        c3 = Sse2.Add(c3, d3);

        b0 = RotateRight7(Sse2.Xor(b0, c0));
        b1 = RotateRight7(Sse2.Xor(b1, c1));
        b2 = RotateRight7(Sse2.Xor(b2, c2));
        b3 = RotateRight7(Sse2.Xor(b3, c3));
    }


    /// <summary>
    /// 4x4 transpose: converts 4 rows where each lane is from a different chunk
    /// into 4 columns where each lane is a word from the same chunk.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Transpose4X4(
        Vector128<uint> r0, Vector128<uint> r1,
        Vector128<uint> r2, Vector128<uint> r3,
        out Vector128<uint> m0, out Vector128<uint> m1,
        out Vector128<uint> m2, out Vector128<uint> m3)
    {
        var t0 = Sse2.UnpackLow(r0, r2);   // {r0[0],r2[0],r0[1],r2[1]}
        var t1 = Sse2.UnpackHigh(r0, r2);  // {r0[2],r2[2],r0[3],r2[3]}
        var t2 = Sse2.UnpackLow(r1, r3);   // {r1[0],r3[0],r1[1],r3[1]}
        var t3 = Sse2.UnpackHigh(r1, r3);  // {r1[2],r3[2],r1[3],r3[3]}
        m0 = Sse2.UnpackLow(t0, t2);       // {r0[0],r1[0],r2[0],r3[0]}
        m1 = Sse2.UnpackHigh(t0, t2);      // {r0[1],r1[1],r2[1],r3[1]}
        m2 = Sse2.UnpackLow(t1, t3);       // {r0[2],r1[2],r2[2],r3[2]}
        m3 = Sse2.UnpackHigh(t1, t3);      // {r0[3],r1[3],r2[3],r3[3]}
    }

    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static unsafe void HashMany(ReadOnlySpan<byte> chunks, int numChunks,
                                       ReadOnlySpan<uint> key, ulong startCounter,
                                       uint flags, Span<uint> cvs)
    {
        const int blocksPerChunk = Blake3Constants.ChunkLen / Blake3Constants.BlockLen; // 16

        Vector128<uint> cv0 = Vector128.Create(key[0]);
        Vector128<uint> cv1 = Vector128.Create(key[1]);
        Vector128<uint> cv2 = Vector128.Create(key[2]);
        Vector128<uint> cv3 = Vector128.Create(key[3]);
        Vector128<uint> cv4 = Vector128.Create(key[4]);
        Vector128<uint> cv5 = Vector128.Create(key[5]);
        Vector128<uint> cv6 = Vector128.Create(key[6]);
        Vector128<uint> cv7 = Vector128.Create(key[7]);

        var counterLo = Vector128.Create(
            (uint)(startCounter + 0), (uint)(startCounter + 1),
            (uint)(startCounter + 2), (uint)(startCounter + 3));
        var counterHi = Vector128.Create(
            (uint)((startCounter + 0) >> 32), (uint)((startCounter + 1) >> 32),
            (uint)((startCounter + 2) >> 32), (uint)((startCounter + 3) >> 32));

        var ivVec0 = Vector128.Create(Blake3Constants.Iv0);
        var ivVec1 = Vector128.Create(Blake3Constants.Iv1);
        var ivVec2 = Vector128.Create(Blake3Constants.Iv2);
        var ivVec3 = Vector128.Create(Blake3Constants.Iv3);
        var blockLenVec = Vector128.Create((uint)Blake3Constants.BlockLen);

        fixed (byte* chunksPtr = chunks)
        {
            Vector128<uint>* m = stackalloc Vector128<uint>[16];

            for (int blockIdx = 0; blockIdx < blocksPerChunk; blockIdx++)
            {
                byte* blockBase = chunksPtr + blockIdx * 64;

                // Load 4 words (16 bytes) from each of 4 chunks, then transpose
                var r0 = Unsafe.ReadUnaligned<Vector128<uint>>(blockBase + 0 * Blake3Constants.ChunkLen);
                var r1 = Unsafe.ReadUnaligned<Vector128<uint>>(blockBase + 1 * Blake3Constants.ChunkLen);
                var r2 = Unsafe.ReadUnaligned<Vector128<uint>>(blockBase + 2 * Blake3Constants.ChunkLen);
                var r3 = Unsafe.ReadUnaligned<Vector128<uint>>(blockBase + 3 * Blake3Constants.ChunkLen);
                Transpose4X4(r0, r1, r2, r3, out m[0], out m[1], out m[2], out m[3]);

                r0 = Unsafe.ReadUnaligned<Vector128<uint>>(blockBase + 0 * Blake3Constants.ChunkLen + 16);
                r1 = Unsafe.ReadUnaligned<Vector128<uint>>(blockBase + 1 * Blake3Constants.ChunkLen + 16);
                r2 = Unsafe.ReadUnaligned<Vector128<uint>>(blockBase + 2 * Blake3Constants.ChunkLen + 16);
                r3 = Unsafe.ReadUnaligned<Vector128<uint>>(blockBase + 3 * Blake3Constants.ChunkLen + 16);
                Transpose4X4(r0, r1, r2, r3, out m[4], out m[5], out m[6], out m[7]);

                r0 = Unsafe.ReadUnaligned<Vector128<uint>>(blockBase + 0 * Blake3Constants.ChunkLen + 32);
                r1 = Unsafe.ReadUnaligned<Vector128<uint>>(blockBase + 1 * Blake3Constants.ChunkLen + 32);
                r2 = Unsafe.ReadUnaligned<Vector128<uint>>(blockBase + 2 * Blake3Constants.ChunkLen + 32);
                r3 = Unsafe.ReadUnaligned<Vector128<uint>>(blockBase + 3 * Blake3Constants.ChunkLen + 32);
                Transpose4X4(r0, r1, r2, r3, out m[8], out m[9], out m[10], out m[11]);

                r0 = Unsafe.ReadUnaligned<Vector128<uint>>(blockBase + 0 * Blake3Constants.ChunkLen + 48);
                r1 = Unsafe.ReadUnaligned<Vector128<uint>>(blockBase + 1 * Blake3Constants.ChunkLen + 48);
                r2 = Unsafe.ReadUnaligned<Vector128<uint>>(blockBase + 2 * Blake3Constants.ChunkLen + 48);
                r3 = Unsafe.ReadUnaligned<Vector128<uint>>(blockBase + 3 * Blake3Constants.ChunkLen + 48);
                Transpose4X4(r0, r1, r2, r3, out m[12], out m[13], out m[14], out m[15]);

                // Block flags
                uint blockFlags = flags;
                if (blockIdx == 0) blockFlags |= Blake3Constants.ChunkStart;
                if (blockIdx == blocksPerChunk - 1) blockFlags |= Blake3Constants.ChunkEnd;
                var flagsVec = Vector128.Create(blockFlags);

                Vector128<uint> s0 = cv0, s1 = cv1, s2 = cv2, s3 = cv3;
                Vector128<uint> s4 = cv4, s5 = cv5, s6 = cv6, s7 = cv7;
                Vector128<uint> s8 = ivVec0, s9 = ivVec1, s10 = ivVec2, s11 = ivVec3;
                Vector128<uint> s12 = counterLo, s13 = counterHi;
                Vector128<uint> s14 = blockLenVec, s15 = flagsVec;

                // Round 0: 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
                G128x4(
                    ref s0, ref s4, ref s8, ref s12,
                    ref s1, ref s5, ref s9, ref s13,
                    ref s2, ref s6, ref s10, ref s14,
                    ref s3, ref s7, ref s11, ref s15,
                    m[0], m[1],
                    m[2], m[3],
                    m[4], m[5],
                    m[6], m[7]);
                G128x4(
                    ref s0, ref s5, ref s10, ref s15,
                    ref s1, ref s6, ref s11, ref s12,
                    ref s2, ref s7, ref s8, ref s13,
                    ref s3, ref s4, ref s9, ref s14,
                    m[8], m[9],
                    m[10], m[11],
                    m[12], m[13],
                    m[14], m[15]);
                // Round 1: 2,6,3,10,7,0,4,13,1,11,12,5,9,14,15,8
                G128x4(
                    ref s0, ref s4, ref s8, ref s12,
                    ref s1, ref s5, ref s9, ref s13,
                    ref s2, ref s6, ref s10, ref s14,
                    ref s3, ref s7, ref s11, ref s15,
                    m[2], m[6],
                    m[3], m[10],
                    m[7], m[0],
                    m[4], m[13]);
                G128x4(
                    ref s0, ref s5, ref s10, ref s15,
                    ref s1, ref s6, ref s11, ref s12,
                    ref s2, ref s7, ref s8, ref s13,
                    ref s3, ref s4, ref s9, ref s14,
                    m[1], m[11],
                    m[12], m[5],
                    m[9], m[14],
                    m[15], m[8]);
                // Round 2: 3,4,10,12,13,2,7,14,6,5,9,0,11,15,8,1
                G128x4(
                    ref s0, ref s4, ref s8, ref s12,
                    ref s1, ref s5, ref s9, ref s13,
                    ref s2, ref s6, ref s10, ref s14,
                    ref s3, ref s7, ref s11, ref s15,
                    m[3], m[4],
                    m[10], m[12],
                    m[13], m[2],
                    m[7], m[14]);
                G128x4(
                    ref s0, ref s5, ref s10, ref s15,
                    ref s1, ref s6, ref s11, ref s12,
                    ref s2, ref s7, ref s8, ref s13,
                    ref s3, ref s4, ref s9, ref s14,
                    m[6], m[5],
                    m[9], m[0],
                    m[11], m[15],
                    m[8], m[1]);
                // Round 3: 10,7,12,9,14,3,13,15,4,0,11,2,5,8,1,6
                G128x4(
                    ref s0, ref s4, ref s8, ref s12,
                    ref s1, ref s5, ref s9, ref s13,
                    ref s2, ref s6, ref s10, ref s14,
                    ref s3, ref s7, ref s11, ref s15,
                    m[10], m[7],
                    m[12], m[9],
                    m[14], m[3],
                    m[13], m[15]);
                G128x4(
                    ref s0, ref s5, ref s10, ref s15,
                    ref s1, ref s6, ref s11, ref s12,
                    ref s2, ref s7, ref s8, ref s13,
                    ref s3, ref s4, ref s9, ref s14,
                    m[4], m[0],
                    m[11], m[2],
                    m[5], m[8],
                    m[1], m[6]);
                // Round 4: 12,13,9,11,15,10,14,8,7,2,5,3,0,1,6,4
                G128x4(
                    ref s0, ref s4, ref s8, ref s12,
                    ref s1, ref s5, ref s9, ref s13,
                    ref s2, ref s6, ref s10, ref s14,
                    ref s3, ref s7, ref s11, ref s15,
                    m[12], m[13],
                    m[9], m[11],
                    m[15], m[10],
                    m[14], m[8]);
                G128x4(
                    ref s0, ref s5, ref s10, ref s15,
                    ref s1, ref s6, ref s11, ref s12,
                    ref s2, ref s7, ref s8, ref s13,
                    ref s3, ref s4, ref s9, ref s14,
                    m[7], m[2],
                    m[5], m[3],
                    m[0], m[1],
                    m[6], m[4]);
                // Round 5: 9,14,11,5,8,12,15,1,13,3,0,10,2,6,4,7
                G128x4(
                    ref s0, ref s4, ref s8, ref s12,
                    ref s1, ref s5, ref s9, ref s13,
                    ref s2, ref s6, ref s10, ref s14,
                    ref s3, ref s7, ref s11, ref s15,
                    m[9], m[14],
                    m[11], m[5],
                    m[8], m[12],
                    m[15], m[1]);
                G128x4(
                    ref s0, ref s5, ref s10, ref s15,
                    ref s1, ref s6, ref s11, ref s12,
                    ref s2, ref s7, ref s8, ref s13,
                    ref s3, ref s4, ref s9, ref s14,
                    m[13], m[3],
                    m[0], m[10],
                    m[2], m[6],
                    m[4], m[7]);
                // Round 6: 11,15,5,0,1,9,8,6,14,10,2,12,3,4,7,13
                G128x4(
                    ref s0, ref s4, ref s8, ref s12,
                    ref s1, ref s5, ref s9, ref s13,
                    ref s2, ref s6, ref s10, ref s14,
                    ref s3, ref s7, ref s11, ref s15,
                    m[11], m[15],
                    m[5], m[0],
                    m[1], m[9],
                    m[8], m[6]);
                G128x4(
                    ref s0, ref s5, ref s10, ref s15,
                    ref s1, ref s6, ref s11, ref s12,
                    ref s2, ref s7, ref s8, ref s13,
                    ref s3, ref s4, ref s9, ref s14,
                    m[14], m[10],
                    m[2], m[12],
                    m[3], m[4],
                    m[7], m[13]);
                // Post-XOR: only chaining value (first 8 words)
                cv0 = Sse2.Xor(s0, s8);
                cv1 = Sse2.Xor(s1, s9);
                cv2 = Sse2.Xor(s2, s10);
                cv3 = Sse2.Xor(s3, s11);
                cv4 = Sse2.Xor(s4, s12);
                cv5 = Sse2.Xor(s5, s13);
                cv6 = Sse2.Xor(s6, s14);
                cv7 = Sse2.Xor(s7, s15);
            }
        }

        // 4x4 transpose: word-major to chunk-major for output
        Transpose4X4(cv0, cv1, cv2, cv3, out var o0, out var o1, out var o2, out var o3);
        Transpose4X4(cv4, cv5, cv6, cv7, out var o4, out var o5, out var o6, out var o7);

        ref uint outRef = ref MemoryMarshal.GetReference(cvs);
        // chunk 0: 8 words = o0 (first 4 words) + o4 (next 4 words)
        VectorCompat.Store(o0, ref outRef);
        VectorCompat.Store(o4, ref outRef, 4);
        // chunk 1
        VectorCompat.Store(o1, ref outRef, 8);
        VectorCompat.Store(o5, ref outRef, 12);
        // chunk 2
        VectorCompat.Store(o2, ref outRef, 16);
        VectorCompat.Store(o6, ref outRef, 20);
        // chunk 3
        VectorCompat.Store(o3, ref outRef, 24);
        VectorCompat.Store(o7, ref outRef, 28);
    }
}
