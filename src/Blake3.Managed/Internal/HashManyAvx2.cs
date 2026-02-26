using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Blake3.Managed.Internal;

internal static class HashManyAvx2
{
    public static bool IsSupported => Avx2.IsSupported;

    private static readonly Vector256<byte> Rot16Mask256 = Vector256.Create(
        (byte)2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13,
        2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13);

    private static readonly Vector256<byte> Rot8Mask256 = Vector256.Create(
        (byte)1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12,
        1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<uint> RotateRight16(Vector256<uint> v)
    {
#if NET8_0_OR_GREATER
        if (Avx512F.VL.IsSupported)
            return Avx512F.VL.RotateRight(v, 16);
#endif
        return Avx2.Shuffle(v.AsByte(), Rot16Mask256).AsUInt32();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<uint> RotateRight12(Vector256<uint> v)
    {
#if NET8_0_OR_GREATER
        if (Avx512F.VL.IsSupported)
            return Avx512F.VL.RotateRight(v, 12);
#endif
        return Avx2.Or(Avx2.ShiftRightLogical(v, 12), Avx2.ShiftLeftLogical(v, 20));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<uint> RotateRight8(Vector256<uint> v)
    {
#if NET8_0_OR_GREATER
        if (Avx512F.VL.IsSupported)
            return Avx512F.VL.RotateRight(v, 8);
#endif
        return Avx2.Shuffle(v.AsByte(), Rot8Mask256).AsUInt32();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<uint> RotateRight7(Vector256<uint> v)
    {
#if NET8_0_OR_GREATER
        if (Avx512F.VL.IsSupported)
            return Avx512F.VL.RotateRight(v, 7);
#endif
        return Avx2.Or(Avx2.ShiftRightLogical(v, 7), Avx2.ShiftLeftLogical(v, 25));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void G256(ref Vector256<uint> a, ref Vector256<uint> b,
                             ref Vector256<uint> c, ref Vector256<uint> d,
                             Vector256<uint> mx, Vector256<uint> my)
    {
        a = Avx2.Add(Avx2.Add(a, b), mx);
        d = RotateRight16(Avx2.Xor(d, a));
        c = Avx2.Add(c, d);
        b = RotateRight12(Avx2.Xor(b, c));
        a = Avx2.Add(Avx2.Add(a, b), my);
        d = RotateRight8(Avx2.Xor(d, a));
        c = Avx2.Add(c, d);
        b = RotateRight7(Avx2.Xor(b, c));
    }

    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static unsafe void HashMany(ReadOnlySpan<byte> chunks, int numChunks,
                                       ReadOnlySpan<uint> key, ulong startCounter,
                                       uint flags, Span<uint> cvs)
    {
        const int blocksPerChunk = Blake3Constants.ChunkLen / Blake3Constants.BlockLen; // 16

        Vector256<uint> cv0 = Vector256.Create(key[0]);
        Vector256<uint> cv1 = Vector256.Create(key[1]);
        Vector256<uint> cv2 = Vector256.Create(key[2]);
        Vector256<uint> cv3 = Vector256.Create(key[3]);
        Vector256<uint> cv4 = Vector256.Create(key[4]);
        Vector256<uint> cv5 = Vector256.Create(key[5]);
        Vector256<uint> cv6 = Vector256.Create(key[6]);
        Vector256<uint> cv7 = Vector256.Create(key[7]);

        var counterLo = Vector256.Create(
            (uint)(startCounter + 0), (uint)(startCounter + 1),
            (uint)(startCounter + 2), (uint)(startCounter + 3),
            (uint)(startCounter + 4), (uint)(startCounter + 5),
            (uint)(startCounter + 6), (uint)(startCounter + 7));
        var counterHi = Vector256.Create(
            (uint)((startCounter + 0) >> 32), (uint)((startCounter + 1) >> 32),
            (uint)((startCounter + 2) >> 32), (uint)((startCounter + 3) >> 32),
            (uint)((startCounter + 4) >> 32), (uint)((startCounter + 5) >> 32),
            (uint)((startCounter + 6) >> 32), (uint)((startCounter + 7) >> 32));

        var ivVec0 = Vector256.Create(Blake3Constants.IV[0]);
        var ivVec1 = Vector256.Create(Blake3Constants.IV[1]);
        var ivVec2 = Vector256.Create(Blake3Constants.IV[2]);
        var ivVec3 = Vector256.Create(Blake3Constants.IV[3]);
        var blockLenVec = Vector256.Create((uint)Blake3Constants.BlockLen);

        fixed (byte* chunksPtr = chunks)
        {
            Vector256<uint>* m = stackalloc Vector256<uint>[16];

            for (int blockIdx = 0; blockIdx < blocksPerChunk; blockIdx++)
            {
                byte* blockBase = chunksPtr + blockIdx * 64;

                // Load lower 8 words (0-7) from each of 8 chunks contiguously
                var r0 = Unsafe.ReadUnaligned<Vector256<uint>>(blockBase + 0 * Blake3Constants.ChunkLen);
                var r1 = Unsafe.ReadUnaligned<Vector256<uint>>(blockBase + 1 * Blake3Constants.ChunkLen);
                var r2 = Unsafe.ReadUnaligned<Vector256<uint>>(blockBase + 2 * Blake3Constants.ChunkLen);
                var r3 = Unsafe.ReadUnaligned<Vector256<uint>>(blockBase + 3 * Blake3Constants.ChunkLen);
                var r4 = Unsafe.ReadUnaligned<Vector256<uint>>(blockBase + 4 * Blake3Constants.ChunkLen);
                var r5 = Unsafe.ReadUnaligned<Vector256<uint>>(blockBase + 5 * Blake3Constants.ChunkLen);
                var r6 = Unsafe.ReadUnaligned<Vector256<uint>>(blockBase + 6 * Blake3Constants.ChunkLen);
                var r7 = Unsafe.ReadUnaligned<Vector256<uint>>(blockBase + 7 * Blake3Constants.ChunkLen);

                // 8x8 transpose: chunk-major -> word-major (words 0-7)
                var bt0 = Avx2.UnpackLow(r0, r1);
                var bt1 = Avx2.UnpackHigh(r0, r1);
                var bt2 = Avx2.UnpackLow(r2, r3);
                var bt3 = Avx2.UnpackHigh(r2, r3);
                var bt4 = Avx2.UnpackLow(r4, r5);
                var bt5 = Avx2.UnpackHigh(r4, r5);
                var bt6 = Avx2.UnpackLow(r6, r7);
                var bt7 = Avx2.UnpackHigh(r6, r7);

                var bu0 = Avx2.UnpackLow(bt0.AsUInt64(), bt2.AsUInt64()).AsUInt32();
                var bu1 = Avx2.UnpackHigh(bt0.AsUInt64(), bt2.AsUInt64()).AsUInt32();
                var bu2 = Avx2.UnpackLow(bt1.AsUInt64(), bt3.AsUInt64()).AsUInt32();
                var bu3 = Avx2.UnpackHigh(bt1.AsUInt64(), bt3.AsUInt64()).AsUInt32();
                var bu4 = Avx2.UnpackLow(bt4.AsUInt64(), bt6.AsUInt64()).AsUInt32();
                var bu5 = Avx2.UnpackHigh(bt4.AsUInt64(), bt6.AsUInt64()).AsUInt32();
                var bu6 = Avx2.UnpackLow(bt5.AsUInt64(), bt7.AsUInt64()).AsUInt32();
                var bu7 = Avx2.UnpackHigh(bt5.AsUInt64(), bt7.AsUInt64()).AsUInt32();

                m[0] = Avx2.Permute2x128(bu0, bu4, 0x20);
                m[1] = Avx2.Permute2x128(bu1, bu5, 0x20);
                m[2] = Avx2.Permute2x128(bu2, bu6, 0x20);
                m[3] = Avx2.Permute2x128(bu3, bu7, 0x20);
                m[4] = Avx2.Permute2x128(bu0, bu4, 0x31);
                m[5] = Avx2.Permute2x128(bu1, bu5, 0x31);
                m[6] = Avx2.Permute2x128(bu2, bu6, 0x31);
                m[7] = Avx2.Permute2x128(bu3, bu7, 0x31);

                // Load upper 8 words (8-15) from each of 8 chunks contiguously
                r0 = Unsafe.ReadUnaligned<Vector256<uint>>(blockBase + 0 * Blake3Constants.ChunkLen + 32);
                r1 = Unsafe.ReadUnaligned<Vector256<uint>>(blockBase + 1 * Blake3Constants.ChunkLen + 32);
                r2 = Unsafe.ReadUnaligned<Vector256<uint>>(blockBase + 2 * Blake3Constants.ChunkLen + 32);
                r3 = Unsafe.ReadUnaligned<Vector256<uint>>(blockBase + 3 * Blake3Constants.ChunkLen + 32);
                r4 = Unsafe.ReadUnaligned<Vector256<uint>>(blockBase + 4 * Blake3Constants.ChunkLen + 32);
                r5 = Unsafe.ReadUnaligned<Vector256<uint>>(blockBase + 5 * Blake3Constants.ChunkLen + 32);
                r6 = Unsafe.ReadUnaligned<Vector256<uint>>(blockBase + 6 * Blake3Constants.ChunkLen + 32);
                r7 = Unsafe.ReadUnaligned<Vector256<uint>>(blockBase + 7 * Blake3Constants.ChunkLen + 32);

                // 8x8 transpose: chunk-major -> word-major (words 8-15)
                bt0 = Avx2.UnpackLow(r0, r1);
                bt1 = Avx2.UnpackHigh(r0, r1);
                bt2 = Avx2.UnpackLow(r2, r3);
                bt3 = Avx2.UnpackHigh(r2, r3);
                bt4 = Avx2.UnpackLow(r4, r5);
                bt5 = Avx2.UnpackHigh(r4, r5);
                bt6 = Avx2.UnpackLow(r6, r7);
                bt7 = Avx2.UnpackHigh(r6, r7);

                bu0 = Avx2.UnpackLow(bt0.AsUInt64(), bt2.AsUInt64()).AsUInt32();
                bu1 = Avx2.UnpackHigh(bt0.AsUInt64(), bt2.AsUInt64()).AsUInt32();
                bu2 = Avx2.UnpackLow(bt1.AsUInt64(), bt3.AsUInt64()).AsUInt32();
                bu3 = Avx2.UnpackHigh(bt1.AsUInt64(), bt3.AsUInt64()).AsUInt32();
                bu4 = Avx2.UnpackLow(bt4.AsUInt64(), bt6.AsUInt64()).AsUInt32();
                bu5 = Avx2.UnpackHigh(bt4.AsUInt64(), bt6.AsUInt64()).AsUInt32();
                bu6 = Avx2.UnpackLow(bt5.AsUInt64(), bt7.AsUInt64()).AsUInt32();
                bu7 = Avx2.UnpackHigh(bt5.AsUInt64(), bt7.AsUInt64()).AsUInt32();

                m[8]  = Avx2.Permute2x128(bu0, bu4, 0x20);
                m[9]  = Avx2.Permute2x128(bu1, bu5, 0x20);
                m[10] = Avx2.Permute2x128(bu2, bu6, 0x20);
                m[11] = Avx2.Permute2x128(bu3, bu7, 0x20);
                m[12] = Avx2.Permute2x128(bu0, bu4, 0x31);
                m[13] = Avx2.Permute2x128(bu1, bu5, 0x31);
                m[14] = Avx2.Permute2x128(bu2, bu6, 0x31);
                m[15] = Avx2.Permute2x128(bu3, bu7, 0x31);

                // Block flags
                uint blockFlags = flags;
                if (blockIdx == 0) blockFlags |= Blake3Constants.ChunkStart;
                if (blockIdx == blocksPerChunk - 1) blockFlags |= Blake3Constants.ChunkEnd;
                var flagsVec = Vector256.Create(blockFlags);

                Vector256<uint> s0 = cv0, s1 = cv1, s2 = cv2, s3 = cv3;
                Vector256<uint> s4 = cv4, s5 = cv5, s6 = cv6, s7 = cv7;
                Vector256<uint> s8 = ivVec0, s9 = ivVec1, s10 = ivVec2, s11 = ivVec3;
                Vector256<uint> s12 = counterLo, s13 = counterHi;
                Vector256<uint> s14 = blockLenVec, s15 = flagsVec;

                // Round 0: 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
                G256(ref s0, ref s4, ref s8,  ref s12, m[0],  m[1]);
                G256(ref s1, ref s5, ref s9,  ref s13, m[2],  m[3]);
                G256(ref s2, ref s6, ref s10, ref s14, m[4],  m[5]);
                G256(ref s3, ref s7, ref s11, ref s15, m[6],  m[7]);
                G256(ref s0, ref s5, ref s10, ref s15, m[8],  m[9]);
                G256(ref s1, ref s6, ref s11, ref s12, m[10], m[11]);
                G256(ref s2, ref s7, ref s8,  ref s13, m[12], m[13]);
                G256(ref s3, ref s4, ref s9,  ref s14, m[14], m[15]);
                // Round 1: 2,6,3,10,7,0,4,13,1,11,12,5,9,14,15,8
                G256(ref s0, ref s4, ref s8,  ref s12, m[2],  m[6]);
                G256(ref s1, ref s5, ref s9,  ref s13, m[3],  m[10]);
                G256(ref s2, ref s6, ref s10, ref s14, m[7],  m[0]);
                G256(ref s3, ref s7, ref s11, ref s15, m[4],  m[13]);
                G256(ref s0, ref s5, ref s10, ref s15, m[1],  m[11]);
                G256(ref s1, ref s6, ref s11, ref s12, m[12], m[5]);
                G256(ref s2, ref s7, ref s8,  ref s13, m[9],  m[14]);
                G256(ref s3, ref s4, ref s9,  ref s14, m[15], m[8]);
                // Round 2: 3,4,10,12,13,2,7,14,6,5,9,0,11,15,8,1
                G256(ref s0, ref s4, ref s8,  ref s12, m[3],  m[4]);
                G256(ref s1, ref s5, ref s9,  ref s13, m[10], m[12]);
                G256(ref s2, ref s6, ref s10, ref s14, m[13], m[2]);
                G256(ref s3, ref s7, ref s11, ref s15, m[7],  m[14]);
                G256(ref s0, ref s5, ref s10, ref s15, m[6],  m[5]);
                G256(ref s1, ref s6, ref s11, ref s12, m[9],  m[0]);
                G256(ref s2, ref s7, ref s8,  ref s13, m[11], m[15]);
                G256(ref s3, ref s4, ref s9,  ref s14, m[8],  m[1]);
                // Round 3: 10,7,12,9,14,3,13,15,4,0,11,2,5,8,1,6
                G256(ref s0, ref s4, ref s8,  ref s12, m[10], m[7]);
                G256(ref s1, ref s5, ref s9,  ref s13, m[12], m[9]);
                G256(ref s2, ref s6, ref s10, ref s14, m[14], m[3]);
                G256(ref s3, ref s7, ref s11, ref s15, m[13], m[15]);
                G256(ref s0, ref s5, ref s10, ref s15, m[4],  m[0]);
                G256(ref s1, ref s6, ref s11, ref s12, m[11], m[2]);
                G256(ref s2, ref s7, ref s8,  ref s13, m[5],  m[8]);
                G256(ref s3, ref s4, ref s9,  ref s14, m[1],  m[6]);
                // Round 4: 12,13,9,11,15,10,14,8,7,2,5,3,0,1,6,4
                G256(ref s0, ref s4, ref s8,  ref s12, m[12], m[13]);
                G256(ref s1, ref s5, ref s9,  ref s13, m[9],  m[11]);
                G256(ref s2, ref s6, ref s10, ref s14, m[15], m[10]);
                G256(ref s3, ref s7, ref s11, ref s15, m[14], m[8]);
                G256(ref s0, ref s5, ref s10, ref s15, m[7],  m[2]);
                G256(ref s1, ref s6, ref s11, ref s12, m[5],  m[3]);
                G256(ref s2, ref s7, ref s8,  ref s13, m[0],  m[1]);
                G256(ref s3, ref s4, ref s9,  ref s14, m[6],  m[4]);
                // Round 5: 9,14,11,5,8,12,15,1,13,3,0,10,2,6,4,7
                G256(ref s0, ref s4, ref s8,  ref s12, m[9],  m[14]);
                G256(ref s1, ref s5, ref s9,  ref s13, m[11], m[5]);
                G256(ref s2, ref s6, ref s10, ref s14, m[8],  m[12]);
                G256(ref s3, ref s7, ref s11, ref s15, m[15], m[1]);
                G256(ref s0, ref s5, ref s10, ref s15, m[13], m[3]);
                G256(ref s1, ref s6, ref s11, ref s12, m[0],  m[10]);
                G256(ref s2, ref s7, ref s8,  ref s13, m[2],  m[6]);
                G256(ref s3, ref s4, ref s9,  ref s14, m[4],  m[7]);
                // Round 6: 11,15,5,0,1,9,8,6,14,10,2,12,3,4,7,13
                G256(ref s0, ref s4, ref s8,  ref s12, m[11], m[15]);
                G256(ref s1, ref s5, ref s9,  ref s13, m[5],  m[0]);
                G256(ref s2, ref s6, ref s10, ref s14, m[1],  m[9]);
                G256(ref s3, ref s7, ref s11, ref s15, m[8],  m[6]);
                G256(ref s0, ref s5, ref s10, ref s15, m[14], m[10]);
                G256(ref s1, ref s6, ref s11, ref s12, m[2],  m[12]);
                G256(ref s2, ref s7, ref s8,  ref s13, m[3],  m[4]);
                G256(ref s3, ref s4, ref s9,  ref s14, m[7],  m[13]);

                // Post-XOR: only chaining value (first 8 words)
                cv0 = Avx2.Xor(s0, s8);
                cv1 = Avx2.Xor(s1, s9);
                cv2 = Avx2.Xor(s2, s10);
                cv3 = Avx2.Xor(s3, s11);
                cv4 = Avx2.Xor(s4, s12);
                cv5 = Avx2.Xor(s5, s13);
                cv6 = Avx2.Xor(s6, s14);
                cv7 = Avx2.Xor(s7, s15);
            }
        }

        // 8x8 transpose: word-major to chunk-major
        var t0 = Avx2.UnpackLow(cv0, cv1);
        var t1 = Avx2.UnpackHigh(cv0, cv1);
        var t2 = Avx2.UnpackLow(cv2, cv3);
        var t3 = Avx2.UnpackHigh(cv2, cv3);
        var t4 = Avx2.UnpackLow(cv4, cv5);
        var t5 = Avx2.UnpackHigh(cv4, cv5);
        var t6 = Avx2.UnpackLow(cv6, cv7);
        var t7 = Avx2.UnpackHigh(cv6, cv7);

        var u0 = Avx2.UnpackLow(t0.AsUInt64(), t2.AsUInt64()).AsUInt32();
        var u1 = Avx2.UnpackHigh(t0.AsUInt64(), t2.AsUInt64()).AsUInt32();
        var u2 = Avx2.UnpackLow(t1.AsUInt64(), t3.AsUInt64()).AsUInt32();
        var u3 = Avx2.UnpackHigh(t1.AsUInt64(), t3.AsUInt64()).AsUInt32();
        var u4 = Avx2.UnpackLow(t4.AsUInt64(), t6.AsUInt64()).AsUInt32();
        var u5 = Avx2.UnpackHigh(t4.AsUInt64(), t6.AsUInt64()).AsUInt32();
        var u6 = Avx2.UnpackLow(t5.AsUInt64(), t7.AsUInt64()).AsUInt32();
        var u7 = Avx2.UnpackHigh(t5.AsUInt64(), t7.AsUInt64()).AsUInt32();

        ref uint outRef = ref MemoryMarshal.GetReference(cvs);
        VectorCompat.Store(Avx2.Permute2x128(u0, u4, 0x20), ref outRef);       // chunk 0
        VectorCompat.Store(Avx2.Permute2x128(u1, u5, 0x20), ref outRef, 8);    // chunk 1
        VectorCompat.Store(Avx2.Permute2x128(u2, u6, 0x20), ref outRef, 16);   // chunk 2
        VectorCompat.Store(Avx2.Permute2x128(u3, u7, 0x20), ref outRef, 24);   // chunk 3
        VectorCompat.Store(Avx2.Permute2x128(u0, u4, 0x31), ref outRef, 32);   // chunk 4
        VectorCompat.Store(Avx2.Permute2x128(u1, u5, 0x31), ref outRef, 40);   // chunk 5
        VectorCompat.Store(Avx2.Permute2x128(u2, u6, 0x31), ref outRef, 48);   // chunk 6
        VectorCompat.Store(Avx2.Permute2x128(u3, u7, 0x31), ref outRef, 56);   // chunk 7
    }
}
