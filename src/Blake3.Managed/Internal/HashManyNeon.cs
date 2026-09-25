using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;

namespace Blake3.Managed.Internal;

/// <summary>
/// ARM NEON kernels: four chunks (<see cref="HashMany"/>, <see cref="HashManyPartial"/>) or four
/// parent nodes (<see cref="HashParents4"/>) in the four 32-bit lanes.
/// </summary>
/// <remarks>
/// The seven rounds of those three kernels are written out statement by statement (generated),
/// in the shape that won an out-of-tree lab on 2026-09-25: each half-round is emitted step by
/// step across its four independent G's, the message word is added first -- (a + m) + b -- and
/// every round ends with a store to <c>m[16]</c> that stops the JIT hoisting all sixteen message
/// loads into registers. Against the previous body (a G helper taking the state by <c>ref</c>,
/// which spilled about 100 vectors per block) the 4-way kernel ran in <b>0.560</b> of the time on
/// Cortex-A73 and <b>0.683</b> on Neoverse-V1 (Graviton3), and the listing has no spills left.
/// Stepping is most of it on the A73 (0.574 alone): RyuJIT does not schedule, so source order is
/// emission order, and the A73 cannot look far enough ahead to overlap whole G's by itself.
/// <para>
/// This mattered more than a percentage: with the old body the kernels had fallen behind the
/// rewritten scalar compressor on the A73 (four scalar chunks 0.940, three 0.696 of the padded
/// kernel, four scalar parents 0.911), while on V1 they still won (scalar 1.35x slower). The new
/// body wins on both. The 8-bit rotate as shift-or instead of TBL was slower on both cores.
/// </para>
/// </remarks>
internal static class HashManyNeon
{
    public static bool IsSupported => AdvSimd.Arm64.IsSupported;

    private static readonly Vector128<byte> Rot16Mask128 = Vector128.Create(
        (byte)2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13);

    private static readonly Vector128<byte> Rot8Mask128 = Vector128.Create(
        (byte)1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<uint> RotateRight16(Vector128<uint> v)
    {
        return AdvSimd.ReverseElement16(v.AsInt32()).AsUInt32();
    }

    // Deliberately three ops rather than SRI. ShiftRightAndInsert is one instruction fewer,
    // but its destination is also a source, so it serialises behind the shift feeding it,
    // while the two shifts here are independent and dual-issue on the A73's two NEON pipes.
    // Measured on a Cortex-A73 (2026-09-19): SRI was 3-5% slower at every size from 4 KB up.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<uint> RotateRight12(Vector128<uint> v)
    {
        return AdvSimd.Or(AdvSimd.ShiftRightLogical(v, 12), AdvSimd.ShiftLeftLogical(v, 20));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<uint> RotateRight8(Vector128<uint> v)
    {
        return AdvSimd.Arm64.VectorTableLookup(v.AsByte(), Rot8Mask128).AsUInt32();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<uint> RotateRight7(Vector128<uint> v)
    {
        return AdvSimd.Or(AdvSimd.ShiftRightLogical(v, 7), AdvSimd.ShiftLeftLogical(v, 25));
    }

    /// <summary>
    /// 4x4 transpose: converts 4 rows where each lane is from a different chunk
    /// into 4 columns where each lane is a word from the same chunk.
    /// Uses ZipLow/ZipHigh pairs.
    /// Input:  r0={a0,b0,c0,d0} r1={a1,b1,c1,d1} r2={a2,b2,c2,d2} r3={a3,b3,c3,d3}
    /// Output: m0={a0,a1,a2,a3} m1={b0,b1,b2,b3} m2={c0,c1,c2,c3} m3={d0,d1,d2,d3}
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Transpose4X4(
        Vector128<uint> r0, Vector128<uint> r1,
        Vector128<uint> r2, Vector128<uint> r3,
        out Vector128<uint> m0, out Vector128<uint> m1,
        out Vector128<uint> m2, out Vector128<uint> m3)
    {
        var t0 = AdvSimd.Arm64.ZipLow(r0, r2);   // {r0[0],r2[0],r0[1],r2[1]}
        var t1 = AdvSimd.Arm64.ZipHigh(r0, r2);   // {r0[2],r2[2],r0[3],r2[3]}
        var t2 = AdvSimd.Arm64.ZipLow(r1, r3);    // {r1[0],r3[0],r1[1],r3[1]}
        var t3 = AdvSimd.Arm64.ZipHigh(r1, r3);   // {r1[2],r3[2],r1[3],r3[3]}
        m0 = AdvSimd.Arm64.ZipLow(t0, t2);        // {r0[0],r1[0],r2[0],r3[0]}
        m1 = AdvSimd.Arm64.ZipHigh(t0, t2);       // {r0[1],r1[1],r2[1],r3[1]}
        m2 = AdvSimd.Arm64.ZipLow(t1, t3);        // {r0[2],r1[2],r2[2],r3[2]}
        m3 = AdvSimd.Arm64.ZipHigh(t1, t3);       // {r0[3],r1[3],r2[3],r3[3]}
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void G128P(Vector128<uint>* a, Vector128<uint>* b,
                                     Vector128<uint>* c, Vector128<uint>* d,
                                     Vector128<uint> mx, Vector128<uint> my)
    {
        *a = AdvSimd.Add(AdvSimd.Add(*a, *b), mx);
        *d = RotateRight16(AdvSimd.Xor(*d, *a));
        *c = AdvSimd.Add(*c, *d);
        *b = RotateRight12(AdvSimd.Xor(*b, *c));
        *a = AdvSimd.Add(AdvSimd.Add(*a, *b), my);
        *d = RotateRight8(AdvSimd.Xor(*d, *a));
        *c = AdvSimd.Add(*c, *d);
        *b = RotateRight7(AdvSimd.Xor(*b, *c));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void DoColumnStep(Vector128<uint>* s, Vector128<uint>* m,
        byte i0, byte i1, byte i2, byte i3, byte i4, byte i5, byte i6, byte i7)
    {
        G128P(s+0, s+4, s+8,  s+12, m[i0], m[i1]);
        G128P(s+1, s+5, s+9,  s+13, m[i2], m[i3]);
        G128P(s+2, s+6, s+10, s+14, m[i4], m[i5]);
        G128P(s+3, s+7, s+11, s+15, m[i6], m[i7]);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void DoDiagonalStep(Vector128<uint>* s, Vector128<uint>* m,
        byte i0, byte i1, byte i2, byte i3, byte i4, byte i5, byte i6, byte i7)
    {
        G128P(s+0, s+5, s+10, s+15, m[i0], m[i1]);
        G128P(s+1, s+6, s+11, s+12, m[i2], m[i3]);
        G128P(s+2, s+7, s+8,  s+13, m[i4], m[i5]);
        G128P(s+3, s+4, s+9,  s+14, m[i6], m[i7]);
    }

    /// <summary>
    /// Not called anywhere. Wiring it in measured 2.8x slower at 8 KB on Cortex-A73 (2026-09-21),
    /// which was put down to register spilling; it is in fact the JIT's inlining budget. Built from
    /// nested helpers (<see cref="DoColumnStep"/> -> <see cref="G128P"/> -> rotates), it runs out
    /// partway through: its listing has 33 real calls on both Cortex-A73 and Neoverse-V1, and it
    /// ran 2.9x and 2.5x slower than two <see cref="HashMany"/> calls there (2026-09-25). Written
    /// out statement by statement, an 8-chunk two-chain kernel still did not beat two calls of the
    /// current 4-way kernel on either core (0.754 against 0.556 on A73; 0.68-0.74 against 0.682 on
    /// V1, bimodal across processes), so the 4-way kernel stays the leaf.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static unsafe void HashMany8(ReadOnlySpan<byte> chunks,
                                        ReadOnlySpan<uint> key, ulong startCounter,
                                        uint flags, Span<uint> cvs)
    {
        const int blocksPerChunk = Blake3Constants.ChunkLen / Blake3Constants.BlockLen;

        // Two sets of state: A (chunks 0-3), B (chunks 4-7)
        Vector128<uint>* sA = stackalloc Vector128<uint>[16];
        Vector128<uint>* sB = stackalloc Vector128<uint>[16];
        Vector128<uint>* mA = stackalloc Vector128<uint>[16];
        Vector128<uint>* mB = stackalloc Vector128<uint>[16];

        // Initialize CVs for both sets (same key)
        var kv0 = Vector128.Create(key[0]); var kv1 = Vector128.Create(key[1]);
        var kv2 = Vector128.Create(key[2]); var kv3 = Vector128.Create(key[3]);
        var kv4 = Vector128.Create(key[4]); var kv5 = Vector128.Create(key[5]);
        var kv6 = Vector128.Create(key[6]); var kv7 = Vector128.Create(key[7]);

        // These are the running CV state across blocks
        Vector128<uint> cvA0=kv0, cvA1=kv1, cvA2=kv2, cvA3=kv3;
        Vector128<uint> cvA4=kv4, cvA5=kv5, cvA6=kv6, cvA7=kv7;
        Vector128<uint> cvB0=kv0, cvB1=kv1, cvB2=kv2, cvB3=kv3;
        Vector128<uint> cvB4=kv4, cvB5=kv5, cvB6=kv6, cvB7=kv7;

        var counterLoA = Vector128.Create(
            (uint)(startCounter+0), (uint)(startCounter+1),
            (uint)(startCounter+2), (uint)(startCounter+3));
        var counterHiA = Vector128.Create(
            (uint)((startCounter+0)>>32), (uint)((startCounter+1)>>32),
            (uint)((startCounter+2)>>32), (uint)((startCounter+3)>>32));
        var counterLoB = Vector128.Create(
            (uint)(startCounter+4), (uint)(startCounter+5),
            (uint)(startCounter+6), (uint)(startCounter+7));
        var counterHiB = Vector128.Create(
            (uint)((startCounter+4)>>32), (uint)((startCounter+5)>>32),
            (uint)((startCounter+6)>>32), (uint)((startCounter+7)>>32));

        var iv0 = Vector128.Create(Blake3Constants.Iv0);
        var iv1 = Vector128.Create(Blake3Constants.Iv1);
        var iv2 = Vector128.Create(Blake3Constants.Iv2);
        var iv3 = Vector128.Create(Blake3Constants.Iv3);
        var blkVec = Vector128.Create((uint)Blake3Constants.BlockLen);

        fixed (byte* chunksPtr = chunks)
        {
            for (int blockIdx = 0; blockIdx < blocksPerChunk; blockIdx++)
            {
                byte* blockBase = chunksPtr + blockIdx * 64;

                // Load + transpose messages for set A (chunks 0-3)
                LoadTranspose(blockBase, 0, mA);
                // Load + transpose messages for set B (chunks 4-7)
                LoadTranspose(blockBase, 4 * Blake3Constants.ChunkLen, mB);

                uint blockFlags = flags;
                if (blockIdx == 0) blockFlags |= Blake3Constants.ChunkStart;
                if (blockIdx == blocksPerChunk - 1) blockFlags |= Blake3Constants.ChunkEnd;
                var fv = Vector128.Create(blockFlags);

                // Init state A
                sA[0]=cvA0; sA[1]=cvA1; sA[2]=cvA2; sA[3]=cvA3;
                sA[4]=cvA4; sA[5]=cvA5; sA[6]=cvA6; sA[7]=cvA7;
                sA[8]=iv0; sA[9]=iv1; sA[10]=iv2; sA[11]=iv3;
                sA[12]=counterLoA; sA[13]=counterHiA; sA[14]=blkVec; sA[15]=fv;

                // Init state B
                sB[0]=cvB0; sB[1]=cvB1; sB[2]=cvB2; sB[3]=cvB3;
                sB[4]=cvB4; sB[5]=cvB5; sB[6]=cvB6; sB[7]=cvB7;
                sB[8]=iv0; sB[9]=iv1; sB[10]=iv2; sB[11]=iv3;
                sB[12]=counterLoB; sB[13]=counterHiB; sB[14]=blkVec; sB[15]=fv;

                // 7 rounds, interleaved column-A/column-B/diagonal-A/diagonal-B
                // Round 0
                DoColumnStep(sA, mA, 0,1,2,3,4,5,6,7);
                DoColumnStep(sB, mB, 0,1,2,3,4,5,6,7);
                DoDiagonalStep(sA, mA, 8,9,10,11,12,13,14,15);
                DoDiagonalStep(sB, mB, 8,9,10,11,12,13,14,15);
                // Round 1
                DoColumnStep(sA, mA, 2,6,3,10,7,0,4,13);
                DoColumnStep(sB, mB, 2,6,3,10,7,0,4,13);
                DoDiagonalStep(sA, mA, 1,11,12,5,9,14,15,8);
                DoDiagonalStep(sB, mB, 1,11,12,5,9,14,15,8);
                // Round 2
                DoColumnStep(sA, mA, 3,4,10,12,13,2,7,14);
                DoColumnStep(sB, mB, 3,4,10,12,13,2,7,14);
                DoDiagonalStep(sA, mA, 6,5,9,0,11,15,8,1);
                DoDiagonalStep(sB, mB, 6,5,9,0,11,15,8,1);
                // Round 3
                DoColumnStep(sA, mA, 10,7,12,9,14,3,13,15);
                DoColumnStep(sB, mB, 10,7,12,9,14,3,13,15);
                DoDiagonalStep(sA, mA, 4,0,11,2,5,8,1,6);
                DoDiagonalStep(sB, mB, 4,0,11,2,5,8,1,6);
                // Round 4
                DoColumnStep(sA, mA, 12,13,9,11,15,10,14,8);
                DoColumnStep(sB, mB, 12,13,9,11,15,10,14,8);
                DoDiagonalStep(sA, mA, 7,2,5,3,0,1,6,4);
                DoDiagonalStep(sB, mB, 7,2,5,3,0,1,6,4);
                // Round 5
                DoColumnStep(sA, mA, 9,14,11,5,8,12,15,1);
                DoColumnStep(sB, mB, 9,14,11,5,8,12,15,1);
                DoDiagonalStep(sA, mA, 13,3,0,10,2,6,4,7);
                DoDiagonalStep(sB, mB, 13,3,0,10,2,6,4,7);
                // Round 6
                DoColumnStep(sA, mA, 11,15,5,0,1,9,8,6);
                DoColumnStep(sB, mB, 11,15,5,0,1,9,8,6);
                DoDiagonalStep(sA, mA, 14,10,2,12,3,4,7,13);
                DoDiagonalStep(sB, mB, 14,10,2,12,3,4,7,13);

                // Post-XOR
                cvA0=AdvSimd.Xor(sA[0],sA[8]);  cvA1=AdvSimd.Xor(sA[1],sA[9]);
                cvA2=AdvSimd.Xor(sA[2],sA[10]); cvA3=AdvSimd.Xor(sA[3],sA[11]);
                cvA4=AdvSimd.Xor(sA[4],sA[12]); cvA5=AdvSimd.Xor(sA[5],sA[13]);
                cvA6=AdvSimd.Xor(sA[6],sA[14]); cvA7=AdvSimd.Xor(sA[7],sA[15]);

                cvB0=AdvSimd.Xor(sB[0],sB[8]);  cvB1=AdvSimd.Xor(sB[1],sB[9]);
                cvB2=AdvSimd.Xor(sB[2],sB[10]); cvB3=AdvSimd.Xor(sB[3],sB[11]);
                cvB4=AdvSimd.Xor(sB[4],sB[12]); cvB5=AdvSimd.Xor(sB[5],sB[13]);
                cvB6=AdvSimd.Xor(sB[6],sB[14]); cvB7=AdvSimd.Xor(sB[7],sB[15]);
            }
        }

        // Transpose and store set A (chunks 0-3)
        Transpose4X4(cvA0, cvA1, cvA2, cvA3, out var oA0, out var oA1, out var oA2, out var oA3);
        Transpose4X4(cvA4, cvA5, cvA6, cvA7, out var oA4, out var oA5, out var oA6, out var oA7);
        // Transpose and store set B (chunks 4-7)
        Transpose4X4(cvB0, cvB1, cvB2, cvB3, out var oB0, out var oB1, out var oB2, out var oB3);
        Transpose4X4(cvB4, cvB5, cvB6, cvB7, out var oB4, out var oB5, out var oB6, out var oB7);

        ref uint outRef = ref MemoryMarshal.GetReference(cvs);
        VectorCompat.Store(oA0, ref outRef);      VectorCompat.Store(oA4, ref outRef, 4);   // chunk 0
        VectorCompat.Store(oA1, ref outRef, 8);   VectorCompat.Store(oA5, ref outRef, 12);  // chunk 1
        VectorCompat.Store(oA2, ref outRef, 16);  VectorCompat.Store(oA6, ref outRef, 20);  // chunk 2
        VectorCompat.Store(oA3, ref outRef, 24);  VectorCompat.Store(oA7, ref outRef, 28);  // chunk 3
        VectorCompat.Store(oB0, ref outRef, 32);  VectorCompat.Store(oB4, ref outRef, 36);  // chunk 4
        VectorCompat.Store(oB1, ref outRef, 40);  VectorCompat.Store(oB5, ref outRef, 44);  // chunk 5
        VectorCompat.Store(oB2, ref outRef, 48);  VectorCompat.Store(oB6, ref outRef, 52);  // chunk 6
        VectorCompat.Store(oB3, ref outRef, 56);  VectorCompat.Store(oB7, ref outRef, 60);  // chunk 7
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void LoadTranspose(byte* blockBase, int chunkOffset,
                                             Vector128<uint>* m)
    {
        var r0 = Unsafe.ReadUnaligned<Vector128<uint>>(blockBase + chunkOffset + 0 * Blake3Constants.ChunkLen);
        var r1 = Unsafe.ReadUnaligned<Vector128<uint>>(blockBase + chunkOffset + 1 * Blake3Constants.ChunkLen);
        var r2 = Unsafe.ReadUnaligned<Vector128<uint>>(blockBase + chunkOffset + 2 * Blake3Constants.ChunkLen);
        var r3 = Unsafe.ReadUnaligned<Vector128<uint>>(blockBase + chunkOffset + 3 * Blake3Constants.ChunkLen);
        Transpose4X4(r0, r1, r2, r3, out m[0], out m[1], out m[2], out m[3]);

        r0 = Unsafe.ReadUnaligned<Vector128<uint>>(blockBase + chunkOffset + 0 * Blake3Constants.ChunkLen + 16);
        r1 = Unsafe.ReadUnaligned<Vector128<uint>>(blockBase + chunkOffset + 1 * Blake3Constants.ChunkLen + 16);
        r2 = Unsafe.ReadUnaligned<Vector128<uint>>(blockBase + chunkOffset + 2 * Blake3Constants.ChunkLen + 16);
        r3 = Unsafe.ReadUnaligned<Vector128<uint>>(blockBase + chunkOffset + 3 * Blake3Constants.ChunkLen + 16);
        Transpose4X4(r0, r1, r2, r3, out m[4], out m[5], out m[6], out m[7]);

        r0 = Unsafe.ReadUnaligned<Vector128<uint>>(blockBase + chunkOffset + 0 * Blake3Constants.ChunkLen + 32);
        r1 = Unsafe.ReadUnaligned<Vector128<uint>>(blockBase + chunkOffset + 1 * Blake3Constants.ChunkLen + 32);
        r2 = Unsafe.ReadUnaligned<Vector128<uint>>(blockBase + chunkOffset + 2 * Blake3Constants.ChunkLen + 32);
        r3 = Unsafe.ReadUnaligned<Vector128<uint>>(blockBase + chunkOffset + 3 * Blake3Constants.ChunkLen + 32);
        Transpose4X4(r0, r1, r2, r3, out m[8], out m[9], out m[10], out m[11]);

        r0 = Unsafe.ReadUnaligned<Vector128<uint>>(blockBase + chunkOffset + 0 * Blake3Constants.ChunkLen + 48);
        r1 = Unsafe.ReadUnaligned<Vector128<uint>>(blockBase + chunkOffset + 1 * Blake3Constants.ChunkLen + 48);
        r2 = Unsafe.ReadUnaligned<Vector128<uint>>(blockBase + chunkOffset + 2 * Blake3Constants.ChunkLen + 48);
        r3 = Unsafe.ReadUnaligned<Vector128<uint>>(blockBase + chunkOffset + 3 * Blake3Constants.ChunkLen + 48);
        Transpose4X4(r0, r1, r2, r3, out m[12], out m[13], out m[14], out m[15]);
    }

    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static unsafe void HashMany(ReadOnlySpan<byte> chunks, int numChunks,
                                       ReadOnlySpan<uint> key, ulong startCounter,
                                       uint flags, Span<uint> cvs)
    {
#if NET10_0_OR_GREATER
        if (HashManySve2.IsSupported)
        {
            HashManySve2.HashMany(chunks, numChunks, key, startCounter, flags, cvs);
            return;
        }
#endif
        var rot8Mask = Vector128.Create((byte)1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12);
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
            Vector128<uint>* m = stackalloc Vector128<uint>[17]; // [16]: the per-round barrier slot

            for (int blockIdx = 0; blockIdx < blocksPerChunk; blockIdx++)
            {
                byte* blockBase = chunksPtr + blockIdx * 64;

                // Load 4 words (16 bytes) from each of 4 chunks, then transpose
                // Lower 8 words (0-7): two groups of 4
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

                // Round 0
                s0 = AdvSimd.Add(AdvSimd.Add(s0, m[0]), s4); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[2]), s5); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[4]), s6); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[6]), s7);
                s12 = AdvSimd.ReverseElement16(AdvSimd.Xor(s12, s0).AsInt32()).AsUInt32(); s13 = AdvSimd.ReverseElement16(AdvSimd.Xor(s13, s1).AsInt32()).AsUInt32(); s14 = AdvSimd.ReverseElement16(AdvSimd.Xor(s14, s2).AsInt32()).AsUInt32(); s15 = AdvSimd.ReverseElement16(AdvSimd.Xor(s15, s3).AsInt32()).AsUInt32();
                s8 = AdvSimd.Add(s8, s12); s9 = AdvSimd.Add(s9, s13); s10 = AdvSimd.Add(s10, s14); s11 = AdvSimd.Add(s11, s15);
                { var t = AdvSimd.Xor(s4, s8); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s5, s9); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s6, s10); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s7, s11); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); }
                s0 = AdvSimd.Add(AdvSimd.Add(s0, m[1]), s4); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[3]), s5); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[5]), s6); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[7]), s7);
                s12 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s12, s0).AsByte(), rot8Mask).AsUInt32(); s13 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s13, s1).AsByte(), rot8Mask).AsUInt32(); s14 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s14, s2).AsByte(), rot8Mask).AsUInt32(); s15 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s15, s3).AsByte(), rot8Mask).AsUInt32();
                s8 = AdvSimd.Add(s8, s12); s9 = AdvSimd.Add(s9, s13); s10 = AdvSimd.Add(s10, s14); s11 = AdvSimd.Add(s11, s15);
                { var t = AdvSimd.Xor(s4, s8); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s5, s9); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s6, s10); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s7, s11); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); }
                s0 = AdvSimd.Add(AdvSimd.Add(s0, m[8]), s5); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[10]), s6); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[12]), s7); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[14]), s4);
                s15 = AdvSimd.ReverseElement16(AdvSimd.Xor(s15, s0).AsInt32()).AsUInt32(); s12 = AdvSimd.ReverseElement16(AdvSimd.Xor(s12, s1).AsInt32()).AsUInt32(); s13 = AdvSimd.ReverseElement16(AdvSimd.Xor(s13, s2).AsInt32()).AsUInt32(); s14 = AdvSimd.ReverseElement16(AdvSimd.Xor(s14, s3).AsInt32()).AsUInt32();
                s10 = AdvSimd.Add(s10, s15); s11 = AdvSimd.Add(s11, s12); s8 = AdvSimd.Add(s8, s13); s9 = AdvSimd.Add(s9, s14);
                { var t = AdvSimd.Xor(s5, s10); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s6, s11); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s7, s8); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s4, s9); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); }
                s0 = AdvSimd.Add(AdvSimd.Add(s0, m[9]), s5); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[11]), s6); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[13]), s7); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[15]), s4);
                s15 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s15, s0).AsByte(), rot8Mask).AsUInt32(); s12 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s12, s1).AsByte(), rot8Mask).AsUInt32(); s13 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s13, s2).AsByte(), rot8Mask).AsUInt32(); s14 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s14, s3).AsByte(), rot8Mask).AsUInt32();
                s10 = AdvSimd.Add(s10, s15); s11 = AdvSimd.Add(s11, s12); s8 = AdvSimd.Add(s8, s13); s9 = AdvSimd.Add(s9, s14);
                { var t = AdvSimd.Xor(s5, s10); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s6, s11); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s7, s8); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s4, s9); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); }
                m[16] = s0; // barrier: stops the message loads being hoisted
                // Round 1
                s0 = AdvSimd.Add(AdvSimd.Add(s0, m[2]), s4); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[3]), s5); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[7]), s6); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[4]), s7);
                s12 = AdvSimd.ReverseElement16(AdvSimd.Xor(s12, s0).AsInt32()).AsUInt32(); s13 = AdvSimd.ReverseElement16(AdvSimd.Xor(s13, s1).AsInt32()).AsUInt32(); s14 = AdvSimd.ReverseElement16(AdvSimd.Xor(s14, s2).AsInt32()).AsUInt32(); s15 = AdvSimd.ReverseElement16(AdvSimd.Xor(s15, s3).AsInt32()).AsUInt32();
                s8 = AdvSimd.Add(s8, s12); s9 = AdvSimd.Add(s9, s13); s10 = AdvSimd.Add(s10, s14); s11 = AdvSimd.Add(s11, s15);
                { var t = AdvSimd.Xor(s4, s8); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s5, s9); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s6, s10); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s7, s11); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); }
                s0 = AdvSimd.Add(AdvSimd.Add(s0, m[6]), s4); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[10]), s5); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[0]), s6); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[13]), s7);
                s12 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s12, s0).AsByte(), rot8Mask).AsUInt32(); s13 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s13, s1).AsByte(), rot8Mask).AsUInt32(); s14 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s14, s2).AsByte(), rot8Mask).AsUInt32(); s15 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s15, s3).AsByte(), rot8Mask).AsUInt32();
                s8 = AdvSimd.Add(s8, s12); s9 = AdvSimd.Add(s9, s13); s10 = AdvSimd.Add(s10, s14); s11 = AdvSimd.Add(s11, s15);
                { var t = AdvSimd.Xor(s4, s8); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s5, s9); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s6, s10); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s7, s11); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); }
                s0 = AdvSimd.Add(AdvSimd.Add(s0, m[1]), s5); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[12]), s6); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[9]), s7); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[15]), s4);
                s15 = AdvSimd.ReverseElement16(AdvSimd.Xor(s15, s0).AsInt32()).AsUInt32(); s12 = AdvSimd.ReverseElement16(AdvSimd.Xor(s12, s1).AsInt32()).AsUInt32(); s13 = AdvSimd.ReverseElement16(AdvSimd.Xor(s13, s2).AsInt32()).AsUInt32(); s14 = AdvSimd.ReverseElement16(AdvSimd.Xor(s14, s3).AsInt32()).AsUInt32();
                s10 = AdvSimd.Add(s10, s15); s11 = AdvSimd.Add(s11, s12); s8 = AdvSimd.Add(s8, s13); s9 = AdvSimd.Add(s9, s14);
                { var t = AdvSimd.Xor(s5, s10); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s6, s11); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s7, s8); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s4, s9); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); }
                s0 = AdvSimd.Add(AdvSimd.Add(s0, m[11]), s5); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[5]), s6); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[14]), s7); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[8]), s4);
                s15 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s15, s0).AsByte(), rot8Mask).AsUInt32(); s12 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s12, s1).AsByte(), rot8Mask).AsUInt32(); s13 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s13, s2).AsByte(), rot8Mask).AsUInt32(); s14 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s14, s3).AsByte(), rot8Mask).AsUInt32();
                s10 = AdvSimd.Add(s10, s15); s11 = AdvSimd.Add(s11, s12); s8 = AdvSimd.Add(s8, s13); s9 = AdvSimd.Add(s9, s14);
                { var t = AdvSimd.Xor(s5, s10); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s6, s11); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s7, s8); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s4, s9); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); }
                m[16] = s0; // barrier: stops the message loads being hoisted
                // Round 2
                s0 = AdvSimd.Add(AdvSimd.Add(s0, m[3]), s4); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[10]), s5); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[13]), s6); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[7]), s7);
                s12 = AdvSimd.ReverseElement16(AdvSimd.Xor(s12, s0).AsInt32()).AsUInt32(); s13 = AdvSimd.ReverseElement16(AdvSimd.Xor(s13, s1).AsInt32()).AsUInt32(); s14 = AdvSimd.ReverseElement16(AdvSimd.Xor(s14, s2).AsInt32()).AsUInt32(); s15 = AdvSimd.ReverseElement16(AdvSimd.Xor(s15, s3).AsInt32()).AsUInt32();
                s8 = AdvSimd.Add(s8, s12); s9 = AdvSimd.Add(s9, s13); s10 = AdvSimd.Add(s10, s14); s11 = AdvSimd.Add(s11, s15);
                { var t = AdvSimd.Xor(s4, s8); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s5, s9); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s6, s10); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s7, s11); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); }
                s0 = AdvSimd.Add(AdvSimd.Add(s0, m[4]), s4); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[12]), s5); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[2]), s6); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[14]), s7);
                s12 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s12, s0).AsByte(), rot8Mask).AsUInt32(); s13 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s13, s1).AsByte(), rot8Mask).AsUInt32(); s14 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s14, s2).AsByte(), rot8Mask).AsUInt32(); s15 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s15, s3).AsByte(), rot8Mask).AsUInt32();
                s8 = AdvSimd.Add(s8, s12); s9 = AdvSimd.Add(s9, s13); s10 = AdvSimd.Add(s10, s14); s11 = AdvSimd.Add(s11, s15);
                { var t = AdvSimd.Xor(s4, s8); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s5, s9); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s6, s10); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s7, s11); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); }
                s0 = AdvSimd.Add(AdvSimd.Add(s0, m[6]), s5); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[9]), s6); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[11]), s7); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[8]), s4);
                s15 = AdvSimd.ReverseElement16(AdvSimd.Xor(s15, s0).AsInt32()).AsUInt32(); s12 = AdvSimd.ReverseElement16(AdvSimd.Xor(s12, s1).AsInt32()).AsUInt32(); s13 = AdvSimd.ReverseElement16(AdvSimd.Xor(s13, s2).AsInt32()).AsUInt32(); s14 = AdvSimd.ReverseElement16(AdvSimd.Xor(s14, s3).AsInt32()).AsUInt32();
                s10 = AdvSimd.Add(s10, s15); s11 = AdvSimd.Add(s11, s12); s8 = AdvSimd.Add(s8, s13); s9 = AdvSimd.Add(s9, s14);
                { var t = AdvSimd.Xor(s5, s10); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s6, s11); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s7, s8); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s4, s9); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); }
                s0 = AdvSimd.Add(AdvSimd.Add(s0, m[5]), s5); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[0]), s6); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[15]), s7); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[1]), s4);
                s15 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s15, s0).AsByte(), rot8Mask).AsUInt32(); s12 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s12, s1).AsByte(), rot8Mask).AsUInt32(); s13 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s13, s2).AsByte(), rot8Mask).AsUInt32(); s14 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s14, s3).AsByte(), rot8Mask).AsUInt32();
                s10 = AdvSimd.Add(s10, s15); s11 = AdvSimd.Add(s11, s12); s8 = AdvSimd.Add(s8, s13); s9 = AdvSimd.Add(s9, s14);
                { var t = AdvSimd.Xor(s5, s10); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s6, s11); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s7, s8); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s4, s9); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); }
                m[16] = s0; // barrier: stops the message loads being hoisted
                // Round 3
                s0 = AdvSimd.Add(AdvSimd.Add(s0, m[10]), s4); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[12]), s5); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[14]), s6); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[13]), s7);
                s12 = AdvSimd.ReverseElement16(AdvSimd.Xor(s12, s0).AsInt32()).AsUInt32(); s13 = AdvSimd.ReverseElement16(AdvSimd.Xor(s13, s1).AsInt32()).AsUInt32(); s14 = AdvSimd.ReverseElement16(AdvSimd.Xor(s14, s2).AsInt32()).AsUInt32(); s15 = AdvSimd.ReverseElement16(AdvSimd.Xor(s15, s3).AsInt32()).AsUInt32();
                s8 = AdvSimd.Add(s8, s12); s9 = AdvSimd.Add(s9, s13); s10 = AdvSimd.Add(s10, s14); s11 = AdvSimd.Add(s11, s15);
                { var t = AdvSimd.Xor(s4, s8); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s5, s9); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s6, s10); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s7, s11); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); }
                s0 = AdvSimd.Add(AdvSimd.Add(s0, m[7]), s4); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[9]), s5); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[3]), s6); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[15]), s7);
                s12 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s12, s0).AsByte(), rot8Mask).AsUInt32(); s13 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s13, s1).AsByte(), rot8Mask).AsUInt32(); s14 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s14, s2).AsByte(), rot8Mask).AsUInt32(); s15 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s15, s3).AsByte(), rot8Mask).AsUInt32();
                s8 = AdvSimd.Add(s8, s12); s9 = AdvSimd.Add(s9, s13); s10 = AdvSimd.Add(s10, s14); s11 = AdvSimd.Add(s11, s15);
                { var t = AdvSimd.Xor(s4, s8); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s5, s9); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s6, s10); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s7, s11); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); }
                s0 = AdvSimd.Add(AdvSimd.Add(s0, m[4]), s5); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[11]), s6); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[5]), s7); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[1]), s4);
                s15 = AdvSimd.ReverseElement16(AdvSimd.Xor(s15, s0).AsInt32()).AsUInt32(); s12 = AdvSimd.ReverseElement16(AdvSimd.Xor(s12, s1).AsInt32()).AsUInt32(); s13 = AdvSimd.ReverseElement16(AdvSimd.Xor(s13, s2).AsInt32()).AsUInt32(); s14 = AdvSimd.ReverseElement16(AdvSimd.Xor(s14, s3).AsInt32()).AsUInt32();
                s10 = AdvSimd.Add(s10, s15); s11 = AdvSimd.Add(s11, s12); s8 = AdvSimd.Add(s8, s13); s9 = AdvSimd.Add(s9, s14);
                { var t = AdvSimd.Xor(s5, s10); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s6, s11); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s7, s8); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s4, s9); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); }
                s0 = AdvSimd.Add(AdvSimd.Add(s0, m[0]), s5); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[2]), s6); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[8]), s7); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[6]), s4);
                s15 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s15, s0).AsByte(), rot8Mask).AsUInt32(); s12 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s12, s1).AsByte(), rot8Mask).AsUInt32(); s13 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s13, s2).AsByte(), rot8Mask).AsUInt32(); s14 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s14, s3).AsByte(), rot8Mask).AsUInt32();
                s10 = AdvSimd.Add(s10, s15); s11 = AdvSimd.Add(s11, s12); s8 = AdvSimd.Add(s8, s13); s9 = AdvSimd.Add(s9, s14);
                { var t = AdvSimd.Xor(s5, s10); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s6, s11); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s7, s8); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s4, s9); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); }
                m[16] = s0; // barrier: stops the message loads being hoisted
                // Round 4
                s0 = AdvSimd.Add(AdvSimd.Add(s0, m[12]), s4); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[9]), s5); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[15]), s6); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[14]), s7);
                s12 = AdvSimd.ReverseElement16(AdvSimd.Xor(s12, s0).AsInt32()).AsUInt32(); s13 = AdvSimd.ReverseElement16(AdvSimd.Xor(s13, s1).AsInt32()).AsUInt32(); s14 = AdvSimd.ReverseElement16(AdvSimd.Xor(s14, s2).AsInt32()).AsUInt32(); s15 = AdvSimd.ReverseElement16(AdvSimd.Xor(s15, s3).AsInt32()).AsUInt32();
                s8 = AdvSimd.Add(s8, s12); s9 = AdvSimd.Add(s9, s13); s10 = AdvSimd.Add(s10, s14); s11 = AdvSimd.Add(s11, s15);
                { var t = AdvSimd.Xor(s4, s8); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s5, s9); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s6, s10); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s7, s11); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); }
                s0 = AdvSimd.Add(AdvSimd.Add(s0, m[13]), s4); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[11]), s5); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[10]), s6); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[8]), s7);
                s12 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s12, s0).AsByte(), rot8Mask).AsUInt32(); s13 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s13, s1).AsByte(), rot8Mask).AsUInt32(); s14 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s14, s2).AsByte(), rot8Mask).AsUInt32(); s15 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s15, s3).AsByte(), rot8Mask).AsUInt32();
                s8 = AdvSimd.Add(s8, s12); s9 = AdvSimd.Add(s9, s13); s10 = AdvSimd.Add(s10, s14); s11 = AdvSimd.Add(s11, s15);
                { var t = AdvSimd.Xor(s4, s8); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s5, s9); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s6, s10); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s7, s11); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); }
                s0 = AdvSimd.Add(AdvSimd.Add(s0, m[7]), s5); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[5]), s6); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[0]), s7); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[6]), s4);
                s15 = AdvSimd.ReverseElement16(AdvSimd.Xor(s15, s0).AsInt32()).AsUInt32(); s12 = AdvSimd.ReverseElement16(AdvSimd.Xor(s12, s1).AsInt32()).AsUInt32(); s13 = AdvSimd.ReverseElement16(AdvSimd.Xor(s13, s2).AsInt32()).AsUInt32(); s14 = AdvSimd.ReverseElement16(AdvSimd.Xor(s14, s3).AsInt32()).AsUInt32();
                s10 = AdvSimd.Add(s10, s15); s11 = AdvSimd.Add(s11, s12); s8 = AdvSimd.Add(s8, s13); s9 = AdvSimd.Add(s9, s14);
                { var t = AdvSimd.Xor(s5, s10); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s6, s11); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s7, s8); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s4, s9); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); }
                s0 = AdvSimd.Add(AdvSimd.Add(s0, m[2]), s5); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[3]), s6); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[1]), s7); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[4]), s4);
                s15 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s15, s0).AsByte(), rot8Mask).AsUInt32(); s12 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s12, s1).AsByte(), rot8Mask).AsUInt32(); s13 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s13, s2).AsByte(), rot8Mask).AsUInt32(); s14 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s14, s3).AsByte(), rot8Mask).AsUInt32();
                s10 = AdvSimd.Add(s10, s15); s11 = AdvSimd.Add(s11, s12); s8 = AdvSimd.Add(s8, s13); s9 = AdvSimd.Add(s9, s14);
                { var t = AdvSimd.Xor(s5, s10); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s6, s11); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s7, s8); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s4, s9); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); }
                m[16] = s0; // barrier: stops the message loads being hoisted
                // Round 5
                s0 = AdvSimd.Add(AdvSimd.Add(s0, m[9]), s4); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[11]), s5); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[8]), s6); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[15]), s7);
                s12 = AdvSimd.ReverseElement16(AdvSimd.Xor(s12, s0).AsInt32()).AsUInt32(); s13 = AdvSimd.ReverseElement16(AdvSimd.Xor(s13, s1).AsInt32()).AsUInt32(); s14 = AdvSimd.ReverseElement16(AdvSimd.Xor(s14, s2).AsInt32()).AsUInt32(); s15 = AdvSimd.ReverseElement16(AdvSimd.Xor(s15, s3).AsInt32()).AsUInt32();
                s8 = AdvSimd.Add(s8, s12); s9 = AdvSimd.Add(s9, s13); s10 = AdvSimd.Add(s10, s14); s11 = AdvSimd.Add(s11, s15);
                { var t = AdvSimd.Xor(s4, s8); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s5, s9); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s6, s10); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s7, s11); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); }
                s0 = AdvSimd.Add(AdvSimd.Add(s0, m[14]), s4); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[5]), s5); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[12]), s6); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[1]), s7);
                s12 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s12, s0).AsByte(), rot8Mask).AsUInt32(); s13 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s13, s1).AsByte(), rot8Mask).AsUInt32(); s14 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s14, s2).AsByte(), rot8Mask).AsUInt32(); s15 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s15, s3).AsByte(), rot8Mask).AsUInt32();
                s8 = AdvSimd.Add(s8, s12); s9 = AdvSimd.Add(s9, s13); s10 = AdvSimd.Add(s10, s14); s11 = AdvSimd.Add(s11, s15);
                { var t = AdvSimd.Xor(s4, s8); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s5, s9); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s6, s10); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s7, s11); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); }
                s0 = AdvSimd.Add(AdvSimd.Add(s0, m[13]), s5); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[0]), s6); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[2]), s7); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[4]), s4);
                s15 = AdvSimd.ReverseElement16(AdvSimd.Xor(s15, s0).AsInt32()).AsUInt32(); s12 = AdvSimd.ReverseElement16(AdvSimd.Xor(s12, s1).AsInt32()).AsUInt32(); s13 = AdvSimd.ReverseElement16(AdvSimd.Xor(s13, s2).AsInt32()).AsUInt32(); s14 = AdvSimd.ReverseElement16(AdvSimd.Xor(s14, s3).AsInt32()).AsUInt32();
                s10 = AdvSimd.Add(s10, s15); s11 = AdvSimd.Add(s11, s12); s8 = AdvSimd.Add(s8, s13); s9 = AdvSimd.Add(s9, s14);
                { var t = AdvSimd.Xor(s5, s10); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s6, s11); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s7, s8); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s4, s9); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); }
                s0 = AdvSimd.Add(AdvSimd.Add(s0, m[3]), s5); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[10]), s6); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[6]), s7); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[7]), s4);
                s15 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s15, s0).AsByte(), rot8Mask).AsUInt32(); s12 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s12, s1).AsByte(), rot8Mask).AsUInt32(); s13 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s13, s2).AsByte(), rot8Mask).AsUInt32(); s14 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s14, s3).AsByte(), rot8Mask).AsUInt32();
                s10 = AdvSimd.Add(s10, s15); s11 = AdvSimd.Add(s11, s12); s8 = AdvSimd.Add(s8, s13); s9 = AdvSimd.Add(s9, s14);
                { var t = AdvSimd.Xor(s5, s10); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s6, s11); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s7, s8); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s4, s9); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); }
                m[16] = s0; // barrier: stops the message loads being hoisted
                // Round 6
                s0 = AdvSimd.Add(AdvSimd.Add(s0, m[11]), s4); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[5]), s5); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[1]), s6); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[8]), s7);
                s12 = AdvSimd.ReverseElement16(AdvSimd.Xor(s12, s0).AsInt32()).AsUInt32(); s13 = AdvSimd.ReverseElement16(AdvSimd.Xor(s13, s1).AsInt32()).AsUInt32(); s14 = AdvSimd.ReverseElement16(AdvSimd.Xor(s14, s2).AsInt32()).AsUInt32(); s15 = AdvSimd.ReverseElement16(AdvSimd.Xor(s15, s3).AsInt32()).AsUInt32();
                s8 = AdvSimd.Add(s8, s12); s9 = AdvSimd.Add(s9, s13); s10 = AdvSimd.Add(s10, s14); s11 = AdvSimd.Add(s11, s15);
                { var t = AdvSimd.Xor(s4, s8); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s5, s9); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s6, s10); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s7, s11); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); }
                s0 = AdvSimd.Add(AdvSimd.Add(s0, m[15]), s4); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[0]), s5); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[9]), s6); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[6]), s7);
                s12 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s12, s0).AsByte(), rot8Mask).AsUInt32(); s13 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s13, s1).AsByte(), rot8Mask).AsUInt32(); s14 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s14, s2).AsByte(), rot8Mask).AsUInt32(); s15 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s15, s3).AsByte(), rot8Mask).AsUInt32();
                s8 = AdvSimd.Add(s8, s12); s9 = AdvSimd.Add(s9, s13); s10 = AdvSimd.Add(s10, s14); s11 = AdvSimd.Add(s11, s15);
                { var t = AdvSimd.Xor(s4, s8); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s5, s9); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s6, s10); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s7, s11); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); }
                s0 = AdvSimd.Add(AdvSimd.Add(s0, m[14]), s5); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[2]), s6); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[3]), s7); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[7]), s4);
                s15 = AdvSimd.ReverseElement16(AdvSimd.Xor(s15, s0).AsInt32()).AsUInt32(); s12 = AdvSimd.ReverseElement16(AdvSimd.Xor(s12, s1).AsInt32()).AsUInt32(); s13 = AdvSimd.ReverseElement16(AdvSimd.Xor(s13, s2).AsInt32()).AsUInt32(); s14 = AdvSimd.ReverseElement16(AdvSimd.Xor(s14, s3).AsInt32()).AsUInt32();
                s10 = AdvSimd.Add(s10, s15); s11 = AdvSimd.Add(s11, s12); s8 = AdvSimd.Add(s8, s13); s9 = AdvSimd.Add(s9, s14);
                { var t = AdvSimd.Xor(s5, s10); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s6, s11); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s7, s8); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s4, s9); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); }
                s0 = AdvSimd.Add(AdvSimd.Add(s0, m[10]), s5); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[12]), s6); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[4]), s7); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[13]), s4);
                s15 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s15, s0).AsByte(), rot8Mask).AsUInt32(); s12 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s12, s1).AsByte(), rot8Mask).AsUInt32(); s13 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s13, s2).AsByte(), rot8Mask).AsUInt32(); s14 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s14, s3).AsByte(), rot8Mask).AsUInt32();
                s10 = AdvSimd.Add(s10, s15); s11 = AdvSimd.Add(s11, s12); s8 = AdvSimd.Add(s8, s13); s9 = AdvSimd.Add(s9, s14);
                { var t = AdvSimd.Xor(s5, s10); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s6, s11); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s7, s8); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s4, s9); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); }
                m[16] = s0; // barrier: stops the message loads being hoisted

                // Post-XOR: only chaining value (first 8 words)
                cv0 = AdvSimd.Xor(s0, s8);
                cv1 = AdvSimd.Xor(s1, s9);
                cv2 = AdvSimd.Xor(s2, s10);
                cv3 = AdvSimd.Xor(s3, s11);
                cv4 = AdvSimd.Xor(s4, s12);
                cv5 = AdvSimd.Xor(s5, s13);
                cv6 = AdvSimd.Xor(s6, s14);
                cv7 = AdvSimd.Xor(s7, s15);
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

    /// <summary>
    /// Three complete chunks through the 4-way NEON kernel, with the spare lane pointed at chunk
    /// zero so the batch stays inside the input span. ARM otherwise drops from the 4-way kernel
    /// straight to per-chunk scalar compression for this remainder.
    /// <para>
    /// Callers should pass only <c>numChunks == 3</c>, or 2 when <c>HashManySve2</c> is
    /// supported. Two chunks measured 24-28% SLOWER than the scalar path on Cortex-A73
    /// (2026-09-21): the kernel always pays for four lanes, and NEON was only about 1.5x scalar
    /// per lane on that core, so padding pays off only when at most one lane is wasted.
    /// </para>
    /// <para>
    /// With the current round body (2026-09-25) NEON is about 1.7x scalar per lane on the A73 and
    /// 2x on Neoverse-V1. Three chunks here now take 0.78 of the scalar time on the A73 -- after
    /// the scalar rewrite and before this body they took 1.44x -- and 0.67 on V1. Two would still
    /// lose on the A73 (about 1.18x scalar) and only tie on V1, so the rule above stands.
    /// </para>
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static unsafe void HashManyPartial(ReadOnlySpan<byte> chunks, int numChunks,
                                       ReadOnlySpan<uint> key, ulong startCounter,
                                       uint flags, Span<uint> cvs)
    {
#if NET10_0_OR_GREATER
        if (HashManySve2.IsSupported)
        {
            HashManySve2.HashManyPartial(chunks, numChunks, key, startCounter, flags, cvs);
            return;
        }
#endif
        var rot8Mask = Vector128.Create((byte)1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12);
        const int blocksPerChunk = Blake3Constants.ChunkLen / Blake3Constants.BlockLen; // 16

        // Unused lanes reread chunk zero, keeping partial batches inside the input span.
        // All four CVs are computed; only numChunks of them are written out.
        int offset1 = numChunks > 1 ? 1 * Blake3Constants.ChunkLen : 0;
        int offset2 = numChunks > 2 ? 2 * Blake3Constants.ChunkLen : 0;
        int offset3 = numChunks > 3 ? 3 * Blake3Constants.ChunkLen : 0;

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
            Vector128<uint>* m = stackalloc Vector128<uint>[17]; // [16]: the per-round barrier slot

            for (int blockIdx = 0; blockIdx < blocksPerChunk; blockIdx++)
            {
                byte* blockBase = chunksPtr + blockIdx * 64;

                // Load 4 words (16 bytes) from each of 4 chunks, then transpose
                // Lower 8 words (0-7): two groups of 4
                var r0 = Unsafe.ReadUnaligned<Vector128<uint>>(blockBase + 0 * Blake3Constants.ChunkLen);
                var r1 = Unsafe.ReadUnaligned<Vector128<uint>>(blockBase + offset1);
                var r2 = Unsafe.ReadUnaligned<Vector128<uint>>(blockBase + offset2);
                var r3 = Unsafe.ReadUnaligned<Vector128<uint>>(blockBase + offset3);
                Transpose4X4(r0, r1, r2, r3, out m[0], out m[1], out m[2], out m[3]);

                r0 = Unsafe.ReadUnaligned<Vector128<uint>>(blockBase + 0 * Blake3Constants.ChunkLen + 16);
                r1 = Unsafe.ReadUnaligned<Vector128<uint>>(blockBase + offset1 + 16);
                r2 = Unsafe.ReadUnaligned<Vector128<uint>>(blockBase + offset2 + 16);
                r3 = Unsafe.ReadUnaligned<Vector128<uint>>(blockBase + offset3 + 16);
                Transpose4X4(r0, r1, r2, r3, out m[4], out m[5], out m[6], out m[7]);

                r0 = Unsafe.ReadUnaligned<Vector128<uint>>(blockBase + 0 * Blake3Constants.ChunkLen + 32);
                r1 = Unsafe.ReadUnaligned<Vector128<uint>>(blockBase + offset1 + 32);
                r2 = Unsafe.ReadUnaligned<Vector128<uint>>(blockBase + offset2 + 32);
                r3 = Unsafe.ReadUnaligned<Vector128<uint>>(blockBase + offset3 + 32);
                Transpose4X4(r0, r1, r2, r3, out m[8], out m[9], out m[10], out m[11]);

                r0 = Unsafe.ReadUnaligned<Vector128<uint>>(blockBase + 0 * Blake3Constants.ChunkLen + 48);
                r1 = Unsafe.ReadUnaligned<Vector128<uint>>(blockBase + offset1 + 48);
                r2 = Unsafe.ReadUnaligned<Vector128<uint>>(blockBase + offset2 + 48);
                r3 = Unsafe.ReadUnaligned<Vector128<uint>>(blockBase + offset3 + 48);
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

                // Round 0
                s0 = AdvSimd.Add(AdvSimd.Add(s0, m[0]), s4); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[2]), s5); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[4]), s6); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[6]), s7);
                s12 = AdvSimd.ReverseElement16(AdvSimd.Xor(s12, s0).AsInt32()).AsUInt32(); s13 = AdvSimd.ReverseElement16(AdvSimd.Xor(s13, s1).AsInt32()).AsUInt32(); s14 = AdvSimd.ReverseElement16(AdvSimd.Xor(s14, s2).AsInt32()).AsUInt32(); s15 = AdvSimd.ReverseElement16(AdvSimd.Xor(s15, s3).AsInt32()).AsUInt32();
                s8 = AdvSimd.Add(s8, s12); s9 = AdvSimd.Add(s9, s13); s10 = AdvSimd.Add(s10, s14); s11 = AdvSimd.Add(s11, s15);
                { var t = AdvSimd.Xor(s4, s8); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s5, s9); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s6, s10); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s7, s11); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); }
                s0 = AdvSimd.Add(AdvSimd.Add(s0, m[1]), s4); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[3]), s5); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[5]), s6); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[7]), s7);
                s12 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s12, s0).AsByte(), rot8Mask).AsUInt32(); s13 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s13, s1).AsByte(), rot8Mask).AsUInt32(); s14 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s14, s2).AsByte(), rot8Mask).AsUInt32(); s15 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s15, s3).AsByte(), rot8Mask).AsUInt32();
                s8 = AdvSimd.Add(s8, s12); s9 = AdvSimd.Add(s9, s13); s10 = AdvSimd.Add(s10, s14); s11 = AdvSimd.Add(s11, s15);
                { var t = AdvSimd.Xor(s4, s8); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s5, s9); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s6, s10); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s7, s11); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); }
                s0 = AdvSimd.Add(AdvSimd.Add(s0, m[8]), s5); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[10]), s6); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[12]), s7); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[14]), s4);
                s15 = AdvSimd.ReverseElement16(AdvSimd.Xor(s15, s0).AsInt32()).AsUInt32(); s12 = AdvSimd.ReverseElement16(AdvSimd.Xor(s12, s1).AsInt32()).AsUInt32(); s13 = AdvSimd.ReverseElement16(AdvSimd.Xor(s13, s2).AsInt32()).AsUInt32(); s14 = AdvSimd.ReverseElement16(AdvSimd.Xor(s14, s3).AsInt32()).AsUInt32();
                s10 = AdvSimd.Add(s10, s15); s11 = AdvSimd.Add(s11, s12); s8 = AdvSimd.Add(s8, s13); s9 = AdvSimd.Add(s9, s14);
                { var t = AdvSimd.Xor(s5, s10); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s6, s11); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s7, s8); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s4, s9); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); }
                s0 = AdvSimd.Add(AdvSimd.Add(s0, m[9]), s5); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[11]), s6); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[13]), s7); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[15]), s4);
                s15 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s15, s0).AsByte(), rot8Mask).AsUInt32(); s12 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s12, s1).AsByte(), rot8Mask).AsUInt32(); s13 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s13, s2).AsByte(), rot8Mask).AsUInt32(); s14 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s14, s3).AsByte(), rot8Mask).AsUInt32();
                s10 = AdvSimd.Add(s10, s15); s11 = AdvSimd.Add(s11, s12); s8 = AdvSimd.Add(s8, s13); s9 = AdvSimd.Add(s9, s14);
                { var t = AdvSimd.Xor(s5, s10); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s6, s11); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s7, s8); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s4, s9); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); }
                m[16] = s0; // barrier: stops the message loads being hoisted
                // Round 1
                s0 = AdvSimd.Add(AdvSimd.Add(s0, m[2]), s4); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[3]), s5); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[7]), s6); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[4]), s7);
                s12 = AdvSimd.ReverseElement16(AdvSimd.Xor(s12, s0).AsInt32()).AsUInt32(); s13 = AdvSimd.ReverseElement16(AdvSimd.Xor(s13, s1).AsInt32()).AsUInt32(); s14 = AdvSimd.ReverseElement16(AdvSimd.Xor(s14, s2).AsInt32()).AsUInt32(); s15 = AdvSimd.ReverseElement16(AdvSimd.Xor(s15, s3).AsInt32()).AsUInt32();
                s8 = AdvSimd.Add(s8, s12); s9 = AdvSimd.Add(s9, s13); s10 = AdvSimd.Add(s10, s14); s11 = AdvSimd.Add(s11, s15);
                { var t = AdvSimd.Xor(s4, s8); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s5, s9); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s6, s10); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s7, s11); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); }
                s0 = AdvSimd.Add(AdvSimd.Add(s0, m[6]), s4); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[10]), s5); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[0]), s6); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[13]), s7);
                s12 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s12, s0).AsByte(), rot8Mask).AsUInt32(); s13 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s13, s1).AsByte(), rot8Mask).AsUInt32(); s14 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s14, s2).AsByte(), rot8Mask).AsUInt32(); s15 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s15, s3).AsByte(), rot8Mask).AsUInt32();
                s8 = AdvSimd.Add(s8, s12); s9 = AdvSimd.Add(s9, s13); s10 = AdvSimd.Add(s10, s14); s11 = AdvSimd.Add(s11, s15);
                { var t = AdvSimd.Xor(s4, s8); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s5, s9); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s6, s10); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s7, s11); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); }
                s0 = AdvSimd.Add(AdvSimd.Add(s0, m[1]), s5); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[12]), s6); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[9]), s7); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[15]), s4);
                s15 = AdvSimd.ReverseElement16(AdvSimd.Xor(s15, s0).AsInt32()).AsUInt32(); s12 = AdvSimd.ReverseElement16(AdvSimd.Xor(s12, s1).AsInt32()).AsUInt32(); s13 = AdvSimd.ReverseElement16(AdvSimd.Xor(s13, s2).AsInt32()).AsUInt32(); s14 = AdvSimd.ReverseElement16(AdvSimd.Xor(s14, s3).AsInt32()).AsUInt32();
                s10 = AdvSimd.Add(s10, s15); s11 = AdvSimd.Add(s11, s12); s8 = AdvSimd.Add(s8, s13); s9 = AdvSimd.Add(s9, s14);
                { var t = AdvSimd.Xor(s5, s10); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s6, s11); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s7, s8); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s4, s9); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); }
                s0 = AdvSimd.Add(AdvSimd.Add(s0, m[11]), s5); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[5]), s6); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[14]), s7); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[8]), s4);
                s15 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s15, s0).AsByte(), rot8Mask).AsUInt32(); s12 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s12, s1).AsByte(), rot8Mask).AsUInt32(); s13 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s13, s2).AsByte(), rot8Mask).AsUInt32(); s14 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s14, s3).AsByte(), rot8Mask).AsUInt32();
                s10 = AdvSimd.Add(s10, s15); s11 = AdvSimd.Add(s11, s12); s8 = AdvSimd.Add(s8, s13); s9 = AdvSimd.Add(s9, s14);
                { var t = AdvSimd.Xor(s5, s10); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s6, s11); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s7, s8); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s4, s9); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); }
                m[16] = s0; // barrier: stops the message loads being hoisted
                // Round 2
                s0 = AdvSimd.Add(AdvSimd.Add(s0, m[3]), s4); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[10]), s5); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[13]), s6); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[7]), s7);
                s12 = AdvSimd.ReverseElement16(AdvSimd.Xor(s12, s0).AsInt32()).AsUInt32(); s13 = AdvSimd.ReverseElement16(AdvSimd.Xor(s13, s1).AsInt32()).AsUInt32(); s14 = AdvSimd.ReverseElement16(AdvSimd.Xor(s14, s2).AsInt32()).AsUInt32(); s15 = AdvSimd.ReverseElement16(AdvSimd.Xor(s15, s3).AsInt32()).AsUInt32();
                s8 = AdvSimd.Add(s8, s12); s9 = AdvSimd.Add(s9, s13); s10 = AdvSimd.Add(s10, s14); s11 = AdvSimd.Add(s11, s15);
                { var t = AdvSimd.Xor(s4, s8); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s5, s9); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s6, s10); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s7, s11); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); }
                s0 = AdvSimd.Add(AdvSimd.Add(s0, m[4]), s4); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[12]), s5); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[2]), s6); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[14]), s7);
                s12 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s12, s0).AsByte(), rot8Mask).AsUInt32(); s13 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s13, s1).AsByte(), rot8Mask).AsUInt32(); s14 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s14, s2).AsByte(), rot8Mask).AsUInt32(); s15 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s15, s3).AsByte(), rot8Mask).AsUInt32();
                s8 = AdvSimd.Add(s8, s12); s9 = AdvSimd.Add(s9, s13); s10 = AdvSimd.Add(s10, s14); s11 = AdvSimd.Add(s11, s15);
                { var t = AdvSimd.Xor(s4, s8); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s5, s9); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s6, s10); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s7, s11); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); }
                s0 = AdvSimd.Add(AdvSimd.Add(s0, m[6]), s5); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[9]), s6); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[11]), s7); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[8]), s4);
                s15 = AdvSimd.ReverseElement16(AdvSimd.Xor(s15, s0).AsInt32()).AsUInt32(); s12 = AdvSimd.ReverseElement16(AdvSimd.Xor(s12, s1).AsInt32()).AsUInt32(); s13 = AdvSimd.ReverseElement16(AdvSimd.Xor(s13, s2).AsInt32()).AsUInt32(); s14 = AdvSimd.ReverseElement16(AdvSimd.Xor(s14, s3).AsInt32()).AsUInt32();
                s10 = AdvSimd.Add(s10, s15); s11 = AdvSimd.Add(s11, s12); s8 = AdvSimd.Add(s8, s13); s9 = AdvSimd.Add(s9, s14);
                { var t = AdvSimd.Xor(s5, s10); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s6, s11); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s7, s8); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s4, s9); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); }
                s0 = AdvSimd.Add(AdvSimd.Add(s0, m[5]), s5); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[0]), s6); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[15]), s7); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[1]), s4);
                s15 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s15, s0).AsByte(), rot8Mask).AsUInt32(); s12 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s12, s1).AsByte(), rot8Mask).AsUInt32(); s13 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s13, s2).AsByte(), rot8Mask).AsUInt32(); s14 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s14, s3).AsByte(), rot8Mask).AsUInt32();
                s10 = AdvSimd.Add(s10, s15); s11 = AdvSimd.Add(s11, s12); s8 = AdvSimd.Add(s8, s13); s9 = AdvSimd.Add(s9, s14);
                { var t = AdvSimd.Xor(s5, s10); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s6, s11); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s7, s8); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s4, s9); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); }
                m[16] = s0; // barrier: stops the message loads being hoisted
                // Round 3
                s0 = AdvSimd.Add(AdvSimd.Add(s0, m[10]), s4); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[12]), s5); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[14]), s6); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[13]), s7);
                s12 = AdvSimd.ReverseElement16(AdvSimd.Xor(s12, s0).AsInt32()).AsUInt32(); s13 = AdvSimd.ReverseElement16(AdvSimd.Xor(s13, s1).AsInt32()).AsUInt32(); s14 = AdvSimd.ReverseElement16(AdvSimd.Xor(s14, s2).AsInt32()).AsUInt32(); s15 = AdvSimd.ReverseElement16(AdvSimd.Xor(s15, s3).AsInt32()).AsUInt32();
                s8 = AdvSimd.Add(s8, s12); s9 = AdvSimd.Add(s9, s13); s10 = AdvSimd.Add(s10, s14); s11 = AdvSimd.Add(s11, s15);
                { var t = AdvSimd.Xor(s4, s8); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s5, s9); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s6, s10); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s7, s11); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); }
                s0 = AdvSimd.Add(AdvSimd.Add(s0, m[7]), s4); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[9]), s5); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[3]), s6); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[15]), s7);
                s12 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s12, s0).AsByte(), rot8Mask).AsUInt32(); s13 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s13, s1).AsByte(), rot8Mask).AsUInt32(); s14 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s14, s2).AsByte(), rot8Mask).AsUInt32(); s15 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s15, s3).AsByte(), rot8Mask).AsUInt32();
                s8 = AdvSimd.Add(s8, s12); s9 = AdvSimd.Add(s9, s13); s10 = AdvSimd.Add(s10, s14); s11 = AdvSimd.Add(s11, s15);
                { var t = AdvSimd.Xor(s4, s8); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s5, s9); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s6, s10); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s7, s11); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); }
                s0 = AdvSimd.Add(AdvSimd.Add(s0, m[4]), s5); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[11]), s6); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[5]), s7); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[1]), s4);
                s15 = AdvSimd.ReverseElement16(AdvSimd.Xor(s15, s0).AsInt32()).AsUInt32(); s12 = AdvSimd.ReverseElement16(AdvSimd.Xor(s12, s1).AsInt32()).AsUInt32(); s13 = AdvSimd.ReverseElement16(AdvSimd.Xor(s13, s2).AsInt32()).AsUInt32(); s14 = AdvSimd.ReverseElement16(AdvSimd.Xor(s14, s3).AsInt32()).AsUInt32();
                s10 = AdvSimd.Add(s10, s15); s11 = AdvSimd.Add(s11, s12); s8 = AdvSimd.Add(s8, s13); s9 = AdvSimd.Add(s9, s14);
                { var t = AdvSimd.Xor(s5, s10); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s6, s11); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s7, s8); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s4, s9); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); }
                s0 = AdvSimd.Add(AdvSimd.Add(s0, m[0]), s5); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[2]), s6); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[8]), s7); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[6]), s4);
                s15 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s15, s0).AsByte(), rot8Mask).AsUInt32(); s12 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s12, s1).AsByte(), rot8Mask).AsUInt32(); s13 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s13, s2).AsByte(), rot8Mask).AsUInt32(); s14 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s14, s3).AsByte(), rot8Mask).AsUInt32();
                s10 = AdvSimd.Add(s10, s15); s11 = AdvSimd.Add(s11, s12); s8 = AdvSimd.Add(s8, s13); s9 = AdvSimd.Add(s9, s14);
                { var t = AdvSimd.Xor(s5, s10); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s6, s11); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s7, s8); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s4, s9); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); }
                m[16] = s0; // barrier: stops the message loads being hoisted
                // Round 4
                s0 = AdvSimd.Add(AdvSimd.Add(s0, m[12]), s4); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[9]), s5); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[15]), s6); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[14]), s7);
                s12 = AdvSimd.ReverseElement16(AdvSimd.Xor(s12, s0).AsInt32()).AsUInt32(); s13 = AdvSimd.ReverseElement16(AdvSimd.Xor(s13, s1).AsInt32()).AsUInt32(); s14 = AdvSimd.ReverseElement16(AdvSimd.Xor(s14, s2).AsInt32()).AsUInt32(); s15 = AdvSimd.ReverseElement16(AdvSimd.Xor(s15, s3).AsInt32()).AsUInt32();
                s8 = AdvSimd.Add(s8, s12); s9 = AdvSimd.Add(s9, s13); s10 = AdvSimd.Add(s10, s14); s11 = AdvSimd.Add(s11, s15);
                { var t = AdvSimd.Xor(s4, s8); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s5, s9); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s6, s10); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s7, s11); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); }
                s0 = AdvSimd.Add(AdvSimd.Add(s0, m[13]), s4); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[11]), s5); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[10]), s6); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[8]), s7);
                s12 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s12, s0).AsByte(), rot8Mask).AsUInt32(); s13 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s13, s1).AsByte(), rot8Mask).AsUInt32(); s14 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s14, s2).AsByte(), rot8Mask).AsUInt32(); s15 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s15, s3).AsByte(), rot8Mask).AsUInt32();
                s8 = AdvSimd.Add(s8, s12); s9 = AdvSimd.Add(s9, s13); s10 = AdvSimd.Add(s10, s14); s11 = AdvSimd.Add(s11, s15);
                { var t = AdvSimd.Xor(s4, s8); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s5, s9); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s6, s10); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s7, s11); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); }
                s0 = AdvSimd.Add(AdvSimd.Add(s0, m[7]), s5); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[5]), s6); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[0]), s7); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[6]), s4);
                s15 = AdvSimd.ReverseElement16(AdvSimd.Xor(s15, s0).AsInt32()).AsUInt32(); s12 = AdvSimd.ReverseElement16(AdvSimd.Xor(s12, s1).AsInt32()).AsUInt32(); s13 = AdvSimd.ReverseElement16(AdvSimd.Xor(s13, s2).AsInt32()).AsUInt32(); s14 = AdvSimd.ReverseElement16(AdvSimd.Xor(s14, s3).AsInt32()).AsUInt32();
                s10 = AdvSimd.Add(s10, s15); s11 = AdvSimd.Add(s11, s12); s8 = AdvSimd.Add(s8, s13); s9 = AdvSimd.Add(s9, s14);
                { var t = AdvSimd.Xor(s5, s10); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s6, s11); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s7, s8); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s4, s9); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); }
                s0 = AdvSimd.Add(AdvSimd.Add(s0, m[2]), s5); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[3]), s6); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[1]), s7); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[4]), s4);
                s15 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s15, s0).AsByte(), rot8Mask).AsUInt32(); s12 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s12, s1).AsByte(), rot8Mask).AsUInt32(); s13 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s13, s2).AsByte(), rot8Mask).AsUInt32(); s14 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s14, s3).AsByte(), rot8Mask).AsUInt32();
                s10 = AdvSimd.Add(s10, s15); s11 = AdvSimd.Add(s11, s12); s8 = AdvSimd.Add(s8, s13); s9 = AdvSimd.Add(s9, s14);
                { var t = AdvSimd.Xor(s5, s10); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s6, s11); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s7, s8); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s4, s9); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); }
                m[16] = s0; // barrier: stops the message loads being hoisted
                // Round 5
                s0 = AdvSimd.Add(AdvSimd.Add(s0, m[9]), s4); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[11]), s5); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[8]), s6); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[15]), s7);
                s12 = AdvSimd.ReverseElement16(AdvSimd.Xor(s12, s0).AsInt32()).AsUInt32(); s13 = AdvSimd.ReverseElement16(AdvSimd.Xor(s13, s1).AsInt32()).AsUInt32(); s14 = AdvSimd.ReverseElement16(AdvSimd.Xor(s14, s2).AsInt32()).AsUInt32(); s15 = AdvSimd.ReverseElement16(AdvSimd.Xor(s15, s3).AsInt32()).AsUInt32();
                s8 = AdvSimd.Add(s8, s12); s9 = AdvSimd.Add(s9, s13); s10 = AdvSimd.Add(s10, s14); s11 = AdvSimd.Add(s11, s15);
                { var t = AdvSimd.Xor(s4, s8); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s5, s9); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s6, s10); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s7, s11); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); }
                s0 = AdvSimd.Add(AdvSimd.Add(s0, m[14]), s4); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[5]), s5); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[12]), s6); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[1]), s7);
                s12 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s12, s0).AsByte(), rot8Mask).AsUInt32(); s13 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s13, s1).AsByte(), rot8Mask).AsUInt32(); s14 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s14, s2).AsByte(), rot8Mask).AsUInt32(); s15 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s15, s3).AsByte(), rot8Mask).AsUInt32();
                s8 = AdvSimd.Add(s8, s12); s9 = AdvSimd.Add(s9, s13); s10 = AdvSimd.Add(s10, s14); s11 = AdvSimd.Add(s11, s15);
                { var t = AdvSimd.Xor(s4, s8); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s5, s9); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s6, s10); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s7, s11); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); }
                s0 = AdvSimd.Add(AdvSimd.Add(s0, m[13]), s5); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[0]), s6); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[2]), s7); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[4]), s4);
                s15 = AdvSimd.ReverseElement16(AdvSimd.Xor(s15, s0).AsInt32()).AsUInt32(); s12 = AdvSimd.ReverseElement16(AdvSimd.Xor(s12, s1).AsInt32()).AsUInt32(); s13 = AdvSimd.ReverseElement16(AdvSimd.Xor(s13, s2).AsInt32()).AsUInt32(); s14 = AdvSimd.ReverseElement16(AdvSimd.Xor(s14, s3).AsInt32()).AsUInt32();
                s10 = AdvSimd.Add(s10, s15); s11 = AdvSimd.Add(s11, s12); s8 = AdvSimd.Add(s8, s13); s9 = AdvSimd.Add(s9, s14);
                { var t = AdvSimd.Xor(s5, s10); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s6, s11); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s7, s8); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s4, s9); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); }
                s0 = AdvSimd.Add(AdvSimd.Add(s0, m[3]), s5); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[10]), s6); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[6]), s7); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[7]), s4);
                s15 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s15, s0).AsByte(), rot8Mask).AsUInt32(); s12 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s12, s1).AsByte(), rot8Mask).AsUInt32(); s13 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s13, s2).AsByte(), rot8Mask).AsUInt32(); s14 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s14, s3).AsByte(), rot8Mask).AsUInt32();
                s10 = AdvSimd.Add(s10, s15); s11 = AdvSimd.Add(s11, s12); s8 = AdvSimd.Add(s8, s13); s9 = AdvSimd.Add(s9, s14);
                { var t = AdvSimd.Xor(s5, s10); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s6, s11); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s7, s8); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s4, s9); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); }
                m[16] = s0; // barrier: stops the message loads being hoisted
                // Round 6
                s0 = AdvSimd.Add(AdvSimd.Add(s0, m[11]), s4); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[5]), s5); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[1]), s6); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[8]), s7);
                s12 = AdvSimd.ReverseElement16(AdvSimd.Xor(s12, s0).AsInt32()).AsUInt32(); s13 = AdvSimd.ReverseElement16(AdvSimd.Xor(s13, s1).AsInt32()).AsUInt32(); s14 = AdvSimd.ReverseElement16(AdvSimd.Xor(s14, s2).AsInt32()).AsUInt32(); s15 = AdvSimd.ReverseElement16(AdvSimd.Xor(s15, s3).AsInt32()).AsUInt32();
                s8 = AdvSimd.Add(s8, s12); s9 = AdvSimd.Add(s9, s13); s10 = AdvSimd.Add(s10, s14); s11 = AdvSimd.Add(s11, s15);
                { var t = AdvSimd.Xor(s4, s8); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s5, s9); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s6, s10); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s7, s11); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); }
                s0 = AdvSimd.Add(AdvSimd.Add(s0, m[15]), s4); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[0]), s5); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[9]), s6); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[6]), s7);
                s12 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s12, s0).AsByte(), rot8Mask).AsUInt32(); s13 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s13, s1).AsByte(), rot8Mask).AsUInt32(); s14 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s14, s2).AsByte(), rot8Mask).AsUInt32(); s15 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s15, s3).AsByte(), rot8Mask).AsUInt32();
                s8 = AdvSimd.Add(s8, s12); s9 = AdvSimd.Add(s9, s13); s10 = AdvSimd.Add(s10, s14); s11 = AdvSimd.Add(s11, s15);
                { var t = AdvSimd.Xor(s4, s8); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s5, s9); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s6, s10); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s7, s11); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); }
                s0 = AdvSimd.Add(AdvSimd.Add(s0, m[14]), s5); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[2]), s6); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[3]), s7); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[7]), s4);
                s15 = AdvSimd.ReverseElement16(AdvSimd.Xor(s15, s0).AsInt32()).AsUInt32(); s12 = AdvSimd.ReverseElement16(AdvSimd.Xor(s12, s1).AsInt32()).AsUInt32(); s13 = AdvSimd.ReverseElement16(AdvSimd.Xor(s13, s2).AsInt32()).AsUInt32(); s14 = AdvSimd.ReverseElement16(AdvSimd.Xor(s14, s3).AsInt32()).AsUInt32();
                s10 = AdvSimd.Add(s10, s15); s11 = AdvSimd.Add(s11, s12); s8 = AdvSimd.Add(s8, s13); s9 = AdvSimd.Add(s9, s14);
                { var t = AdvSimd.Xor(s5, s10); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s6, s11); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s7, s8); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s4, s9); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); }
                s0 = AdvSimd.Add(AdvSimd.Add(s0, m[10]), s5); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[12]), s6); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[4]), s7); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[13]), s4);
                s15 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s15, s0).AsByte(), rot8Mask).AsUInt32(); s12 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s12, s1).AsByte(), rot8Mask).AsUInt32(); s13 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s13, s2).AsByte(), rot8Mask).AsUInt32(); s14 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s14, s3).AsByte(), rot8Mask).AsUInt32();
                s10 = AdvSimd.Add(s10, s15); s11 = AdvSimd.Add(s11, s12); s8 = AdvSimd.Add(s8, s13); s9 = AdvSimd.Add(s9, s14);
                { var t = AdvSimd.Xor(s5, s10); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s6, s11); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s7, s8); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s4, s9); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); }
                m[16] = s0; // barrier: stops the message loads being hoisted

                // Post-XOR: only chaining value (first 8 words)
                cv0 = AdvSimd.Xor(s0, s8);
                cv1 = AdvSimd.Xor(s1, s9);
                cv2 = AdvSimd.Xor(s2, s10);
                cv3 = AdvSimd.Xor(s3, s11);
                cv4 = AdvSimd.Xor(s4, s12);
                cv5 = AdvSimd.Xor(s5, s13);
                cv6 = AdvSimd.Xor(s6, s14);
                cv7 = AdvSimd.Xor(s7, s15);
            }
        }

        // 4x4 transpose: word-major to chunk-major for output
        Transpose4X4(cv0, cv1, cv2, cv3, out var o0, out var o1, out var o2, out var o3);
        Transpose4X4(cv4, cv5, cv6, cv7, out var o4, out var o5, out var o6, out var o7);

        ref uint outRef = ref MemoryMarshal.GetReference(cvs);
        VectorCompat.Store(o0, ref outRef);
        VectorCompat.Store(o4, ref outRef, 4);
        if (numChunks > 1)
        {
            VectorCompat.Store(o1, ref outRef, 8);
            VectorCompat.Store(o5, ref outRef, 12);
        }
        if (numChunks > 2)
        {
            VectorCompat.Store(o2, ref outRef, 16);
            VectorCompat.Store(o6, ref outRef, 20);
        }
        if (numChunks > 3)
        {
            VectorCompat.Store(o3, ref outRef, 24);
            VectorCompat.Store(o7, ref outRef, 28);
        }
    }

    /// <summary>
    /// Four parent nodes compressed together in the four NEON lanes. Each parent block is 64
    /// bytes (two child chaining values). Without this every internal tree node on ARM is a
    /// separate scalar compression; the AVX2 tier has had <c>HashParents8</c> from the start.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static unsafe void HashParents4(ReadOnlySpan<uint> parentBlocks,
                                           ReadOnlySpan<uint> key, uint flags,
                                           Span<uint> cvs)
    {
#if NET10_0_OR_GREATER
        if (HashManySve2.IsSupported)
        {
            HashManySve2.HashParents4(parentBlocks, key, flags, cvs);
            return;
        }
#endif
        var rot8Mask = Vector128.Create((byte)1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12);
        fixed (uint* inPtr = parentBlocks)
        {
            byte* basePtr = (byte*)inPtr;
            Vector128<uint>* m = stackalloc Vector128<uint>[17]; // [16]: the per-round barrier slot

            // Four parent blocks, stride 64 bytes, transposed block-major -> word-major.
            for (int g = 0; g < 4; g++)
            {
                var r0 = Unsafe.ReadUnaligned<Vector128<uint>>(basePtr + 0 * 64 + g * 16);
                var r1 = Unsafe.ReadUnaligned<Vector128<uint>>(basePtr + 1 * 64 + g * 16);
                var r2 = Unsafe.ReadUnaligned<Vector128<uint>>(basePtr + 2 * 64 + g * 16);
                var r3 = Unsafe.ReadUnaligned<Vector128<uint>>(basePtr + 3 * 64 + g * 16);
                Transpose4X4(r0, r1, r2, r3, out m[g * 4 + 0], out m[g * 4 + 1],
                             out m[g * 4 + 2], out m[g * 4 + 3]);
            }

            // Parent nodes always start from the key, counter 0, a full 64-byte block.
            Vector128<uint> s0 = Vector128.Create(key[0]);
            Vector128<uint> s1 = Vector128.Create(key[1]);
            Vector128<uint> s2 = Vector128.Create(key[2]);
            Vector128<uint> s3 = Vector128.Create(key[3]);
            Vector128<uint> s4 = Vector128.Create(key[4]);
            Vector128<uint> s5 = Vector128.Create(key[5]);
            Vector128<uint> s6 = Vector128.Create(key[6]);
            Vector128<uint> s7 = Vector128.Create(key[7]);
            Vector128<uint> s8 = Vector128.Create(Blake3Constants.Iv0);
            Vector128<uint> s9 = Vector128.Create(Blake3Constants.Iv1);
            Vector128<uint> s10 = Vector128.Create(Blake3Constants.Iv2);
            Vector128<uint> s11 = Vector128.Create(Blake3Constants.Iv3);
            Vector128<uint> s12 = Vector128<uint>.Zero;
            Vector128<uint> s13 = Vector128<uint>.Zero;
            Vector128<uint> s14 = Vector128.Create((uint)Blake3Constants.BlockLen);
            Vector128<uint> s15 = Vector128.Create(flags | Blake3Constants.Parent);

            // Round 0
            s0 = AdvSimd.Add(AdvSimd.Add(s0, m[0]), s4); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[2]), s5); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[4]), s6); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[6]), s7);
            s12 = AdvSimd.ReverseElement16(AdvSimd.Xor(s12, s0).AsInt32()).AsUInt32(); s13 = AdvSimd.ReverseElement16(AdvSimd.Xor(s13, s1).AsInt32()).AsUInt32(); s14 = AdvSimd.ReverseElement16(AdvSimd.Xor(s14, s2).AsInt32()).AsUInt32(); s15 = AdvSimd.ReverseElement16(AdvSimd.Xor(s15, s3).AsInt32()).AsUInt32();
            s8 = AdvSimd.Add(s8, s12); s9 = AdvSimd.Add(s9, s13); s10 = AdvSimd.Add(s10, s14); s11 = AdvSimd.Add(s11, s15);
            { var t = AdvSimd.Xor(s4, s8); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s5, s9); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s6, s10); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s7, s11); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); }
            s0 = AdvSimd.Add(AdvSimd.Add(s0, m[1]), s4); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[3]), s5); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[5]), s6); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[7]), s7);
            s12 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s12, s0).AsByte(), rot8Mask).AsUInt32(); s13 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s13, s1).AsByte(), rot8Mask).AsUInt32(); s14 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s14, s2).AsByte(), rot8Mask).AsUInt32(); s15 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s15, s3).AsByte(), rot8Mask).AsUInt32();
            s8 = AdvSimd.Add(s8, s12); s9 = AdvSimd.Add(s9, s13); s10 = AdvSimd.Add(s10, s14); s11 = AdvSimd.Add(s11, s15);
            { var t = AdvSimd.Xor(s4, s8); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s5, s9); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s6, s10); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s7, s11); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); }
            s0 = AdvSimd.Add(AdvSimd.Add(s0, m[8]), s5); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[10]), s6); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[12]), s7); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[14]), s4);
            s15 = AdvSimd.ReverseElement16(AdvSimd.Xor(s15, s0).AsInt32()).AsUInt32(); s12 = AdvSimd.ReverseElement16(AdvSimd.Xor(s12, s1).AsInt32()).AsUInt32(); s13 = AdvSimd.ReverseElement16(AdvSimd.Xor(s13, s2).AsInt32()).AsUInt32(); s14 = AdvSimd.ReverseElement16(AdvSimd.Xor(s14, s3).AsInt32()).AsUInt32();
            s10 = AdvSimd.Add(s10, s15); s11 = AdvSimd.Add(s11, s12); s8 = AdvSimd.Add(s8, s13); s9 = AdvSimd.Add(s9, s14);
            { var t = AdvSimd.Xor(s5, s10); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s6, s11); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s7, s8); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s4, s9); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); }
            s0 = AdvSimd.Add(AdvSimd.Add(s0, m[9]), s5); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[11]), s6); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[13]), s7); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[15]), s4);
            s15 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s15, s0).AsByte(), rot8Mask).AsUInt32(); s12 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s12, s1).AsByte(), rot8Mask).AsUInt32(); s13 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s13, s2).AsByte(), rot8Mask).AsUInt32(); s14 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s14, s3).AsByte(), rot8Mask).AsUInt32();
            s10 = AdvSimd.Add(s10, s15); s11 = AdvSimd.Add(s11, s12); s8 = AdvSimd.Add(s8, s13); s9 = AdvSimd.Add(s9, s14);
            { var t = AdvSimd.Xor(s5, s10); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s6, s11); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s7, s8); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s4, s9); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); }
            m[16] = s0; // barrier: stops the message loads being hoisted
            // Round 1
            s0 = AdvSimd.Add(AdvSimd.Add(s0, m[2]), s4); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[3]), s5); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[7]), s6); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[4]), s7);
            s12 = AdvSimd.ReverseElement16(AdvSimd.Xor(s12, s0).AsInt32()).AsUInt32(); s13 = AdvSimd.ReverseElement16(AdvSimd.Xor(s13, s1).AsInt32()).AsUInt32(); s14 = AdvSimd.ReverseElement16(AdvSimd.Xor(s14, s2).AsInt32()).AsUInt32(); s15 = AdvSimd.ReverseElement16(AdvSimd.Xor(s15, s3).AsInt32()).AsUInt32();
            s8 = AdvSimd.Add(s8, s12); s9 = AdvSimd.Add(s9, s13); s10 = AdvSimd.Add(s10, s14); s11 = AdvSimd.Add(s11, s15);
            { var t = AdvSimd.Xor(s4, s8); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s5, s9); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s6, s10); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s7, s11); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); }
            s0 = AdvSimd.Add(AdvSimd.Add(s0, m[6]), s4); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[10]), s5); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[0]), s6); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[13]), s7);
            s12 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s12, s0).AsByte(), rot8Mask).AsUInt32(); s13 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s13, s1).AsByte(), rot8Mask).AsUInt32(); s14 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s14, s2).AsByte(), rot8Mask).AsUInt32(); s15 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s15, s3).AsByte(), rot8Mask).AsUInt32();
            s8 = AdvSimd.Add(s8, s12); s9 = AdvSimd.Add(s9, s13); s10 = AdvSimd.Add(s10, s14); s11 = AdvSimd.Add(s11, s15);
            { var t = AdvSimd.Xor(s4, s8); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s5, s9); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s6, s10); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s7, s11); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); }
            s0 = AdvSimd.Add(AdvSimd.Add(s0, m[1]), s5); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[12]), s6); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[9]), s7); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[15]), s4);
            s15 = AdvSimd.ReverseElement16(AdvSimd.Xor(s15, s0).AsInt32()).AsUInt32(); s12 = AdvSimd.ReverseElement16(AdvSimd.Xor(s12, s1).AsInt32()).AsUInt32(); s13 = AdvSimd.ReverseElement16(AdvSimd.Xor(s13, s2).AsInt32()).AsUInt32(); s14 = AdvSimd.ReverseElement16(AdvSimd.Xor(s14, s3).AsInt32()).AsUInt32();
            s10 = AdvSimd.Add(s10, s15); s11 = AdvSimd.Add(s11, s12); s8 = AdvSimd.Add(s8, s13); s9 = AdvSimd.Add(s9, s14);
            { var t = AdvSimd.Xor(s5, s10); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s6, s11); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s7, s8); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s4, s9); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); }
            s0 = AdvSimd.Add(AdvSimd.Add(s0, m[11]), s5); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[5]), s6); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[14]), s7); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[8]), s4);
            s15 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s15, s0).AsByte(), rot8Mask).AsUInt32(); s12 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s12, s1).AsByte(), rot8Mask).AsUInt32(); s13 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s13, s2).AsByte(), rot8Mask).AsUInt32(); s14 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s14, s3).AsByte(), rot8Mask).AsUInt32();
            s10 = AdvSimd.Add(s10, s15); s11 = AdvSimd.Add(s11, s12); s8 = AdvSimd.Add(s8, s13); s9 = AdvSimd.Add(s9, s14);
            { var t = AdvSimd.Xor(s5, s10); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s6, s11); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s7, s8); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s4, s9); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); }
            m[16] = s0; // barrier: stops the message loads being hoisted
            // Round 2
            s0 = AdvSimd.Add(AdvSimd.Add(s0, m[3]), s4); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[10]), s5); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[13]), s6); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[7]), s7);
            s12 = AdvSimd.ReverseElement16(AdvSimd.Xor(s12, s0).AsInt32()).AsUInt32(); s13 = AdvSimd.ReverseElement16(AdvSimd.Xor(s13, s1).AsInt32()).AsUInt32(); s14 = AdvSimd.ReverseElement16(AdvSimd.Xor(s14, s2).AsInt32()).AsUInt32(); s15 = AdvSimd.ReverseElement16(AdvSimd.Xor(s15, s3).AsInt32()).AsUInt32();
            s8 = AdvSimd.Add(s8, s12); s9 = AdvSimd.Add(s9, s13); s10 = AdvSimd.Add(s10, s14); s11 = AdvSimd.Add(s11, s15);
            { var t = AdvSimd.Xor(s4, s8); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s5, s9); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s6, s10); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s7, s11); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); }
            s0 = AdvSimd.Add(AdvSimd.Add(s0, m[4]), s4); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[12]), s5); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[2]), s6); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[14]), s7);
            s12 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s12, s0).AsByte(), rot8Mask).AsUInt32(); s13 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s13, s1).AsByte(), rot8Mask).AsUInt32(); s14 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s14, s2).AsByte(), rot8Mask).AsUInt32(); s15 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s15, s3).AsByte(), rot8Mask).AsUInt32();
            s8 = AdvSimd.Add(s8, s12); s9 = AdvSimd.Add(s9, s13); s10 = AdvSimd.Add(s10, s14); s11 = AdvSimd.Add(s11, s15);
            { var t = AdvSimd.Xor(s4, s8); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s5, s9); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s6, s10); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s7, s11); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); }
            s0 = AdvSimd.Add(AdvSimd.Add(s0, m[6]), s5); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[9]), s6); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[11]), s7); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[8]), s4);
            s15 = AdvSimd.ReverseElement16(AdvSimd.Xor(s15, s0).AsInt32()).AsUInt32(); s12 = AdvSimd.ReverseElement16(AdvSimd.Xor(s12, s1).AsInt32()).AsUInt32(); s13 = AdvSimd.ReverseElement16(AdvSimd.Xor(s13, s2).AsInt32()).AsUInt32(); s14 = AdvSimd.ReverseElement16(AdvSimd.Xor(s14, s3).AsInt32()).AsUInt32();
            s10 = AdvSimd.Add(s10, s15); s11 = AdvSimd.Add(s11, s12); s8 = AdvSimd.Add(s8, s13); s9 = AdvSimd.Add(s9, s14);
            { var t = AdvSimd.Xor(s5, s10); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s6, s11); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s7, s8); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s4, s9); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); }
            s0 = AdvSimd.Add(AdvSimd.Add(s0, m[5]), s5); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[0]), s6); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[15]), s7); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[1]), s4);
            s15 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s15, s0).AsByte(), rot8Mask).AsUInt32(); s12 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s12, s1).AsByte(), rot8Mask).AsUInt32(); s13 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s13, s2).AsByte(), rot8Mask).AsUInt32(); s14 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s14, s3).AsByte(), rot8Mask).AsUInt32();
            s10 = AdvSimd.Add(s10, s15); s11 = AdvSimd.Add(s11, s12); s8 = AdvSimd.Add(s8, s13); s9 = AdvSimd.Add(s9, s14);
            { var t = AdvSimd.Xor(s5, s10); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s6, s11); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s7, s8); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s4, s9); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); }
            m[16] = s0; // barrier: stops the message loads being hoisted
            // Round 3
            s0 = AdvSimd.Add(AdvSimd.Add(s0, m[10]), s4); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[12]), s5); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[14]), s6); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[13]), s7);
            s12 = AdvSimd.ReverseElement16(AdvSimd.Xor(s12, s0).AsInt32()).AsUInt32(); s13 = AdvSimd.ReverseElement16(AdvSimd.Xor(s13, s1).AsInt32()).AsUInt32(); s14 = AdvSimd.ReverseElement16(AdvSimd.Xor(s14, s2).AsInt32()).AsUInt32(); s15 = AdvSimd.ReverseElement16(AdvSimd.Xor(s15, s3).AsInt32()).AsUInt32();
            s8 = AdvSimd.Add(s8, s12); s9 = AdvSimd.Add(s9, s13); s10 = AdvSimd.Add(s10, s14); s11 = AdvSimd.Add(s11, s15);
            { var t = AdvSimd.Xor(s4, s8); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s5, s9); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s6, s10); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s7, s11); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); }
            s0 = AdvSimd.Add(AdvSimd.Add(s0, m[7]), s4); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[9]), s5); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[3]), s6); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[15]), s7);
            s12 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s12, s0).AsByte(), rot8Mask).AsUInt32(); s13 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s13, s1).AsByte(), rot8Mask).AsUInt32(); s14 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s14, s2).AsByte(), rot8Mask).AsUInt32(); s15 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s15, s3).AsByte(), rot8Mask).AsUInt32();
            s8 = AdvSimd.Add(s8, s12); s9 = AdvSimd.Add(s9, s13); s10 = AdvSimd.Add(s10, s14); s11 = AdvSimd.Add(s11, s15);
            { var t = AdvSimd.Xor(s4, s8); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s5, s9); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s6, s10); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s7, s11); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); }
            s0 = AdvSimd.Add(AdvSimd.Add(s0, m[4]), s5); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[11]), s6); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[5]), s7); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[1]), s4);
            s15 = AdvSimd.ReverseElement16(AdvSimd.Xor(s15, s0).AsInt32()).AsUInt32(); s12 = AdvSimd.ReverseElement16(AdvSimd.Xor(s12, s1).AsInt32()).AsUInt32(); s13 = AdvSimd.ReverseElement16(AdvSimd.Xor(s13, s2).AsInt32()).AsUInt32(); s14 = AdvSimd.ReverseElement16(AdvSimd.Xor(s14, s3).AsInt32()).AsUInt32();
            s10 = AdvSimd.Add(s10, s15); s11 = AdvSimd.Add(s11, s12); s8 = AdvSimd.Add(s8, s13); s9 = AdvSimd.Add(s9, s14);
            { var t = AdvSimd.Xor(s5, s10); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s6, s11); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s7, s8); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s4, s9); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); }
            s0 = AdvSimd.Add(AdvSimd.Add(s0, m[0]), s5); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[2]), s6); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[8]), s7); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[6]), s4);
            s15 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s15, s0).AsByte(), rot8Mask).AsUInt32(); s12 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s12, s1).AsByte(), rot8Mask).AsUInt32(); s13 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s13, s2).AsByte(), rot8Mask).AsUInt32(); s14 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s14, s3).AsByte(), rot8Mask).AsUInt32();
            s10 = AdvSimd.Add(s10, s15); s11 = AdvSimd.Add(s11, s12); s8 = AdvSimd.Add(s8, s13); s9 = AdvSimd.Add(s9, s14);
            { var t = AdvSimd.Xor(s5, s10); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s6, s11); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s7, s8); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s4, s9); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); }
            m[16] = s0; // barrier: stops the message loads being hoisted
            // Round 4
            s0 = AdvSimd.Add(AdvSimd.Add(s0, m[12]), s4); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[9]), s5); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[15]), s6); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[14]), s7);
            s12 = AdvSimd.ReverseElement16(AdvSimd.Xor(s12, s0).AsInt32()).AsUInt32(); s13 = AdvSimd.ReverseElement16(AdvSimd.Xor(s13, s1).AsInt32()).AsUInt32(); s14 = AdvSimd.ReverseElement16(AdvSimd.Xor(s14, s2).AsInt32()).AsUInt32(); s15 = AdvSimd.ReverseElement16(AdvSimd.Xor(s15, s3).AsInt32()).AsUInt32();
            s8 = AdvSimd.Add(s8, s12); s9 = AdvSimd.Add(s9, s13); s10 = AdvSimd.Add(s10, s14); s11 = AdvSimd.Add(s11, s15);
            { var t = AdvSimd.Xor(s4, s8); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s5, s9); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s6, s10); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s7, s11); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); }
            s0 = AdvSimd.Add(AdvSimd.Add(s0, m[13]), s4); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[11]), s5); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[10]), s6); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[8]), s7);
            s12 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s12, s0).AsByte(), rot8Mask).AsUInt32(); s13 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s13, s1).AsByte(), rot8Mask).AsUInt32(); s14 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s14, s2).AsByte(), rot8Mask).AsUInt32(); s15 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s15, s3).AsByte(), rot8Mask).AsUInt32();
            s8 = AdvSimd.Add(s8, s12); s9 = AdvSimd.Add(s9, s13); s10 = AdvSimd.Add(s10, s14); s11 = AdvSimd.Add(s11, s15);
            { var t = AdvSimd.Xor(s4, s8); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s5, s9); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s6, s10); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s7, s11); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); }
            s0 = AdvSimd.Add(AdvSimd.Add(s0, m[7]), s5); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[5]), s6); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[0]), s7); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[6]), s4);
            s15 = AdvSimd.ReverseElement16(AdvSimd.Xor(s15, s0).AsInt32()).AsUInt32(); s12 = AdvSimd.ReverseElement16(AdvSimd.Xor(s12, s1).AsInt32()).AsUInt32(); s13 = AdvSimd.ReverseElement16(AdvSimd.Xor(s13, s2).AsInt32()).AsUInt32(); s14 = AdvSimd.ReverseElement16(AdvSimd.Xor(s14, s3).AsInt32()).AsUInt32();
            s10 = AdvSimd.Add(s10, s15); s11 = AdvSimd.Add(s11, s12); s8 = AdvSimd.Add(s8, s13); s9 = AdvSimd.Add(s9, s14);
            { var t = AdvSimd.Xor(s5, s10); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s6, s11); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s7, s8); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s4, s9); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); }
            s0 = AdvSimd.Add(AdvSimd.Add(s0, m[2]), s5); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[3]), s6); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[1]), s7); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[4]), s4);
            s15 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s15, s0).AsByte(), rot8Mask).AsUInt32(); s12 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s12, s1).AsByte(), rot8Mask).AsUInt32(); s13 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s13, s2).AsByte(), rot8Mask).AsUInt32(); s14 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s14, s3).AsByte(), rot8Mask).AsUInt32();
            s10 = AdvSimd.Add(s10, s15); s11 = AdvSimd.Add(s11, s12); s8 = AdvSimd.Add(s8, s13); s9 = AdvSimd.Add(s9, s14);
            { var t = AdvSimd.Xor(s5, s10); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s6, s11); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s7, s8); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s4, s9); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); }
            m[16] = s0; // barrier: stops the message loads being hoisted
            // Round 5
            s0 = AdvSimd.Add(AdvSimd.Add(s0, m[9]), s4); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[11]), s5); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[8]), s6); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[15]), s7);
            s12 = AdvSimd.ReverseElement16(AdvSimd.Xor(s12, s0).AsInt32()).AsUInt32(); s13 = AdvSimd.ReverseElement16(AdvSimd.Xor(s13, s1).AsInt32()).AsUInt32(); s14 = AdvSimd.ReverseElement16(AdvSimd.Xor(s14, s2).AsInt32()).AsUInt32(); s15 = AdvSimd.ReverseElement16(AdvSimd.Xor(s15, s3).AsInt32()).AsUInt32();
            s8 = AdvSimd.Add(s8, s12); s9 = AdvSimd.Add(s9, s13); s10 = AdvSimd.Add(s10, s14); s11 = AdvSimd.Add(s11, s15);
            { var t = AdvSimd.Xor(s4, s8); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s5, s9); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s6, s10); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s7, s11); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); }
            s0 = AdvSimd.Add(AdvSimd.Add(s0, m[14]), s4); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[5]), s5); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[12]), s6); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[1]), s7);
            s12 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s12, s0).AsByte(), rot8Mask).AsUInt32(); s13 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s13, s1).AsByte(), rot8Mask).AsUInt32(); s14 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s14, s2).AsByte(), rot8Mask).AsUInt32(); s15 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s15, s3).AsByte(), rot8Mask).AsUInt32();
            s8 = AdvSimd.Add(s8, s12); s9 = AdvSimd.Add(s9, s13); s10 = AdvSimd.Add(s10, s14); s11 = AdvSimd.Add(s11, s15);
            { var t = AdvSimd.Xor(s4, s8); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s5, s9); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s6, s10); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s7, s11); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); }
            s0 = AdvSimd.Add(AdvSimd.Add(s0, m[13]), s5); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[0]), s6); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[2]), s7); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[4]), s4);
            s15 = AdvSimd.ReverseElement16(AdvSimd.Xor(s15, s0).AsInt32()).AsUInt32(); s12 = AdvSimd.ReverseElement16(AdvSimd.Xor(s12, s1).AsInt32()).AsUInt32(); s13 = AdvSimd.ReverseElement16(AdvSimd.Xor(s13, s2).AsInt32()).AsUInt32(); s14 = AdvSimd.ReverseElement16(AdvSimd.Xor(s14, s3).AsInt32()).AsUInt32();
            s10 = AdvSimd.Add(s10, s15); s11 = AdvSimd.Add(s11, s12); s8 = AdvSimd.Add(s8, s13); s9 = AdvSimd.Add(s9, s14);
            { var t = AdvSimd.Xor(s5, s10); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s6, s11); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s7, s8); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s4, s9); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); }
            s0 = AdvSimd.Add(AdvSimd.Add(s0, m[3]), s5); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[10]), s6); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[6]), s7); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[7]), s4);
            s15 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s15, s0).AsByte(), rot8Mask).AsUInt32(); s12 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s12, s1).AsByte(), rot8Mask).AsUInt32(); s13 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s13, s2).AsByte(), rot8Mask).AsUInt32(); s14 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s14, s3).AsByte(), rot8Mask).AsUInt32();
            s10 = AdvSimd.Add(s10, s15); s11 = AdvSimd.Add(s11, s12); s8 = AdvSimd.Add(s8, s13); s9 = AdvSimd.Add(s9, s14);
            { var t = AdvSimd.Xor(s5, s10); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s6, s11); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s7, s8); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s4, s9); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); }
            m[16] = s0; // barrier: stops the message loads being hoisted
            // Round 6
            s0 = AdvSimd.Add(AdvSimd.Add(s0, m[11]), s4); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[5]), s5); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[1]), s6); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[8]), s7);
            s12 = AdvSimd.ReverseElement16(AdvSimd.Xor(s12, s0).AsInt32()).AsUInt32(); s13 = AdvSimd.ReverseElement16(AdvSimd.Xor(s13, s1).AsInt32()).AsUInt32(); s14 = AdvSimd.ReverseElement16(AdvSimd.Xor(s14, s2).AsInt32()).AsUInt32(); s15 = AdvSimd.ReverseElement16(AdvSimd.Xor(s15, s3).AsInt32()).AsUInt32();
            s8 = AdvSimd.Add(s8, s12); s9 = AdvSimd.Add(s9, s13); s10 = AdvSimd.Add(s10, s14); s11 = AdvSimd.Add(s11, s15);
            { var t = AdvSimd.Xor(s4, s8); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s5, s9); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s6, s10); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s7, s11); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); }
            s0 = AdvSimd.Add(AdvSimd.Add(s0, m[15]), s4); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[0]), s5); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[9]), s6); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[6]), s7);
            s12 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s12, s0).AsByte(), rot8Mask).AsUInt32(); s13 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s13, s1).AsByte(), rot8Mask).AsUInt32(); s14 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s14, s2).AsByte(), rot8Mask).AsUInt32(); s15 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s15, s3).AsByte(), rot8Mask).AsUInt32();
            s8 = AdvSimd.Add(s8, s12); s9 = AdvSimd.Add(s9, s13); s10 = AdvSimd.Add(s10, s14); s11 = AdvSimd.Add(s11, s15);
            { var t = AdvSimd.Xor(s4, s8); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s5, s9); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s6, s10); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s7, s11); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); }
            s0 = AdvSimd.Add(AdvSimd.Add(s0, m[14]), s5); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[2]), s6); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[3]), s7); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[7]), s4);
            s15 = AdvSimd.ReverseElement16(AdvSimd.Xor(s15, s0).AsInt32()).AsUInt32(); s12 = AdvSimd.ReverseElement16(AdvSimd.Xor(s12, s1).AsInt32()).AsUInt32(); s13 = AdvSimd.ReverseElement16(AdvSimd.Xor(s13, s2).AsInt32()).AsUInt32(); s14 = AdvSimd.ReverseElement16(AdvSimd.Xor(s14, s3).AsInt32()).AsUInt32();
            s10 = AdvSimd.Add(s10, s15); s11 = AdvSimd.Add(s11, s12); s8 = AdvSimd.Add(s8, s13); s9 = AdvSimd.Add(s9, s14);
            { var t = AdvSimd.Xor(s5, s10); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s6, s11); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s7, s8); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); } { var t = AdvSimd.Xor(s4, s9); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 12), AdvSimd.ShiftLeftLogical(t, 20)); }
            s0 = AdvSimd.Add(AdvSimd.Add(s0, m[10]), s5); s1 = AdvSimd.Add(AdvSimd.Add(s1, m[12]), s6); s2 = AdvSimd.Add(AdvSimd.Add(s2, m[4]), s7); s3 = AdvSimd.Add(AdvSimd.Add(s3, m[13]), s4);
            s15 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s15, s0).AsByte(), rot8Mask).AsUInt32(); s12 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s12, s1).AsByte(), rot8Mask).AsUInt32(); s13 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s13, s2).AsByte(), rot8Mask).AsUInt32(); s14 = AdvSimd.Arm64.VectorTableLookup(AdvSimd.Xor(s14, s3).AsByte(), rot8Mask).AsUInt32();
            s10 = AdvSimd.Add(s10, s15); s11 = AdvSimd.Add(s11, s12); s8 = AdvSimd.Add(s8, s13); s9 = AdvSimd.Add(s9, s14);
            { var t = AdvSimd.Xor(s5, s10); s5 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s6, s11); s6 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s7, s8); s7 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); } { var t = AdvSimd.Xor(s4, s9); s4 = AdvSimd.Or(AdvSimd.ShiftRightLogical(t, 7), AdvSimd.ShiftLeftLogical(t, 25)); }
            m[16] = s0; // barrier: stops the message loads being hoisted

            var cv0 = AdvSimd.Xor(s0, s8);
            var cv1 = AdvSimd.Xor(s1, s9);
            var cv2 = AdvSimd.Xor(s2, s10);
            var cv3 = AdvSimd.Xor(s3, s11);
            var cv4 = AdvSimd.Xor(s4, s12);
            var cv5 = AdvSimd.Xor(s5, s13);
            var cv6 = AdvSimd.Xor(s6, s14);
            var cv7 = AdvSimd.Xor(s7, s15);

            Transpose4X4(cv0, cv1, cv2, cv3, out var o0, out var o1, out var o2, out var o3);
            Transpose4X4(cv4, cv5, cv6, cv7, out var o4, out var o5, out var o6, out var o7);

            ref uint outRef = ref MemoryMarshal.GetReference(cvs);
            VectorCompat.Store(o0, ref outRef);
            VectorCompat.Store(o4, ref outRef, 4);
            VectorCompat.Store(o1, ref outRef, 8);
            VectorCompat.Store(o5, ref outRef, 12);
            VectorCompat.Store(o2, ref outRef, 16);
            VectorCompat.Store(o6, ref outRef, 20);
            VectorCompat.Store(o3, ref outRef, 24);
            VectorCompat.Store(o7, ref outRef, 28);
        }
    }
}
