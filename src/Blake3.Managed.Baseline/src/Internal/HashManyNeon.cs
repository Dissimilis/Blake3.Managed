using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;

namespace Blake3.Managed.Internal;

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

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void G128(ref Vector128<uint> a, ref Vector128<uint> b,
                             ref Vector128<uint> c, ref Vector128<uint> d,
                             Vector128<uint> mx, Vector128<uint> my)
    {
        a = AdvSimd.Add(AdvSimd.Add(a, b), mx);
        d = RotateRight16(AdvSimd.Xor(d, a));
        c = AdvSimd.Add(c, d);
        b = RotateRight12(AdvSimd.Xor(b, c));
        a = AdvSimd.Add(AdvSimd.Add(a, b), my);
        d = RotateRight8(AdvSimd.Xor(d, a));
        c = AdvSimd.Add(c, d);
        b = RotateRight7(AdvSimd.Xor(b, c));
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

        var iv0 = Vector128.Create(Blake3Constants.IV[0]);
        var iv1 = Vector128.Create(Blake3Constants.IV[1]);
        var iv2 = Vector128.Create(Blake3Constants.IV[2]);
        var iv3 = Vector128.Create(Blake3Constants.IV[3]);
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

        var ivVec0 = Vector128.Create(Blake3Constants.IV[0]);
        var ivVec1 = Vector128.Create(Blake3Constants.IV[1]);
        var ivVec2 = Vector128.Create(Blake3Constants.IV[2]);
        var ivVec3 = Vector128.Create(Blake3Constants.IV[3]);
        var blockLenVec = Vector128.Create((uint)Blake3Constants.BlockLen);

        fixed (byte* chunksPtr = chunks)
        {
            Vector128<uint>* m = stackalloc Vector128<uint>[16];

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

                // Round 0: 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
                G128(ref s0, ref s4, ref s8,  ref s12, m[0],  m[1]);
                G128(ref s1, ref s5, ref s9,  ref s13, m[2],  m[3]);
                G128(ref s2, ref s6, ref s10, ref s14, m[4],  m[5]);
                G128(ref s3, ref s7, ref s11, ref s15, m[6],  m[7]);
                G128(ref s0, ref s5, ref s10, ref s15, m[8],  m[9]);
                G128(ref s1, ref s6, ref s11, ref s12, m[10], m[11]);
                G128(ref s2, ref s7, ref s8,  ref s13, m[12], m[13]);
                G128(ref s3, ref s4, ref s9,  ref s14, m[14], m[15]);
                // Round 1: 2,6,3,10,7,0,4,13,1,11,12,5,9,14,15,8
                G128(ref s0, ref s4, ref s8,  ref s12, m[2],  m[6]);
                G128(ref s1, ref s5, ref s9,  ref s13, m[3],  m[10]);
                G128(ref s2, ref s6, ref s10, ref s14, m[7],  m[0]);
                G128(ref s3, ref s7, ref s11, ref s15, m[4],  m[13]);
                G128(ref s0, ref s5, ref s10, ref s15, m[1],  m[11]);
                G128(ref s1, ref s6, ref s11, ref s12, m[12], m[5]);
                G128(ref s2, ref s7, ref s8,  ref s13, m[9],  m[14]);
                G128(ref s3, ref s4, ref s9,  ref s14, m[15], m[8]);
                // Round 2: 3,4,10,12,13,2,7,14,6,5,9,0,11,15,8,1
                G128(ref s0, ref s4, ref s8,  ref s12, m[3],  m[4]);
                G128(ref s1, ref s5, ref s9,  ref s13, m[10], m[12]);
                G128(ref s2, ref s6, ref s10, ref s14, m[13], m[2]);
                G128(ref s3, ref s7, ref s11, ref s15, m[7],  m[14]);
                G128(ref s0, ref s5, ref s10, ref s15, m[6],  m[5]);
                G128(ref s1, ref s6, ref s11, ref s12, m[9],  m[0]);
                G128(ref s2, ref s7, ref s8,  ref s13, m[11], m[15]);
                G128(ref s3, ref s4, ref s9,  ref s14, m[8],  m[1]);
                // Round 3: 10,7,12,9,14,3,13,15,4,0,11,2,5,8,1,6
                G128(ref s0, ref s4, ref s8,  ref s12, m[10], m[7]);
                G128(ref s1, ref s5, ref s9,  ref s13, m[12], m[9]);
                G128(ref s2, ref s6, ref s10, ref s14, m[14], m[3]);
                G128(ref s3, ref s7, ref s11, ref s15, m[13], m[15]);
                G128(ref s0, ref s5, ref s10, ref s15, m[4],  m[0]);
                G128(ref s1, ref s6, ref s11, ref s12, m[11], m[2]);
                G128(ref s2, ref s7, ref s8,  ref s13, m[5],  m[8]);
                G128(ref s3, ref s4, ref s9,  ref s14, m[1],  m[6]);
                // Round 4: 12,13,9,11,15,10,14,8,7,2,5,3,0,1,6,4
                G128(ref s0, ref s4, ref s8,  ref s12, m[12], m[13]);
                G128(ref s1, ref s5, ref s9,  ref s13, m[9],  m[11]);
                G128(ref s2, ref s6, ref s10, ref s14, m[15], m[10]);
                G128(ref s3, ref s7, ref s11, ref s15, m[14], m[8]);
                G128(ref s0, ref s5, ref s10, ref s15, m[7],  m[2]);
                G128(ref s1, ref s6, ref s11, ref s12, m[5],  m[3]);
                G128(ref s2, ref s7, ref s8,  ref s13, m[0],  m[1]);
                G128(ref s3, ref s4, ref s9,  ref s14, m[6],  m[4]);
                // Round 5: 9,14,11,5,8,12,15,1,13,3,0,10,2,6,4,7
                G128(ref s0, ref s4, ref s8,  ref s12, m[9],  m[14]);
                G128(ref s1, ref s5, ref s9,  ref s13, m[11], m[5]);
                G128(ref s2, ref s6, ref s10, ref s14, m[8],  m[12]);
                G128(ref s3, ref s7, ref s11, ref s15, m[15], m[1]);
                G128(ref s0, ref s5, ref s10, ref s15, m[13], m[3]);
                G128(ref s1, ref s6, ref s11, ref s12, m[0],  m[10]);
                G128(ref s2, ref s7, ref s8,  ref s13, m[2],  m[6]);
                G128(ref s3, ref s4, ref s9,  ref s14, m[4],  m[7]);
                // Round 6: 11,15,5,0,1,9,8,6,14,10,2,12,3,4,7,13
                G128(ref s0, ref s4, ref s8,  ref s12, m[11], m[15]);
                G128(ref s1, ref s5, ref s9,  ref s13, m[5],  m[0]);
                G128(ref s2, ref s6, ref s10, ref s14, m[1],  m[9]);
                G128(ref s3, ref s7, ref s11, ref s15, m[8],  m[6]);
                G128(ref s0, ref s5, ref s10, ref s15, m[14], m[10]);
                G128(ref s1, ref s6, ref s11, ref s12, m[2],  m[12]);
                G128(ref s2, ref s7, ref s8,  ref s13, m[3],  m[4]);
                G128(ref s3, ref s4, ref s9,  ref s14, m[7],  m[13]);

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
}
