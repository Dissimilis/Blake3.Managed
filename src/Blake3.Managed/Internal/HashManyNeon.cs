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
