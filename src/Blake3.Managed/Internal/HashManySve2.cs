#if NET10_0_OR_GREATER
// Sve2 is [Experimental] in .NET 10 (SYSLIB5003). Only XorRotateRight is used, on Vector128
// values reinterpreted as Vector<T>; XAR is lane-wise, so this is correct at any SVE vector length.
#pragma warning disable SYSLIB5003
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;

namespace Blake3.Managed.Internal;

/// <summary>
/// The <see cref="HashManyNeon"/> kernels with every xor-then-rotate fused into one SVE2
/// <c>XAR</c>, plus an 8-chunk kernel. A NEON G step costs 18 vector ops (6 adds, 4 xors, 8 for
/// the rotates: REV32, TBL and two shift-shift-or triples); with XAR it costs 10. The 4-chunk
/// bodies are copied statement for statement from <see cref="HashManyNeon"/>; only
/// <see cref="G128"/> differs.
/// <para>
/// Measured on Graviton4 (Neoverse-V2, 128-bit SVE, .NET 10.0.12, 2026-09-23): the 4-way kernel
/// ran in 0.675 of the NEON time with 31% fewer instructions. Fusing only the 12- and 7-bit
/// rotates (0.718) or 12/8/7 (0.702) was slower than fusing all four.
/// </para>
/// <para>
/// The 4-chunk kernels are reached through the <see cref="HashManyNeon"/> entry points, which
/// forward here; <see cref="HashMany8"/> is called directly from <see cref="Blake3Tree"/> and
/// <c>Blake3Core.HasherState</c>. Every call site is guarded by <see cref="IsSupported"/>, a
/// JIT-time constant, and was placed so that the NEON and x86 code compiles byte for byte as
/// before (checked by disassembly on Cortex-A73 and four x86 tiers; see CLAUDE.md).
/// </para>
/// </summary>
internal static class HashManySve2
{
    public static bool IsSupported => Sve2.IsSupported && AdvSimd.Arm64.IsSupported;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<uint> XorRotateRight(Vector128<uint> x, Vector128<uint> y,
                                                  [System.Diagnostics.CodeAnalysis.ConstantExpected] byte count)
    {
        return Sve2.XorRotateRight(x.AsVector(), y.AsVector(), count).AsVector128();
    }

    // XAR's first operand is also its destination, so the value being overwritten goes first.
    //
    // The kernel is latency-bound: each G is one dependent chain of adds and XARs, and XAR is the
    // slow link. b is always the newest input (it comes straight out of the previous XAR), so the
    // message word is added to a first, off the critical path: (a + m) + b rather than
    // (a + b) + m. That removes one add from the chain twice per G and measured 0.866 of the
    // (a + b) + m time on Graviton4 (2026-09-23), in line with a 28 -> 24 cycle chain.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void G128(ref Vector128<uint> a, ref Vector128<uint> b,
                             ref Vector128<uint> c, ref Vector128<uint> d,
                             Vector128<uint> mx, Vector128<uint> my)
    {
        a = AdvSimd.Add(AdvSimd.Add(a, mx), b);
        d = XorRotateRight(d, a, 16);
        c = AdvSimd.Add(c, d);
        b = XorRotateRight(b, c, 12);
        a = AdvSimd.Add(AdvSimd.Add(a, my), b);
        d = XorRotateRight(d, a, 8);
        c = AdvSimd.Add(c, d);
        b = XorRotateRight(b, c, 7);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Transpose4X4(
        Vector128<uint> r0, Vector128<uint> r1,
        Vector128<uint> r2, Vector128<uint> r3,
        out Vector128<uint> m0, out Vector128<uint> m1,
        out Vector128<uint> m2, out Vector128<uint> m3)
    {
        var t0 = AdvSimd.Arm64.ZipLow(r0, r2);
        var t1 = AdvSimd.Arm64.ZipHigh(r0, r2);
        var t2 = AdvSimd.Arm64.ZipLow(r1, r3);
        var t3 = AdvSimd.Arm64.ZipHigh(r1, r3);
        m0 = AdvSimd.Arm64.ZipLow(t0, t2);
        m1 = AdvSimd.Arm64.ZipHigh(t0, t2);
        m2 = AdvSimd.Arm64.ZipLow(t1, t3);
        m3 = AdvSimd.Arm64.ZipHigh(t1, t3);
    }

    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.NoInlining | MethodImplOptions.AggressiveOptimization)]
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

    /// <summary>As <see cref="HashManyNeon.HashManyPartial"/>; called with 2 or 3 chunks.</summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.NoInlining | MethodImplOptions.AggressiveOptimization)]
    public static unsafe void HashManyPartial(ReadOnlySpan<byte> chunks, int numChunks,
                                       ReadOnlySpan<uint> key, ulong startCounter,
                                       uint flags, Span<uint> cvs)
    {
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
            Vector128<uint>* m = stackalloc Vector128<uint>[16];

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

    /// <summary>As <see cref="HashManyNeon.HashParents4"/>.</summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.NoInlining | MethodImplOptions.AggressiveOptimization)]
    public static unsafe void HashParents4(ReadOnlySpan<uint> parentBlocks,
                                           ReadOnlySpan<uint> key, uint flags,
                                           Span<uint> cvs)
    {
        fixed (uint* inPtr = parentBlocks)
        {
            byte* basePtr = (byte*)inPtr;
            Vector128<uint>* m = stackalloc Vector128<uint>[16];

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

    /// <summary>
    /// Eight complete chunks as two independent 4-chunk batches (A: chunks 0-3, B: 4-7) whose
    /// rounds are interleaved per half-round. One batch is a latency-bound chain -- each G is a
    /// serial run of adds and XARs, and four of them only half-fill V2's vector pipes -- so a
    /// second, independent batch runs in the gaps.
    /// <para>
    /// Written out statement by statement, with no G helper. As a helper, the 112 G calls ran the
    /// JIT out of inlining budget partway through the method; the rest became real calls and the
    /// kernel ran 5.6x slower. Expanded, it has 32 live state vectors and spills some of them, and
    /// still measured 0.791 of two <see cref="HashMany"/> calls on Graviton4 (2026-09-23).
    /// Interleaving per G instead of per half-round measured 0.899.
    /// </para>
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.NoInlining | MethodImplOptions.AggressiveOptimization)]
    public static unsafe void HashMany8(ReadOnlySpan<byte> chunks, ReadOnlySpan<uint> key,
                                        ulong startCounter, uint flags, Span<uint> cvs)
    {
        const int blocksPerChunk = Blake3Constants.ChunkLen / Blake3Constants.BlockLen; // 16
        const int chunkLen = Blake3Constants.ChunkLen;

        Vector128<uint> ca0 = Vector128.Create(key[0]), ca1 = Vector128.Create(key[1]);
        Vector128<uint> ca2 = Vector128.Create(key[2]), ca3 = Vector128.Create(key[3]);
        Vector128<uint> ca4 = Vector128.Create(key[4]), ca5 = Vector128.Create(key[5]);
        Vector128<uint> ca6 = Vector128.Create(key[6]), ca7 = Vector128.Create(key[7]);
        Vector128<uint> cb0 = ca0, cb1 = ca1, cb2 = ca2, cb3 = ca3;
        Vector128<uint> cb4 = ca4, cb5 = ca5, cb6 = ca6, cb7 = ca7;

        var counterLoA = Vector128.Create(
            (uint)(startCounter + 0), (uint)(startCounter + 1),
            (uint)(startCounter + 2), (uint)(startCounter + 3));
        var counterHiA = Vector128.Create(
            (uint)((startCounter + 0) >> 32), (uint)((startCounter + 1) >> 32),
            (uint)((startCounter + 2) >> 32), (uint)((startCounter + 3) >> 32));
        var counterLoB = Vector128.Create(
            (uint)(startCounter + 4), (uint)(startCounter + 5),
            (uint)(startCounter + 6), (uint)(startCounter + 7));
        var counterHiB = Vector128.Create(
            (uint)((startCounter + 4) >> 32), (uint)((startCounter + 5) >> 32),
            (uint)((startCounter + 6) >> 32), (uint)((startCounter + 7) >> 32));

        fixed (byte* chunksPtr = chunks)
        {
            Vector128<uint>* mA = stackalloc Vector128<uint>[32];
            Vector128<uint>* mB = mA + 16;

            for (int blockIdx = 0; blockIdx < blocksPerChunk; blockIdx++)
            {
                byte* blockBase = chunksPtr + blockIdx * 64;
                for (int g = 0; g < 4; g++)
                {
                    Transpose4X4(
                        Unsafe.ReadUnaligned<Vector128<uint>>(blockBase + 0 * chunkLen + g * 16),
                        Unsafe.ReadUnaligned<Vector128<uint>>(blockBase + 1 * chunkLen + g * 16),
                        Unsafe.ReadUnaligned<Vector128<uint>>(blockBase + 2 * chunkLen + g * 16),
                        Unsafe.ReadUnaligned<Vector128<uint>>(blockBase + 3 * chunkLen + g * 16),
                        out mA[g * 4], out mA[g * 4 + 1], out mA[g * 4 + 2], out mA[g * 4 + 3]);
                    Transpose4X4(
                        Unsafe.ReadUnaligned<Vector128<uint>>(blockBase + 4 * chunkLen + g * 16),
                        Unsafe.ReadUnaligned<Vector128<uint>>(blockBase + 5 * chunkLen + g * 16),
                        Unsafe.ReadUnaligned<Vector128<uint>>(blockBase + 6 * chunkLen + g * 16),
                        Unsafe.ReadUnaligned<Vector128<uint>>(blockBase + 7 * chunkLen + g * 16),
                        out mB[g * 4], out mB[g * 4 + 1], out mB[g * 4 + 2], out mB[g * 4 + 3]);
                }

                uint blockFlags = flags;
                if (blockIdx == 0) blockFlags |= Blake3Constants.ChunkStart;
                if (blockIdx == blocksPerChunk - 1) blockFlags |= Blake3Constants.ChunkEnd;
                var flagsVec = Vector128.Create(blockFlags);
                var blockLenVec = Vector128.Create((uint)Blake3Constants.BlockLen);

                Vector128<uint> a0 = ca0, a1 = ca1, a2 = ca2, a3 = ca3, a4 = ca4, a5 = ca5, a6 = ca6, a7 = ca7;
                Vector128<uint> a8 = Vector128.Create(Blake3Constants.Iv0), a9 = Vector128.Create(Blake3Constants.Iv1);
                Vector128<uint> a10 = Vector128.Create(Blake3Constants.Iv2), a11 = Vector128.Create(Blake3Constants.Iv3);
                Vector128<uint> a12 = counterLoA, a13 = counterHiA, a14 = blockLenVec, a15 = flagsVec;
                Vector128<uint> b0 = cb0, b1 = cb1, b2 = cb2, b3 = cb3, b4 = cb4, b5 = cb5, b6 = cb6, b7 = cb7;
                Vector128<uint> b8 = a8, b9 = a9, b10 = a10, b11 = a11;
                Vector128<uint> b12 = counterLoB, b13 = counterHiB, b14 = blockLenVec, b15 = flagsVec;

                // Round 0: 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
                // columns, batch A
                a0 = AdvSimd.Add(AdvSimd.Add(a0, mA[0]), a4); a12 = Sve2.XorRotateRight(a12.AsVector(), a0.AsVector(), 16).AsVector128(); a8 = AdvSimd.Add(a8, a12); a4 = Sve2.XorRotateRight(a4.AsVector(), a8.AsVector(), 12).AsVector128();
                a0 = AdvSimd.Add(AdvSimd.Add(a0, mA[1]), a4); a12 = Sve2.XorRotateRight(a12.AsVector(), a0.AsVector(), 8).AsVector128(); a8 = AdvSimd.Add(a8, a12); a4 = Sve2.XorRotateRight(a4.AsVector(), a8.AsVector(), 7).AsVector128();
                a1 = AdvSimd.Add(AdvSimd.Add(a1, mA[2]), a5); a13 = Sve2.XorRotateRight(a13.AsVector(), a1.AsVector(), 16).AsVector128(); a9 = AdvSimd.Add(a9, a13); a5 = Sve2.XorRotateRight(a5.AsVector(), a9.AsVector(), 12).AsVector128();
                a1 = AdvSimd.Add(AdvSimd.Add(a1, mA[3]), a5); a13 = Sve2.XorRotateRight(a13.AsVector(), a1.AsVector(), 8).AsVector128(); a9 = AdvSimd.Add(a9, a13); a5 = Sve2.XorRotateRight(a5.AsVector(), a9.AsVector(), 7).AsVector128();
                a2 = AdvSimd.Add(AdvSimd.Add(a2, mA[4]), a6); a14 = Sve2.XorRotateRight(a14.AsVector(), a2.AsVector(), 16).AsVector128(); a10 = AdvSimd.Add(a10, a14); a6 = Sve2.XorRotateRight(a6.AsVector(), a10.AsVector(), 12).AsVector128();
                a2 = AdvSimd.Add(AdvSimd.Add(a2, mA[5]), a6); a14 = Sve2.XorRotateRight(a14.AsVector(), a2.AsVector(), 8).AsVector128(); a10 = AdvSimd.Add(a10, a14); a6 = Sve2.XorRotateRight(a6.AsVector(), a10.AsVector(), 7).AsVector128();
                a3 = AdvSimd.Add(AdvSimd.Add(a3, mA[6]), a7); a15 = Sve2.XorRotateRight(a15.AsVector(), a3.AsVector(), 16).AsVector128(); a11 = AdvSimd.Add(a11, a15); a7 = Sve2.XorRotateRight(a7.AsVector(), a11.AsVector(), 12).AsVector128();
                a3 = AdvSimd.Add(AdvSimd.Add(a3, mA[7]), a7); a15 = Sve2.XorRotateRight(a15.AsVector(), a3.AsVector(), 8).AsVector128(); a11 = AdvSimd.Add(a11, a15); a7 = Sve2.XorRotateRight(a7.AsVector(), a11.AsVector(), 7).AsVector128();
                // columns, batch B
                b0 = AdvSimd.Add(AdvSimd.Add(b0, mB[0]), b4); b12 = Sve2.XorRotateRight(b12.AsVector(), b0.AsVector(), 16).AsVector128(); b8 = AdvSimd.Add(b8, b12); b4 = Sve2.XorRotateRight(b4.AsVector(), b8.AsVector(), 12).AsVector128();
                b0 = AdvSimd.Add(AdvSimd.Add(b0, mB[1]), b4); b12 = Sve2.XorRotateRight(b12.AsVector(), b0.AsVector(), 8).AsVector128(); b8 = AdvSimd.Add(b8, b12); b4 = Sve2.XorRotateRight(b4.AsVector(), b8.AsVector(), 7).AsVector128();
                b1 = AdvSimd.Add(AdvSimd.Add(b1, mB[2]), b5); b13 = Sve2.XorRotateRight(b13.AsVector(), b1.AsVector(), 16).AsVector128(); b9 = AdvSimd.Add(b9, b13); b5 = Sve2.XorRotateRight(b5.AsVector(), b9.AsVector(), 12).AsVector128();
                b1 = AdvSimd.Add(AdvSimd.Add(b1, mB[3]), b5); b13 = Sve2.XorRotateRight(b13.AsVector(), b1.AsVector(), 8).AsVector128(); b9 = AdvSimd.Add(b9, b13); b5 = Sve2.XorRotateRight(b5.AsVector(), b9.AsVector(), 7).AsVector128();
                b2 = AdvSimd.Add(AdvSimd.Add(b2, mB[4]), b6); b14 = Sve2.XorRotateRight(b14.AsVector(), b2.AsVector(), 16).AsVector128(); b10 = AdvSimd.Add(b10, b14); b6 = Sve2.XorRotateRight(b6.AsVector(), b10.AsVector(), 12).AsVector128();
                b2 = AdvSimd.Add(AdvSimd.Add(b2, mB[5]), b6); b14 = Sve2.XorRotateRight(b14.AsVector(), b2.AsVector(), 8).AsVector128(); b10 = AdvSimd.Add(b10, b14); b6 = Sve2.XorRotateRight(b6.AsVector(), b10.AsVector(), 7).AsVector128();
                b3 = AdvSimd.Add(AdvSimd.Add(b3, mB[6]), b7); b15 = Sve2.XorRotateRight(b15.AsVector(), b3.AsVector(), 16).AsVector128(); b11 = AdvSimd.Add(b11, b15); b7 = Sve2.XorRotateRight(b7.AsVector(), b11.AsVector(), 12).AsVector128();
                b3 = AdvSimd.Add(AdvSimd.Add(b3, mB[7]), b7); b15 = Sve2.XorRotateRight(b15.AsVector(), b3.AsVector(), 8).AsVector128(); b11 = AdvSimd.Add(b11, b15); b7 = Sve2.XorRotateRight(b7.AsVector(), b11.AsVector(), 7).AsVector128();
                // diagonals, batch A
                a0 = AdvSimd.Add(AdvSimd.Add(a0, mA[8]), a5); a15 = Sve2.XorRotateRight(a15.AsVector(), a0.AsVector(), 16).AsVector128(); a10 = AdvSimd.Add(a10, a15); a5 = Sve2.XorRotateRight(a5.AsVector(), a10.AsVector(), 12).AsVector128();
                a0 = AdvSimd.Add(AdvSimd.Add(a0, mA[9]), a5); a15 = Sve2.XorRotateRight(a15.AsVector(), a0.AsVector(), 8).AsVector128(); a10 = AdvSimd.Add(a10, a15); a5 = Sve2.XorRotateRight(a5.AsVector(), a10.AsVector(), 7).AsVector128();
                a1 = AdvSimd.Add(AdvSimd.Add(a1, mA[10]), a6); a12 = Sve2.XorRotateRight(a12.AsVector(), a1.AsVector(), 16).AsVector128(); a11 = AdvSimd.Add(a11, a12); a6 = Sve2.XorRotateRight(a6.AsVector(), a11.AsVector(), 12).AsVector128();
                a1 = AdvSimd.Add(AdvSimd.Add(a1, mA[11]), a6); a12 = Sve2.XorRotateRight(a12.AsVector(), a1.AsVector(), 8).AsVector128(); a11 = AdvSimd.Add(a11, a12); a6 = Sve2.XorRotateRight(a6.AsVector(), a11.AsVector(), 7).AsVector128();
                a2 = AdvSimd.Add(AdvSimd.Add(a2, mA[12]), a7); a13 = Sve2.XorRotateRight(a13.AsVector(), a2.AsVector(), 16).AsVector128(); a8 = AdvSimd.Add(a8, a13); a7 = Sve2.XorRotateRight(a7.AsVector(), a8.AsVector(), 12).AsVector128();
                a2 = AdvSimd.Add(AdvSimd.Add(a2, mA[13]), a7); a13 = Sve2.XorRotateRight(a13.AsVector(), a2.AsVector(), 8).AsVector128(); a8 = AdvSimd.Add(a8, a13); a7 = Sve2.XorRotateRight(a7.AsVector(), a8.AsVector(), 7).AsVector128();
                a3 = AdvSimd.Add(AdvSimd.Add(a3, mA[14]), a4); a14 = Sve2.XorRotateRight(a14.AsVector(), a3.AsVector(), 16).AsVector128(); a9 = AdvSimd.Add(a9, a14); a4 = Sve2.XorRotateRight(a4.AsVector(), a9.AsVector(), 12).AsVector128();
                a3 = AdvSimd.Add(AdvSimd.Add(a3, mA[15]), a4); a14 = Sve2.XorRotateRight(a14.AsVector(), a3.AsVector(), 8).AsVector128(); a9 = AdvSimd.Add(a9, a14); a4 = Sve2.XorRotateRight(a4.AsVector(), a9.AsVector(), 7).AsVector128();
                // diagonals, batch B
                b0 = AdvSimd.Add(AdvSimd.Add(b0, mB[8]), b5); b15 = Sve2.XorRotateRight(b15.AsVector(), b0.AsVector(), 16).AsVector128(); b10 = AdvSimd.Add(b10, b15); b5 = Sve2.XorRotateRight(b5.AsVector(), b10.AsVector(), 12).AsVector128();
                b0 = AdvSimd.Add(AdvSimd.Add(b0, mB[9]), b5); b15 = Sve2.XorRotateRight(b15.AsVector(), b0.AsVector(), 8).AsVector128(); b10 = AdvSimd.Add(b10, b15); b5 = Sve2.XorRotateRight(b5.AsVector(), b10.AsVector(), 7).AsVector128();
                b1 = AdvSimd.Add(AdvSimd.Add(b1, mB[10]), b6); b12 = Sve2.XorRotateRight(b12.AsVector(), b1.AsVector(), 16).AsVector128(); b11 = AdvSimd.Add(b11, b12); b6 = Sve2.XorRotateRight(b6.AsVector(), b11.AsVector(), 12).AsVector128();
                b1 = AdvSimd.Add(AdvSimd.Add(b1, mB[11]), b6); b12 = Sve2.XorRotateRight(b12.AsVector(), b1.AsVector(), 8).AsVector128(); b11 = AdvSimd.Add(b11, b12); b6 = Sve2.XorRotateRight(b6.AsVector(), b11.AsVector(), 7).AsVector128();
                b2 = AdvSimd.Add(AdvSimd.Add(b2, mB[12]), b7); b13 = Sve2.XorRotateRight(b13.AsVector(), b2.AsVector(), 16).AsVector128(); b8 = AdvSimd.Add(b8, b13); b7 = Sve2.XorRotateRight(b7.AsVector(), b8.AsVector(), 12).AsVector128();
                b2 = AdvSimd.Add(AdvSimd.Add(b2, mB[13]), b7); b13 = Sve2.XorRotateRight(b13.AsVector(), b2.AsVector(), 8).AsVector128(); b8 = AdvSimd.Add(b8, b13); b7 = Sve2.XorRotateRight(b7.AsVector(), b8.AsVector(), 7).AsVector128();
                b3 = AdvSimd.Add(AdvSimd.Add(b3, mB[14]), b4); b14 = Sve2.XorRotateRight(b14.AsVector(), b3.AsVector(), 16).AsVector128(); b9 = AdvSimd.Add(b9, b14); b4 = Sve2.XorRotateRight(b4.AsVector(), b9.AsVector(), 12).AsVector128();
                b3 = AdvSimd.Add(AdvSimd.Add(b3, mB[15]), b4); b14 = Sve2.XorRotateRight(b14.AsVector(), b3.AsVector(), 8).AsVector128(); b9 = AdvSimd.Add(b9, b14); b4 = Sve2.XorRotateRight(b4.AsVector(), b9.AsVector(), 7).AsVector128();
                // Round 1: 2,6,3,10,7,0,4,13,1,11,12,5,9,14,15,8
                // columns, batch A
                a0 = AdvSimd.Add(AdvSimd.Add(a0, mA[2]), a4); a12 = Sve2.XorRotateRight(a12.AsVector(), a0.AsVector(), 16).AsVector128(); a8 = AdvSimd.Add(a8, a12); a4 = Sve2.XorRotateRight(a4.AsVector(), a8.AsVector(), 12).AsVector128();
                a0 = AdvSimd.Add(AdvSimd.Add(a0, mA[6]), a4); a12 = Sve2.XorRotateRight(a12.AsVector(), a0.AsVector(), 8).AsVector128(); a8 = AdvSimd.Add(a8, a12); a4 = Sve2.XorRotateRight(a4.AsVector(), a8.AsVector(), 7).AsVector128();
                a1 = AdvSimd.Add(AdvSimd.Add(a1, mA[3]), a5); a13 = Sve2.XorRotateRight(a13.AsVector(), a1.AsVector(), 16).AsVector128(); a9 = AdvSimd.Add(a9, a13); a5 = Sve2.XorRotateRight(a5.AsVector(), a9.AsVector(), 12).AsVector128();
                a1 = AdvSimd.Add(AdvSimd.Add(a1, mA[10]), a5); a13 = Sve2.XorRotateRight(a13.AsVector(), a1.AsVector(), 8).AsVector128(); a9 = AdvSimd.Add(a9, a13); a5 = Sve2.XorRotateRight(a5.AsVector(), a9.AsVector(), 7).AsVector128();
                a2 = AdvSimd.Add(AdvSimd.Add(a2, mA[7]), a6); a14 = Sve2.XorRotateRight(a14.AsVector(), a2.AsVector(), 16).AsVector128(); a10 = AdvSimd.Add(a10, a14); a6 = Sve2.XorRotateRight(a6.AsVector(), a10.AsVector(), 12).AsVector128();
                a2 = AdvSimd.Add(AdvSimd.Add(a2, mA[0]), a6); a14 = Sve2.XorRotateRight(a14.AsVector(), a2.AsVector(), 8).AsVector128(); a10 = AdvSimd.Add(a10, a14); a6 = Sve2.XorRotateRight(a6.AsVector(), a10.AsVector(), 7).AsVector128();
                a3 = AdvSimd.Add(AdvSimd.Add(a3, mA[4]), a7); a15 = Sve2.XorRotateRight(a15.AsVector(), a3.AsVector(), 16).AsVector128(); a11 = AdvSimd.Add(a11, a15); a7 = Sve2.XorRotateRight(a7.AsVector(), a11.AsVector(), 12).AsVector128();
                a3 = AdvSimd.Add(AdvSimd.Add(a3, mA[13]), a7); a15 = Sve2.XorRotateRight(a15.AsVector(), a3.AsVector(), 8).AsVector128(); a11 = AdvSimd.Add(a11, a15); a7 = Sve2.XorRotateRight(a7.AsVector(), a11.AsVector(), 7).AsVector128();
                // columns, batch B
                b0 = AdvSimd.Add(AdvSimd.Add(b0, mB[2]), b4); b12 = Sve2.XorRotateRight(b12.AsVector(), b0.AsVector(), 16).AsVector128(); b8 = AdvSimd.Add(b8, b12); b4 = Sve2.XorRotateRight(b4.AsVector(), b8.AsVector(), 12).AsVector128();
                b0 = AdvSimd.Add(AdvSimd.Add(b0, mB[6]), b4); b12 = Sve2.XorRotateRight(b12.AsVector(), b0.AsVector(), 8).AsVector128(); b8 = AdvSimd.Add(b8, b12); b4 = Sve2.XorRotateRight(b4.AsVector(), b8.AsVector(), 7).AsVector128();
                b1 = AdvSimd.Add(AdvSimd.Add(b1, mB[3]), b5); b13 = Sve2.XorRotateRight(b13.AsVector(), b1.AsVector(), 16).AsVector128(); b9 = AdvSimd.Add(b9, b13); b5 = Sve2.XorRotateRight(b5.AsVector(), b9.AsVector(), 12).AsVector128();
                b1 = AdvSimd.Add(AdvSimd.Add(b1, mB[10]), b5); b13 = Sve2.XorRotateRight(b13.AsVector(), b1.AsVector(), 8).AsVector128(); b9 = AdvSimd.Add(b9, b13); b5 = Sve2.XorRotateRight(b5.AsVector(), b9.AsVector(), 7).AsVector128();
                b2 = AdvSimd.Add(AdvSimd.Add(b2, mB[7]), b6); b14 = Sve2.XorRotateRight(b14.AsVector(), b2.AsVector(), 16).AsVector128(); b10 = AdvSimd.Add(b10, b14); b6 = Sve2.XorRotateRight(b6.AsVector(), b10.AsVector(), 12).AsVector128();
                b2 = AdvSimd.Add(AdvSimd.Add(b2, mB[0]), b6); b14 = Sve2.XorRotateRight(b14.AsVector(), b2.AsVector(), 8).AsVector128(); b10 = AdvSimd.Add(b10, b14); b6 = Sve2.XorRotateRight(b6.AsVector(), b10.AsVector(), 7).AsVector128();
                b3 = AdvSimd.Add(AdvSimd.Add(b3, mB[4]), b7); b15 = Sve2.XorRotateRight(b15.AsVector(), b3.AsVector(), 16).AsVector128(); b11 = AdvSimd.Add(b11, b15); b7 = Sve2.XorRotateRight(b7.AsVector(), b11.AsVector(), 12).AsVector128();
                b3 = AdvSimd.Add(AdvSimd.Add(b3, mB[13]), b7); b15 = Sve2.XorRotateRight(b15.AsVector(), b3.AsVector(), 8).AsVector128(); b11 = AdvSimd.Add(b11, b15); b7 = Sve2.XorRotateRight(b7.AsVector(), b11.AsVector(), 7).AsVector128();
                // diagonals, batch A
                a0 = AdvSimd.Add(AdvSimd.Add(a0, mA[1]), a5); a15 = Sve2.XorRotateRight(a15.AsVector(), a0.AsVector(), 16).AsVector128(); a10 = AdvSimd.Add(a10, a15); a5 = Sve2.XorRotateRight(a5.AsVector(), a10.AsVector(), 12).AsVector128();
                a0 = AdvSimd.Add(AdvSimd.Add(a0, mA[11]), a5); a15 = Sve2.XorRotateRight(a15.AsVector(), a0.AsVector(), 8).AsVector128(); a10 = AdvSimd.Add(a10, a15); a5 = Sve2.XorRotateRight(a5.AsVector(), a10.AsVector(), 7).AsVector128();
                a1 = AdvSimd.Add(AdvSimd.Add(a1, mA[12]), a6); a12 = Sve2.XorRotateRight(a12.AsVector(), a1.AsVector(), 16).AsVector128(); a11 = AdvSimd.Add(a11, a12); a6 = Sve2.XorRotateRight(a6.AsVector(), a11.AsVector(), 12).AsVector128();
                a1 = AdvSimd.Add(AdvSimd.Add(a1, mA[5]), a6); a12 = Sve2.XorRotateRight(a12.AsVector(), a1.AsVector(), 8).AsVector128(); a11 = AdvSimd.Add(a11, a12); a6 = Sve2.XorRotateRight(a6.AsVector(), a11.AsVector(), 7).AsVector128();
                a2 = AdvSimd.Add(AdvSimd.Add(a2, mA[9]), a7); a13 = Sve2.XorRotateRight(a13.AsVector(), a2.AsVector(), 16).AsVector128(); a8 = AdvSimd.Add(a8, a13); a7 = Sve2.XorRotateRight(a7.AsVector(), a8.AsVector(), 12).AsVector128();
                a2 = AdvSimd.Add(AdvSimd.Add(a2, mA[14]), a7); a13 = Sve2.XorRotateRight(a13.AsVector(), a2.AsVector(), 8).AsVector128(); a8 = AdvSimd.Add(a8, a13); a7 = Sve2.XorRotateRight(a7.AsVector(), a8.AsVector(), 7).AsVector128();
                a3 = AdvSimd.Add(AdvSimd.Add(a3, mA[15]), a4); a14 = Sve2.XorRotateRight(a14.AsVector(), a3.AsVector(), 16).AsVector128(); a9 = AdvSimd.Add(a9, a14); a4 = Sve2.XorRotateRight(a4.AsVector(), a9.AsVector(), 12).AsVector128();
                a3 = AdvSimd.Add(AdvSimd.Add(a3, mA[8]), a4); a14 = Sve2.XorRotateRight(a14.AsVector(), a3.AsVector(), 8).AsVector128(); a9 = AdvSimd.Add(a9, a14); a4 = Sve2.XorRotateRight(a4.AsVector(), a9.AsVector(), 7).AsVector128();
                // diagonals, batch B
                b0 = AdvSimd.Add(AdvSimd.Add(b0, mB[1]), b5); b15 = Sve2.XorRotateRight(b15.AsVector(), b0.AsVector(), 16).AsVector128(); b10 = AdvSimd.Add(b10, b15); b5 = Sve2.XorRotateRight(b5.AsVector(), b10.AsVector(), 12).AsVector128();
                b0 = AdvSimd.Add(AdvSimd.Add(b0, mB[11]), b5); b15 = Sve2.XorRotateRight(b15.AsVector(), b0.AsVector(), 8).AsVector128(); b10 = AdvSimd.Add(b10, b15); b5 = Sve2.XorRotateRight(b5.AsVector(), b10.AsVector(), 7).AsVector128();
                b1 = AdvSimd.Add(AdvSimd.Add(b1, mB[12]), b6); b12 = Sve2.XorRotateRight(b12.AsVector(), b1.AsVector(), 16).AsVector128(); b11 = AdvSimd.Add(b11, b12); b6 = Sve2.XorRotateRight(b6.AsVector(), b11.AsVector(), 12).AsVector128();
                b1 = AdvSimd.Add(AdvSimd.Add(b1, mB[5]), b6); b12 = Sve2.XorRotateRight(b12.AsVector(), b1.AsVector(), 8).AsVector128(); b11 = AdvSimd.Add(b11, b12); b6 = Sve2.XorRotateRight(b6.AsVector(), b11.AsVector(), 7).AsVector128();
                b2 = AdvSimd.Add(AdvSimd.Add(b2, mB[9]), b7); b13 = Sve2.XorRotateRight(b13.AsVector(), b2.AsVector(), 16).AsVector128(); b8 = AdvSimd.Add(b8, b13); b7 = Sve2.XorRotateRight(b7.AsVector(), b8.AsVector(), 12).AsVector128();
                b2 = AdvSimd.Add(AdvSimd.Add(b2, mB[14]), b7); b13 = Sve2.XorRotateRight(b13.AsVector(), b2.AsVector(), 8).AsVector128(); b8 = AdvSimd.Add(b8, b13); b7 = Sve2.XorRotateRight(b7.AsVector(), b8.AsVector(), 7).AsVector128();
                b3 = AdvSimd.Add(AdvSimd.Add(b3, mB[15]), b4); b14 = Sve2.XorRotateRight(b14.AsVector(), b3.AsVector(), 16).AsVector128(); b9 = AdvSimd.Add(b9, b14); b4 = Sve2.XorRotateRight(b4.AsVector(), b9.AsVector(), 12).AsVector128();
                b3 = AdvSimd.Add(AdvSimd.Add(b3, mB[8]), b4); b14 = Sve2.XorRotateRight(b14.AsVector(), b3.AsVector(), 8).AsVector128(); b9 = AdvSimd.Add(b9, b14); b4 = Sve2.XorRotateRight(b4.AsVector(), b9.AsVector(), 7).AsVector128();
                // Round 2: 3,4,10,12,13,2,7,14,6,5,9,0,11,15,8,1
                // columns, batch A
                a0 = AdvSimd.Add(AdvSimd.Add(a0, mA[3]), a4); a12 = Sve2.XorRotateRight(a12.AsVector(), a0.AsVector(), 16).AsVector128(); a8 = AdvSimd.Add(a8, a12); a4 = Sve2.XorRotateRight(a4.AsVector(), a8.AsVector(), 12).AsVector128();
                a0 = AdvSimd.Add(AdvSimd.Add(a0, mA[4]), a4); a12 = Sve2.XorRotateRight(a12.AsVector(), a0.AsVector(), 8).AsVector128(); a8 = AdvSimd.Add(a8, a12); a4 = Sve2.XorRotateRight(a4.AsVector(), a8.AsVector(), 7).AsVector128();
                a1 = AdvSimd.Add(AdvSimd.Add(a1, mA[10]), a5); a13 = Sve2.XorRotateRight(a13.AsVector(), a1.AsVector(), 16).AsVector128(); a9 = AdvSimd.Add(a9, a13); a5 = Sve2.XorRotateRight(a5.AsVector(), a9.AsVector(), 12).AsVector128();
                a1 = AdvSimd.Add(AdvSimd.Add(a1, mA[12]), a5); a13 = Sve2.XorRotateRight(a13.AsVector(), a1.AsVector(), 8).AsVector128(); a9 = AdvSimd.Add(a9, a13); a5 = Sve2.XorRotateRight(a5.AsVector(), a9.AsVector(), 7).AsVector128();
                a2 = AdvSimd.Add(AdvSimd.Add(a2, mA[13]), a6); a14 = Sve2.XorRotateRight(a14.AsVector(), a2.AsVector(), 16).AsVector128(); a10 = AdvSimd.Add(a10, a14); a6 = Sve2.XorRotateRight(a6.AsVector(), a10.AsVector(), 12).AsVector128();
                a2 = AdvSimd.Add(AdvSimd.Add(a2, mA[2]), a6); a14 = Sve2.XorRotateRight(a14.AsVector(), a2.AsVector(), 8).AsVector128(); a10 = AdvSimd.Add(a10, a14); a6 = Sve2.XorRotateRight(a6.AsVector(), a10.AsVector(), 7).AsVector128();
                a3 = AdvSimd.Add(AdvSimd.Add(a3, mA[7]), a7); a15 = Sve2.XorRotateRight(a15.AsVector(), a3.AsVector(), 16).AsVector128(); a11 = AdvSimd.Add(a11, a15); a7 = Sve2.XorRotateRight(a7.AsVector(), a11.AsVector(), 12).AsVector128();
                a3 = AdvSimd.Add(AdvSimd.Add(a3, mA[14]), a7); a15 = Sve2.XorRotateRight(a15.AsVector(), a3.AsVector(), 8).AsVector128(); a11 = AdvSimd.Add(a11, a15); a7 = Sve2.XorRotateRight(a7.AsVector(), a11.AsVector(), 7).AsVector128();
                // columns, batch B
                b0 = AdvSimd.Add(AdvSimd.Add(b0, mB[3]), b4); b12 = Sve2.XorRotateRight(b12.AsVector(), b0.AsVector(), 16).AsVector128(); b8 = AdvSimd.Add(b8, b12); b4 = Sve2.XorRotateRight(b4.AsVector(), b8.AsVector(), 12).AsVector128();
                b0 = AdvSimd.Add(AdvSimd.Add(b0, mB[4]), b4); b12 = Sve2.XorRotateRight(b12.AsVector(), b0.AsVector(), 8).AsVector128(); b8 = AdvSimd.Add(b8, b12); b4 = Sve2.XorRotateRight(b4.AsVector(), b8.AsVector(), 7).AsVector128();
                b1 = AdvSimd.Add(AdvSimd.Add(b1, mB[10]), b5); b13 = Sve2.XorRotateRight(b13.AsVector(), b1.AsVector(), 16).AsVector128(); b9 = AdvSimd.Add(b9, b13); b5 = Sve2.XorRotateRight(b5.AsVector(), b9.AsVector(), 12).AsVector128();
                b1 = AdvSimd.Add(AdvSimd.Add(b1, mB[12]), b5); b13 = Sve2.XorRotateRight(b13.AsVector(), b1.AsVector(), 8).AsVector128(); b9 = AdvSimd.Add(b9, b13); b5 = Sve2.XorRotateRight(b5.AsVector(), b9.AsVector(), 7).AsVector128();
                b2 = AdvSimd.Add(AdvSimd.Add(b2, mB[13]), b6); b14 = Sve2.XorRotateRight(b14.AsVector(), b2.AsVector(), 16).AsVector128(); b10 = AdvSimd.Add(b10, b14); b6 = Sve2.XorRotateRight(b6.AsVector(), b10.AsVector(), 12).AsVector128();
                b2 = AdvSimd.Add(AdvSimd.Add(b2, mB[2]), b6); b14 = Sve2.XorRotateRight(b14.AsVector(), b2.AsVector(), 8).AsVector128(); b10 = AdvSimd.Add(b10, b14); b6 = Sve2.XorRotateRight(b6.AsVector(), b10.AsVector(), 7).AsVector128();
                b3 = AdvSimd.Add(AdvSimd.Add(b3, mB[7]), b7); b15 = Sve2.XorRotateRight(b15.AsVector(), b3.AsVector(), 16).AsVector128(); b11 = AdvSimd.Add(b11, b15); b7 = Sve2.XorRotateRight(b7.AsVector(), b11.AsVector(), 12).AsVector128();
                b3 = AdvSimd.Add(AdvSimd.Add(b3, mB[14]), b7); b15 = Sve2.XorRotateRight(b15.AsVector(), b3.AsVector(), 8).AsVector128(); b11 = AdvSimd.Add(b11, b15); b7 = Sve2.XorRotateRight(b7.AsVector(), b11.AsVector(), 7).AsVector128();
                // diagonals, batch A
                a0 = AdvSimd.Add(AdvSimd.Add(a0, mA[6]), a5); a15 = Sve2.XorRotateRight(a15.AsVector(), a0.AsVector(), 16).AsVector128(); a10 = AdvSimd.Add(a10, a15); a5 = Sve2.XorRotateRight(a5.AsVector(), a10.AsVector(), 12).AsVector128();
                a0 = AdvSimd.Add(AdvSimd.Add(a0, mA[5]), a5); a15 = Sve2.XorRotateRight(a15.AsVector(), a0.AsVector(), 8).AsVector128(); a10 = AdvSimd.Add(a10, a15); a5 = Sve2.XorRotateRight(a5.AsVector(), a10.AsVector(), 7).AsVector128();
                a1 = AdvSimd.Add(AdvSimd.Add(a1, mA[9]), a6); a12 = Sve2.XorRotateRight(a12.AsVector(), a1.AsVector(), 16).AsVector128(); a11 = AdvSimd.Add(a11, a12); a6 = Sve2.XorRotateRight(a6.AsVector(), a11.AsVector(), 12).AsVector128();
                a1 = AdvSimd.Add(AdvSimd.Add(a1, mA[0]), a6); a12 = Sve2.XorRotateRight(a12.AsVector(), a1.AsVector(), 8).AsVector128(); a11 = AdvSimd.Add(a11, a12); a6 = Sve2.XorRotateRight(a6.AsVector(), a11.AsVector(), 7).AsVector128();
                a2 = AdvSimd.Add(AdvSimd.Add(a2, mA[11]), a7); a13 = Sve2.XorRotateRight(a13.AsVector(), a2.AsVector(), 16).AsVector128(); a8 = AdvSimd.Add(a8, a13); a7 = Sve2.XorRotateRight(a7.AsVector(), a8.AsVector(), 12).AsVector128();
                a2 = AdvSimd.Add(AdvSimd.Add(a2, mA[15]), a7); a13 = Sve2.XorRotateRight(a13.AsVector(), a2.AsVector(), 8).AsVector128(); a8 = AdvSimd.Add(a8, a13); a7 = Sve2.XorRotateRight(a7.AsVector(), a8.AsVector(), 7).AsVector128();
                a3 = AdvSimd.Add(AdvSimd.Add(a3, mA[8]), a4); a14 = Sve2.XorRotateRight(a14.AsVector(), a3.AsVector(), 16).AsVector128(); a9 = AdvSimd.Add(a9, a14); a4 = Sve2.XorRotateRight(a4.AsVector(), a9.AsVector(), 12).AsVector128();
                a3 = AdvSimd.Add(AdvSimd.Add(a3, mA[1]), a4); a14 = Sve2.XorRotateRight(a14.AsVector(), a3.AsVector(), 8).AsVector128(); a9 = AdvSimd.Add(a9, a14); a4 = Sve2.XorRotateRight(a4.AsVector(), a9.AsVector(), 7).AsVector128();
                // diagonals, batch B
                b0 = AdvSimd.Add(AdvSimd.Add(b0, mB[6]), b5); b15 = Sve2.XorRotateRight(b15.AsVector(), b0.AsVector(), 16).AsVector128(); b10 = AdvSimd.Add(b10, b15); b5 = Sve2.XorRotateRight(b5.AsVector(), b10.AsVector(), 12).AsVector128();
                b0 = AdvSimd.Add(AdvSimd.Add(b0, mB[5]), b5); b15 = Sve2.XorRotateRight(b15.AsVector(), b0.AsVector(), 8).AsVector128(); b10 = AdvSimd.Add(b10, b15); b5 = Sve2.XorRotateRight(b5.AsVector(), b10.AsVector(), 7).AsVector128();
                b1 = AdvSimd.Add(AdvSimd.Add(b1, mB[9]), b6); b12 = Sve2.XorRotateRight(b12.AsVector(), b1.AsVector(), 16).AsVector128(); b11 = AdvSimd.Add(b11, b12); b6 = Sve2.XorRotateRight(b6.AsVector(), b11.AsVector(), 12).AsVector128();
                b1 = AdvSimd.Add(AdvSimd.Add(b1, mB[0]), b6); b12 = Sve2.XorRotateRight(b12.AsVector(), b1.AsVector(), 8).AsVector128(); b11 = AdvSimd.Add(b11, b12); b6 = Sve2.XorRotateRight(b6.AsVector(), b11.AsVector(), 7).AsVector128();
                b2 = AdvSimd.Add(AdvSimd.Add(b2, mB[11]), b7); b13 = Sve2.XorRotateRight(b13.AsVector(), b2.AsVector(), 16).AsVector128(); b8 = AdvSimd.Add(b8, b13); b7 = Sve2.XorRotateRight(b7.AsVector(), b8.AsVector(), 12).AsVector128();
                b2 = AdvSimd.Add(AdvSimd.Add(b2, mB[15]), b7); b13 = Sve2.XorRotateRight(b13.AsVector(), b2.AsVector(), 8).AsVector128(); b8 = AdvSimd.Add(b8, b13); b7 = Sve2.XorRotateRight(b7.AsVector(), b8.AsVector(), 7).AsVector128();
                b3 = AdvSimd.Add(AdvSimd.Add(b3, mB[8]), b4); b14 = Sve2.XorRotateRight(b14.AsVector(), b3.AsVector(), 16).AsVector128(); b9 = AdvSimd.Add(b9, b14); b4 = Sve2.XorRotateRight(b4.AsVector(), b9.AsVector(), 12).AsVector128();
                b3 = AdvSimd.Add(AdvSimd.Add(b3, mB[1]), b4); b14 = Sve2.XorRotateRight(b14.AsVector(), b3.AsVector(), 8).AsVector128(); b9 = AdvSimd.Add(b9, b14); b4 = Sve2.XorRotateRight(b4.AsVector(), b9.AsVector(), 7).AsVector128();
                // Round 3: 10,7,12,9,14,3,13,15,4,0,11,2,5,8,1,6
                // columns, batch A
                a0 = AdvSimd.Add(AdvSimd.Add(a0, mA[10]), a4); a12 = Sve2.XorRotateRight(a12.AsVector(), a0.AsVector(), 16).AsVector128(); a8 = AdvSimd.Add(a8, a12); a4 = Sve2.XorRotateRight(a4.AsVector(), a8.AsVector(), 12).AsVector128();
                a0 = AdvSimd.Add(AdvSimd.Add(a0, mA[7]), a4); a12 = Sve2.XorRotateRight(a12.AsVector(), a0.AsVector(), 8).AsVector128(); a8 = AdvSimd.Add(a8, a12); a4 = Sve2.XorRotateRight(a4.AsVector(), a8.AsVector(), 7).AsVector128();
                a1 = AdvSimd.Add(AdvSimd.Add(a1, mA[12]), a5); a13 = Sve2.XorRotateRight(a13.AsVector(), a1.AsVector(), 16).AsVector128(); a9 = AdvSimd.Add(a9, a13); a5 = Sve2.XorRotateRight(a5.AsVector(), a9.AsVector(), 12).AsVector128();
                a1 = AdvSimd.Add(AdvSimd.Add(a1, mA[9]), a5); a13 = Sve2.XorRotateRight(a13.AsVector(), a1.AsVector(), 8).AsVector128(); a9 = AdvSimd.Add(a9, a13); a5 = Sve2.XorRotateRight(a5.AsVector(), a9.AsVector(), 7).AsVector128();
                a2 = AdvSimd.Add(AdvSimd.Add(a2, mA[14]), a6); a14 = Sve2.XorRotateRight(a14.AsVector(), a2.AsVector(), 16).AsVector128(); a10 = AdvSimd.Add(a10, a14); a6 = Sve2.XorRotateRight(a6.AsVector(), a10.AsVector(), 12).AsVector128();
                a2 = AdvSimd.Add(AdvSimd.Add(a2, mA[3]), a6); a14 = Sve2.XorRotateRight(a14.AsVector(), a2.AsVector(), 8).AsVector128(); a10 = AdvSimd.Add(a10, a14); a6 = Sve2.XorRotateRight(a6.AsVector(), a10.AsVector(), 7).AsVector128();
                a3 = AdvSimd.Add(AdvSimd.Add(a3, mA[13]), a7); a15 = Sve2.XorRotateRight(a15.AsVector(), a3.AsVector(), 16).AsVector128(); a11 = AdvSimd.Add(a11, a15); a7 = Sve2.XorRotateRight(a7.AsVector(), a11.AsVector(), 12).AsVector128();
                a3 = AdvSimd.Add(AdvSimd.Add(a3, mA[15]), a7); a15 = Sve2.XorRotateRight(a15.AsVector(), a3.AsVector(), 8).AsVector128(); a11 = AdvSimd.Add(a11, a15); a7 = Sve2.XorRotateRight(a7.AsVector(), a11.AsVector(), 7).AsVector128();
                // columns, batch B
                b0 = AdvSimd.Add(AdvSimd.Add(b0, mB[10]), b4); b12 = Sve2.XorRotateRight(b12.AsVector(), b0.AsVector(), 16).AsVector128(); b8 = AdvSimd.Add(b8, b12); b4 = Sve2.XorRotateRight(b4.AsVector(), b8.AsVector(), 12).AsVector128();
                b0 = AdvSimd.Add(AdvSimd.Add(b0, mB[7]), b4); b12 = Sve2.XorRotateRight(b12.AsVector(), b0.AsVector(), 8).AsVector128(); b8 = AdvSimd.Add(b8, b12); b4 = Sve2.XorRotateRight(b4.AsVector(), b8.AsVector(), 7).AsVector128();
                b1 = AdvSimd.Add(AdvSimd.Add(b1, mB[12]), b5); b13 = Sve2.XorRotateRight(b13.AsVector(), b1.AsVector(), 16).AsVector128(); b9 = AdvSimd.Add(b9, b13); b5 = Sve2.XorRotateRight(b5.AsVector(), b9.AsVector(), 12).AsVector128();
                b1 = AdvSimd.Add(AdvSimd.Add(b1, mB[9]), b5); b13 = Sve2.XorRotateRight(b13.AsVector(), b1.AsVector(), 8).AsVector128(); b9 = AdvSimd.Add(b9, b13); b5 = Sve2.XorRotateRight(b5.AsVector(), b9.AsVector(), 7).AsVector128();
                b2 = AdvSimd.Add(AdvSimd.Add(b2, mB[14]), b6); b14 = Sve2.XorRotateRight(b14.AsVector(), b2.AsVector(), 16).AsVector128(); b10 = AdvSimd.Add(b10, b14); b6 = Sve2.XorRotateRight(b6.AsVector(), b10.AsVector(), 12).AsVector128();
                b2 = AdvSimd.Add(AdvSimd.Add(b2, mB[3]), b6); b14 = Sve2.XorRotateRight(b14.AsVector(), b2.AsVector(), 8).AsVector128(); b10 = AdvSimd.Add(b10, b14); b6 = Sve2.XorRotateRight(b6.AsVector(), b10.AsVector(), 7).AsVector128();
                b3 = AdvSimd.Add(AdvSimd.Add(b3, mB[13]), b7); b15 = Sve2.XorRotateRight(b15.AsVector(), b3.AsVector(), 16).AsVector128(); b11 = AdvSimd.Add(b11, b15); b7 = Sve2.XorRotateRight(b7.AsVector(), b11.AsVector(), 12).AsVector128();
                b3 = AdvSimd.Add(AdvSimd.Add(b3, mB[15]), b7); b15 = Sve2.XorRotateRight(b15.AsVector(), b3.AsVector(), 8).AsVector128(); b11 = AdvSimd.Add(b11, b15); b7 = Sve2.XorRotateRight(b7.AsVector(), b11.AsVector(), 7).AsVector128();
                // diagonals, batch A
                a0 = AdvSimd.Add(AdvSimd.Add(a0, mA[4]), a5); a15 = Sve2.XorRotateRight(a15.AsVector(), a0.AsVector(), 16).AsVector128(); a10 = AdvSimd.Add(a10, a15); a5 = Sve2.XorRotateRight(a5.AsVector(), a10.AsVector(), 12).AsVector128();
                a0 = AdvSimd.Add(AdvSimd.Add(a0, mA[0]), a5); a15 = Sve2.XorRotateRight(a15.AsVector(), a0.AsVector(), 8).AsVector128(); a10 = AdvSimd.Add(a10, a15); a5 = Sve2.XorRotateRight(a5.AsVector(), a10.AsVector(), 7).AsVector128();
                a1 = AdvSimd.Add(AdvSimd.Add(a1, mA[11]), a6); a12 = Sve2.XorRotateRight(a12.AsVector(), a1.AsVector(), 16).AsVector128(); a11 = AdvSimd.Add(a11, a12); a6 = Sve2.XorRotateRight(a6.AsVector(), a11.AsVector(), 12).AsVector128();
                a1 = AdvSimd.Add(AdvSimd.Add(a1, mA[2]), a6); a12 = Sve2.XorRotateRight(a12.AsVector(), a1.AsVector(), 8).AsVector128(); a11 = AdvSimd.Add(a11, a12); a6 = Sve2.XorRotateRight(a6.AsVector(), a11.AsVector(), 7).AsVector128();
                a2 = AdvSimd.Add(AdvSimd.Add(a2, mA[5]), a7); a13 = Sve2.XorRotateRight(a13.AsVector(), a2.AsVector(), 16).AsVector128(); a8 = AdvSimd.Add(a8, a13); a7 = Sve2.XorRotateRight(a7.AsVector(), a8.AsVector(), 12).AsVector128();
                a2 = AdvSimd.Add(AdvSimd.Add(a2, mA[8]), a7); a13 = Sve2.XorRotateRight(a13.AsVector(), a2.AsVector(), 8).AsVector128(); a8 = AdvSimd.Add(a8, a13); a7 = Sve2.XorRotateRight(a7.AsVector(), a8.AsVector(), 7).AsVector128();
                a3 = AdvSimd.Add(AdvSimd.Add(a3, mA[1]), a4); a14 = Sve2.XorRotateRight(a14.AsVector(), a3.AsVector(), 16).AsVector128(); a9 = AdvSimd.Add(a9, a14); a4 = Sve2.XorRotateRight(a4.AsVector(), a9.AsVector(), 12).AsVector128();
                a3 = AdvSimd.Add(AdvSimd.Add(a3, mA[6]), a4); a14 = Sve2.XorRotateRight(a14.AsVector(), a3.AsVector(), 8).AsVector128(); a9 = AdvSimd.Add(a9, a14); a4 = Sve2.XorRotateRight(a4.AsVector(), a9.AsVector(), 7).AsVector128();
                // diagonals, batch B
                b0 = AdvSimd.Add(AdvSimd.Add(b0, mB[4]), b5); b15 = Sve2.XorRotateRight(b15.AsVector(), b0.AsVector(), 16).AsVector128(); b10 = AdvSimd.Add(b10, b15); b5 = Sve2.XorRotateRight(b5.AsVector(), b10.AsVector(), 12).AsVector128();
                b0 = AdvSimd.Add(AdvSimd.Add(b0, mB[0]), b5); b15 = Sve2.XorRotateRight(b15.AsVector(), b0.AsVector(), 8).AsVector128(); b10 = AdvSimd.Add(b10, b15); b5 = Sve2.XorRotateRight(b5.AsVector(), b10.AsVector(), 7).AsVector128();
                b1 = AdvSimd.Add(AdvSimd.Add(b1, mB[11]), b6); b12 = Sve2.XorRotateRight(b12.AsVector(), b1.AsVector(), 16).AsVector128(); b11 = AdvSimd.Add(b11, b12); b6 = Sve2.XorRotateRight(b6.AsVector(), b11.AsVector(), 12).AsVector128();
                b1 = AdvSimd.Add(AdvSimd.Add(b1, mB[2]), b6); b12 = Sve2.XorRotateRight(b12.AsVector(), b1.AsVector(), 8).AsVector128(); b11 = AdvSimd.Add(b11, b12); b6 = Sve2.XorRotateRight(b6.AsVector(), b11.AsVector(), 7).AsVector128();
                b2 = AdvSimd.Add(AdvSimd.Add(b2, mB[5]), b7); b13 = Sve2.XorRotateRight(b13.AsVector(), b2.AsVector(), 16).AsVector128(); b8 = AdvSimd.Add(b8, b13); b7 = Sve2.XorRotateRight(b7.AsVector(), b8.AsVector(), 12).AsVector128();
                b2 = AdvSimd.Add(AdvSimd.Add(b2, mB[8]), b7); b13 = Sve2.XorRotateRight(b13.AsVector(), b2.AsVector(), 8).AsVector128(); b8 = AdvSimd.Add(b8, b13); b7 = Sve2.XorRotateRight(b7.AsVector(), b8.AsVector(), 7).AsVector128();
                b3 = AdvSimd.Add(AdvSimd.Add(b3, mB[1]), b4); b14 = Sve2.XorRotateRight(b14.AsVector(), b3.AsVector(), 16).AsVector128(); b9 = AdvSimd.Add(b9, b14); b4 = Sve2.XorRotateRight(b4.AsVector(), b9.AsVector(), 12).AsVector128();
                b3 = AdvSimd.Add(AdvSimd.Add(b3, mB[6]), b4); b14 = Sve2.XorRotateRight(b14.AsVector(), b3.AsVector(), 8).AsVector128(); b9 = AdvSimd.Add(b9, b14); b4 = Sve2.XorRotateRight(b4.AsVector(), b9.AsVector(), 7).AsVector128();
                // Round 4: 12,13,9,11,15,10,14,8,7,2,5,3,0,1,6,4
                // columns, batch A
                a0 = AdvSimd.Add(AdvSimd.Add(a0, mA[12]), a4); a12 = Sve2.XorRotateRight(a12.AsVector(), a0.AsVector(), 16).AsVector128(); a8 = AdvSimd.Add(a8, a12); a4 = Sve2.XorRotateRight(a4.AsVector(), a8.AsVector(), 12).AsVector128();
                a0 = AdvSimd.Add(AdvSimd.Add(a0, mA[13]), a4); a12 = Sve2.XorRotateRight(a12.AsVector(), a0.AsVector(), 8).AsVector128(); a8 = AdvSimd.Add(a8, a12); a4 = Sve2.XorRotateRight(a4.AsVector(), a8.AsVector(), 7).AsVector128();
                a1 = AdvSimd.Add(AdvSimd.Add(a1, mA[9]), a5); a13 = Sve2.XorRotateRight(a13.AsVector(), a1.AsVector(), 16).AsVector128(); a9 = AdvSimd.Add(a9, a13); a5 = Sve2.XorRotateRight(a5.AsVector(), a9.AsVector(), 12).AsVector128();
                a1 = AdvSimd.Add(AdvSimd.Add(a1, mA[11]), a5); a13 = Sve2.XorRotateRight(a13.AsVector(), a1.AsVector(), 8).AsVector128(); a9 = AdvSimd.Add(a9, a13); a5 = Sve2.XorRotateRight(a5.AsVector(), a9.AsVector(), 7).AsVector128();
                a2 = AdvSimd.Add(AdvSimd.Add(a2, mA[15]), a6); a14 = Sve2.XorRotateRight(a14.AsVector(), a2.AsVector(), 16).AsVector128(); a10 = AdvSimd.Add(a10, a14); a6 = Sve2.XorRotateRight(a6.AsVector(), a10.AsVector(), 12).AsVector128();
                a2 = AdvSimd.Add(AdvSimd.Add(a2, mA[10]), a6); a14 = Sve2.XorRotateRight(a14.AsVector(), a2.AsVector(), 8).AsVector128(); a10 = AdvSimd.Add(a10, a14); a6 = Sve2.XorRotateRight(a6.AsVector(), a10.AsVector(), 7).AsVector128();
                a3 = AdvSimd.Add(AdvSimd.Add(a3, mA[14]), a7); a15 = Sve2.XorRotateRight(a15.AsVector(), a3.AsVector(), 16).AsVector128(); a11 = AdvSimd.Add(a11, a15); a7 = Sve2.XorRotateRight(a7.AsVector(), a11.AsVector(), 12).AsVector128();
                a3 = AdvSimd.Add(AdvSimd.Add(a3, mA[8]), a7); a15 = Sve2.XorRotateRight(a15.AsVector(), a3.AsVector(), 8).AsVector128(); a11 = AdvSimd.Add(a11, a15); a7 = Sve2.XorRotateRight(a7.AsVector(), a11.AsVector(), 7).AsVector128();
                // columns, batch B
                b0 = AdvSimd.Add(AdvSimd.Add(b0, mB[12]), b4); b12 = Sve2.XorRotateRight(b12.AsVector(), b0.AsVector(), 16).AsVector128(); b8 = AdvSimd.Add(b8, b12); b4 = Sve2.XorRotateRight(b4.AsVector(), b8.AsVector(), 12).AsVector128();
                b0 = AdvSimd.Add(AdvSimd.Add(b0, mB[13]), b4); b12 = Sve2.XorRotateRight(b12.AsVector(), b0.AsVector(), 8).AsVector128(); b8 = AdvSimd.Add(b8, b12); b4 = Sve2.XorRotateRight(b4.AsVector(), b8.AsVector(), 7).AsVector128();
                b1 = AdvSimd.Add(AdvSimd.Add(b1, mB[9]), b5); b13 = Sve2.XorRotateRight(b13.AsVector(), b1.AsVector(), 16).AsVector128(); b9 = AdvSimd.Add(b9, b13); b5 = Sve2.XorRotateRight(b5.AsVector(), b9.AsVector(), 12).AsVector128();
                b1 = AdvSimd.Add(AdvSimd.Add(b1, mB[11]), b5); b13 = Sve2.XorRotateRight(b13.AsVector(), b1.AsVector(), 8).AsVector128(); b9 = AdvSimd.Add(b9, b13); b5 = Sve2.XorRotateRight(b5.AsVector(), b9.AsVector(), 7).AsVector128();
                b2 = AdvSimd.Add(AdvSimd.Add(b2, mB[15]), b6); b14 = Sve2.XorRotateRight(b14.AsVector(), b2.AsVector(), 16).AsVector128(); b10 = AdvSimd.Add(b10, b14); b6 = Sve2.XorRotateRight(b6.AsVector(), b10.AsVector(), 12).AsVector128();
                b2 = AdvSimd.Add(AdvSimd.Add(b2, mB[10]), b6); b14 = Sve2.XorRotateRight(b14.AsVector(), b2.AsVector(), 8).AsVector128(); b10 = AdvSimd.Add(b10, b14); b6 = Sve2.XorRotateRight(b6.AsVector(), b10.AsVector(), 7).AsVector128();
                b3 = AdvSimd.Add(AdvSimd.Add(b3, mB[14]), b7); b15 = Sve2.XorRotateRight(b15.AsVector(), b3.AsVector(), 16).AsVector128(); b11 = AdvSimd.Add(b11, b15); b7 = Sve2.XorRotateRight(b7.AsVector(), b11.AsVector(), 12).AsVector128();
                b3 = AdvSimd.Add(AdvSimd.Add(b3, mB[8]), b7); b15 = Sve2.XorRotateRight(b15.AsVector(), b3.AsVector(), 8).AsVector128(); b11 = AdvSimd.Add(b11, b15); b7 = Sve2.XorRotateRight(b7.AsVector(), b11.AsVector(), 7).AsVector128();
                // diagonals, batch A
                a0 = AdvSimd.Add(AdvSimd.Add(a0, mA[7]), a5); a15 = Sve2.XorRotateRight(a15.AsVector(), a0.AsVector(), 16).AsVector128(); a10 = AdvSimd.Add(a10, a15); a5 = Sve2.XorRotateRight(a5.AsVector(), a10.AsVector(), 12).AsVector128();
                a0 = AdvSimd.Add(AdvSimd.Add(a0, mA[2]), a5); a15 = Sve2.XorRotateRight(a15.AsVector(), a0.AsVector(), 8).AsVector128(); a10 = AdvSimd.Add(a10, a15); a5 = Sve2.XorRotateRight(a5.AsVector(), a10.AsVector(), 7).AsVector128();
                a1 = AdvSimd.Add(AdvSimd.Add(a1, mA[5]), a6); a12 = Sve2.XorRotateRight(a12.AsVector(), a1.AsVector(), 16).AsVector128(); a11 = AdvSimd.Add(a11, a12); a6 = Sve2.XorRotateRight(a6.AsVector(), a11.AsVector(), 12).AsVector128();
                a1 = AdvSimd.Add(AdvSimd.Add(a1, mA[3]), a6); a12 = Sve2.XorRotateRight(a12.AsVector(), a1.AsVector(), 8).AsVector128(); a11 = AdvSimd.Add(a11, a12); a6 = Sve2.XorRotateRight(a6.AsVector(), a11.AsVector(), 7).AsVector128();
                a2 = AdvSimd.Add(AdvSimd.Add(a2, mA[0]), a7); a13 = Sve2.XorRotateRight(a13.AsVector(), a2.AsVector(), 16).AsVector128(); a8 = AdvSimd.Add(a8, a13); a7 = Sve2.XorRotateRight(a7.AsVector(), a8.AsVector(), 12).AsVector128();
                a2 = AdvSimd.Add(AdvSimd.Add(a2, mA[1]), a7); a13 = Sve2.XorRotateRight(a13.AsVector(), a2.AsVector(), 8).AsVector128(); a8 = AdvSimd.Add(a8, a13); a7 = Sve2.XorRotateRight(a7.AsVector(), a8.AsVector(), 7).AsVector128();
                a3 = AdvSimd.Add(AdvSimd.Add(a3, mA[6]), a4); a14 = Sve2.XorRotateRight(a14.AsVector(), a3.AsVector(), 16).AsVector128(); a9 = AdvSimd.Add(a9, a14); a4 = Sve2.XorRotateRight(a4.AsVector(), a9.AsVector(), 12).AsVector128();
                a3 = AdvSimd.Add(AdvSimd.Add(a3, mA[4]), a4); a14 = Sve2.XorRotateRight(a14.AsVector(), a3.AsVector(), 8).AsVector128(); a9 = AdvSimd.Add(a9, a14); a4 = Sve2.XorRotateRight(a4.AsVector(), a9.AsVector(), 7).AsVector128();
                // diagonals, batch B
                b0 = AdvSimd.Add(AdvSimd.Add(b0, mB[7]), b5); b15 = Sve2.XorRotateRight(b15.AsVector(), b0.AsVector(), 16).AsVector128(); b10 = AdvSimd.Add(b10, b15); b5 = Sve2.XorRotateRight(b5.AsVector(), b10.AsVector(), 12).AsVector128();
                b0 = AdvSimd.Add(AdvSimd.Add(b0, mB[2]), b5); b15 = Sve2.XorRotateRight(b15.AsVector(), b0.AsVector(), 8).AsVector128(); b10 = AdvSimd.Add(b10, b15); b5 = Sve2.XorRotateRight(b5.AsVector(), b10.AsVector(), 7).AsVector128();
                b1 = AdvSimd.Add(AdvSimd.Add(b1, mB[5]), b6); b12 = Sve2.XorRotateRight(b12.AsVector(), b1.AsVector(), 16).AsVector128(); b11 = AdvSimd.Add(b11, b12); b6 = Sve2.XorRotateRight(b6.AsVector(), b11.AsVector(), 12).AsVector128();
                b1 = AdvSimd.Add(AdvSimd.Add(b1, mB[3]), b6); b12 = Sve2.XorRotateRight(b12.AsVector(), b1.AsVector(), 8).AsVector128(); b11 = AdvSimd.Add(b11, b12); b6 = Sve2.XorRotateRight(b6.AsVector(), b11.AsVector(), 7).AsVector128();
                b2 = AdvSimd.Add(AdvSimd.Add(b2, mB[0]), b7); b13 = Sve2.XorRotateRight(b13.AsVector(), b2.AsVector(), 16).AsVector128(); b8 = AdvSimd.Add(b8, b13); b7 = Sve2.XorRotateRight(b7.AsVector(), b8.AsVector(), 12).AsVector128();
                b2 = AdvSimd.Add(AdvSimd.Add(b2, mB[1]), b7); b13 = Sve2.XorRotateRight(b13.AsVector(), b2.AsVector(), 8).AsVector128(); b8 = AdvSimd.Add(b8, b13); b7 = Sve2.XorRotateRight(b7.AsVector(), b8.AsVector(), 7).AsVector128();
                b3 = AdvSimd.Add(AdvSimd.Add(b3, mB[6]), b4); b14 = Sve2.XorRotateRight(b14.AsVector(), b3.AsVector(), 16).AsVector128(); b9 = AdvSimd.Add(b9, b14); b4 = Sve2.XorRotateRight(b4.AsVector(), b9.AsVector(), 12).AsVector128();
                b3 = AdvSimd.Add(AdvSimd.Add(b3, mB[4]), b4); b14 = Sve2.XorRotateRight(b14.AsVector(), b3.AsVector(), 8).AsVector128(); b9 = AdvSimd.Add(b9, b14); b4 = Sve2.XorRotateRight(b4.AsVector(), b9.AsVector(), 7).AsVector128();
                // Round 5: 9,14,11,5,8,12,15,1,13,3,0,10,2,6,4,7
                // columns, batch A
                a0 = AdvSimd.Add(AdvSimd.Add(a0, mA[9]), a4); a12 = Sve2.XorRotateRight(a12.AsVector(), a0.AsVector(), 16).AsVector128(); a8 = AdvSimd.Add(a8, a12); a4 = Sve2.XorRotateRight(a4.AsVector(), a8.AsVector(), 12).AsVector128();
                a0 = AdvSimd.Add(AdvSimd.Add(a0, mA[14]), a4); a12 = Sve2.XorRotateRight(a12.AsVector(), a0.AsVector(), 8).AsVector128(); a8 = AdvSimd.Add(a8, a12); a4 = Sve2.XorRotateRight(a4.AsVector(), a8.AsVector(), 7).AsVector128();
                a1 = AdvSimd.Add(AdvSimd.Add(a1, mA[11]), a5); a13 = Sve2.XorRotateRight(a13.AsVector(), a1.AsVector(), 16).AsVector128(); a9 = AdvSimd.Add(a9, a13); a5 = Sve2.XorRotateRight(a5.AsVector(), a9.AsVector(), 12).AsVector128();
                a1 = AdvSimd.Add(AdvSimd.Add(a1, mA[5]), a5); a13 = Sve2.XorRotateRight(a13.AsVector(), a1.AsVector(), 8).AsVector128(); a9 = AdvSimd.Add(a9, a13); a5 = Sve2.XorRotateRight(a5.AsVector(), a9.AsVector(), 7).AsVector128();
                a2 = AdvSimd.Add(AdvSimd.Add(a2, mA[8]), a6); a14 = Sve2.XorRotateRight(a14.AsVector(), a2.AsVector(), 16).AsVector128(); a10 = AdvSimd.Add(a10, a14); a6 = Sve2.XorRotateRight(a6.AsVector(), a10.AsVector(), 12).AsVector128();
                a2 = AdvSimd.Add(AdvSimd.Add(a2, mA[12]), a6); a14 = Sve2.XorRotateRight(a14.AsVector(), a2.AsVector(), 8).AsVector128(); a10 = AdvSimd.Add(a10, a14); a6 = Sve2.XorRotateRight(a6.AsVector(), a10.AsVector(), 7).AsVector128();
                a3 = AdvSimd.Add(AdvSimd.Add(a3, mA[15]), a7); a15 = Sve2.XorRotateRight(a15.AsVector(), a3.AsVector(), 16).AsVector128(); a11 = AdvSimd.Add(a11, a15); a7 = Sve2.XorRotateRight(a7.AsVector(), a11.AsVector(), 12).AsVector128();
                a3 = AdvSimd.Add(AdvSimd.Add(a3, mA[1]), a7); a15 = Sve2.XorRotateRight(a15.AsVector(), a3.AsVector(), 8).AsVector128(); a11 = AdvSimd.Add(a11, a15); a7 = Sve2.XorRotateRight(a7.AsVector(), a11.AsVector(), 7).AsVector128();
                // columns, batch B
                b0 = AdvSimd.Add(AdvSimd.Add(b0, mB[9]), b4); b12 = Sve2.XorRotateRight(b12.AsVector(), b0.AsVector(), 16).AsVector128(); b8 = AdvSimd.Add(b8, b12); b4 = Sve2.XorRotateRight(b4.AsVector(), b8.AsVector(), 12).AsVector128();
                b0 = AdvSimd.Add(AdvSimd.Add(b0, mB[14]), b4); b12 = Sve2.XorRotateRight(b12.AsVector(), b0.AsVector(), 8).AsVector128(); b8 = AdvSimd.Add(b8, b12); b4 = Sve2.XorRotateRight(b4.AsVector(), b8.AsVector(), 7).AsVector128();
                b1 = AdvSimd.Add(AdvSimd.Add(b1, mB[11]), b5); b13 = Sve2.XorRotateRight(b13.AsVector(), b1.AsVector(), 16).AsVector128(); b9 = AdvSimd.Add(b9, b13); b5 = Sve2.XorRotateRight(b5.AsVector(), b9.AsVector(), 12).AsVector128();
                b1 = AdvSimd.Add(AdvSimd.Add(b1, mB[5]), b5); b13 = Sve2.XorRotateRight(b13.AsVector(), b1.AsVector(), 8).AsVector128(); b9 = AdvSimd.Add(b9, b13); b5 = Sve2.XorRotateRight(b5.AsVector(), b9.AsVector(), 7).AsVector128();
                b2 = AdvSimd.Add(AdvSimd.Add(b2, mB[8]), b6); b14 = Sve2.XorRotateRight(b14.AsVector(), b2.AsVector(), 16).AsVector128(); b10 = AdvSimd.Add(b10, b14); b6 = Sve2.XorRotateRight(b6.AsVector(), b10.AsVector(), 12).AsVector128();
                b2 = AdvSimd.Add(AdvSimd.Add(b2, mB[12]), b6); b14 = Sve2.XorRotateRight(b14.AsVector(), b2.AsVector(), 8).AsVector128(); b10 = AdvSimd.Add(b10, b14); b6 = Sve2.XorRotateRight(b6.AsVector(), b10.AsVector(), 7).AsVector128();
                b3 = AdvSimd.Add(AdvSimd.Add(b3, mB[15]), b7); b15 = Sve2.XorRotateRight(b15.AsVector(), b3.AsVector(), 16).AsVector128(); b11 = AdvSimd.Add(b11, b15); b7 = Sve2.XorRotateRight(b7.AsVector(), b11.AsVector(), 12).AsVector128();
                b3 = AdvSimd.Add(AdvSimd.Add(b3, mB[1]), b7); b15 = Sve2.XorRotateRight(b15.AsVector(), b3.AsVector(), 8).AsVector128(); b11 = AdvSimd.Add(b11, b15); b7 = Sve2.XorRotateRight(b7.AsVector(), b11.AsVector(), 7).AsVector128();
                // diagonals, batch A
                a0 = AdvSimd.Add(AdvSimd.Add(a0, mA[13]), a5); a15 = Sve2.XorRotateRight(a15.AsVector(), a0.AsVector(), 16).AsVector128(); a10 = AdvSimd.Add(a10, a15); a5 = Sve2.XorRotateRight(a5.AsVector(), a10.AsVector(), 12).AsVector128();
                a0 = AdvSimd.Add(AdvSimd.Add(a0, mA[3]), a5); a15 = Sve2.XorRotateRight(a15.AsVector(), a0.AsVector(), 8).AsVector128(); a10 = AdvSimd.Add(a10, a15); a5 = Sve2.XorRotateRight(a5.AsVector(), a10.AsVector(), 7).AsVector128();
                a1 = AdvSimd.Add(AdvSimd.Add(a1, mA[0]), a6); a12 = Sve2.XorRotateRight(a12.AsVector(), a1.AsVector(), 16).AsVector128(); a11 = AdvSimd.Add(a11, a12); a6 = Sve2.XorRotateRight(a6.AsVector(), a11.AsVector(), 12).AsVector128();
                a1 = AdvSimd.Add(AdvSimd.Add(a1, mA[10]), a6); a12 = Sve2.XorRotateRight(a12.AsVector(), a1.AsVector(), 8).AsVector128(); a11 = AdvSimd.Add(a11, a12); a6 = Sve2.XorRotateRight(a6.AsVector(), a11.AsVector(), 7).AsVector128();
                a2 = AdvSimd.Add(AdvSimd.Add(a2, mA[2]), a7); a13 = Sve2.XorRotateRight(a13.AsVector(), a2.AsVector(), 16).AsVector128(); a8 = AdvSimd.Add(a8, a13); a7 = Sve2.XorRotateRight(a7.AsVector(), a8.AsVector(), 12).AsVector128();
                a2 = AdvSimd.Add(AdvSimd.Add(a2, mA[6]), a7); a13 = Sve2.XorRotateRight(a13.AsVector(), a2.AsVector(), 8).AsVector128(); a8 = AdvSimd.Add(a8, a13); a7 = Sve2.XorRotateRight(a7.AsVector(), a8.AsVector(), 7).AsVector128();
                a3 = AdvSimd.Add(AdvSimd.Add(a3, mA[4]), a4); a14 = Sve2.XorRotateRight(a14.AsVector(), a3.AsVector(), 16).AsVector128(); a9 = AdvSimd.Add(a9, a14); a4 = Sve2.XorRotateRight(a4.AsVector(), a9.AsVector(), 12).AsVector128();
                a3 = AdvSimd.Add(AdvSimd.Add(a3, mA[7]), a4); a14 = Sve2.XorRotateRight(a14.AsVector(), a3.AsVector(), 8).AsVector128(); a9 = AdvSimd.Add(a9, a14); a4 = Sve2.XorRotateRight(a4.AsVector(), a9.AsVector(), 7).AsVector128();
                // diagonals, batch B
                b0 = AdvSimd.Add(AdvSimd.Add(b0, mB[13]), b5); b15 = Sve2.XorRotateRight(b15.AsVector(), b0.AsVector(), 16).AsVector128(); b10 = AdvSimd.Add(b10, b15); b5 = Sve2.XorRotateRight(b5.AsVector(), b10.AsVector(), 12).AsVector128();
                b0 = AdvSimd.Add(AdvSimd.Add(b0, mB[3]), b5); b15 = Sve2.XorRotateRight(b15.AsVector(), b0.AsVector(), 8).AsVector128(); b10 = AdvSimd.Add(b10, b15); b5 = Sve2.XorRotateRight(b5.AsVector(), b10.AsVector(), 7).AsVector128();
                b1 = AdvSimd.Add(AdvSimd.Add(b1, mB[0]), b6); b12 = Sve2.XorRotateRight(b12.AsVector(), b1.AsVector(), 16).AsVector128(); b11 = AdvSimd.Add(b11, b12); b6 = Sve2.XorRotateRight(b6.AsVector(), b11.AsVector(), 12).AsVector128();
                b1 = AdvSimd.Add(AdvSimd.Add(b1, mB[10]), b6); b12 = Sve2.XorRotateRight(b12.AsVector(), b1.AsVector(), 8).AsVector128(); b11 = AdvSimd.Add(b11, b12); b6 = Sve2.XorRotateRight(b6.AsVector(), b11.AsVector(), 7).AsVector128();
                b2 = AdvSimd.Add(AdvSimd.Add(b2, mB[2]), b7); b13 = Sve2.XorRotateRight(b13.AsVector(), b2.AsVector(), 16).AsVector128(); b8 = AdvSimd.Add(b8, b13); b7 = Sve2.XorRotateRight(b7.AsVector(), b8.AsVector(), 12).AsVector128();
                b2 = AdvSimd.Add(AdvSimd.Add(b2, mB[6]), b7); b13 = Sve2.XorRotateRight(b13.AsVector(), b2.AsVector(), 8).AsVector128(); b8 = AdvSimd.Add(b8, b13); b7 = Sve2.XorRotateRight(b7.AsVector(), b8.AsVector(), 7).AsVector128();
                b3 = AdvSimd.Add(AdvSimd.Add(b3, mB[4]), b4); b14 = Sve2.XorRotateRight(b14.AsVector(), b3.AsVector(), 16).AsVector128(); b9 = AdvSimd.Add(b9, b14); b4 = Sve2.XorRotateRight(b4.AsVector(), b9.AsVector(), 12).AsVector128();
                b3 = AdvSimd.Add(AdvSimd.Add(b3, mB[7]), b4); b14 = Sve2.XorRotateRight(b14.AsVector(), b3.AsVector(), 8).AsVector128(); b9 = AdvSimd.Add(b9, b14); b4 = Sve2.XorRotateRight(b4.AsVector(), b9.AsVector(), 7).AsVector128();
                // Round 6: 11,15,5,0,1,9,8,6,14,10,2,12,3,4,7,13
                // columns, batch A
                a0 = AdvSimd.Add(AdvSimd.Add(a0, mA[11]), a4); a12 = Sve2.XorRotateRight(a12.AsVector(), a0.AsVector(), 16).AsVector128(); a8 = AdvSimd.Add(a8, a12); a4 = Sve2.XorRotateRight(a4.AsVector(), a8.AsVector(), 12).AsVector128();
                a0 = AdvSimd.Add(AdvSimd.Add(a0, mA[15]), a4); a12 = Sve2.XorRotateRight(a12.AsVector(), a0.AsVector(), 8).AsVector128(); a8 = AdvSimd.Add(a8, a12); a4 = Sve2.XorRotateRight(a4.AsVector(), a8.AsVector(), 7).AsVector128();
                a1 = AdvSimd.Add(AdvSimd.Add(a1, mA[5]), a5); a13 = Sve2.XorRotateRight(a13.AsVector(), a1.AsVector(), 16).AsVector128(); a9 = AdvSimd.Add(a9, a13); a5 = Sve2.XorRotateRight(a5.AsVector(), a9.AsVector(), 12).AsVector128();
                a1 = AdvSimd.Add(AdvSimd.Add(a1, mA[0]), a5); a13 = Sve2.XorRotateRight(a13.AsVector(), a1.AsVector(), 8).AsVector128(); a9 = AdvSimd.Add(a9, a13); a5 = Sve2.XorRotateRight(a5.AsVector(), a9.AsVector(), 7).AsVector128();
                a2 = AdvSimd.Add(AdvSimd.Add(a2, mA[1]), a6); a14 = Sve2.XorRotateRight(a14.AsVector(), a2.AsVector(), 16).AsVector128(); a10 = AdvSimd.Add(a10, a14); a6 = Sve2.XorRotateRight(a6.AsVector(), a10.AsVector(), 12).AsVector128();
                a2 = AdvSimd.Add(AdvSimd.Add(a2, mA[9]), a6); a14 = Sve2.XorRotateRight(a14.AsVector(), a2.AsVector(), 8).AsVector128(); a10 = AdvSimd.Add(a10, a14); a6 = Sve2.XorRotateRight(a6.AsVector(), a10.AsVector(), 7).AsVector128();
                a3 = AdvSimd.Add(AdvSimd.Add(a3, mA[8]), a7); a15 = Sve2.XorRotateRight(a15.AsVector(), a3.AsVector(), 16).AsVector128(); a11 = AdvSimd.Add(a11, a15); a7 = Sve2.XorRotateRight(a7.AsVector(), a11.AsVector(), 12).AsVector128();
                a3 = AdvSimd.Add(AdvSimd.Add(a3, mA[6]), a7); a15 = Sve2.XorRotateRight(a15.AsVector(), a3.AsVector(), 8).AsVector128(); a11 = AdvSimd.Add(a11, a15); a7 = Sve2.XorRotateRight(a7.AsVector(), a11.AsVector(), 7).AsVector128();
                // columns, batch B
                b0 = AdvSimd.Add(AdvSimd.Add(b0, mB[11]), b4); b12 = Sve2.XorRotateRight(b12.AsVector(), b0.AsVector(), 16).AsVector128(); b8 = AdvSimd.Add(b8, b12); b4 = Sve2.XorRotateRight(b4.AsVector(), b8.AsVector(), 12).AsVector128();
                b0 = AdvSimd.Add(AdvSimd.Add(b0, mB[15]), b4); b12 = Sve2.XorRotateRight(b12.AsVector(), b0.AsVector(), 8).AsVector128(); b8 = AdvSimd.Add(b8, b12); b4 = Sve2.XorRotateRight(b4.AsVector(), b8.AsVector(), 7).AsVector128();
                b1 = AdvSimd.Add(AdvSimd.Add(b1, mB[5]), b5); b13 = Sve2.XorRotateRight(b13.AsVector(), b1.AsVector(), 16).AsVector128(); b9 = AdvSimd.Add(b9, b13); b5 = Sve2.XorRotateRight(b5.AsVector(), b9.AsVector(), 12).AsVector128();
                b1 = AdvSimd.Add(AdvSimd.Add(b1, mB[0]), b5); b13 = Sve2.XorRotateRight(b13.AsVector(), b1.AsVector(), 8).AsVector128(); b9 = AdvSimd.Add(b9, b13); b5 = Sve2.XorRotateRight(b5.AsVector(), b9.AsVector(), 7).AsVector128();
                b2 = AdvSimd.Add(AdvSimd.Add(b2, mB[1]), b6); b14 = Sve2.XorRotateRight(b14.AsVector(), b2.AsVector(), 16).AsVector128(); b10 = AdvSimd.Add(b10, b14); b6 = Sve2.XorRotateRight(b6.AsVector(), b10.AsVector(), 12).AsVector128();
                b2 = AdvSimd.Add(AdvSimd.Add(b2, mB[9]), b6); b14 = Sve2.XorRotateRight(b14.AsVector(), b2.AsVector(), 8).AsVector128(); b10 = AdvSimd.Add(b10, b14); b6 = Sve2.XorRotateRight(b6.AsVector(), b10.AsVector(), 7).AsVector128();
                b3 = AdvSimd.Add(AdvSimd.Add(b3, mB[8]), b7); b15 = Sve2.XorRotateRight(b15.AsVector(), b3.AsVector(), 16).AsVector128(); b11 = AdvSimd.Add(b11, b15); b7 = Sve2.XorRotateRight(b7.AsVector(), b11.AsVector(), 12).AsVector128();
                b3 = AdvSimd.Add(AdvSimd.Add(b3, mB[6]), b7); b15 = Sve2.XorRotateRight(b15.AsVector(), b3.AsVector(), 8).AsVector128(); b11 = AdvSimd.Add(b11, b15); b7 = Sve2.XorRotateRight(b7.AsVector(), b11.AsVector(), 7).AsVector128();
                // diagonals, batch A
                a0 = AdvSimd.Add(AdvSimd.Add(a0, mA[14]), a5); a15 = Sve2.XorRotateRight(a15.AsVector(), a0.AsVector(), 16).AsVector128(); a10 = AdvSimd.Add(a10, a15); a5 = Sve2.XorRotateRight(a5.AsVector(), a10.AsVector(), 12).AsVector128();
                a0 = AdvSimd.Add(AdvSimd.Add(a0, mA[10]), a5); a15 = Sve2.XorRotateRight(a15.AsVector(), a0.AsVector(), 8).AsVector128(); a10 = AdvSimd.Add(a10, a15); a5 = Sve2.XorRotateRight(a5.AsVector(), a10.AsVector(), 7).AsVector128();
                a1 = AdvSimd.Add(AdvSimd.Add(a1, mA[2]), a6); a12 = Sve2.XorRotateRight(a12.AsVector(), a1.AsVector(), 16).AsVector128(); a11 = AdvSimd.Add(a11, a12); a6 = Sve2.XorRotateRight(a6.AsVector(), a11.AsVector(), 12).AsVector128();
                a1 = AdvSimd.Add(AdvSimd.Add(a1, mA[12]), a6); a12 = Sve2.XorRotateRight(a12.AsVector(), a1.AsVector(), 8).AsVector128(); a11 = AdvSimd.Add(a11, a12); a6 = Sve2.XorRotateRight(a6.AsVector(), a11.AsVector(), 7).AsVector128();
                a2 = AdvSimd.Add(AdvSimd.Add(a2, mA[3]), a7); a13 = Sve2.XorRotateRight(a13.AsVector(), a2.AsVector(), 16).AsVector128(); a8 = AdvSimd.Add(a8, a13); a7 = Sve2.XorRotateRight(a7.AsVector(), a8.AsVector(), 12).AsVector128();
                a2 = AdvSimd.Add(AdvSimd.Add(a2, mA[4]), a7); a13 = Sve2.XorRotateRight(a13.AsVector(), a2.AsVector(), 8).AsVector128(); a8 = AdvSimd.Add(a8, a13); a7 = Sve2.XorRotateRight(a7.AsVector(), a8.AsVector(), 7).AsVector128();
                a3 = AdvSimd.Add(AdvSimd.Add(a3, mA[7]), a4); a14 = Sve2.XorRotateRight(a14.AsVector(), a3.AsVector(), 16).AsVector128(); a9 = AdvSimd.Add(a9, a14); a4 = Sve2.XorRotateRight(a4.AsVector(), a9.AsVector(), 12).AsVector128();
                a3 = AdvSimd.Add(AdvSimd.Add(a3, mA[13]), a4); a14 = Sve2.XorRotateRight(a14.AsVector(), a3.AsVector(), 8).AsVector128(); a9 = AdvSimd.Add(a9, a14); a4 = Sve2.XorRotateRight(a4.AsVector(), a9.AsVector(), 7).AsVector128();
                // diagonals, batch B
                b0 = AdvSimd.Add(AdvSimd.Add(b0, mB[14]), b5); b15 = Sve2.XorRotateRight(b15.AsVector(), b0.AsVector(), 16).AsVector128(); b10 = AdvSimd.Add(b10, b15); b5 = Sve2.XorRotateRight(b5.AsVector(), b10.AsVector(), 12).AsVector128();
                b0 = AdvSimd.Add(AdvSimd.Add(b0, mB[10]), b5); b15 = Sve2.XorRotateRight(b15.AsVector(), b0.AsVector(), 8).AsVector128(); b10 = AdvSimd.Add(b10, b15); b5 = Sve2.XorRotateRight(b5.AsVector(), b10.AsVector(), 7).AsVector128();
                b1 = AdvSimd.Add(AdvSimd.Add(b1, mB[2]), b6); b12 = Sve2.XorRotateRight(b12.AsVector(), b1.AsVector(), 16).AsVector128(); b11 = AdvSimd.Add(b11, b12); b6 = Sve2.XorRotateRight(b6.AsVector(), b11.AsVector(), 12).AsVector128();
                b1 = AdvSimd.Add(AdvSimd.Add(b1, mB[12]), b6); b12 = Sve2.XorRotateRight(b12.AsVector(), b1.AsVector(), 8).AsVector128(); b11 = AdvSimd.Add(b11, b12); b6 = Sve2.XorRotateRight(b6.AsVector(), b11.AsVector(), 7).AsVector128();
                b2 = AdvSimd.Add(AdvSimd.Add(b2, mB[3]), b7); b13 = Sve2.XorRotateRight(b13.AsVector(), b2.AsVector(), 16).AsVector128(); b8 = AdvSimd.Add(b8, b13); b7 = Sve2.XorRotateRight(b7.AsVector(), b8.AsVector(), 12).AsVector128();
                b2 = AdvSimd.Add(AdvSimd.Add(b2, mB[4]), b7); b13 = Sve2.XorRotateRight(b13.AsVector(), b2.AsVector(), 8).AsVector128(); b8 = AdvSimd.Add(b8, b13); b7 = Sve2.XorRotateRight(b7.AsVector(), b8.AsVector(), 7).AsVector128();
                b3 = AdvSimd.Add(AdvSimd.Add(b3, mB[7]), b4); b14 = Sve2.XorRotateRight(b14.AsVector(), b3.AsVector(), 16).AsVector128(); b9 = AdvSimd.Add(b9, b14); b4 = Sve2.XorRotateRight(b4.AsVector(), b9.AsVector(), 12).AsVector128();
                b3 = AdvSimd.Add(AdvSimd.Add(b3, mB[13]), b4); b14 = Sve2.XorRotateRight(b14.AsVector(), b3.AsVector(), 8).AsVector128(); b9 = AdvSimd.Add(b9, b14); b4 = Sve2.XorRotateRight(b4.AsVector(), b9.AsVector(), 7).AsVector128();

                ca0 = AdvSimd.Xor(a0, a8); ca1 = AdvSimd.Xor(a1, a9); ca2 = AdvSimd.Xor(a2, a10); ca3 = AdvSimd.Xor(a3, a11);
                ca4 = AdvSimd.Xor(a4, a12); ca5 = AdvSimd.Xor(a5, a13); ca6 = AdvSimd.Xor(a6, a14); ca7 = AdvSimd.Xor(a7, a15);
                cb0 = AdvSimd.Xor(b0, b8); cb1 = AdvSimd.Xor(b1, b9); cb2 = AdvSimd.Xor(b2, b10); cb3 = AdvSimd.Xor(b3, b11);
                cb4 = AdvSimd.Xor(b4, b12); cb5 = AdvSimd.Xor(b5, b13); cb6 = AdvSimd.Xor(b6, b14); cb7 = AdvSimd.Xor(b7, b15);
            }
        }

        ref uint outRef = ref MemoryMarshal.GetReference(cvs);
        Transpose4X4(ca0, ca1, ca2, ca3, out var o0, out var o1, out var o2, out var o3);
        Transpose4X4(ca4, ca5, ca6, ca7, out var o4, out var o5, out var o6, out var o7);
        VectorCompat.Store(o0, ref outRef);      VectorCompat.Store(o4, ref outRef, 4);   // chunk 0
        VectorCompat.Store(o1, ref outRef, 8);   VectorCompat.Store(o5, ref outRef, 12);  // chunk 1
        VectorCompat.Store(o2, ref outRef, 16);  VectorCompat.Store(o6, ref outRef, 20);  // chunk 2
        VectorCompat.Store(o3, ref outRef, 24);  VectorCompat.Store(o7, ref outRef, 28);  // chunk 3
        Transpose4X4(cb0, cb1, cb2, cb3, out o0, out o1, out o2, out o3);
        Transpose4X4(cb4, cb5, cb6, cb7, out o4, out o5, out o6, out o7);
        VectorCompat.Store(o0, ref outRef, 32);  VectorCompat.Store(o4, ref outRef, 36);  // chunk 4
        VectorCompat.Store(o1, ref outRef, 40);  VectorCompat.Store(o5, ref outRef, 44);  // chunk 5
        VectorCompat.Store(o2, ref outRef, 48);  VectorCompat.Store(o6, ref outRef, 52);  // chunk 6
        VectorCompat.Store(o3, ref outRef, 56);  VectorCompat.Store(o7, ref outRef, 60);  // chunk 7
    }
}
#endif
