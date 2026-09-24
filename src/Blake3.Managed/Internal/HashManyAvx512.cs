#if NET8_0_OR_GREATER
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Blake3.Managed.Internal;

/// <summary>
/// Sixteen chunks at a time in 512-bit vectors, one chunk per 32-bit lane.
/// </summary>
/// <remarks>
/// <para>
/// Written out statement by statement (generated), with each half-round emitted step by step
/// across its four G's and the message word added before <c>b</c>. Everything that lives across
/// blocks -- the message, the chaining values, the counters -- sits in one 64-byte-aligned
/// scratch area and is read as memory operands, so the rounds need only the sixteen state
/// registers and nothing is spilled.
/// </para>
/// <para>
/// An earlier 16-way attempt was recorded as 2.5x slower and 512-bit code in general as
/// "bimodal": fast in some processes and slow in others. The cause was the transpose helper.
/// The kernel's IL is too large for the inliner, so the helper is a real call, and under
/// tiered compilation it can stay unoptimized tier-0 code indefinitely. Marked
/// <c>NoInlining | AggressiveOptimization</c> it is compiled optimized up front, and the kernel
/// measured 0.73-0.86 of two <see cref="HashManyAvx2.HashManySerial"/> calls on Zen 4, in every
/// process (2026-09-23).
/// </para>
/// </remarks>
internal static class HashManyAvx512
{
    public static bool IsSupported => Avx512F.IsSupported && Avx512F.VL.IsSupported;

    private const int ChunkLen = Blake3Constants.ChunkLen;

    /// <summary>
    /// Chaining values of sixteen complete chunks: <paramref name="chunks"/> is 16 KiB and
    /// <paramref name="cvs"/> receives 128 words, chunk-major.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.NoInlining | MethodImplOptions.AggressiveOptimization)]
    public static unsafe void HashMany16(ReadOnlySpan<byte> chunks, ReadOnlySpan<uint> key,
        ulong counter, uint flags, Span<uint> cvs)
    {
        _ = chunks[16 * ChunkLen - 1];
        _ = cvs[16 * 8 - 1];

        byte* raw = stackalloc byte[26 * 64 + 64];
        Vector512<uint>* m = (Vector512<uint>*)(((nuint)raw + 63) & ~(nuint)63); // 16 message words
        Vector512<uint>* cv = m + 16;                                           // 8 chaining values
        Vector512<uint>* ctr = m + 24;                                          // counter low, high

        for (int i = 0; i < 8; i++) cv[i] = Vector512.Create(key[i]);
        var lane = Vector512.Create(0ul, 1, 2, 3, 4, 5, 6, 7);
        var lo8 = Avx512F.Add(Vector512.Create(counter), lane);
        var hi8 = Avx512F.Add(Vector512.Create(counter + 8), lane);
        ctr[0] = Avx512F.PermuteVar16x32x2(lo8.AsUInt32(),
            Vector512.Create(0u, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30), hi8.AsUInt32());
        ctr[1] = Avx512F.PermuteVar16x32x2(lo8.AsUInt32(),
            Vector512.Create(1u, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31), hi8.AsUInt32());

        fixed (byte* chunksPtr = chunks)
        {
            for (int blk = 0; blk < 16; blk++)
            {
                Transpose16(chunksPtr + blk * 64, ChunkLen, m);
                uint blockFlags = flags;
                if (blk == 0) blockFlags |= Blake3Constants.ChunkStart;
                if (blk == 15) blockFlags |= Blake3Constants.ChunkEnd;

                Vector512<uint> s0 = cv[0], s1 = cv[1], s2 = cv[2], s3 = cv[3];
                Vector512<uint> s4 = cv[4], s5 = cv[5], s6 = cv[6], s7 = cv[7];
                Vector512<uint> s8 = Vector512.Create(Blake3Constants.Iv0), s9 = Vector512.Create(Blake3Constants.Iv1);
                Vector512<uint> s10 = Vector512.Create(Blake3Constants.Iv2), s11 = Vector512.Create(Blake3Constants.Iv3);
                Vector512<uint> s12 = ctr[0], s13 = ctr[1];
                Vector512<uint> s14 = Vector512.Create((uint)Blake3Constants.BlockLen), s15 = Vector512.Create(blockFlags);

                // Round 0
                s0 = Avx512F.Add(Avx512F.Add(s0, m[0]), s4); s1 = Avx512F.Add(Avx512F.Add(s1, m[2]), s5); s2 = Avx512F.Add(Avx512F.Add(s2, m[4]), s6); s3 = Avx512F.Add(Avx512F.Add(s3, m[6]), s7);
                s12 = Avx512F.RotateRight(Avx512F.Xor(s12, s0), 16); s13 = Avx512F.RotateRight(Avx512F.Xor(s13, s1), 16); s14 = Avx512F.RotateRight(Avx512F.Xor(s14, s2), 16); s15 = Avx512F.RotateRight(Avx512F.Xor(s15, s3), 16);
                s8 = Avx512F.Add(s8, s12); s9 = Avx512F.Add(s9, s13); s10 = Avx512F.Add(s10, s14); s11 = Avx512F.Add(s11, s15);
                s4 = Avx512F.RotateRight(Avx512F.Xor(s4, s8), 12); s5 = Avx512F.RotateRight(Avx512F.Xor(s5, s9), 12); s6 = Avx512F.RotateRight(Avx512F.Xor(s6, s10), 12); s7 = Avx512F.RotateRight(Avx512F.Xor(s7, s11), 12);
                s0 = Avx512F.Add(Avx512F.Add(s0, m[1]), s4); s1 = Avx512F.Add(Avx512F.Add(s1, m[3]), s5); s2 = Avx512F.Add(Avx512F.Add(s2, m[5]), s6); s3 = Avx512F.Add(Avx512F.Add(s3, m[7]), s7);
                s12 = Avx512F.RotateRight(Avx512F.Xor(s12, s0), 8); s13 = Avx512F.RotateRight(Avx512F.Xor(s13, s1), 8); s14 = Avx512F.RotateRight(Avx512F.Xor(s14, s2), 8); s15 = Avx512F.RotateRight(Avx512F.Xor(s15, s3), 8);
                s8 = Avx512F.Add(s8, s12); s9 = Avx512F.Add(s9, s13); s10 = Avx512F.Add(s10, s14); s11 = Avx512F.Add(s11, s15);
                s4 = Avx512F.RotateRight(Avx512F.Xor(s4, s8), 7); s5 = Avx512F.RotateRight(Avx512F.Xor(s5, s9), 7); s6 = Avx512F.RotateRight(Avx512F.Xor(s6, s10), 7); s7 = Avx512F.RotateRight(Avx512F.Xor(s7, s11), 7);
                s0 = Avx512F.Add(Avx512F.Add(s0, m[8]), s5); s1 = Avx512F.Add(Avx512F.Add(s1, m[10]), s6); s2 = Avx512F.Add(Avx512F.Add(s2, m[12]), s7); s3 = Avx512F.Add(Avx512F.Add(s3, m[14]), s4);
                s15 = Avx512F.RotateRight(Avx512F.Xor(s15, s0), 16); s12 = Avx512F.RotateRight(Avx512F.Xor(s12, s1), 16); s13 = Avx512F.RotateRight(Avx512F.Xor(s13, s2), 16); s14 = Avx512F.RotateRight(Avx512F.Xor(s14, s3), 16);
                s10 = Avx512F.Add(s10, s15); s11 = Avx512F.Add(s11, s12); s8 = Avx512F.Add(s8, s13); s9 = Avx512F.Add(s9, s14);
                s5 = Avx512F.RotateRight(Avx512F.Xor(s5, s10), 12); s6 = Avx512F.RotateRight(Avx512F.Xor(s6, s11), 12); s7 = Avx512F.RotateRight(Avx512F.Xor(s7, s8), 12); s4 = Avx512F.RotateRight(Avx512F.Xor(s4, s9), 12);
                s0 = Avx512F.Add(Avx512F.Add(s0, m[9]), s5); s1 = Avx512F.Add(Avx512F.Add(s1, m[11]), s6); s2 = Avx512F.Add(Avx512F.Add(s2, m[13]), s7); s3 = Avx512F.Add(Avx512F.Add(s3, m[15]), s4);
                s15 = Avx512F.RotateRight(Avx512F.Xor(s15, s0), 8); s12 = Avx512F.RotateRight(Avx512F.Xor(s12, s1), 8); s13 = Avx512F.RotateRight(Avx512F.Xor(s13, s2), 8); s14 = Avx512F.RotateRight(Avx512F.Xor(s14, s3), 8);
                s10 = Avx512F.Add(s10, s15); s11 = Avx512F.Add(s11, s12); s8 = Avx512F.Add(s8, s13); s9 = Avx512F.Add(s9, s14);
                s5 = Avx512F.RotateRight(Avx512F.Xor(s5, s10), 7); s6 = Avx512F.RotateRight(Avx512F.Xor(s6, s11), 7); s7 = Avx512F.RotateRight(Avx512F.Xor(s7, s8), 7); s4 = Avx512F.RotateRight(Avx512F.Xor(s4, s9), 7);
                // Round 1
                s0 = Avx512F.Add(Avx512F.Add(s0, m[2]), s4); s1 = Avx512F.Add(Avx512F.Add(s1, m[3]), s5); s2 = Avx512F.Add(Avx512F.Add(s2, m[7]), s6); s3 = Avx512F.Add(Avx512F.Add(s3, m[4]), s7);
                s12 = Avx512F.RotateRight(Avx512F.Xor(s12, s0), 16); s13 = Avx512F.RotateRight(Avx512F.Xor(s13, s1), 16); s14 = Avx512F.RotateRight(Avx512F.Xor(s14, s2), 16); s15 = Avx512F.RotateRight(Avx512F.Xor(s15, s3), 16);
                s8 = Avx512F.Add(s8, s12); s9 = Avx512F.Add(s9, s13); s10 = Avx512F.Add(s10, s14); s11 = Avx512F.Add(s11, s15);
                s4 = Avx512F.RotateRight(Avx512F.Xor(s4, s8), 12); s5 = Avx512F.RotateRight(Avx512F.Xor(s5, s9), 12); s6 = Avx512F.RotateRight(Avx512F.Xor(s6, s10), 12); s7 = Avx512F.RotateRight(Avx512F.Xor(s7, s11), 12);
                s0 = Avx512F.Add(Avx512F.Add(s0, m[6]), s4); s1 = Avx512F.Add(Avx512F.Add(s1, m[10]), s5); s2 = Avx512F.Add(Avx512F.Add(s2, m[0]), s6); s3 = Avx512F.Add(Avx512F.Add(s3, m[13]), s7);
                s12 = Avx512F.RotateRight(Avx512F.Xor(s12, s0), 8); s13 = Avx512F.RotateRight(Avx512F.Xor(s13, s1), 8); s14 = Avx512F.RotateRight(Avx512F.Xor(s14, s2), 8); s15 = Avx512F.RotateRight(Avx512F.Xor(s15, s3), 8);
                s8 = Avx512F.Add(s8, s12); s9 = Avx512F.Add(s9, s13); s10 = Avx512F.Add(s10, s14); s11 = Avx512F.Add(s11, s15);
                s4 = Avx512F.RotateRight(Avx512F.Xor(s4, s8), 7); s5 = Avx512F.RotateRight(Avx512F.Xor(s5, s9), 7); s6 = Avx512F.RotateRight(Avx512F.Xor(s6, s10), 7); s7 = Avx512F.RotateRight(Avx512F.Xor(s7, s11), 7);
                s0 = Avx512F.Add(Avx512F.Add(s0, m[1]), s5); s1 = Avx512F.Add(Avx512F.Add(s1, m[12]), s6); s2 = Avx512F.Add(Avx512F.Add(s2, m[9]), s7); s3 = Avx512F.Add(Avx512F.Add(s3, m[15]), s4);
                s15 = Avx512F.RotateRight(Avx512F.Xor(s15, s0), 16); s12 = Avx512F.RotateRight(Avx512F.Xor(s12, s1), 16); s13 = Avx512F.RotateRight(Avx512F.Xor(s13, s2), 16); s14 = Avx512F.RotateRight(Avx512F.Xor(s14, s3), 16);
                s10 = Avx512F.Add(s10, s15); s11 = Avx512F.Add(s11, s12); s8 = Avx512F.Add(s8, s13); s9 = Avx512F.Add(s9, s14);
                s5 = Avx512F.RotateRight(Avx512F.Xor(s5, s10), 12); s6 = Avx512F.RotateRight(Avx512F.Xor(s6, s11), 12); s7 = Avx512F.RotateRight(Avx512F.Xor(s7, s8), 12); s4 = Avx512F.RotateRight(Avx512F.Xor(s4, s9), 12);
                s0 = Avx512F.Add(Avx512F.Add(s0, m[11]), s5); s1 = Avx512F.Add(Avx512F.Add(s1, m[5]), s6); s2 = Avx512F.Add(Avx512F.Add(s2, m[14]), s7); s3 = Avx512F.Add(Avx512F.Add(s3, m[8]), s4);
                s15 = Avx512F.RotateRight(Avx512F.Xor(s15, s0), 8); s12 = Avx512F.RotateRight(Avx512F.Xor(s12, s1), 8); s13 = Avx512F.RotateRight(Avx512F.Xor(s13, s2), 8); s14 = Avx512F.RotateRight(Avx512F.Xor(s14, s3), 8);
                s10 = Avx512F.Add(s10, s15); s11 = Avx512F.Add(s11, s12); s8 = Avx512F.Add(s8, s13); s9 = Avx512F.Add(s9, s14);
                s5 = Avx512F.RotateRight(Avx512F.Xor(s5, s10), 7); s6 = Avx512F.RotateRight(Avx512F.Xor(s6, s11), 7); s7 = Avx512F.RotateRight(Avx512F.Xor(s7, s8), 7); s4 = Avx512F.RotateRight(Avx512F.Xor(s4, s9), 7);
                // Round 2
                s0 = Avx512F.Add(Avx512F.Add(s0, m[3]), s4); s1 = Avx512F.Add(Avx512F.Add(s1, m[10]), s5); s2 = Avx512F.Add(Avx512F.Add(s2, m[13]), s6); s3 = Avx512F.Add(Avx512F.Add(s3, m[7]), s7);
                s12 = Avx512F.RotateRight(Avx512F.Xor(s12, s0), 16); s13 = Avx512F.RotateRight(Avx512F.Xor(s13, s1), 16); s14 = Avx512F.RotateRight(Avx512F.Xor(s14, s2), 16); s15 = Avx512F.RotateRight(Avx512F.Xor(s15, s3), 16);
                s8 = Avx512F.Add(s8, s12); s9 = Avx512F.Add(s9, s13); s10 = Avx512F.Add(s10, s14); s11 = Avx512F.Add(s11, s15);
                s4 = Avx512F.RotateRight(Avx512F.Xor(s4, s8), 12); s5 = Avx512F.RotateRight(Avx512F.Xor(s5, s9), 12); s6 = Avx512F.RotateRight(Avx512F.Xor(s6, s10), 12); s7 = Avx512F.RotateRight(Avx512F.Xor(s7, s11), 12);
                s0 = Avx512F.Add(Avx512F.Add(s0, m[4]), s4); s1 = Avx512F.Add(Avx512F.Add(s1, m[12]), s5); s2 = Avx512F.Add(Avx512F.Add(s2, m[2]), s6); s3 = Avx512F.Add(Avx512F.Add(s3, m[14]), s7);
                s12 = Avx512F.RotateRight(Avx512F.Xor(s12, s0), 8); s13 = Avx512F.RotateRight(Avx512F.Xor(s13, s1), 8); s14 = Avx512F.RotateRight(Avx512F.Xor(s14, s2), 8); s15 = Avx512F.RotateRight(Avx512F.Xor(s15, s3), 8);
                s8 = Avx512F.Add(s8, s12); s9 = Avx512F.Add(s9, s13); s10 = Avx512F.Add(s10, s14); s11 = Avx512F.Add(s11, s15);
                s4 = Avx512F.RotateRight(Avx512F.Xor(s4, s8), 7); s5 = Avx512F.RotateRight(Avx512F.Xor(s5, s9), 7); s6 = Avx512F.RotateRight(Avx512F.Xor(s6, s10), 7); s7 = Avx512F.RotateRight(Avx512F.Xor(s7, s11), 7);
                s0 = Avx512F.Add(Avx512F.Add(s0, m[6]), s5); s1 = Avx512F.Add(Avx512F.Add(s1, m[9]), s6); s2 = Avx512F.Add(Avx512F.Add(s2, m[11]), s7); s3 = Avx512F.Add(Avx512F.Add(s3, m[8]), s4);
                s15 = Avx512F.RotateRight(Avx512F.Xor(s15, s0), 16); s12 = Avx512F.RotateRight(Avx512F.Xor(s12, s1), 16); s13 = Avx512F.RotateRight(Avx512F.Xor(s13, s2), 16); s14 = Avx512F.RotateRight(Avx512F.Xor(s14, s3), 16);
                s10 = Avx512F.Add(s10, s15); s11 = Avx512F.Add(s11, s12); s8 = Avx512F.Add(s8, s13); s9 = Avx512F.Add(s9, s14);
                s5 = Avx512F.RotateRight(Avx512F.Xor(s5, s10), 12); s6 = Avx512F.RotateRight(Avx512F.Xor(s6, s11), 12); s7 = Avx512F.RotateRight(Avx512F.Xor(s7, s8), 12); s4 = Avx512F.RotateRight(Avx512F.Xor(s4, s9), 12);
                s0 = Avx512F.Add(Avx512F.Add(s0, m[5]), s5); s1 = Avx512F.Add(Avx512F.Add(s1, m[0]), s6); s2 = Avx512F.Add(Avx512F.Add(s2, m[15]), s7); s3 = Avx512F.Add(Avx512F.Add(s3, m[1]), s4);
                s15 = Avx512F.RotateRight(Avx512F.Xor(s15, s0), 8); s12 = Avx512F.RotateRight(Avx512F.Xor(s12, s1), 8); s13 = Avx512F.RotateRight(Avx512F.Xor(s13, s2), 8); s14 = Avx512F.RotateRight(Avx512F.Xor(s14, s3), 8);
                s10 = Avx512F.Add(s10, s15); s11 = Avx512F.Add(s11, s12); s8 = Avx512F.Add(s8, s13); s9 = Avx512F.Add(s9, s14);
                s5 = Avx512F.RotateRight(Avx512F.Xor(s5, s10), 7); s6 = Avx512F.RotateRight(Avx512F.Xor(s6, s11), 7); s7 = Avx512F.RotateRight(Avx512F.Xor(s7, s8), 7); s4 = Avx512F.RotateRight(Avx512F.Xor(s4, s9), 7);
                // Round 3
                s0 = Avx512F.Add(Avx512F.Add(s0, m[10]), s4); s1 = Avx512F.Add(Avx512F.Add(s1, m[12]), s5); s2 = Avx512F.Add(Avx512F.Add(s2, m[14]), s6); s3 = Avx512F.Add(Avx512F.Add(s3, m[13]), s7);
                s12 = Avx512F.RotateRight(Avx512F.Xor(s12, s0), 16); s13 = Avx512F.RotateRight(Avx512F.Xor(s13, s1), 16); s14 = Avx512F.RotateRight(Avx512F.Xor(s14, s2), 16); s15 = Avx512F.RotateRight(Avx512F.Xor(s15, s3), 16);
                s8 = Avx512F.Add(s8, s12); s9 = Avx512F.Add(s9, s13); s10 = Avx512F.Add(s10, s14); s11 = Avx512F.Add(s11, s15);
                s4 = Avx512F.RotateRight(Avx512F.Xor(s4, s8), 12); s5 = Avx512F.RotateRight(Avx512F.Xor(s5, s9), 12); s6 = Avx512F.RotateRight(Avx512F.Xor(s6, s10), 12); s7 = Avx512F.RotateRight(Avx512F.Xor(s7, s11), 12);
                s0 = Avx512F.Add(Avx512F.Add(s0, m[7]), s4); s1 = Avx512F.Add(Avx512F.Add(s1, m[9]), s5); s2 = Avx512F.Add(Avx512F.Add(s2, m[3]), s6); s3 = Avx512F.Add(Avx512F.Add(s3, m[15]), s7);
                s12 = Avx512F.RotateRight(Avx512F.Xor(s12, s0), 8); s13 = Avx512F.RotateRight(Avx512F.Xor(s13, s1), 8); s14 = Avx512F.RotateRight(Avx512F.Xor(s14, s2), 8); s15 = Avx512F.RotateRight(Avx512F.Xor(s15, s3), 8);
                s8 = Avx512F.Add(s8, s12); s9 = Avx512F.Add(s9, s13); s10 = Avx512F.Add(s10, s14); s11 = Avx512F.Add(s11, s15);
                s4 = Avx512F.RotateRight(Avx512F.Xor(s4, s8), 7); s5 = Avx512F.RotateRight(Avx512F.Xor(s5, s9), 7); s6 = Avx512F.RotateRight(Avx512F.Xor(s6, s10), 7); s7 = Avx512F.RotateRight(Avx512F.Xor(s7, s11), 7);
                s0 = Avx512F.Add(Avx512F.Add(s0, m[4]), s5); s1 = Avx512F.Add(Avx512F.Add(s1, m[11]), s6); s2 = Avx512F.Add(Avx512F.Add(s2, m[5]), s7); s3 = Avx512F.Add(Avx512F.Add(s3, m[1]), s4);
                s15 = Avx512F.RotateRight(Avx512F.Xor(s15, s0), 16); s12 = Avx512F.RotateRight(Avx512F.Xor(s12, s1), 16); s13 = Avx512F.RotateRight(Avx512F.Xor(s13, s2), 16); s14 = Avx512F.RotateRight(Avx512F.Xor(s14, s3), 16);
                s10 = Avx512F.Add(s10, s15); s11 = Avx512F.Add(s11, s12); s8 = Avx512F.Add(s8, s13); s9 = Avx512F.Add(s9, s14);
                s5 = Avx512F.RotateRight(Avx512F.Xor(s5, s10), 12); s6 = Avx512F.RotateRight(Avx512F.Xor(s6, s11), 12); s7 = Avx512F.RotateRight(Avx512F.Xor(s7, s8), 12); s4 = Avx512F.RotateRight(Avx512F.Xor(s4, s9), 12);
                s0 = Avx512F.Add(Avx512F.Add(s0, m[0]), s5); s1 = Avx512F.Add(Avx512F.Add(s1, m[2]), s6); s2 = Avx512F.Add(Avx512F.Add(s2, m[8]), s7); s3 = Avx512F.Add(Avx512F.Add(s3, m[6]), s4);
                s15 = Avx512F.RotateRight(Avx512F.Xor(s15, s0), 8); s12 = Avx512F.RotateRight(Avx512F.Xor(s12, s1), 8); s13 = Avx512F.RotateRight(Avx512F.Xor(s13, s2), 8); s14 = Avx512F.RotateRight(Avx512F.Xor(s14, s3), 8);
                s10 = Avx512F.Add(s10, s15); s11 = Avx512F.Add(s11, s12); s8 = Avx512F.Add(s8, s13); s9 = Avx512F.Add(s9, s14);
                s5 = Avx512F.RotateRight(Avx512F.Xor(s5, s10), 7); s6 = Avx512F.RotateRight(Avx512F.Xor(s6, s11), 7); s7 = Avx512F.RotateRight(Avx512F.Xor(s7, s8), 7); s4 = Avx512F.RotateRight(Avx512F.Xor(s4, s9), 7);
                // Round 4
                s0 = Avx512F.Add(Avx512F.Add(s0, m[12]), s4); s1 = Avx512F.Add(Avx512F.Add(s1, m[9]), s5); s2 = Avx512F.Add(Avx512F.Add(s2, m[15]), s6); s3 = Avx512F.Add(Avx512F.Add(s3, m[14]), s7);
                s12 = Avx512F.RotateRight(Avx512F.Xor(s12, s0), 16); s13 = Avx512F.RotateRight(Avx512F.Xor(s13, s1), 16); s14 = Avx512F.RotateRight(Avx512F.Xor(s14, s2), 16); s15 = Avx512F.RotateRight(Avx512F.Xor(s15, s3), 16);
                s8 = Avx512F.Add(s8, s12); s9 = Avx512F.Add(s9, s13); s10 = Avx512F.Add(s10, s14); s11 = Avx512F.Add(s11, s15);
                s4 = Avx512F.RotateRight(Avx512F.Xor(s4, s8), 12); s5 = Avx512F.RotateRight(Avx512F.Xor(s5, s9), 12); s6 = Avx512F.RotateRight(Avx512F.Xor(s6, s10), 12); s7 = Avx512F.RotateRight(Avx512F.Xor(s7, s11), 12);
                s0 = Avx512F.Add(Avx512F.Add(s0, m[13]), s4); s1 = Avx512F.Add(Avx512F.Add(s1, m[11]), s5); s2 = Avx512F.Add(Avx512F.Add(s2, m[10]), s6); s3 = Avx512F.Add(Avx512F.Add(s3, m[8]), s7);
                s12 = Avx512F.RotateRight(Avx512F.Xor(s12, s0), 8); s13 = Avx512F.RotateRight(Avx512F.Xor(s13, s1), 8); s14 = Avx512F.RotateRight(Avx512F.Xor(s14, s2), 8); s15 = Avx512F.RotateRight(Avx512F.Xor(s15, s3), 8);
                s8 = Avx512F.Add(s8, s12); s9 = Avx512F.Add(s9, s13); s10 = Avx512F.Add(s10, s14); s11 = Avx512F.Add(s11, s15);
                s4 = Avx512F.RotateRight(Avx512F.Xor(s4, s8), 7); s5 = Avx512F.RotateRight(Avx512F.Xor(s5, s9), 7); s6 = Avx512F.RotateRight(Avx512F.Xor(s6, s10), 7); s7 = Avx512F.RotateRight(Avx512F.Xor(s7, s11), 7);
                s0 = Avx512F.Add(Avx512F.Add(s0, m[7]), s5); s1 = Avx512F.Add(Avx512F.Add(s1, m[5]), s6); s2 = Avx512F.Add(Avx512F.Add(s2, m[0]), s7); s3 = Avx512F.Add(Avx512F.Add(s3, m[6]), s4);
                s15 = Avx512F.RotateRight(Avx512F.Xor(s15, s0), 16); s12 = Avx512F.RotateRight(Avx512F.Xor(s12, s1), 16); s13 = Avx512F.RotateRight(Avx512F.Xor(s13, s2), 16); s14 = Avx512F.RotateRight(Avx512F.Xor(s14, s3), 16);
                s10 = Avx512F.Add(s10, s15); s11 = Avx512F.Add(s11, s12); s8 = Avx512F.Add(s8, s13); s9 = Avx512F.Add(s9, s14);
                s5 = Avx512F.RotateRight(Avx512F.Xor(s5, s10), 12); s6 = Avx512F.RotateRight(Avx512F.Xor(s6, s11), 12); s7 = Avx512F.RotateRight(Avx512F.Xor(s7, s8), 12); s4 = Avx512F.RotateRight(Avx512F.Xor(s4, s9), 12);
                s0 = Avx512F.Add(Avx512F.Add(s0, m[2]), s5); s1 = Avx512F.Add(Avx512F.Add(s1, m[3]), s6); s2 = Avx512F.Add(Avx512F.Add(s2, m[1]), s7); s3 = Avx512F.Add(Avx512F.Add(s3, m[4]), s4);
                s15 = Avx512F.RotateRight(Avx512F.Xor(s15, s0), 8); s12 = Avx512F.RotateRight(Avx512F.Xor(s12, s1), 8); s13 = Avx512F.RotateRight(Avx512F.Xor(s13, s2), 8); s14 = Avx512F.RotateRight(Avx512F.Xor(s14, s3), 8);
                s10 = Avx512F.Add(s10, s15); s11 = Avx512F.Add(s11, s12); s8 = Avx512F.Add(s8, s13); s9 = Avx512F.Add(s9, s14);
                s5 = Avx512F.RotateRight(Avx512F.Xor(s5, s10), 7); s6 = Avx512F.RotateRight(Avx512F.Xor(s6, s11), 7); s7 = Avx512F.RotateRight(Avx512F.Xor(s7, s8), 7); s4 = Avx512F.RotateRight(Avx512F.Xor(s4, s9), 7);
                // Round 5
                s0 = Avx512F.Add(Avx512F.Add(s0, m[9]), s4); s1 = Avx512F.Add(Avx512F.Add(s1, m[11]), s5); s2 = Avx512F.Add(Avx512F.Add(s2, m[8]), s6); s3 = Avx512F.Add(Avx512F.Add(s3, m[15]), s7);
                s12 = Avx512F.RotateRight(Avx512F.Xor(s12, s0), 16); s13 = Avx512F.RotateRight(Avx512F.Xor(s13, s1), 16); s14 = Avx512F.RotateRight(Avx512F.Xor(s14, s2), 16); s15 = Avx512F.RotateRight(Avx512F.Xor(s15, s3), 16);
                s8 = Avx512F.Add(s8, s12); s9 = Avx512F.Add(s9, s13); s10 = Avx512F.Add(s10, s14); s11 = Avx512F.Add(s11, s15);
                s4 = Avx512F.RotateRight(Avx512F.Xor(s4, s8), 12); s5 = Avx512F.RotateRight(Avx512F.Xor(s5, s9), 12); s6 = Avx512F.RotateRight(Avx512F.Xor(s6, s10), 12); s7 = Avx512F.RotateRight(Avx512F.Xor(s7, s11), 12);
                s0 = Avx512F.Add(Avx512F.Add(s0, m[14]), s4); s1 = Avx512F.Add(Avx512F.Add(s1, m[5]), s5); s2 = Avx512F.Add(Avx512F.Add(s2, m[12]), s6); s3 = Avx512F.Add(Avx512F.Add(s3, m[1]), s7);
                s12 = Avx512F.RotateRight(Avx512F.Xor(s12, s0), 8); s13 = Avx512F.RotateRight(Avx512F.Xor(s13, s1), 8); s14 = Avx512F.RotateRight(Avx512F.Xor(s14, s2), 8); s15 = Avx512F.RotateRight(Avx512F.Xor(s15, s3), 8);
                s8 = Avx512F.Add(s8, s12); s9 = Avx512F.Add(s9, s13); s10 = Avx512F.Add(s10, s14); s11 = Avx512F.Add(s11, s15);
                s4 = Avx512F.RotateRight(Avx512F.Xor(s4, s8), 7); s5 = Avx512F.RotateRight(Avx512F.Xor(s5, s9), 7); s6 = Avx512F.RotateRight(Avx512F.Xor(s6, s10), 7); s7 = Avx512F.RotateRight(Avx512F.Xor(s7, s11), 7);
                s0 = Avx512F.Add(Avx512F.Add(s0, m[13]), s5); s1 = Avx512F.Add(Avx512F.Add(s1, m[0]), s6); s2 = Avx512F.Add(Avx512F.Add(s2, m[2]), s7); s3 = Avx512F.Add(Avx512F.Add(s3, m[4]), s4);
                s15 = Avx512F.RotateRight(Avx512F.Xor(s15, s0), 16); s12 = Avx512F.RotateRight(Avx512F.Xor(s12, s1), 16); s13 = Avx512F.RotateRight(Avx512F.Xor(s13, s2), 16); s14 = Avx512F.RotateRight(Avx512F.Xor(s14, s3), 16);
                s10 = Avx512F.Add(s10, s15); s11 = Avx512F.Add(s11, s12); s8 = Avx512F.Add(s8, s13); s9 = Avx512F.Add(s9, s14);
                s5 = Avx512F.RotateRight(Avx512F.Xor(s5, s10), 12); s6 = Avx512F.RotateRight(Avx512F.Xor(s6, s11), 12); s7 = Avx512F.RotateRight(Avx512F.Xor(s7, s8), 12); s4 = Avx512F.RotateRight(Avx512F.Xor(s4, s9), 12);
                s0 = Avx512F.Add(Avx512F.Add(s0, m[3]), s5); s1 = Avx512F.Add(Avx512F.Add(s1, m[10]), s6); s2 = Avx512F.Add(Avx512F.Add(s2, m[6]), s7); s3 = Avx512F.Add(Avx512F.Add(s3, m[7]), s4);
                s15 = Avx512F.RotateRight(Avx512F.Xor(s15, s0), 8); s12 = Avx512F.RotateRight(Avx512F.Xor(s12, s1), 8); s13 = Avx512F.RotateRight(Avx512F.Xor(s13, s2), 8); s14 = Avx512F.RotateRight(Avx512F.Xor(s14, s3), 8);
                s10 = Avx512F.Add(s10, s15); s11 = Avx512F.Add(s11, s12); s8 = Avx512F.Add(s8, s13); s9 = Avx512F.Add(s9, s14);
                s5 = Avx512F.RotateRight(Avx512F.Xor(s5, s10), 7); s6 = Avx512F.RotateRight(Avx512F.Xor(s6, s11), 7); s7 = Avx512F.RotateRight(Avx512F.Xor(s7, s8), 7); s4 = Avx512F.RotateRight(Avx512F.Xor(s4, s9), 7);
                // Round 6
                s0 = Avx512F.Add(Avx512F.Add(s0, m[11]), s4); s1 = Avx512F.Add(Avx512F.Add(s1, m[5]), s5); s2 = Avx512F.Add(Avx512F.Add(s2, m[1]), s6); s3 = Avx512F.Add(Avx512F.Add(s3, m[8]), s7);
                s12 = Avx512F.RotateRight(Avx512F.Xor(s12, s0), 16); s13 = Avx512F.RotateRight(Avx512F.Xor(s13, s1), 16); s14 = Avx512F.RotateRight(Avx512F.Xor(s14, s2), 16); s15 = Avx512F.RotateRight(Avx512F.Xor(s15, s3), 16);
                s8 = Avx512F.Add(s8, s12); s9 = Avx512F.Add(s9, s13); s10 = Avx512F.Add(s10, s14); s11 = Avx512F.Add(s11, s15);
                s4 = Avx512F.RotateRight(Avx512F.Xor(s4, s8), 12); s5 = Avx512F.RotateRight(Avx512F.Xor(s5, s9), 12); s6 = Avx512F.RotateRight(Avx512F.Xor(s6, s10), 12); s7 = Avx512F.RotateRight(Avx512F.Xor(s7, s11), 12);
                s0 = Avx512F.Add(Avx512F.Add(s0, m[15]), s4); s1 = Avx512F.Add(Avx512F.Add(s1, m[0]), s5); s2 = Avx512F.Add(Avx512F.Add(s2, m[9]), s6); s3 = Avx512F.Add(Avx512F.Add(s3, m[6]), s7);
                s12 = Avx512F.RotateRight(Avx512F.Xor(s12, s0), 8); s13 = Avx512F.RotateRight(Avx512F.Xor(s13, s1), 8); s14 = Avx512F.RotateRight(Avx512F.Xor(s14, s2), 8); s15 = Avx512F.RotateRight(Avx512F.Xor(s15, s3), 8);
                s8 = Avx512F.Add(s8, s12); s9 = Avx512F.Add(s9, s13); s10 = Avx512F.Add(s10, s14); s11 = Avx512F.Add(s11, s15);
                s4 = Avx512F.RotateRight(Avx512F.Xor(s4, s8), 7); s5 = Avx512F.RotateRight(Avx512F.Xor(s5, s9), 7); s6 = Avx512F.RotateRight(Avx512F.Xor(s6, s10), 7); s7 = Avx512F.RotateRight(Avx512F.Xor(s7, s11), 7);
                s0 = Avx512F.Add(Avx512F.Add(s0, m[14]), s5); s1 = Avx512F.Add(Avx512F.Add(s1, m[2]), s6); s2 = Avx512F.Add(Avx512F.Add(s2, m[3]), s7); s3 = Avx512F.Add(Avx512F.Add(s3, m[7]), s4);
                s15 = Avx512F.RotateRight(Avx512F.Xor(s15, s0), 16); s12 = Avx512F.RotateRight(Avx512F.Xor(s12, s1), 16); s13 = Avx512F.RotateRight(Avx512F.Xor(s13, s2), 16); s14 = Avx512F.RotateRight(Avx512F.Xor(s14, s3), 16);
                s10 = Avx512F.Add(s10, s15); s11 = Avx512F.Add(s11, s12); s8 = Avx512F.Add(s8, s13); s9 = Avx512F.Add(s9, s14);
                s5 = Avx512F.RotateRight(Avx512F.Xor(s5, s10), 12); s6 = Avx512F.RotateRight(Avx512F.Xor(s6, s11), 12); s7 = Avx512F.RotateRight(Avx512F.Xor(s7, s8), 12); s4 = Avx512F.RotateRight(Avx512F.Xor(s4, s9), 12);
                s0 = Avx512F.Add(Avx512F.Add(s0, m[10]), s5); s1 = Avx512F.Add(Avx512F.Add(s1, m[12]), s6); s2 = Avx512F.Add(Avx512F.Add(s2, m[4]), s7); s3 = Avx512F.Add(Avx512F.Add(s3, m[13]), s4);
                s15 = Avx512F.RotateRight(Avx512F.Xor(s15, s0), 8); s12 = Avx512F.RotateRight(Avx512F.Xor(s12, s1), 8); s13 = Avx512F.RotateRight(Avx512F.Xor(s13, s2), 8); s14 = Avx512F.RotateRight(Avx512F.Xor(s14, s3), 8);
                s10 = Avx512F.Add(s10, s15); s11 = Avx512F.Add(s11, s12); s8 = Avx512F.Add(s8, s13); s9 = Avx512F.Add(s9, s14);
                s5 = Avx512F.RotateRight(Avx512F.Xor(s5, s10), 7); s6 = Avx512F.RotateRight(Avx512F.Xor(s6, s11), 7); s7 = Avx512F.RotateRight(Avx512F.Xor(s7, s8), 7); s4 = Avx512F.RotateRight(Avx512F.Xor(s4, s9), 7);

                cv[0] = Avx512F.Xor(s0, s8); cv[1] = Avx512F.Xor(s1, s9);
                cv[2] = Avx512F.Xor(s2, s10); cv[3] = Avx512F.Xor(s3, s11);
                cv[4] = Avx512F.Xor(s4, s12); cv[5] = Avx512F.Xor(s5, s13);
                cv[6] = Avx512F.Xor(s6, s14); cv[7] = Avx512F.Xor(s7, s15);
            }
        }

        // Word-major -> chunk-major. Once per sixteen chunks, so plain scalar moves are fine.
        uint* words = (uint*)cv;
        ref uint outRef = ref MemoryMarshal.GetReference(cvs);
        for (int j = 0; j < 16; j++)
            for (int i = 0; i < 8; i++)
                Unsafe.Add(ref outRef, j * 8 + i) = words[i * 16 + j];
    }

    /// <summary>
    /// Writes <paramref name="output"/>, a positive multiple of 1 KiB, as root output blocks
    /// <paramref name="counter"/>, <paramref name="counter"/> + 1, ...: sixteen blocks per
    /// iteration, one per lane. Every lane shares the root CV and message.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.NoInlining | MethodImplOptions.AggressiveOptimization)]
    public static unsafe void HashOutput16(ReadOnlySpan<uint> inputCv, ReadOnlySpan<uint> block,
        ulong counter, uint blockLen, uint flags, Span<byte> output)
    {
        _ = inputCv[7];
        _ = block[15];

        byte* raw = stackalloc byte[40 * 64 + 64];
        Vector512<uint>* m = (Vector512<uint>*)(((nuint)raw + 63) & ~(nuint)63); // 16 message words
        Vector512<uint>* cv = m + 16;                                           // 8 input CV words
        Vector512<uint>* words = m + 24;                                        // 16 output words
        for (int i = 0; i < 16; i++) m[i] = Vector512.Create(block[i]);
        for (int i = 0; i < 8; i++) cv[i] = Vector512.Create(inputCv[i]);
        var lane = Vector512.Create(0ul, 1, 2, 3, 4, 5, 6, 7);
        var loIdx = Vector512.Create(0u, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
        var hiIdx = Vector512.Create(1u, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31);

        fixed (byte* outPtr = output)
        {
            for (int pos = 0; pos < output.Length; pos += 1024, counter += 16)
            {
                var lo8 = Avx512F.Add(Vector512.Create(counter), lane);
                var hi8 = Avx512F.Add(Vector512.Create(counter + 8), lane);
                Vector512<uint> s0 = cv[0], s1 = cv[1], s2 = cv[2], s3 = cv[3];
                Vector512<uint> s4 = cv[4], s5 = cv[5], s6 = cv[6], s7 = cv[7];
                Vector512<uint> s8 = Vector512.Create(Blake3Constants.Iv0), s9 = Vector512.Create(Blake3Constants.Iv1);
                Vector512<uint> s10 = Vector512.Create(Blake3Constants.Iv2), s11 = Vector512.Create(Blake3Constants.Iv3);
                Vector512<uint> s12 = Avx512F.PermuteVar16x32x2(lo8.AsUInt32(), loIdx, hi8.AsUInt32());
                Vector512<uint> s13 = Avx512F.PermuteVar16x32x2(lo8.AsUInt32(), hiIdx, hi8.AsUInt32());
                Vector512<uint> s14 = Vector512.Create(blockLen), s15 = Vector512.Create(flags | Blake3Constants.Root);

                // Round 0
                s0 = Avx512F.Add(Avx512F.Add(s0, m[0]), s4); s1 = Avx512F.Add(Avx512F.Add(s1, m[2]), s5); s2 = Avx512F.Add(Avx512F.Add(s2, m[4]), s6); s3 = Avx512F.Add(Avx512F.Add(s3, m[6]), s7);
                s12 = Avx512F.RotateRight(Avx512F.Xor(s12, s0), 16); s13 = Avx512F.RotateRight(Avx512F.Xor(s13, s1), 16); s14 = Avx512F.RotateRight(Avx512F.Xor(s14, s2), 16); s15 = Avx512F.RotateRight(Avx512F.Xor(s15, s3), 16);
                s8 = Avx512F.Add(s8, s12); s9 = Avx512F.Add(s9, s13); s10 = Avx512F.Add(s10, s14); s11 = Avx512F.Add(s11, s15);
                s4 = Avx512F.RotateRight(Avx512F.Xor(s4, s8), 12); s5 = Avx512F.RotateRight(Avx512F.Xor(s5, s9), 12); s6 = Avx512F.RotateRight(Avx512F.Xor(s6, s10), 12); s7 = Avx512F.RotateRight(Avx512F.Xor(s7, s11), 12);
                s0 = Avx512F.Add(Avx512F.Add(s0, m[1]), s4); s1 = Avx512F.Add(Avx512F.Add(s1, m[3]), s5); s2 = Avx512F.Add(Avx512F.Add(s2, m[5]), s6); s3 = Avx512F.Add(Avx512F.Add(s3, m[7]), s7);
                s12 = Avx512F.RotateRight(Avx512F.Xor(s12, s0), 8); s13 = Avx512F.RotateRight(Avx512F.Xor(s13, s1), 8); s14 = Avx512F.RotateRight(Avx512F.Xor(s14, s2), 8); s15 = Avx512F.RotateRight(Avx512F.Xor(s15, s3), 8);
                s8 = Avx512F.Add(s8, s12); s9 = Avx512F.Add(s9, s13); s10 = Avx512F.Add(s10, s14); s11 = Avx512F.Add(s11, s15);
                s4 = Avx512F.RotateRight(Avx512F.Xor(s4, s8), 7); s5 = Avx512F.RotateRight(Avx512F.Xor(s5, s9), 7); s6 = Avx512F.RotateRight(Avx512F.Xor(s6, s10), 7); s7 = Avx512F.RotateRight(Avx512F.Xor(s7, s11), 7);
                s0 = Avx512F.Add(Avx512F.Add(s0, m[8]), s5); s1 = Avx512F.Add(Avx512F.Add(s1, m[10]), s6); s2 = Avx512F.Add(Avx512F.Add(s2, m[12]), s7); s3 = Avx512F.Add(Avx512F.Add(s3, m[14]), s4);
                s15 = Avx512F.RotateRight(Avx512F.Xor(s15, s0), 16); s12 = Avx512F.RotateRight(Avx512F.Xor(s12, s1), 16); s13 = Avx512F.RotateRight(Avx512F.Xor(s13, s2), 16); s14 = Avx512F.RotateRight(Avx512F.Xor(s14, s3), 16);
                s10 = Avx512F.Add(s10, s15); s11 = Avx512F.Add(s11, s12); s8 = Avx512F.Add(s8, s13); s9 = Avx512F.Add(s9, s14);
                s5 = Avx512F.RotateRight(Avx512F.Xor(s5, s10), 12); s6 = Avx512F.RotateRight(Avx512F.Xor(s6, s11), 12); s7 = Avx512F.RotateRight(Avx512F.Xor(s7, s8), 12); s4 = Avx512F.RotateRight(Avx512F.Xor(s4, s9), 12);
                s0 = Avx512F.Add(Avx512F.Add(s0, m[9]), s5); s1 = Avx512F.Add(Avx512F.Add(s1, m[11]), s6); s2 = Avx512F.Add(Avx512F.Add(s2, m[13]), s7); s3 = Avx512F.Add(Avx512F.Add(s3, m[15]), s4);
                s15 = Avx512F.RotateRight(Avx512F.Xor(s15, s0), 8); s12 = Avx512F.RotateRight(Avx512F.Xor(s12, s1), 8); s13 = Avx512F.RotateRight(Avx512F.Xor(s13, s2), 8); s14 = Avx512F.RotateRight(Avx512F.Xor(s14, s3), 8);
                s10 = Avx512F.Add(s10, s15); s11 = Avx512F.Add(s11, s12); s8 = Avx512F.Add(s8, s13); s9 = Avx512F.Add(s9, s14);
                s5 = Avx512F.RotateRight(Avx512F.Xor(s5, s10), 7); s6 = Avx512F.RotateRight(Avx512F.Xor(s6, s11), 7); s7 = Avx512F.RotateRight(Avx512F.Xor(s7, s8), 7); s4 = Avx512F.RotateRight(Avx512F.Xor(s4, s9), 7);
                // Round 1
                s0 = Avx512F.Add(Avx512F.Add(s0, m[2]), s4); s1 = Avx512F.Add(Avx512F.Add(s1, m[3]), s5); s2 = Avx512F.Add(Avx512F.Add(s2, m[7]), s6); s3 = Avx512F.Add(Avx512F.Add(s3, m[4]), s7);
                s12 = Avx512F.RotateRight(Avx512F.Xor(s12, s0), 16); s13 = Avx512F.RotateRight(Avx512F.Xor(s13, s1), 16); s14 = Avx512F.RotateRight(Avx512F.Xor(s14, s2), 16); s15 = Avx512F.RotateRight(Avx512F.Xor(s15, s3), 16);
                s8 = Avx512F.Add(s8, s12); s9 = Avx512F.Add(s9, s13); s10 = Avx512F.Add(s10, s14); s11 = Avx512F.Add(s11, s15);
                s4 = Avx512F.RotateRight(Avx512F.Xor(s4, s8), 12); s5 = Avx512F.RotateRight(Avx512F.Xor(s5, s9), 12); s6 = Avx512F.RotateRight(Avx512F.Xor(s6, s10), 12); s7 = Avx512F.RotateRight(Avx512F.Xor(s7, s11), 12);
                s0 = Avx512F.Add(Avx512F.Add(s0, m[6]), s4); s1 = Avx512F.Add(Avx512F.Add(s1, m[10]), s5); s2 = Avx512F.Add(Avx512F.Add(s2, m[0]), s6); s3 = Avx512F.Add(Avx512F.Add(s3, m[13]), s7);
                s12 = Avx512F.RotateRight(Avx512F.Xor(s12, s0), 8); s13 = Avx512F.RotateRight(Avx512F.Xor(s13, s1), 8); s14 = Avx512F.RotateRight(Avx512F.Xor(s14, s2), 8); s15 = Avx512F.RotateRight(Avx512F.Xor(s15, s3), 8);
                s8 = Avx512F.Add(s8, s12); s9 = Avx512F.Add(s9, s13); s10 = Avx512F.Add(s10, s14); s11 = Avx512F.Add(s11, s15);
                s4 = Avx512F.RotateRight(Avx512F.Xor(s4, s8), 7); s5 = Avx512F.RotateRight(Avx512F.Xor(s5, s9), 7); s6 = Avx512F.RotateRight(Avx512F.Xor(s6, s10), 7); s7 = Avx512F.RotateRight(Avx512F.Xor(s7, s11), 7);
                s0 = Avx512F.Add(Avx512F.Add(s0, m[1]), s5); s1 = Avx512F.Add(Avx512F.Add(s1, m[12]), s6); s2 = Avx512F.Add(Avx512F.Add(s2, m[9]), s7); s3 = Avx512F.Add(Avx512F.Add(s3, m[15]), s4);
                s15 = Avx512F.RotateRight(Avx512F.Xor(s15, s0), 16); s12 = Avx512F.RotateRight(Avx512F.Xor(s12, s1), 16); s13 = Avx512F.RotateRight(Avx512F.Xor(s13, s2), 16); s14 = Avx512F.RotateRight(Avx512F.Xor(s14, s3), 16);
                s10 = Avx512F.Add(s10, s15); s11 = Avx512F.Add(s11, s12); s8 = Avx512F.Add(s8, s13); s9 = Avx512F.Add(s9, s14);
                s5 = Avx512F.RotateRight(Avx512F.Xor(s5, s10), 12); s6 = Avx512F.RotateRight(Avx512F.Xor(s6, s11), 12); s7 = Avx512F.RotateRight(Avx512F.Xor(s7, s8), 12); s4 = Avx512F.RotateRight(Avx512F.Xor(s4, s9), 12);
                s0 = Avx512F.Add(Avx512F.Add(s0, m[11]), s5); s1 = Avx512F.Add(Avx512F.Add(s1, m[5]), s6); s2 = Avx512F.Add(Avx512F.Add(s2, m[14]), s7); s3 = Avx512F.Add(Avx512F.Add(s3, m[8]), s4);
                s15 = Avx512F.RotateRight(Avx512F.Xor(s15, s0), 8); s12 = Avx512F.RotateRight(Avx512F.Xor(s12, s1), 8); s13 = Avx512F.RotateRight(Avx512F.Xor(s13, s2), 8); s14 = Avx512F.RotateRight(Avx512F.Xor(s14, s3), 8);
                s10 = Avx512F.Add(s10, s15); s11 = Avx512F.Add(s11, s12); s8 = Avx512F.Add(s8, s13); s9 = Avx512F.Add(s9, s14);
                s5 = Avx512F.RotateRight(Avx512F.Xor(s5, s10), 7); s6 = Avx512F.RotateRight(Avx512F.Xor(s6, s11), 7); s7 = Avx512F.RotateRight(Avx512F.Xor(s7, s8), 7); s4 = Avx512F.RotateRight(Avx512F.Xor(s4, s9), 7);
                // Round 2
                s0 = Avx512F.Add(Avx512F.Add(s0, m[3]), s4); s1 = Avx512F.Add(Avx512F.Add(s1, m[10]), s5); s2 = Avx512F.Add(Avx512F.Add(s2, m[13]), s6); s3 = Avx512F.Add(Avx512F.Add(s3, m[7]), s7);
                s12 = Avx512F.RotateRight(Avx512F.Xor(s12, s0), 16); s13 = Avx512F.RotateRight(Avx512F.Xor(s13, s1), 16); s14 = Avx512F.RotateRight(Avx512F.Xor(s14, s2), 16); s15 = Avx512F.RotateRight(Avx512F.Xor(s15, s3), 16);
                s8 = Avx512F.Add(s8, s12); s9 = Avx512F.Add(s9, s13); s10 = Avx512F.Add(s10, s14); s11 = Avx512F.Add(s11, s15);
                s4 = Avx512F.RotateRight(Avx512F.Xor(s4, s8), 12); s5 = Avx512F.RotateRight(Avx512F.Xor(s5, s9), 12); s6 = Avx512F.RotateRight(Avx512F.Xor(s6, s10), 12); s7 = Avx512F.RotateRight(Avx512F.Xor(s7, s11), 12);
                s0 = Avx512F.Add(Avx512F.Add(s0, m[4]), s4); s1 = Avx512F.Add(Avx512F.Add(s1, m[12]), s5); s2 = Avx512F.Add(Avx512F.Add(s2, m[2]), s6); s3 = Avx512F.Add(Avx512F.Add(s3, m[14]), s7);
                s12 = Avx512F.RotateRight(Avx512F.Xor(s12, s0), 8); s13 = Avx512F.RotateRight(Avx512F.Xor(s13, s1), 8); s14 = Avx512F.RotateRight(Avx512F.Xor(s14, s2), 8); s15 = Avx512F.RotateRight(Avx512F.Xor(s15, s3), 8);
                s8 = Avx512F.Add(s8, s12); s9 = Avx512F.Add(s9, s13); s10 = Avx512F.Add(s10, s14); s11 = Avx512F.Add(s11, s15);
                s4 = Avx512F.RotateRight(Avx512F.Xor(s4, s8), 7); s5 = Avx512F.RotateRight(Avx512F.Xor(s5, s9), 7); s6 = Avx512F.RotateRight(Avx512F.Xor(s6, s10), 7); s7 = Avx512F.RotateRight(Avx512F.Xor(s7, s11), 7);
                s0 = Avx512F.Add(Avx512F.Add(s0, m[6]), s5); s1 = Avx512F.Add(Avx512F.Add(s1, m[9]), s6); s2 = Avx512F.Add(Avx512F.Add(s2, m[11]), s7); s3 = Avx512F.Add(Avx512F.Add(s3, m[8]), s4);
                s15 = Avx512F.RotateRight(Avx512F.Xor(s15, s0), 16); s12 = Avx512F.RotateRight(Avx512F.Xor(s12, s1), 16); s13 = Avx512F.RotateRight(Avx512F.Xor(s13, s2), 16); s14 = Avx512F.RotateRight(Avx512F.Xor(s14, s3), 16);
                s10 = Avx512F.Add(s10, s15); s11 = Avx512F.Add(s11, s12); s8 = Avx512F.Add(s8, s13); s9 = Avx512F.Add(s9, s14);
                s5 = Avx512F.RotateRight(Avx512F.Xor(s5, s10), 12); s6 = Avx512F.RotateRight(Avx512F.Xor(s6, s11), 12); s7 = Avx512F.RotateRight(Avx512F.Xor(s7, s8), 12); s4 = Avx512F.RotateRight(Avx512F.Xor(s4, s9), 12);
                s0 = Avx512F.Add(Avx512F.Add(s0, m[5]), s5); s1 = Avx512F.Add(Avx512F.Add(s1, m[0]), s6); s2 = Avx512F.Add(Avx512F.Add(s2, m[15]), s7); s3 = Avx512F.Add(Avx512F.Add(s3, m[1]), s4);
                s15 = Avx512F.RotateRight(Avx512F.Xor(s15, s0), 8); s12 = Avx512F.RotateRight(Avx512F.Xor(s12, s1), 8); s13 = Avx512F.RotateRight(Avx512F.Xor(s13, s2), 8); s14 = Avx512F.RotateRight(Avx512F.Xor(s14, s3), 8);
                s10 = Avx512F.Add(s10, s15); s11 = Avx512F.Add(s11, s12); s8 = Avx512F.Add(s8, s13); s9 = Avx512F.Add(s9, s14);
                s5 = Avx512F.RotateRight(Avx512F.Xor(s5, s10), 7); s6 = Avx512F.RotateRight(Avx512F.Xor(s6, s11), 7); s7 = Avx512F.RotateRight(Avx512F.Xor(s7, s8), 7); s4 = Avx512F.RotateRight(Avx512F.Xor(s4, s9), 7);
                // Round 3
                s0 = Avx512F.Add(Avx512F.Add(s0, m[10]), s4); s1 = Avx512F.Add(Avx512F.Add(s1, m[12]), s5); s2 = Avx512F.Add(Avx512F.Add(s2, m[14]), s6); s3 = Avx512F.Add(Avx512F.Add(s3, m[13]), s7);
                s12 = Avx512F.RotateRight(Avx512F.Xor(s12, s0), 16); s13 = Avx512F.RotateRight(Avx512F.Xor(s13, s1), 16); s14 = Avx512F.RotateRight(Avx512F.Xor(s14, s2), 16); s15 = Avx512F.RotateRight(Avx512F.Xor(s15, s3), 16);
                s8 = Avx512F.Add(s8, s12); s9 = Avx512F.Add(s9, s13); s10 = Avx512F.Add(s10, s14); s11 = Avx512F.Add(s11, s15);
                s4 = Avx512F.RotateRight(Avx512F.Xor(s4, s8), 12); s5 = Avx512F.RotateRight(Avx512F.Xor(s5, s9), 12); s6 = Avx512F.RotateRight(Avx512F.Xor(s6, s10), 12); s7 = Avx512F.RotateRight(Avx512F.Xor(s7, s11), 12);
                s0 = Avx512F.Add(Avx512F.Add(s0, m[7]), s4); s1 = Avx512F.Add(Avx512F.Add(s1, m[9]), s5); s2 = Avx512F.Add(Avx512F.Add(s2, m[3]), s6); s3 = Avx512F.Add(Avx512F.Add(s3, m[15]), s7);
                s12 = Avx512F.RotateRight(Avx512F.Xor(s12, s0), 8); s13 = Avx512F.RotateRight(Avx512F.Xor(s13, s1), 8); s14 = Avx512F.RotateRight(Avx512F.Xor(s14, s2), 8); s15 = Avx512F.RotateRight(Avx512F.Xor(s15, s3), 8);
                s8 = Avx512F.Add(s8, s12); s9 = Avx512F.Add(s9, s13); s10 = Avx512F.Add(s10, s14); s11 = Avx512F.Add(s11, s15);
                s4 = Avx512F.RotateRight(Avx512F.Xor(s4, s8), 7); s5 = Avx512F.RotateRight(Avx512F.Xor(s5, s9), 7); s6 = Avx512F.RotateRight(Avx512F.Xor(s6, s10), 7); s7 = Avx512F.RotateRight(Avx512F.Xor(s7, s11), 7);
                s0 = Avx512F.Add(Avx512F.Add(s0, m[4]), s5); s1 = Avx512F.Add(Avx512F.Add(s1, m[11]), s6); s2 = Avx512F.Add(Avx512F.Add(s2, m[5]), s7); s3 = Avx512F.Add(Avx512F.Add(s3, m[1]), s4);
                s15 = Avx512F.RotateRight(Avx512F.Xor(s15, s0), 16); s12 = Avx512F.RotateRight(Avx512F.Xor(s12, s1), 16); s13 = Avx512F.RotateRight(Avx512F.Xor(s13, s2), 16); s14 = Avx512F.RotateRight(Avx512F.Xor(s14, s3), 16);
                s10 = Avx512F.Add(s10, s15); s11 = Avx512F.Add(s11, s12); s8 = Avx512F.Add(s8, s13); s9 = Avx512F.Add(s9, s14);
                s5 = Avx512F.RotateRight(Avx512F.Xor(s5, s10), 12); s6 = Avx512F.RotateRight(Avx512F.Xor(s6, s11), 12); s7 = Avx512F.RotateRight(Avx512F.Xor(s7, s8), 12); s4 = Avx512F.RotateRight(Avx512F.Xor(s4, s9), 12);
                s0 = Avx512F.Add(Avx512F.Add(s0, m[0]), s5); s1 = Avx512F.Add(Avx512F.Add(s1, m[2]), s6); s2 = Avx512F.Add(Avx512F.Add(s2, m[8]), s7); s3 = Avx512F.Add(Avx512F.Add(s3, m[6]), s4);
                s15 = Avx512F.RotateRight(Avx512F.Xor(s15, s0), 8); s12 = Avx512F.RotateRight(Avx512F.Xor(s12, s1), 8); s13 = Avx512F.RotateRight(Avx512F.Xor(s13, s2), 8); s14 = Avx512F.RotateRight(Avx512F.Xor(s14, s3), 8);
                s10 = Avx512F.Add(s10, s15); s11 = Avx512F.Add(s11, s12); s8 = Avx512F.Add(s8, s13); s9 = Avx512F.Add(s9, s14);
                s5 = Avx512F.RotateRight(Avx512F.Xor(s5, s10), 7); s6 = Avx512F.RotateRight(Avx512F.Xor(s6, s11), 7); s7 = Avx512F.RotateRight(Avx512F.Xor(s7, s8), 7); s4 = Avx512F.RotateRight(Avx512F.Xor(s4, s9), 7);
                // Round 4
                s0 = Avx512F.Add(Avx512F.Add(s0, m[12]), s4); s1 = Avx512F.Add(Avx512F.Add(s1, m[9]), s5); s2 = Avx512F.Add(Avx512F.Add(s2, m[15]), s6); s3 = Avx512F.Add(Avx512F.Add(s3, m[14]), s7);
                s12 = Avx512F.RotateRight(Avx512F.Xor(s12, s0), 16); s13 = Avx512F.RotateRight(Avx512F.Xor(s13, s1), 16); s14 = Avx512F.RotateRight(Avx512F.Xor(s14, s2), 16); s15 = Avx512F.RotateRight(Avx512F.Xor(s15, s3), 16);
                s8 = Avx512F.Add(s8, s12); s9 = Avx512F.Add(s9, s13); s10 = Avx512F.Add(s10, s14); s11 = Avx512F.Add(s11, s15);
                s4 = Avx512F.RotateRight(Avx512F.Xor(s4, s8), 12); s5 = Avx512F.RotateRight(Avx512F.Xor(s5, s9), 12); s6 = Avx512F.RotateRight(Avx512F.Xor(s6, s10), 12); s7 = Avx512F.RotateRight(Avx512F.Xor(s7, s11), 12);
                s0 = Avx512F.Add(Avx512F.Add(s0, m[13]), s4); s1 = Avx512F.Add(Avx512F.Add(s1, m[11]), s5); s2 = Avx512F.Add(Avx512F.Add(s2, m[10]), s6); s3 = Avx512F.Add(Avx512F.Add(s3, m[8]), s7);
                s12 = Avx512F.RotateRight(Avx512F.Xor(s12, s0), 8); s13 = Avx512F.RotateRight(Avx512F.Xor(s13, s1), 8); s14 = Avx512F.RotateRight(Avx512F.Xor(s14, s2), 8); s15 = Avx512F.RotateRight(Avx512F.Xor(s15, s3), 8);
                s8 = Avx512F.Add(s8, s12); s9 = Avx512F.Add(s9, s13); s10 = Avx512F.Add(s10, s14); s11 = Avx512F.Add(s11, s15);
                s4 = Avx512F.RotateRight(Avx512F.Xor(s4, s8), 7); s5 = Avx512F.RotateRight(Avx512F.Xor(s5, s9), 7); s6 = Avx512F.RotateRight(Avx512F.Xor(s6, s10), 7); s7 = Avx512F.RotateRight(Avx512F.Xor(s7, s11), 7);
                s0 = Avx512F.Add(Avx512F.Add(s0, m[7]), s5); s1 = Avx512F.Add(Avx512F.Add(s1, m[5]), s6); s2 = Avx512F.Add(Avx512F.Add(s2, m[0]), s7); s3 = Avx512F.Add(Avx512F.Add(s3, m[6]), s4);
                s15 = Avx512F.RotateRight(Avx512F.Xor(s15, s0), 16); s12 = Avx512F.RotateRight(Avx512F.Xor(s12, s1), 16); s13 = Avx512F.RotateRight(Avx512F.Xor(s13, s2), 16); s14 = Avx512F.RotateRight(Avx512F.Xor(s14, s3), 16);
                s10 = Avx512F.Add(s10, s15); s11 = Avx512F.Add(s11, s12); s8 = Avx512F.Add(s8, s13); s9 = Avx512F.Add(s9, s14);
                s5 = Avx512F.RotateRight(Avx512F.Xor(s5, s10), 12); s6 = Avx512F.RotateRight(Avx512F.Xor(s6, s11), 12); s7 = Avx512F.RotateRight(Avx512F.Xor(s7, s8), 12); s4 = Avx512F.RotateRight(Avx512F.Xor(s4, s9), 12);
                s0 = Avx512F.Add(Avx512F.Add(s0, m[2]), s5); s1 = Avx512F.Add(Avx512F.Add(s1, m[3]), s6); s2 = Avx512F.Add(Avx512F.Add(s2, m[1]), s7); s3 = Avx512F.Add(Avx512F.Add(s3, m[4]), s4);
                s15 = Avx512F.RotateRight(Avx512F.Xor(s15, s0), 8); s12 = Avx512F.RotateRight(Avx512F.Xor(s12, s1), 8); s13 = Avx512F.RotateRight(Avx512F.Xor(s13, s2), 8); s14 = Avx512F.RotateRight(Avx512F.Xor(s14, s3), 8);
                s10 = Avx512F.Add(s10, s15); s11 = Avx512F.Add(s11, s12); s8 = Avx512F.Add(s8, s13); s9 = Avx512F.Add(s9, s14);
                s5 = Avx512F.RotateRight(Avx512F.Xor(s5, s10), 7); s6 = Avx512F.RotateRight(Avx512F.Xor(s6, s11), 7); s7 = Avx512F.RotateRight(Avx512F.Xor(s7, s8), 7); s4 = Avx512F.RotateRight(Avx512F.Xor(s4, s9), 7);
                // Round 5
                s0 = Avx512F.Add(Avx512F.Add(s0, m[9]), s4); s1 = Avx512F.Add(Avx512F.Add(s1, m[11]), s5); s2 = Avx512F.Add(Avx512F.Add(s2, m[8]), s6); s3 = Avx512F.Add(Avx512F.Add(s3, m[15]), s7);
                s12 = Avx512F.RotateRight(Avx512F.Xor(s12, s0), 16); s13 = Avx512F.RotateRight(Avx512F.Xor(s13, s1), 16); s14 = Avx512F.RotateRight(Avx512F.Xor(s14, s2), 16); s15 = Avx512F.RotateRight(Avx512F.Xor(s15, s3), 16);
                s8 = Avx512F.Add(s8, s12); s9 = Avx512F.Add(s9, s13); s10 = Avx512F.Add(s10, s14); s11 = Avx512F.Add(s11, s15);
                s4 = Avx512F.RotateRight(Avx512F.Xor(s4, s8), 12); s5 = Avx512F.RotateRight(Avx512F.Xor(s5, s9), 12); s6 = Avx512F.RotateRight(Avx512F.Xor(s6, s10), 12); s7 = Avx512F.RotateRight(Avx512F.Xor(s7, s11), 12);
                s0 = Avx512F.Add(Avx512F.Add(s0, m[14]), s4); s1 = Avx512F.Add(Avx512F.Add(s1, m[5]), s5); s2 = Avx512F.Add(Avx512F.Add(s2, m[12]), s6); s3 = Avx512F.Add(Avx512F.Add(s3, m[1]), s7);
                s12 = Avx512F.RotateRight(Avx512F.Xor(s12, s0), 8); s13 = Avx512F.RotateRight(Avx512F.Xor(s13, s1), 8); s14 = Avx512F.RotateRight(Avx512F.Xor(s14, s2), 8); s15 = Avx512F.RotateRight(Avx512F.Xor(s15, s3), 8);
                s8 = Avx512F.Add(s8, s12); s9 = Avx512F.Add(s9, s13); s10 = Avx512F.Add(s10, s14); s11 = Avx512F.Add(s11, s15);
                s4 = Avx512F.RotateRight(Avx512F.Xor(s4, s8), 7); s5 = Avx512F.RotateRight(Avx512F.Xor(s5, s9), 7); s6 = Avx512F.RotateRight(Avx512F.Xor(s6, s10), 7); s7 = Avx512F.RotateRight(Avx512F.Xor(s7, s11), 7);
                s0 = Avx512F.Add(Avx512F.Add(s0, m[13]), s5); s1 = Avx512F.Add(Avx512F.Add(s1, m[0]), s6); s2 = Avx512F.Add(Avx512F.Add(s2, m[2]), s7); s3 = Avx512F.Add(Avx512F.Add(s3, m[4]), s4);
                s15 = Avx512F.RotateRight(Avx512F.Xor(s15, s0), 16); s12 = Avx512F.RotateRight(Avx512F.Xor(s12, s1), 16); s13 = Avx512F.RotateRight(Avx512F.Xor(s13, s2), 16); s14 = Avx512F.RotateRight(Avx512F.Xor(s14, s3), 16);
                s10 = Avx512F.Add(s10, s15); s11 = Avx512F.Add(s11, s12); s8 = Avx512F.Add(s8, s13); s9 = Avx512F.Add(s9, s14);
                s5 = Avx512F.RotateRight(Avx512F.Xor(s5, s10), 12); s6 = Avx512F.RotateRight(Avx512F.Xor(s6, s11), 12); s7 = Avx512F.RotateRight(Avx512F.Xor(s7, s8), 12); s4 = Avx512F.RotateRight(Avx512F.Xor(s4, s9), 12);
                s0 = Avx512F.Add(Avx512F.Add(s0, m[3]), s5); s1 = Avx512F.Add(Avx512F.Add(s1, m[10]), s6); s2 = Avx512F.Add(Avx512F.Add(s2, m[6]), s7); s3 = Avx512F.Add(Avx512F.Add(s3, m[7]), s4);
                s15 = Avx512F.RotateRight(Avx512F.Xor(s15, s0), 8); s12 = Avx512F.RotateRight(Avx512F.Xor(s12, s1), 8); s13 = Avx512F.RotateRight(Avx512F.Xor(s13, s2), 8); s14 = Avx512F.RotateRight(Avx512F.Xor(s14, s3), 8);
                s10 = Avx512F.Add(s10, s15); s11 = Avx512F.Add(s11, s12); s8 = Avx512F.Add(s8, s13); s9 = Avx512F.Add(s9, s14);
                s5 = Avx512F.RotateRight(Avx512F.Xor(s5, s10), 7); s6 = Avx512F.RotateRight(Avx512F.Xor(s6, s11), 7); s7 = Avx512F.RotateRight(Avx512F.Xor(s7, s8), 7); s4 = Avx512F.RotateRight(Avx512F.Xor(s4, s9), 7);
                // Round 6
                s0 = Avx512F.Add(Avx512F.Add(s0, m[11]), s4); s1 = Avx512F.Add(Avx512F.Add(s1, m[5]), s5); s2 = Avx512F.Add(Avx512F.Add(s2, m[1]), s6); s3 = Avx512F.Add(Avx512F.Add(s3, m[8]), s7);
                s12 = Avx512F.RotateRight(Avx512F.Xor(s12, s0), 16); s13 = Avx512F.RotateRight(Avx512F.Xor(s13, s1), 16); s14 = Avx512F.RotateRight(Avx512F.Xor(s14, s2), 16); s15 = Avx512F.RotateRight(Avx512F.Xor(s15, s3), 16);
                s8 = Avx512F.Add(s8, s12); s9 = Avx512F.Add(s9, s13); s10 = Avx512F.Add(s10, s14); s11 = Avx512F.Add(s11, s15);
                s4 = Avx512F.RotateRight(Avx512F.Xor(s4, s8), 12); s5 = Avx512F.RotateRight(Avx512F.Xor(s5, s9), 12); s6 = Avx512F.RotateRight(Avx512F.Xor(s6, s10), 12); s7 = Avx512F.RotateRight(Avx512F.Xor(s7, s11), 12);
                s0 = Avx512F.Add(Avx512F.Add(s0, m[15]), s4); s1 = Avx512F.Add(Avx512F.Add(s1, m[0]), s5); s2 = Avx512F.Add(Avx512F.Add(s2, m[9]), s6); s3 = Avx512F.Add(Avx512F.Add(s3, m[6]), s7);
                s12 = Avx512F.RotateRight(Avx512F.Xor(s12, s0), 8); s13 = Avx512F.RotateRight(Avx512F.Xor(s13, s1), 8); s14 = Avx512F.RotateRight(Avx512F.Xor(s14, s2), 8); s15 = Avx512F.RotateRight(Avx512F.Xor(s15, s3), 8);
                s8 = Avx512F.Add(s8, s12); s9 = Avx512F.Add(s9, s13); s10 = Avx512F.Add(s10, s14); s11 = Avx512F.Add(s11, s15);
                s4 = Avx512F.RotateRight(Avx512F.Xor(s4, s8), 7); s5 = Avx512F.RotateRight(Avx512F.Xor(s5, s9), 7); s6 = Avx512F.RotateRight(Avx512F.Xor(s6, s10), 7); s7 = Avx512F.RotateRight(Avx512F.Xor(s7, s11), 7);
                s0 = Avx512F.Add(Avx512F.Add(s0, m[14]), s5); s1 = Avx512F.Add(Avx512F.Add(s1, m[2]), s6); s2 = Avx512F.Add(Avx512F.Add(s2, m[3]), s7); s3 = Avx512F.Add(Avx512F.Add(s3, m[7]), s4);
                s15 = Avx512F.RotateRight(Avx512F.Xor(s15, s0), 16); s12 = Avx512F.RotateRight(Avx512F.Xor(s12, s1), 16); s13 = Avx512F.RotateRight(Avx512F.Xor(s13, s2), 16); s14 = Avx512F.RotateRight(Avx512F.Xor(s14, s3), 16);
                s10 = Avx512F.Add(s10, s15); s11 = Avx512F.Add(s11, s12); s8 = Avx512F.Add(s8, s13); s9 = Avx512F.Add(s9, s14);
                s5 = Avx512F.RotateRight(Avx512F.Xor(s5, s10), 12); s6 = Avx512F.RotateRight(Avx512F.Xor(s6, s11), 12); s7 = Avx512F.RotateRight(Avx512F.Xor(s7, s8), 12); s4 = Avx512F.RotateRight(Avx512F.Xor(s4, s9), 12);
                s0 = Avx512F.Add(Avx512F.Add(s0, m[10]), s5); s1 = Avx512F.Add(Avx512F.Add(s1, m[12]), s6); s2 = Avx512F.Add(Avx512F.Add(s2, m[4]), s7); s3 = Avx512F.Add(Avx512F.Add(s3, m[13]), s4);
                s15 = Avx512F.RotateRight(Avx512F.Xor(s15, s0), 8); s12 = Avx512F.RotateRight(Avx512F.Xor(s12, s1), 8); s13 = Avx512F.RotateRight(Avx512F.Xor(s13, s2), 8); s14 = Avx512F.RotateRight(Avx512F.Xor(s14, s3), 8);
                s10 = Avx512F.Add(s10, s15); s11 = Avx512F.Add(s11, s12); s8 = Avx512F.Add(s8, s13); s9 = Avx512F.Add(s9, s14);
                s5 = Avx512F.RotateRight(Avx512F.Xor(s5, s10), 7); s6 = Avx512F.RotateRight(Avx512F.Xor(s6, s11), 7); s7 = Avx512F.RotateRight(Avx512F.Xor(s7, s8), 7); s4 = Avx512F.RotateRight(Avx512F.Xor(s4, s9), 7);

                words[0] = Avx512F.Xor(s0, s8); words[1] = Avx512F.Xor(s1, s9);
                words[2] = Avx512F.Xor(s2, s10); words[3] = Avx512F.Xor(s3, s11);
                words[4] = Avx512F.Xor(s4, s12); words[5] = Avx512F.Xor(s5, s13);
                words[6] = Avx512F.Xor(s6, s14); words[7] = Avx512F.Xor(s7, s15);
                words[8] = Avx512F.Xor(s8, cv[0]); words[9] = Avx512F.Xor(s9, cv[1]);
                words[10] = Avx512F.Xor(s10, cv[2]); words[11] = Avx512F.Xor(s11, cv[3]);
                words[12] = Avx512F.Xor(s12, cv[4]); words[13] = Avx512F.Xor(s13, cv[5]);
                words[14] = Avx512F.Xor(s14, cv[6]); words[15] = Avx512F.Xor(s15, cv[7]);

                // words[i] holds word i of all sixteen blocks; block j is output bytes 64j..64j+63.
                Transpose16((byte*)words, 64, (Vector512<uint>*)(outPtr + pos));
            }
        }
    }

    /// <summary>
    /// Transposes sixteen 64-byte rows <paramref name="stride"/> bytes apart so that
    /// <c>m[i]</c> holds word <c>i</c> of every row: one block of each of sixteen chunks going
    /// in (stride 1 KiB), and back to block-major order for XOF output (stride 64).
    /// </summary>
    /// <remarks>
    /// <c>NoInlining | AggressiveOptimization</c> on purpose: the kernel is too large for this to
    /// be inlined anyway, and as an ordinary method it can run as tier-0 code for the life of the
    /// process, which made the kernel 1.6-2x slower in some processes. See the class remarks.
    /// </remarks>
    [MethodImpl(MethodImplOptions.NoInlining | MethodImplOptions.AggressiveOptimization)]
    private static unsafe void Transpose16(byte* bb, int stride, Vector512<uint>* m)
    {
        var r0 = Unsafe.ReadUnaligned<Vector512<uint>>(bb + 0 * stride);
        var r1 = Unsafe.ReadUnaligned<Vector512<uint>>(bb + 1 * stride);
        var r2 = Unsafe.ReadUnaligned<Vector512<uint>>(bb + 2 * stride);
        var r3 = Unsafe.ReadUnaligned<Vector512<uint>>(bb + 3 * stride);
        var r4 = Unsafe.ReadUnaligned<Vector512<uint>>(bb + 4 * stride);
        var r5 = Unsafe.ReadUnaligned<Vector512<uint>>(bb + 5 * stride);
        var r6 = Unsafe.ReadUnaligned<Vector512<uint>>(bb + 6 * stride);
        var r7 = Unsafe.ReadUnaligned<Vector512<uint>>(bb + 7 * stride);
        var r8 = Unsafe.ReadUnaligned<Vector512<uint>>(bb + 8 * stride);
        var r9 = Unsafe.ReadUnaligned<Vector512<uint>>(bb + 9 * stride);
        var r10 = Unsafe.ReadUnaligned<Vector512<uint>>(bb + 10 * stride);
        var r11 = Unsafe.ReadUnaligned<Vector512<uint>>(bb + 11 * stride);
        var r12 = Unsafe.ReadUnaligned<Vector512<uint>>(bb + 12 * stride);
        var r13 = Unsafe.ReadUnaligned<Vector512<uint>>(bb + 13 * stride);
        var r14 = Unsafe.ReadUnaligned<Vector512<uint>>(bb + 14 * stride);
        var r15 = Unsafe.ReadUnaligned<Vector512<uint>>(bb + 15 * stride);

        // 32-bit interleave within 128-bit lanes.
        var a0 = Avx512F.UnpackLow(r0, r1); var a1 = Avx512F.UnpackHigh(r0, r1);
        var a2 = Avx512F.UnpackLow(r2, r3); var a3 = Avx512F.UnpackHigh(r2, r3);
        var a4 = Avx512F.UnpackLow(r4, r5); var a5 = Avx512F.UnpackHigh(r4, r5);
        var a6 = Avx512F.UnpackLow(r6, r7); var a7 = Avx512F.UnpackHigh(r6, r7);
        var a8 = Avx512F.UnpackLow(r8, r9); var a9 = Avx512F.UnpackHigh(r8, r9);
        var a10 = Avx512F.UnpackLow(r10, r11); var a11 = Avx512F.UnpackHigh(r10, r11);
        var a12 = Avx512F.UnpackLow(r12, r13); var a13 = Avx512F.UnpackHigh(r12, r13);
        var a14 = Avx512F.UnpackLow(r14, r15); var a15 = Avx512F.UnpackHigh(r14, r15);

        // 64-bit interleave: b[4g + k], 128-bit lane q = word 4q + k of chunks 4g..4g+3.
        var b0 = Avx512F.UnpackLow(a0.AsUInt64(), a2.AsUInt64()).AsUInt32();
        var b1 = Avx512F.UnpackHigh(a0.AsUInt64(), a2.AsUInt64()).AsUInt32();
        var b2 = Avx512F.UnpackLow(a1.AsUInt64(), a3.AsUInt64()).AsUInt32();
        var b3 = Avx512F.UnpackHigh(a1.AsUInt64(), a3.AsUInt64()).AsUInt32();
        var b4 = Avx512F.UnpackLow(a4.AsUInt64(), a6.AsUInt64()).AsUInt32();
        var b5 = Avx512F.UnpackHigh(a4.AsUInt64(), a6.AsUInt64()).AsUInt32();
        var b6 = Avx512F.UnpackLow(a5.AsUInt64(), a7.AsUInt64()).AsUInt32();
        var b7 = Avx512F.UnpackHigh(a5.AsUInt64(), a7.AsUInt64()).AsUInt32();
        var b8 = Avx512F.UnpackLow(a8.AsUInt64(), a10.AsUInt64()).AsUInt32();
        var b9 = Avx512F.UnpackHigh(a8.AsUInt64(), a10.AsUInt64()).AsUInt32();
        var b10 = Avx512F.UnpackLow(a9.AsUInt64(), a11.AsUInt64()).AsUInt32();
        var b11 = Avx512F.UnpackHigh(a9.AsUInt64(), a11.AsUInt64()).AsUInt32();
        var b12 = Avx512F.UnpackLow(a12.AsUInt64(), a14.AsUInt64()).AsUInt32();
        var b13 = Avx512F.UnpackHigh(a12.AsUInt64(), a14.AsUInt64()).AsUInt32();
        var b14 = Avx512F.UnpackLow(a13.AsUInt64(), a15.AsUInt64()).AsUInt32();
        var b15 = Avx512F.UnpackHigh(a13.AsUInt64(), a15.AsUInt64()).AsUInt32();

        Lanes(b0, b4, b8, b12, m, 0);
        Lanes(b1, b5, b9, b13, m, 1);
        Lanes(b2, b6, b10, b14, m, 2);
        Lanes(b3, b7, b11, b15, m, 3);
    }

    // 4x4 transpose of 128-bit lanes: m[4q + k] = [B0 lane q, B1 lane q, B2 lane q, B3 lane q].
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void Lanes(Vector512<uint> b0, Vector512<uint> b1, Vector512<uint> b2,
        Vector512<uint> b3, Vector512<uint>* m, int k)
    {
        var c0 = Avx512F.Shuffle4x128(b0, b1, 0x44);
        var c1 = Avx512F.Shuffle4x128(b0, b1, 0xEE);
        var c2 = Avx512F.Shuffle4x128(b2, b3, 0x44);
        var c3 = Avx512F.Shuffle4x128(b2, b3, 0xEE);
        m[k] = Avx512F.Shuffle4x128(c0, c2, 0x88);
        m[4 + k] = Avx512F.Shuffle4x128(c0, c2, 0xDD);
        m[8 + k] = Avx512F.Shuffle4x128(c1, c3, 0x88);
        m[12 + k] = Avx512F.Shuffle4x128(c1, c3, 0xDD);
    }
}
#endif
