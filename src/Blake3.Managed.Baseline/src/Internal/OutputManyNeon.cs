using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;

namespace Blake3.Managed.Internal;

/// <summary>
/// Four root output blocks per call for NEON: the XOF counterpart of <see cref="HashManyNeon"/>
/// for ARM64 cores without SVE2 (which have <c>OutputManySve2</c>). Every lane shares the
/// root CV and message and differs only in the output counter, so the four blocks are
/// independent. The round body is the one the NEON hashing kernels use -- generated, stepped
/// across the four G's of each half-round, (a + m) + b, and a store to <c>m[16]</c> after every
/// round -- with the message words broadcast once into <c>m</c>.
/// </summary>
/// <remarks>
/// Before this kernel NEON produced XOF output one scalar compression per 64-byte block.
/// </remarks>
internal static class OutputManyNeon
{
    public static bool IsSupported => AdvSimd.Arm64.IsSupported;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Transpose4X4(
        Vector128<uint> r0, Vector128<uint> r1, Vector128<uint> r2, Vector128<uint> r3,
        out Vector128<uint> m0, out Vector128<uint> m1, out Vector128<uint> m2, out Vector128<uint> m3)
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

    // Words w0..w3 of four lanes -> 16 bytes of each of the four blocks, at word offset `word`.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void StoreQuad(Vector128<uint> w0, Vector128<uint> w1, Vector128<uint> w2,
        Vector128<uint> w3, ref uint blocks, int word)
    {
        Transpose4X4(w0, w1, w2, w3, out var b0, out var b1, out var b2, out var b3);
        VectorCompat.Store(b0, ref blocks, (nuint)word);
        VectorCompat.Store(b1, ref blocks, (nuint)(16 + word));
        VectorCompat.Store(b2, ref blocks, (nuint)(32 + word));
        VectorCompat.Store(b3, ref blocks, (nuint)(48 + word));
    }

    /// <summary>
    /// Writes <paramref name="output"/>, a positive multiple of 256 bytes, as root output blocks
    /// <paramref name="counter"/>, <paramref name="counter"/> + 1, ...
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.NoInlining | MethodImplOptions.AggressiveOptimization)]
    internal static unsafe void HashOutput4(ReadOnlySpan<uint> inputCv, ReadOnlySpan<uint> block,
        ulong counter, uint blockLen, uint flags, Span<byte> output)
    {
        var rot8Mask = Vector128.Create((byte)1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12);
        Vector128<uint>* m = stackalloc Vector128<uint>[17]; // [16]: the per-round barrier slot
        for (int i = 0; i < 16; i++) m[i] = Vector128.Create(block[i]);
        var cv0 = Vector128.Create(inputCv[0]); var cv1 = Vector128.Create(inputCv[1]);
        var cv2 = Vector128.Create(inputCv[2]); var cv3 = Vector128.Create(inputCv[3]);
        var cv4 = Vector128.Create(inputCv[4]); var cv5 = Vector128.Create(inputCv[5]);
        var cv6 = Vector128.Create(inputCv[6]); var cv7 = Vector128.Create(inputCv[7]);
        var blockLenVec = Vector128.Create(blockLen);
        var flagsVec = Vector128.Create(flags | Blake3Constants.Root);

        ref uint outWords = ref MemoryMarshal.GetReference(MemoryMarshal.Cast<byte, uint>(output));
        for (int pos = 0; pos < output.Length; pos += 256, counter += 4)
        {
            Vector128<uint> s0 = cv0, s1 = cv1, s2 = cv2, s3 = cv3, s4 = cv4, s5 = cv5, s6 = cv6, s7 = cv7;
            Vector128<uint> s8 = Vector128.Create(Blake3Constants.Iv0), s9 = Vector128.Create(Blake3Constants.Iv1);
            Vector128<uint> s10 = Vector128.Create(Blake3Constants.Iv2), s11 = Vector128.Create(Blake3Constants.Iv3);
            Vector128<uint> s12 = Vector128.Create((uint)counter, (uint)(counter + 1), (uint)(counter + 2), (uint)(counter + 3));
            Vector128<uint> s13 = Vector128.Create((uint)(counter >> 32), (uint)((counter + 1) >> 32),
                (uint)((counter + 2) >> 32), (uint)((counter + 3) >> 32));
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

            ref uint blocks = ref Unsafe.Add(ref outWords, pos / 4);
            StoreQuad(AdvSimd.Xor(s0, s8), AdvSimd.Xor(s1, s9), AdvSimd.Xor(s2, s10), AdvSimd.Xor(s3, s11), ref blocks, 0);
            StoreQuad(AdvSimd.Xor(s4, s12), AdvSimd.Xor(s5, s13), AdvSimd.Xor(s6, s14), AdvSimd.Xor(s7, s15), ref blocks, 4);
            StoreQuad(AdvSimd.Xor(s8, cv0), AdvSimd.Xor(s9, cv1), AdvSimd.Xor(s10, cv2), AdvSimd.Xor(s11, cv3), ref blocks, 8);
            StoreQuad(AdvSimd.Xor(s12, cv4), AdvSimd.Xor(s13, cv5), AdvSimd.Xor(s14, cv6), AdvSimd.Xor(s15, cv7), ref blocks, 12);
        }
    }
}
