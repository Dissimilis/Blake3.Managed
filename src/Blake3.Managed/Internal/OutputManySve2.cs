#if NET10_0_OR_GREATER
// Sve2 is [Experimental] in .NET 10 (SYSLIB5003); see HashManySve2.
#pragma warning disable SYSLIB5003
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;

namespace Blake3.Managed.Internal;

/// <summary>
/// Eight root output blocks per call for SVE2: the ARM counterpart of <see cref="OutputManyAvx2"/>.
/// Every lane shares the root CV and message and differs only in the output counter, so the
/// blocks are independent. Two 4-lane batches are interleaved per half-round, as in
/// <see cref="HashManySve2.HashMany8"/>, and written out statement by statement for the same
/// reason (a G helper exhausts the JIT's inlining budget). The message words are broadcast once
/// into a scratch array and read from it at each use.
/// </summary>
internal static class OutputManySve2
{
    public static bool IsSupported => HashManySve2.IsSupported;

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
    /// Writes <paramref name="output"/>, a positive multiple of 512 bytes, as root output blocks
    /// <paramref name="counter"/>, <paramref name="counter"/> + 1, ...
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.NoInlining | MethodImplOptions.AggressiveOptimization)]
    internal static unsafe void HashOutput8(ReadOnlySpan<uint> inputCv, ReadOnlySpan<uint> block,
        ulong counter, uint blockLen, uint flags, Span<byte> output)
    {
        Vector128<uint>* mb = stackalloc Vector128<uint>[16];
        for (int i = 0; i < 16; i++) mb[i] = Vector128.Create(block[i]);
        var cv0 = Vector128.Create(inputCv[0]); var cv1 = Vector128.Create(inputCv[1]);
        var cv2 = Vector128.Create(inputCv[2]); var cv3 = Vector128.Create(inputCv[3]);
        var cv4 = Vector128.Create(inputCv[4]); var cv5 = Vector128.Create(inputCv[5]);
        var cv6 = Vector128.Create(inputCv[6]); var cv7 = Vector128.Create(inputCv[7]);
        var blockLenVec = Vector128.Create(blockLen);
        var flagsVec = Vector128.Create(flags | Blake3Constants.Root);

        ref uint outWords = ref MemoryMarshal.GetReference(MemoryMarshal.Cast<byte, uint>(output));
        for (int pos = 0; pos < output.Length; pos += 512, counter += 8)
        {
            Vector128<uint> a0 = cv0, a1 = cv1, a2 = cv2, a3 = cv3, a4 = cv4, a5 = cv5, a6 = cv6, a7 = cv7;
            Vector128<uint> a8 = Vector128.Create(Blake3Constants.Iv0), a9 = Vector128.Create(Blake3Constants.Iv1);
            Vector128<uint> a10 = Vector128.Create(Blake3Constants.Iv2), a11 = Vector128.Create(Blake3Constants.Iv3);
            Vector128<uint> a12 = Vector128.Create((uint)counter, (uint)(counter + 1), (uint)(counter + 2), (uint)(counter + 3));
            Vector128<uint> a13 = Vector128.Create((uint)(counter >> 32), (uint)((counter + 1) >> 32),
                (uint)((counter + 2) >> 32), (uint)((counter + 3) >> 32));
            Vector128<uint> a14 = blockLenVec, a15 = flagsVec;
            Vector128<uint> b0 = cv0, b1 = cv1, b2 = cv2, b3 = cv3, b4 = cv4, b5 = cv5, b6 = cv6, b7 = cv7;
            Vector128<uint> b8 = a8, b9 = a9, b10 = a10, b11 = a11;
            Vector128<uint> b12 = Vector128.Create((uint)(counter + 4), (uint)(counter + 5), (uint)(counter + 6), (uint)(counter + 7));
            Vector128<uint> b13 = Vector128.Create((uint)((counter + 4) >> 32), (uint)((counter + 5) >> 32),
                (uint)((counter + 6) >> 32), (uint)((counter + 7) >> 32));
            Vector128<uint> b14 = blockLenVec, b15 = flagsVec;

                // Round 0
                a0 = AdvSimd.Add(AdvSimd.Add(a0, mb[0]), a4); a12 = Sve2.XorRotateRight(a12.AsVector(), a0.AsVector(), 16).AsVector128(); a8 = AdvSimd.Add(a8, a12); a4 = Sve2.XorRotateRight(a4.AsVector(), a8.AsVector(), 12).AsVector128();
                a0 = AdvSimd.Add(AdvSimd.Add(a0, mb[1]), a4); a12 = Sve2.XorRotateRight(a12.AsVector(), a0.AsVector(), 8).AsVector128(); a8 = AdvSimd.Add(a8, a12); a4 = Sve2.XorRotateRight(a4.AsVector(), a8.AsVector(), 7).AsVector128();
                a1 = AdvSimd.Add(AdvSimd.Add(a1, mb[2]), a5); a13 = Sve2.XorRotateRight(a13.AsVector(), a1.AsVector(), 16).AsVector128(); a9 = AdvSimd.Add(a9, a13); a5 = Sve2.XorRotateRight(a5.AsVector(), a9.AsVector(), 12).AsVector128();
                a1 = AdvSimd.Add(AdvSimd.Add(a1, mb[3]), a5); a13 = Sve2.XorRotateRight(a13.AsVector(), a1.AsVector(), 8).AsVector128(); a9 = AdvSimd.Add(a9, a13); a5 = Sve2.XorRotateRight(a5.AsVector(), a9.AsVector(), 7).AsVector128();
                a2 = AdvSimd.Add(AdvSimd.Add(a2, mb[4]), a6); a14 = Sve2.XorRotateRight(a14.AsVector(), a2.AsVector(), 16).AsVector128(); a10 = AdvSimd.Add(a10, a14); a6 = Sve2.XorRotateRight(a6.AsVector(), a10.AsVector(), 12).AsVector128();
                a2 = AdvSimd.Add(AdvSimd.Add(a2, mb[5]), a6); a14 = Sve2.XorRotateRight(a14.AsVector(), a2.AsVector(), 8).AsVector128(); a10 = AdvSimd.Add(a10, a14); a6 = Sve2.XorRotateRight(a6.AsVector(), a10.AsVector(), 7).AsVector128();
                a3 = AdvSimd.Add(AdvSimd.Add(a3, mb[6]), a7); a15 = Sve2.XorRotateRight(a15.AsVector(), a3.AsVector(), 16).AsVector128(); a11 = AdvSimd.Add(a11, a15); a7 = Sve2.XorRotateRight(a7.AsVector(), a11.AsVector(), 12).AsVector128();
                a3 = AdvSimd.Add(AdvSimd.Add(a3, mb[7]), a7); a15 = Sve2.XorRotateRight(a15.AsVector(), a3.AsVector(), 8).AsVector128(); a11 = AdvSimd.Add(a11, a15); a7 = Sve2.XorRotateRight(a7.AsVector(), a11.AsVector(), 7).AsVector128();
                b0 = AdvSimd.Add(AdvSimd.Add(b0, mb[0]), b4); b12 = Sve2.XorRotateRight(b12.AsVector(), b0.AsVector(), 16).AsVector128(); b8 = AdvSimd.Add(b8, b12); b4 = Sve2.XorRotateRight(b4.AsVector(), b8.AsVector(), 12).AsVector128();
                b0 = AdvSimd.Add(AdvSimd.Add(b0, mb[1]), b4); b12 = Sve2.XorRotateRight(b12.AsVector(), b0.AsVector(), 8).AsVector128(); b8 = AdvSimd.Add(b8, b12); b4 = Sve2.XorRotateRight(b4.AsVector(), b8.AsVector(), 7).AsVector128();
                b1 = AdvSimd.Add(AdvSimd.Add(b1, mb[2]), b5); b13 = Sve2.XorRotateRight(b13.AsVector(), b1.AsVector(), 16).AsVector128(); b9 = AdvSimd.Add(b9, b13); b5 = Sve2.XorRotateRight(b5.AsVector(), b9.AsVector(), 12).AsVector128();
                b1 = AdvSimd.Add(AdvSimd.Add(b1, mb[3]), b5); b13 = Sve2.XorRotateRight(b13.AsVector(), b1.AsVector(), 8).AsVector128(); b9 = AdvSimd.Add(b9, b13); b5 = Sve2.XorRotateRight(b5.AsVector(), b9.AsVector(), 7).AsVector128();
                b2 = AdvSimd.Add(AdvSimd.Add(b2, mb[4]), b6); b14 = Sve2.XorRotateRight(b14.AsVector(), b2.AsVector(), 16).AsVector128(); b10 = AdvSimd.Add(b10, b14); b6 = Sve2.XorRotateRight(b6.AsVector(), b10.AsVector(), 12).AsVector128();
                b2 = AdvSimd.Add(AdvSimd.Add(b2, mb[5]), b6); b14 = Sve2.XorRotateRight(b14.AsVector(), b2.AsVector(), 8).AsVector128(); b10 = AdvSimd.Add(b10, b14); b6 = Sve2.XorRotateRight(b6.AsVector(), b10.AsVector(), 7).AsVector128();
                b3 = AdvSimd.Add(AdvSimd.Add(b3, mb[6]), b7); b15 = Sve2.XorRotateRight(b15.AsVector(), b3.AsVector(), 16).AsVector128(); b11 = AdvSimd.Add(b11, b15); b7 = Sve2.XorRotateRight(b7.AsVector(), b11.AsVector(), 12).AsVector128();
                b3 = AdvSimd.Add(AdvSimd.Add(b3, mb[7]), b7); b15 = Sve2.XorRotateRight(b15.AsVector(), b3.AsVector(), 8).AsVector128(); b11 = AdvSimd.Add(b11, b15); b7 = Sve2.XorRotateRight(b7.AsVector(), b11.AsVector(), 7).AsVector128();
                a0 = AdvSimd.Add(AdvSimd.Add(a0, mb[8]), a5); a15 = Sve2.XorRotateRight(a15.AsVector(), a0.AsVector(), 16).AsVector128(); a10 = AdvSimd.Add(a10, a15); a5 = Sve2.XorRotateRight(a5.AsVector(), a10.AsVector(), 12).AsVector128();
                a0 = AdvSimd.Add(AdvSimd.Add(a0, mb[9]), a5); a15 = Sve2.XorRotateRight(a15.AsVector(), a0.AsVector(), 8).AsVector128(); a10 = AdvSimd.Add(a10, a15); a5 = Sve2.XorRotateRight(a5.AsVector(), a10.AsVector(), 7).AsVector128();
                a1 = AdvSimd.Add(AdvSimd.Add(a1, mb[10]), a6); a12 = Sve2.XorRotateRight(a12.AsVector(), a1.AsVector(), 16).AsVector128(); a11 = AdvSimd.Add(a11, a12); a6 = Sve2.XorRotateRight(a6.AsVector(), a11.AsVector(), 12).AsVector128();
                a1 = AdvSimd.Add(AdvSimd.Add(a1, mb[11]), a6); a12 = Sve2.XorRotateRight(a12.AsVector(), a1.AsVector(), 8).AsVector128(); a11 = AdvSimd.Add(a11, a12); a6 = Sve2.XorRotateRight(a6.AsVector(), a11.AsVector(), 7).AsVector128();
                a2 = AdvSimd.Add(AdvSimd.Add(a2, mb[12]), a7); a13 = Sve2.XorRotateRight(a13.AsVector(), a2.AsVector(), 16).AsVector128(); a8 = AdvSimd.Add(a8, a13); a7 = Sve2.XorRotateRight(a7.AsVector(), a8.AsVector(), 12).AsVector128();
                a2 = AdvSimd.Add(AdvSimd.Add(a2, mb[13]), a7); a13 = Sve2.XorRotateRight(a13.AsVector(), a2.AsVector(), 8).AsVector128(); a8 = AdvSimd.Add(a8, a13); a7 = Sve2.XorRotateRight(a7.AsVector(), a8.AsVector(), 7).AsVector128();
                a3 = AdvSimd.Add(AdvSimd.Add(a3, mb[14]), a4); a14 = Sve2.XorRotateRight(a14.AsVector(), a3.AsVector(), 16).AsVector128(); a9 = AdvSimd.Add(a9, a14); a4 = Sve2.XorRotateRight(a4.AsVector(), a9.AsVector(), 12).AsVector128();
                a3 = AdvSimd.Add(AdvSimd.Add(a3, mb[15]), a4); a14 = Sve2.XorRotateRight(a14.AsVector(), a3.AsVector(), 8).AsVector128(); a9 = AdvSimd.Add(a9, a14); a4 = Sve2.XorRotateRight(a4.AsVector(), a9.AsVector(), 7).AsVector128();
                b0 = AdvSimd.Add(AdvSimd.Add(b0, mb[8]), b5); b15 = Sve2.XorRotateRight(b15.AsVector(), b0.AsVector(), 16).AsVector128(); b10 = AdvSimd.Add(b10, b15); b5 = Sve2.XorRotateRight(b5.AsVector(), b10.AsVector(), 12).AsVector128();
                b0 = AdvSimd.Add(AdvSimd.Add(b0, mb[9]), b5); b15 = Sve2.XorRotateRight(b15.AsVector(), b0.AsVector(), 8).AsVector128(); b10 = AdvSimd.Add(b10, b15); b5 = Sve2.XorRotateRight(b5.AsVector(), b10.AsVector(), 7).AsVector128();
                b1 = AdvSimd.Add(AdvSimd.Add(b1, mb[10]), b6); b12 = Sve2.XorRotateRight(b12.AsVector(), b1.AsVector(), 16).AsVector128(); b11 = AdvSimd.Add(b11, b12); b6 = Sve2.XorRotateRight(b6.AsVector(), b11.AsVector(), 12).AsVector128();
                b1 = AdvSimd.Add(AdvSimd.Add(b1, mb[11]), b6); b12 = Sve2.XorRotateRight(b12.AsVector(), b1.AsVector(), 8).AsVector128(); b11 = AdvSimd.Add(b11, b12); b6 = Sve2.XorRotateRight(b6.AsVector(), b11.AsVector(), 7).AsVector128();
                b2 = AdvSimd.Add(AdvSimd.Add(b2, mb[12]), b7); b13 = Sve2.XorRotateRight(b13.AsVector(), b2.AsVector(), 16).AsVector128(); b8 = AdvSimd.Add(b8, b13); b7 = Sve2.XorRotateRight(b7.AsVector(), b8.AsVector(), 12).AsVector128();
                b2 = AdvSimd.Add(AdvSimd.Add(b2, mb[13]), b7); b13 = Sve2.XorRotateRight(b13.AsVector(), b2.AsVector(), 8).AsVector128(); b8 = AdvSimd.Add(b8, b13); b7 = Sve2.XorRotateRight(b7.AsVector(), b8.AsVector(), 7).AsVector128();
                b3 = AdvSimd.Add(AdvSimd.Add(b3, mb[14]), b4); b14 = Sve2.XorRotateRight(b14.AsVector(), b3.AsVector(), 16).AsVector128(); b9 = AdvSimd.Add(b9, b14); b4 = Sve2.XorRotateRight(b4.AsVector(), b9.AsVector(), 12).AsVector128();
                b3 = AdvSimd.Add(AdvSimd.Add(b3, mb[15]), b4); b14 = Sve2.XorRotateRight(b14.AsVector(), b3.AsVector(), 8).AsVector128(); b9 = AdvSimd.Add(b9, b14); b4 = Sve2.XorRotateRight(b4.AsVector(), b9.AsVector(), 7).AsVector128();
                // Round 1
                a0 = AdvSimd.Add(AdvSimd.Add(a0, mb[2]), a4); a12 = Sve2.XorRotateRight(a12.AsVector(), a0.AsVector(), 16).AsVector128(); a8 = AdvSimd.Add(a8, a12); a4 = Sve2.XorRotateRight(a4.AsVector(), a8.AsVector(), 12).AsVector128();
                a0 = AdvSimd.Add(AdvSimd.Add(a0, mb[6]), a4); a12 = Sve2.XorRotateRight(a12.AsVector(), a0.AsVector(), 8).AsVector128(); a8 = AdvSimd.Add(a8, a12); a4 = Sve2.XorRotateRight(a4.AsVector(), a8.AsVector(), 7).AsVector128();
                a1 = AdvSimd.Add(AdvSimd.Add(a1, mb[3]), a5); a13 = Sve2.XorRotateRight(a13.AsVector(), a1.AsVector(), 16).AsVector128(); a9 = AdvSimd.Add(a9, a13); a5 = Sve2.XorRotateRight(a5.AsVector(), a9.AsVector(), 12).AsVector128();
                a1 = AdvSimd.Add(AdvSimd.Add(a1, mb[10]), a5); a13 = Sve2.XorRotateRight(a13.AsVector(), a1.AsVector(), 8).AsVector128(); a9 = AdvSimd.Add(a9, a13); a5 = Sve2.XorRotateRight(a5.AsVector(), a9.AsVector(), 7).AsVector128();
                a2 = AdvSimd.Add(AdvSimd.Add(a2, mb[7]), a6); a14 = Sve2.XorRotateRight(a14.AsVector(), a2.AsVector(), 16).AsVector128(); a10 = AdvSimd.Add(a10, a14); a6 = Sve2.XorRotateRight(a6.AsVector(), a10.AsVector(), 12).AsVector128();
                a2 = AdvSimd.Add(AdvSimd.Add(a2, mb[0]), a6); a14 = Sve2.XorRotateRight(a14.AsVector(), a2.AsVector(), 8).AsVector128(); a10 = AdvSimd.Add(a10, a14); a6 = Sve2.XorRotateRight(a6.AsVector(), a10.AsVector(), 7).AsVector128();
                a3 = AdvSimd.Add(AdvSimd.Add(a3, mb[4]), a7); a15 = Sve2.XorRotateRight(a15.AsVector(), a3.AsVector(), 16).AsVector128(); a11 = AdvSimd.Add(a11, a15); a7 = Sve2.XorRotateRight(a7.AsVector(), a11.AsVector(), 12).AsVector128();
                a3 = AdvSimd.Add(AdvSimd.Add(a3, mb[13]), a7); a15 = Sve2.XorRotateRight(a15.AsVector(), a3.AsVector(), 8).AsVector128(); a11 = AdvSimd.Add(a11, a15); a7 = Sve2.XorRotateRight(a7.AsVector(), a11.AsVector(), 7).AsVector128();
                b0 = AdvSimd.Add(AdvSimd.Add(b0, mb[2]), b4); b12 = Sve2.XorRotateRight(b12.AsVector(), b0.AsVector(), 16).AsVector128(); b8 = AdvSimd.Add(b8, b12); b4 = Sve2.XorRotateRight(b4.AsVector(), b8.AsVector(), 12).AsVector128();
                b0 = AdvSimd.Add(AdvSimd.Add(b0, mb[6]), b4); b12 = Sve2.XorRotateRight(b12.AsVector(), b0.AsVector(), 8).AsVector128(); b8 = AdvSimd.Add(b8, b12); b4 = Sve2.XorRotateRight(b4.AsVector(), b8.AsVector(), 7).AsVector128();
                b1 = AdvSimd.Add(AdvSimd.Add(b1, mb[3]), b5); b13 = Sve2.XorRotateRight(b13.AsVector(), b1.AsVector(), 16).AsVector128(); b9 = AdvSimd.Add(b9, b13); b5 = Sve2.XorRotateRight(b5.AsVector(), b9.AsVector(), 12).AsVector128();
                b1 = AdvSimd.Add(AdvSimd.Add(b1, mb[10]), b5); b13 = Sve2.XorRotateRight(b13.AsVector(), b1.AsVector(), 8).AsVector128(); b9 = AdvSimd.Add(b9, b13); b5 = Sve2.XorRotateRight(b5.AsVector(), b9.AsVector(), 7).AsVector128();
                b2 = AdvSimd.Add(AdvSimd.Add(b2, mb[7]), b6); b14 = Sve2.XorRotateRight(b14.AsVector(), b2.AsVector(), 16).AsVector128(); b10 = AdvSimd.Add(b10, b14); b6 = Sve2.XorRotateRight(b6.AsVector(), b10.AsVector(), 12).AsVector128();
                b2 = AdvSimd.Add(AdvSimd.Add(b2, mb[0]), b6); b14 = Sve2.XorRotateRight(b14.AsVector(), b2.AsVector(), 8).AsVector128(); b10 = AdvSimd.Add(b10, b14); b6 = Sve2.XorRotateRight(b6.AsVector(), b10.AsVector(), 7).AsVector128();
                b3 = AdvSimd.Add(AdvSimd.Add(b3, mb[4]), b7); b15 = Sve2.XorRotateRight(b15.AsVector(), b3.AsVector(), 16).AsVector128(); b11 = AdvSimd.Add(b11, b15); b7 = Sve2.XorRotateRight(b7.AsVector(), b11.AsVector(), 12).AsVector128();
                b3 = AdvSimd.Add(AdvSimd.Add(b3, mb[13]), b7); b15 = Sve2.XorRotateRight(b15.AsVector(), b3.AsVector(), 8).AsVector128(); b11 = AdvSimd.Add(b11, b15); b7 = Sve2.XorRotateRight(b7.AsVector(), b11.AsVector(), 7).AsVector128();
                a0 = AdvSimd.Add(AdvSimd.Add(a0, mb[1]), a5); a15 = Sve2.XorRotateRight(a15.AsVector(), a0.AsVector(), 16).AsVector128(); a10 = AdvSimd.Add(a10, a15); a5 = Sve2.XorRotateRight(a5.AsVector(), a10.AsVector(), 12).AsVector128();
                a0 = AdvSimd.Add(AdvSimd.Add(a0, mb[11]), a5); a15 = Sve2.XorRotateRight(a15.AsVector(), a0.AsVector(), 8).AsVector128(); a10 = AdvSimd.Add(a10, a15); a5 = Sve2.XorRotateRight(a5.AsVector(), a10.AsVector(), 7).AsVector128();
                a1 = AdvSimd.Add(AdvSimd.Add(a1, mb[12]), a6); a12 = Sve2.XorRotateRight(a12.AsVector(), a1.AsVector(), 16).AsVector128(); a11 = AdvSimd.Add(a11, a12); a6 = Sve2.XorRotateRight(a6.AsVector(), a11.AsVector(), 12).AsVector128();
                a1 = AdvSimd.Add(AdvSimd.Add(a1, mb[5]), a6); a12 = Sve2.XorRotateRight(a12.AsVector(), a1.AsVector(), 8).AsVector128(); a11 = AdvSimd.Add(a11, a12); a6 = Sve2.XorRotateRight(a6.AsVector(), a11.AsVector(), 7).AsVector128();
                a2 = AdvSimd.Add(AdvSimd.Add(a2, mb[9]), a7); a13 = Sve2.XorRotateRight(a13.AsVector(), a2.AsVector(), 16).AsVector128(); a8 = AdvSimd.Add(a8, a13); a7 = Sve2.XorRotateRight(a7.AsVector(), a8.AsVector(), 12).AsVector128();
                a2 = AdvSimd.Add(AdvSimd.Add(a2, mb[14]), a7); a13 = Sve2.XorRotateRight(a13.AsVector(), a2.AsVector(), 8).AsVector128(); a8 = AdvSimd.Add(a8, a13); a7 = Sve2.XorRotateRight(a7.AsVector(), a8.AsVector(), 7).AsVector128();
                a3 = AdvSimd.Add(AdvSimd.Add(a3, mb[15]), a4); a14 = Sve2.XorRotateRight(a14.AsVector(), a3.AsVector(), 16).AsVector128(); a9 = AdvSimd.Add(a9, a14); a4 = Sve2.XorRotateRight(a4.AsVector(), a9.AsVector(), 12).AsVector128();
                a3 = AdvSimd.Add(AdvSimd.Add(a3, mb[8]), a4); a14 = Sve2.XorRotateRight(a14.AsVector(), a3.AsVector(), 8).AsVector128(); a9 = AdvSimd.Add(a9, a14); a4 = Sve2.XorRotateRight(a4.AsVector(), a9.AsVector(), 7).AsVector128();
                b0 = AdvSimd.Add(AdvSimd.Add(b0, mb[1]), b5); b15 = Sve2.XorRotateRight(b15.AsVector(), b0.AsVector(), 16).AsVector128(); b10 = AdvSimd.Add(b10, b15); b5 = Sve2.XorRotateRight(b5.AsVector(), b10.AsVector(), 12).AsVector128();
                b0 = AdvSimd.Add(AdvSimd.Add(b0, mb[11]), b5); b15 = Sve2.XorRotateRight(b15.AsVector(), b0.AsVector(), 8).AsVector128(); b10 = AdvSimd.Add(b10, b15); b5 = Sve2.XorRotateRight(b5.AsVector(), b10.AsVector(), 7).AsVector128();
                b1 = AdvSimd.Add(AdvSimd.Add(b1, mb[12]), b6); b12 = Sve2.XorRotateRight(b12.AsVector(), b1.AsVector(), 16).AsVector128(); b11 = AdvSimd.Add(b11, b12); b6 = Sve2.XorRotateRight(b6.AsVector(), b11.AsVector(), 12).AsVector128();
                b1 = AdvSimd.Add(AdvSimd.Add(b1, mb[5]), b6); b12 = Sve2.XorRotateRight(b12.AsVector(), b1.AsVector(), 8).AsVector128(); b11 = AdvSimd.Add(b11, b12); b6 = Sve2.XorRotateRight(b6.AsVector(), b11.AsVector(), 7).AsVector128();
                b2 = AdvSimd.Add(AdvSimd.Add(b2, mb[9]), b7); b13 = Sve2.XorRotateRight(b13.AsVector(), b2.AsVector(), 16).AsVector128(); b8 = AdvSimd.Add(b8, b13); b7 = Sve2.XorRotateRight(b7.AsVector(), b8.AsVector(), 12).AsVector128();
                b2 = AdvSimd.Add(AdvSimd.Add(b2, mb[14]), b7); b13 = Sve2.XorRotateRight(b13.AsVector(), b2.AsVector(), 8).AsVector128(); b8 = AdvSimd.Add(b8, b13); b7 = Sve2.XorRotateRight(b7.AsVector(), b8.AsVector(), 7).AsVector128();
                b3 = AdvSimd.Add(AdvSimd.Add(b3, mb[15]), b4); b14 = Sve2.XorRotateRight(b14.AsVector(), b3.AsVector(), 16).AsVector128(); b9 = AdvSimd.Add(b9, b14); b4 = Sve2.XorRotateRight(b4.AsVector(), b9.AsVector(), 12).AsVector128();
                b3 = AdvSimd.Add(AdvSimd.Add(b3, mb[8]), b4); b14 = Sve2.XorRotateRight(b14.AsVector(), b3.AsVector(), 8).AsVector128(); b9 = AdvSimd.Add(b9, b14); b4 = Sve2.XorRotateRight(b4.AsVector(), b9.AsVector(), 7).AsVector128();
                // Round 2
                a0 = AdvSimd.Add(AdvSimd.Add(a0, mb[3]), a4); a12 = Sve2.XorRotateRight(a12.AsVector(), a0.AsVector(), 16).AsVector128(); a8 = AdvSimd.Add(a8, a12); a4 = Sve2.XorRotateRight(a4.AsVector(), a8.AsVector(), 12).AsVector128();
                a0 = AdvSimd.Add(AdvSimd.Add(a0, mb[4]), a4); a12 = Sve2.XorRotateRight(a12.AsVector(), a0.AsVector(), 8).AsVector128(); a8 = AdvSimd.Add(a8, a12); a4 = Sve2.XorRotateRight(a4.AsVector(), a8.AsVector(), 7).AsVector128();
                a1 = AdvSimd.Add(AdvSimd.Add(a1, mb[10]), a5); a13 = Sve2.XorRotateRight(a13.AsVector(), a1.AsVector(), 16).AsVector128(); a9 = AdvSimd.Add(a9, a13); a5 = Sve2.XorRotateRight(a5.AsVector(), a9.AsVector(), 12).AsVector128();
                a1 = AdvSimd.Add(AdvSimd.Add(a1, mb[12]), a5); a13 = Sve2.XorRotateRight(a13.AsVector(), a1.AsVector(), 8).AsVector128(); a9 = AdvSimd.Add(a9, a13); a5 = Sve2.XorRotateRight(a5.AsVector(), a9.AsVector(), 7).AsVector128();
                a2 = AdvSimd.Add(AdvSimd.Add(a2, mb[13]), a6); a14 = Sve2.XorRotateRight(a14.AsVector(), a2.AsVector(), 16).AsVector128(); a10 = AdvSimd.Add(a10, a14); a6 = Sve2.XorRotateRight(a6.AsVector(), a10.AsVector(), 12).AsVector128();
                a2 = AdvSimd.Add(AdvSimd.Add(a2, mb[2]), a6); a14 = Sve2.XorRotateRight(a14.AsVector(), a2.AsVector(), 8).AsVector128(); a10 = AdvSimd.Add(a10, a14); a6 = Sve2.XorRotateRight(a6.AsVector(), a10.AsVector(), 7).AsVector128();
                a3 = AdvSimd.Add(AdvSimd.Add(a3, mb[7]), a7); a15 = Sve2.XorRotateRight(a15.AsVector(), a3.AsVector(), 16).AsVector128(); a11 = AdvSimd.Add(a11, a15); a7 = Sve2.XorRotateRight(a7.AsVector(), a11.AsVector(), 12).AsVector128();
                a3 = AdvSimd.Add(AdvSimd.Add(a3, mb[14]), a7); a15 = Sve2.XorRotateRight(a15.AsVector(), a3.AsVector(), 8).AsVector128(); a11 = AdvSimd.Add(a11, a15); a7 = Sve2.XorRotateRight(a7.AsVector(), a11.AsVector(), 7).AsVector128();
                b0 = AdvSimd.Add(AdvSimd.Add(b0, mb[3]), b4); b12 = Sve2.XorRotateRight(b12.AsVector(), b0.AsVector(), 16).AsVector128(); b8 = AdvSimd.Add(b8, b12); b4 = Sve2.XorRotateRight(b4.AsVector(), b8.AsVector(), 12).AsVector128();
                b0 = AdvSimd.Add(AdvSimd.Add(b0, mb[4]), b4); b12 = Sve2.XorRotateRight(b12.AsVector(), b0.AsVector(), 8).AsVector128(); b8 = AdvSimd.Add(b8, b12); b4 = Sve2.XorRotateRight(b4.AsVector(), b8.AsVector(), 7).AsVector128();
                b1 = AdvSimd.Add(AdvSimd.Add(b1, mb[10]), b5); b13 = Sve2.XorRotateRight(b13.AsVector(), b1.AsVector(), 16).AsVector128(); b9 = AdvSimd.Add(b9, b13); b5 = Sve2.XorRotateRight(b5.AsVector(), b9.AsVector(), 12).AsVector128();
                b1 = AdvSimd.Add(AdvSimd.Add(b1, mb[12]), b5); b13 = Sve2.XorRotateRight(b13.AsVector(), b1.AsVector(), 8).AsVector128(); b9 = AdvSimd.Add(b9, b13); b5 = Sve2.XorRotateRight(b5.AsVector(), b9.AsVector(), 7).AsVector128();
                b2 = AdvSimd.Add(AdvSimd.Add(b2, mb[13]), b6); b14 = Sve2.XorRotateRight(b14.AsVector(), b2.AsVector(), 16).AsVector128(); b10 = AdvSimd.Add(b10, b14); b6 = Sve2.XorRotateRight(b6.AsVector(), b10.AsVector(), 12).AsVector128();
                b2 = AdvSimd.Add(AdvSimd.Add(b2, mb[2]), b6); b14 = Sve2.XorRotateRight(b14.AsVector(), b2.AsVector(), 8).AsVector128(); b10 = AdvSimd.Add(b10, b14); b6 = Sve2.XorRotateRight(b6.AsVector(), b10.AsVector(), 7).AsVector128();
                b3 = AdvSimd.Add(AdvSimd.Add(b3, mb[7]), b7); b15 = Sve2.XorRotateRight(b15.AsVector(), b3.AsVector(), 16).AsVector128(); b11 = AdvSimd.Add(b11, b15); b7 = Sve2.XorRotateRight(b7.AsVector(), b11.AsVector(), 12).AsVector128();
                b3 = AdvSimd.Add(AdvSimd.Add(b3, mb[14]), b7); b15 = Sve2.XorRotateRight(b15.AsVector(), b3.AsVector(), 8).AsVector128(); b11 = AdvSimd.Add(b11, b15); b7 = Sve2.XorRotateRight(b7.AsVector(), b11.AsVector(), 7).AsVector128();
                a0 = AdvSimd.Add(AdvSimd.Add(a0, mb[6]), a5); a15 = Sve2.XorRotateRight(a15.AsVector(), a0.AsVector(), 16).AsVector128(); a10 = AdvSimd.Add(a10, a15); a5 = Sve2.XorRotateRight(a5.AsVector(), a10.AsVector(), 12).AsVector128();
                a0 = AdvSimd.Add(AdvSimd.Add(a0, mb[5]), a5); a15 = Sve2.XorRotateRight(a15.AsVector(), a0.AsVector(), 8).AsVector128(); a10 = AdvSimd.Add(a10, a15); a5 = Sve2.XorRotateRight(a5.AsVector(), a10.AsVector(), 7).AsVector128();
                a1 = AdvSimd.Add(AdvSimd.Add(a1, mb[9]), a6); a12 = Sve2.XorRotateRight(a12.AsVector(), a1.AsVector(), 16).AsVector128(); a11 = AdvSimd.Add(a11, a12); a6 = Sve2.XorRotateRight(a6.AsVector(), a11.AsVector(), 12).AsVector128();
                a1 = AdvSimd.Add(AdvSimd.Add(a1, mb[0]), a6); a12 = Sve2.XorRotateRight(a12.AsVector(), a1.AsVector(), 8).AsVector128(); a11 = AdvSimd.Add(a11, a12); a6 = Sve2.XorRotateRight(a6.AsVector(), a11.AsVector(), 7).AsVector128();
                a2 = AdvSimd.Add(AdvSimd.Add(a2, mb[11]), a7); a13 = Sve2.XorRotateRight(a13.AsVector(), a2.AsVector(), 16).AsVector128(); a8 = AdvSimd.Add(a8, a13); a7 = Sve2.XorRotateRight(a7.AsVector(), a8.AsVector(), 12).AsVector128();
                a2 = AdvSimd.Add(AdvSimd.Add(a2, mb[15]), a7); a13 = Sve2.XorRotateRight(a13.AsVector(), a2.AsVector(), 8).AsVector128(); a8 = AdvSimd.Add(a8, a13); a7 = Sve2.XorRotateRight(a7.AsVector(), a8.AsVector(), 7).AsVector128();
                a3 = AdvSimd.Add(AdvSimd.Add(a3, mb[8]), a4); a14 = Sve2.XorRotateRight(a14.AsVector(), a3.AsVector(), 16).AsVector128(); a9 = AdvSimd.Add(a9, a14); a4 = Sve2.XorRotateRight(a4.AsVector(), a9.AsVector(), 12).AsVector128();
                a3 = AdvSimd.Add(AdvSimd.Add(a3, mb[1]), a4); a14 = Sve2.XorRotateRight(a14.AsVector(), a3.AsVector(), 8).AsVector128(); a9 = AdvSimd.Add(a9, a14); a4 = Sve2.XorRotateRight(a4.AsVector(), a9.AsVector(), 7).AsVector128();
                b0 = AdvSimd.Add(AdvSimd.Add(b0, mb[6]), b5); b15 = Sve2.XorRotateRight(b15.AsVector(), b0.AsVector(), 16).AsVector128(); b10 = AdvSimd.Add(b10, b15); b5 = Sve2.XorRotateRight(b5.AsVector(), b10.AsVector(), 12).AsVector128();
                b0 = AdvSimd.Add(AdvSimd.Add(b0, mb[5]), b5); b15 = Sve2.XorRotateRight(b15.AsVector(), b0.AsVector(), 8).AsVector128(); b10 = AdvSimd.Add(b10, b15); b5 = Sve2.XorRotateRight(b5.AsVector(), b10.AsVector(), 7).AsVector128();
                b1 = AdvSimd.Add(AdvSimd.Add(b1, mb[9]), b6); b12 = Sve2.XorRotateRight(b12.AsVector(), b1.AsVector(), 16).AsVector128(); b11 = AdvSimd.Add(b11, b12); b6 = Sve2.XorRotateRight(b6.AsVector(), b11.AsVector(), 12).AsVector128();
                b1 = AdvSimd.Add(AdvSimd.Add(b1, mb[0]), b6); b12 = Sve2.XorRotateRight(b12.AsVector(), b1.AsVector(), 8).AsVector128(); b11 = AdvSimd.Add(b11, b12); b6 = Sve2.XorRotateRight(b6.AsVector(), b11.AsVector(), 7).AsVector128();
                b2 = AdvSimd.Add(AdvSimd.Add(b2, mb[11]), b7); b13 = Sve2.XorRotateRight(b13.AsVector(), b2.AsVector(), 16).AsVector128(); b8 = AdvSimd.Add(b8, b13); b7 = Sve2.XorRotateRight(b7.AsVector(), b8.AsVector(), 12).AsVector128();
                b2 = AdvSimd.Add(AdvSimd.Add(b2, mb[15]), b7); b13 = Sve2.XorRotateRight(b13.AsVector(), b2.AsVector(), 8).AsVector128(); b8 = AdvSimd.Add(b8, b13); b7 = Sve2.XorRotateRight(b7.AsVector(), b8.AsVector(), 7).AsVector128();
                b3 = AdvSimd.Add(AdvSimd.Add(b3, mb[8]), b4); b14 = Sve2.XorRotateRight(b14.AsVector(), b3.AsVector(), 16).AsVector128(); b9 = AdvSimd.Add(b9, b14); b4 = Sve2.XorRotateRight(b4.AsVector(), b9.AsVector(), 12).AsVector128();
                b3 = AdvSimd.Add(AdvSimd.Add(b3, mb[1]), b4); b14 = Sve2.XorRotateRight(b14.AsVector(), b3.AsVector(), 8).AsVector128(); b9 = AdvSimd.Add(b9, b14); b4 = Sve2.XorRotateRight(b4.AsVector(), b9.AsVector(), 7).AsVector128();
                // Round 3
                a0 = AdvSimd.Add(AdvSimd.Add(a0, mb[10]), a4); a12 = Sve2.XorRotateRight(a12.AsVector(), a0.AsVector(), 16).AsVector128(); a8 = AdvSimd.Add(a8, a12); a4 = Sve2.XorRotateRight(a4.AsVector(), a8.AsVector(), 12).AsVector128();
                a0 = AdvSimd.Add(AdvSimd.Add(a0, mb[7]), a4); a12 = Sve2.XorRotateRight(a12.AsVector(), a0.AsVector(), 8).AsVector128(); a8 = AdvSimd.Add(a8, a12); a4 = Sve2.XorRotateRight(a4.AsVector(), a8.AsVector(), 7).AsVector128();
                a1 = AdvSimd.Add(AdvSimd.Add(a1, mb[12]), a5); a13 = Sve2.XorRotateRight(a13.AsVector(), a1.AsVector(), 16).AsVector128(); a9 = AdvSimd.Add(a9, a13); a5 = Sve2.XorRotateRight(a5.AsVector(), a9.AsVector(), 12).AsVector128();
                a1 = AdvSimd.Add(AdvSimd.Add(a1, mb[9]), a5); a13 = Sve2.XorRotateRight(a13.AsVector(), a1.AsVector(), 8).AsVector128(); a9 = AdvSimd.Add(a9, a13); a5 = Sve2.XorRotateRight(a5.AsVector(), a9.AsVector(), 7).AsVector128();
                a2 = AdvSimd.Add(AdvSimd.Add(a2, mb[14]), a6); a14 = Sve2.XorRotateRight(a14.AsVector(), a2.AsVector(), 16).AsVector128(); a10 = AdvSimd.Add(a10, a14); a6 = Sve2.XorRotateRight(a6.AsVector(), a10.AsVector(), 12).AsVector128();
                a2 = AdvSimd.Add(AdvSimd.Add(a2, mb[3]), a6); a14 = Sve2.XorRotateRight(a14.AsVector(), a2.AsVector(), 8).AsVector128(); a10 = AdvSimd.Add(a10, a14); a6 = Sve2.XorRotateRight(a6.AsVector(), a10.AsVector(), 7).AsVector128();
                a3 = AdvSimd.Add(AdvSimd.Add(a3, mb[13]), a7); a15 = Sve2.XorRotateRight(a15.AsVector(), a3.AsVector(), 16).AsVector128(); a11 = AdvSimd.Add(a11, a15); a7 = Sve2.XorRotateRight(a7.AsVector(), a11.AsVector(), 12).AsVector128();
                a3 = AdvSimd.Add(AdvSimd.Add(a3, mb[15]), a7); a15 = Sve2.XorRotateRight(a15.AsVector(), a3.AsVector(), 8).AsVector128(); a11 = AdvSimd.Add(a11, a15); a7 = Sve2.XorRotateRight(a7.AsVector(), a11.AsVector(), 7).AsVector128();
                b0 = AdvSimd.Add(AdvSimd.Add(b0, mb[10]), b4); b12 = Sve2.XorRotateRight(b12.AsVector(), b0.AsVector(), 16).AsVector128(); b8 = AdvSimd.Add(b8, b12); b4 = Sve2.XorRotateRight(b4.AsVector(), b8.AsVector(), 12).AsVector128();
                b0 = AdvSimd.Add(AdvSimd.Add(b0, mb[7]), b4); b12 = Sve2.XorRotateRight(b12.AsVector(), b0.AsVector(), 8).AsVector128(); b8 = AdvSimd.Add(b8, b12); b4 = Sve2.XorRotateRight(b4.AsVector(), b8.AsVector(), 7).AsVector128();
                b1 = AdvSimd.Add(AdvSimd.Add(b1, mb[12]), b5); b13 = Sve2.XorRotateRight(b13.AsVector(), b1.AsVector(), 16).AsVector128(); b9 = AdvSimd.Add(b9, b13); b5 = Sve2.XorRotateRight(b5.AsVector(), b9.AsVector(), 12).AsVector128();
                b1 = AdvSimd.Add(AdvSimd.Add(b1, mb[9]), b5); b13 = Sve2.XorRotateRight(b13.AsVector(), b1.AsVector(), 8).AsVector128(); b9 = AdvSimd.Add(b9, b13); b5 = Sve2.XorRotateRight(b5.AsVector(), b9.AsVector(), 7).AsVector128();
                b2 = AdvSimd.Add(AdvSimd.Add(b2, mb[14]), b6); b14 = Sve2.XorRotateRight(b14.AsVector(), b2.AsVector(), 16).AsVector128(); b10 = AdvSimd.Add(b10, b14); b6 = Sve2.XorRotateRight(b6.AsVector(), b10.AsVector(), 12).AsVector128();
                b2 = AdvSimd.Add(AdvSimd.Add(b2, mb[3]), b6); b14 = Sve2.XorRotateRight(b14.AsVector(), b2.AsVector(), 8).AsVector128(); b10 = AdvSimd.Add(b10, b14); b6 = Sve2.XorRotateRight(b6.AsVector(), b10.AsVector(), 7).AsVector128();
                b3 = AdvSimd.Add(AdvSimd.Add(b3, mb[13]), b7); b15 = Sve2.XorRotateRight(b15.AsVector(), b3.AsVector(), 16).AsVector128(); b11 = AdvSimd.Add(b11, b15); b7 = Sve2.XorRotateRight(b7.AsVector(), b11.AsVector(), 12).AsVector128();
                b3 = AdvSimd.Add(AdvSimd.Add(b3, mb[15]), b7); b15 = Sve2.XorRotateRight(b15.AsVector(), b3.AsVector(), 8).AsVector128(); b11 = AdvSimd.Add(b11, b15); b7 = Sve2.XorRotateRight(b7.AsVector(), b11.AsVector(), 7).AsVector128();
                a0 = AdvSimd.Add(AdvSimd.Add(a0, mb[4]), a5); a15 = Sve2.XorRotateRight(a15.AsVector(), a0.AsVector(), 16).AsVector128(); a10 = AdvSimd.Add(a10, a15); a5 = Sve2.XorRotateRight(a5.AsVector(), a10.AsVector(), 12).AsVector128();
                a0 = AdvSimd.Add(AdvSimd.Add(a0, mb[0]), a5); a15 = Sve2.XorRotateRight(a15.AsVector(), a0.AsVector(), 8).AsVector128(); a10 = AdvSimd.Add(a10, a15); a5 = Sve2.XorRotateRight(a5.AsVector(), a10.AsVector(), 7).AsVector128();
                a1 = AdvSimd.Add(AdvSimd.Add(a1, mb[11]), a6); a12 = Sve2.XorRotateRight(a12.AsVector(), a1.AsVector(), 16).AsVector128(); a11 = AdvSimd.Add(a11, a12); a6 = Sve2.XorRotateRight(a6.AsVector(), a11.AsVector(), 12).AsVector128();
                a1 = AdvSimd.Add(AdvSimd.Add(a1, mb[2]), a6); a12 = Sve2.XorRotateRight(a12.AsVector(), a1.AsVector(), 8).AsVector128(); a11 = AdvSimd.Add(a11, a12); a6 = Sve2.XorRotateRight(a6.AsVector(), a11.AsVector(), 7).AsVector128();
                a2 = AdvSimd.Add(AdvSimd.Add(a2, mb[5]), a7); a13 = Sve2.XorRotateRight(a13.AsVector(), a2.AsVector(), 16).AsVector128(); a8 = AdvSimd.Add(a8, a13); a7 = Sve2.XorRotateRight(a7.AsVector(), a8.AsVector(), 12).AsVector128();
                a2 = AdvSimd.Add(AdvSimd.Add(a2, mb[8]), a7); a13 = Sve2.XorRotateRight(a13.AsVector(), a2.AsVector(), 8).AsVector128(); a8 = AdvSimd.Add(a8, a13); a7 = Sve2.XorRotateRight(a7.AsVector(), a8.AsVector(), 7).AsVector128();
                a3 = AdvSimd.Add(AdvSimd.Add(a3, mb[1]), a4); a14 = Sve2.XorRotateRight(a14.AsVector(), a3.AsVector(), 16).AsVector128(); a9 = AdvSimd.Add(a9, a14); a4 = Sve2.XorRotateRight(a4.AsVector(), a9.AsVector(), 12).AsVector128();
                a3 = AdvSimd.Add(AdvSimd.Add(a3, mb[6]), a4); a14 = Sve2.XorRotateRight(a14.AsVector(), a3.AsVector(), 8).AsVector128(); a9 = AdvSimd.Add(a9, a14); a4 = Sve2.XorRotateRight(a4.AsVector(), a9.AsVector(), 7).AsVector128();
                b0 = AdvSimd.Add(AdvSimd.Add(b0, mb[4]), b5); b15 = Sve2.XorRotateRight(b15.AsVector(), b0.AsVector(), 16).AsVector128(); b10 = AdvSimd.Add(b10, b15); b5 = Sve2.XorRotateRight(b5.AsVector(), b10.AsVector(), 12).AsVector128();
                b0 = AdvSimd.Add(AdvSimd.Add(b0, mb[0]), b5); b15 = Sve2.XorRotateRight(b15.AsVector(), b0.AsVector(), 8).AsVector128(); b10 = AdvSimd.Add(b10, b15); b5 = Sve2.XorRotateRight(b5.AsVector(), b10.AsVector(), 7).AsVector128();
                b1 = AdvSimd.Add(AdvSimd.Add(b1, mb[11]), b6); b12 = Sve2.XorRotateRight(b12.AsVector(), b1.AsVector(), 16).AsVector128(); b11 = AdvSimd.Add(b11, b12); b6 = Sve2.XorRotateRight(b6.AsVector(), b11.AsVector(), 12).AsVector128();
                b1 = AdvSimd.Add(AdvSimd.Add(b1, mb[2]), b6); b12 = Sve2.XorRotateRight(b12.AsVector(), b1.AsVector(), 8).AsVector128(); b11 = AdvSimd.Add(b11, b12); b6 = Sve2.XorRotateRight(b6.AsVector(), b11.AsVector(), 7).AsVector128();
                b2 = AdvSimd.Add(AdvSimd.Add(b2, mb[5]), b7); b13 = Sve2.XorRotateRight(b13.AsVector(), b2.AsVector(), 16).AsVector128(); b8 = AdvSimd.Add(b8, b13); b7 = Sve2.XorRotateRight(b7.AsVector(), b8.AsVector(), 12).AsVector128();
                b2 = AdvSimd.Add(AdvSimd.Add(b2, mb[8]), b7); b13 = Sve2.XorRotateRight(b13.AsVector(), b2.AsVector(), 8).AsVector128(); b8 = AdvSimd.Add(b8, b13); b7 = Sve2.XorRotateRight(b7.AsVector(), b8.AsVector(), 7).AsVector128();
                b3 = AdvSimd.Add(AdvSimd.Add(b3, mb[1]), b4); b14 = Sve2.XorRotateRight(b14.AsVector(), b3.AsVector(), 16).AsVector128(); b9 = AdvSimd.Add(b9, b14); b4 = Sve2.XorRotateRight(b4.AsVector(), b9.AsVector(), 12).AsVector128();
                b3 = AdvSimd.Add(AdvSimd.Add(b3, mb[6]), b4); b14 = Sve2.XorRotateRight(b14.AsVector(), b3.AsVector(), 8).AsVector128(); b9 = AdvSimd.Add(b9, b14); b4 = Sve2.XorRotateRight(b4.AsVector(), b9.AsVector(), 7).AsVector128();
                // Round 4
                a0 = AdvSimd.Add(AdvSimd.Add(a0, mb[12]), a4); a12 = Sve2.XorRotateRight(a12.AsVector(), a0.AsVector(), 16).AsVector128(); a8 = AdvSimd.Add(a8, a12); a4 = Sve2.XorRotateRight(a4.AsVector(), a8.AsVector(), 12).AsVector128();
                a0 = AdvSimd.Add(AdvSimd.Add(a0, mb[13]), a4); a12 = Sve2.XorRotateRight(a12.AsVector(), a0.AsVector(), 8).AsVector128(); a8 = AdvSimd.Add(a8, a12); a4 = Sve2.XorRotateRight(a4.AsVector(), a8.AsVector(), 7).AsVector128();
                a1 = AdvSimd.Add(AdvSimd.Add(a1, mb[9]), a5); a13 = Sve2.XorRotateRight(a13.AsVector(), a1.AsVector(), 16).AsVector128(); a9 = AdvSimd.Add(a9, a13); a5 = Sve2.XorRotateRight(a5.AsVector(), a9.AsVector(), 12).AsVector128();
                a1 = AdvSimd.Add(AdvSimd.Add(a1, mb[11]), a5); a13 = Sve2.XorRotateRight(a13.AsVector(), a1.AsVector(), 8).AsVector128(); a9 = AdvSimd.Add(a9, a13); a5 = Sve2.XorRotateRight(a5.AsVector(), a9.AsVector(), 7).AsVector128();
                a2 = AdvSimd.Add(AdvSimd.Add(a2, mb[15]), a6); a14 = Sve2.XorRotateRight(a14.AsVector(), a2.AsVector(), 16).AsVector128(); a10 = AdvSimd.Add(a10, a14); a6 = Sve2.XorRotateRight(a6.AsVector(), a10.AsVector(), 12).AsVector128();
                a2 = AdvSimd.Add(AdvSimd.Add(a2, mb[10]), a6); a14 = Sve2.XorRotateRight(a14.AsVector(), a2.AsVector(), 8).AsVector128(); a10 = AdvSimd.Add(a10, a14); a6 = Sve2.XorRotateRight(a6.AsVector(), a10.AsVector(), 7).AsVector128();
                a3 = AdvSimd.Add(AdvSimd.Add(a3, mb[14]), a7); a15 = Sve2.XorRotateRight(a15.AsVector(), a3.AsVector(), 16).AsVector128(); a11 = AdvSimd.Add(a11, a15); a7 = Sve2.XorRotateRight(a7.AsVector(), a11.AsVector(), 12).AsVector128();
                a3 = AdvSimd.Add(AdvSimd.Add(a3, mb[8]), a7); a15 = Sve2.XorRotateRight(a15.AsVector(), a3.AsVector(), 8).AsVector128(); a11 = AdvSimd.Add(a11, a15); a7 = Sve2.XorRotateRight(a7.AsVector(), a11.AsVector(), 7).AsVector128();
                b0 = AdvSimd.Add(AdvSimd.Add(b0, mb[12]), b4); b12 = Sve2.XorRotateRight(b12.AsVector(), b0.AsVector(), 16).AsVector128(); b8 = AdvSimd.Add(b8, b12); b4 = Sve2.XorRotateRight(b4.AsVector(), b8.AsVector(), 12).AsVector128();
                b0 = AdvSimd.Add(AdvSimd.Add(b0, mb[13]), b4); b12 = Sve2.XorRotateRight(b12.AsVector(), b0.AsVector(), 8).AsVector128(); b8 = AdvSimd.Add(b8, b12); b4 = Sve2.XorRotateRight(b4.AsVector(), b8.AsVector(), 7).AsVector128();
                b1 = AdvSimd.Add(AdvSimd.Add(b1, mb[9]), b5); b13 = Sve2.XorRotateRight(b13.AsVector(), b1.AsVector(), 16).AsVector128(); b9 = AdvSimd.Add(b9, b13); b5 = Sve2.XorRotateRight(b5.AsVector(), b9.AsVector(), 12).AsVector128();
                b1 = AdvSimd.Add(AdvSimd.Add(b1, mb[11]), b5); b13 = Sve2.XorRotateRight(b13.AsVector(), b1.AsVector(), 8).AsVector128(); b9 = AdvSimd.Add(b9, b13); b5 = Sve2.XorRotateRight(b5.AsVector(), b9.AsVector(), 7).AsVector128();
                b2 = AdvSimd.Add(AdvSimd.Add(b2, mb[15]), b6); b14 = Sve2.XorRotateRight(b14.AsVector(), b2.AsVector(), 16).AsVector128(); b10 = AdvSimd.Add(b10, b14); b6 = Sve2.XorRotateRight(b6.AsVector(), b10.AsVector(), 12).AsVector128();
                b2 = AdvSimd.Add(AdvSimd.Add(b2, mb[10]), b6); b14 = Sve2.XorRotateRight(b14.AsVector(), b2.AsVector(), 8).AsVector128(); b10 = AdvSimd.Add(b10, b14); b6 = Sve2.XorRotateRight(b6.AsVector(), b10.AsVector(), 7).AsVector128();
                b3 = AdvSimd.Add(AdvSimd.Add(b3, mb[14]), b7); b15 = Sve2.XorRotateRight(b15.AsVector(), b3.AsVector(), 16).AsVector128(); b11 = AdvSimd.Add(b11, b15); b7 = Sve2.XorRotateRight(b7.AsVector(), b11.AsVector(), 12).AsVector128();
                b3 = AdvSimd.Add(AdvSimd.Add(b3, mb[8]), b7); b15 = Sve2.XorRotateRight(b15.AsVector(), b3.AsVector(), 8).AsVector128(); b11 = AdvSimd.Add(b11, b15); b7 = Sve2.XorRotateRight(b7.AsVector(), b11.AsVector(), 7).AsVector128();
                a0 = AdvSimd.Add(AdvSimd.Add(a0, mb[7]), a5); a15 = Sve2.XorRotateRight(a15.AsVector(), a0.AsVector(), 16).AsVector128(); a10 = AdvSimd.Add(a10, a15); a5 = Sve2.XorRotateRight(a5.AsVector(), a10.AsVector(), 12).AsVector128();
                a0 = AdvSimd.Add(AdvSimd.Add(a0, mb[2]), a5); a15 = Sve2.XorRotateRight(a15.AsVector(), a0.AsVector(), 8).AsVector128(); a10 = AdvSimd.Add(a10, a15); a5 = Sve2.XorRotateRight(a5.AsVector(), a10.AsVector(), 7).AsVector128();
                a1 = AdvSimd.Add(AdvSimd.Add(a1, mb[5]), a6); a12 = Sve2.XorRotateRight(a12.AsVector(), a1.AsVector(), 16).AsVector128(); a11 = AdvSimd.Add(a11, a12); a6 = Sve2.XorRotateRight(a6.AsVector(), a11.AsVector(), 12).AsVector128();
                a1 = AdvSimd.Add(AdvSimd.Add(a1, mb[3]), a6); a12 = Sve2.XorRotateRight(a12.AsVector(), a1.AsVector(), 8).AsVector128(); a11 = AdvSimd.Add(a11, a12); a6 = Sve2.XorRotateRight(a6.AsVector(), a11.AsVector(), 7).AsVector128();
                a2 = AdvSimd.Add(AdvSimd.Add(a2, mb[0]), a7); a13 = Sve2.XorRotateRight(a13.AsVector(), a2.AsVector(), 16).AsVector128(); a8 = AdvSimd.Add(a8, a13); a7 = Sve2.XorRotateRight(a7.AsVector(), a8.AsVector(), 12).AsVector128();
                a2 = AdvSimd.Add(AdvSimd.Add(a2, mb[1]), a7); a13 = Sve2.XorRotateRight(a13.AsVector(), a2.AsVector(), 8).AsVector128(); a8 = AdvSimd.Add(a8, a13); a7 = Sve2.XorRotateRight(a7.AsVector(), a8.AsVector(), 7).AsVector128();
                a3 = AdvSimd.Add(AdvSimd.Add(a3, mb[6]), a4); a14 = Sve2.XorRotateRight(a14.AsVector(), a3.AsVector(), 16).AsVector128(); a9 = AdvSimd.Add(a9, a14); a4 = Sve2.XorRotateRight(a4.AsVector(), a9.AsVector(), 12).AsVector128();
                a3 = AdvSimd.Add(AdvSimd.Add(a3, mb[4]), a4); a14 = Sve2.XorRotateRight(a14.AsVector(), a3.AsVector(), 8).AsVector128(); a9 = AdvSimd.Add(a9, a14); a4 = Sve2.XorRotateRight(a4.AsVector(), a9.AsVector(), 7).AsVector128();
                b0 = AdvSimd.Add(AdvSimd.Add(b0, mb[7]), b5); b15 = Sve2.XorRotateRight(b15.AsVector(), b0.AsVector(), 16).AsVector128(); b10 = AdvSimd.Add(b10, b15); b5 = Sve2.XorRotateRight(b5.AsVector(), b10.AsVector(), 12).AsVector128();
                b0 = AdvSimd.Add(AdvSimd.Add(b0, mb[2]), b5); b15 = Sve2.XorRotateRight(b15.AsVector(), b0.AsVector(), 8).AsVector128(); b10 = AdvSimd.Add(b10, b15); b5 = Sve2.XorRotateRight(b5.AsVector(), b10.AsVector(), 7).AsVector128();
                b1 = AdvSimd.Add(AdvSimd.Add(b1, mb[5]), b6); b12 = Sve2.XorRotateRight(b12.AsVector(), b1.AsVector(), 16).AsVector128(); b11 = AdvSimd.Add(b11, b12); b6 = Sve2.XorRotateRight(b6.AsVector(), b11.AsVector(), 12).AsVector128();
                b1 = AdvSimd.Add(AdvSimd.Add(b1, mb[3]), b6); b12 = Sve2.XorRotateRight(b12.AsVector(), b1.AsVector(), 8).AsVector128(); b11 = AdvSimd.Add(b11, b12); b6 = Sve2.XorRotateRight(b6.AsVector(), b11.AsVector(), 7).AsVector128();
                b2 = AdvSimd.Add(AdvSimd.Add(b2, mb[0]), b7); b13 = Sve2.XorRotateRight(b13.AsVector(), b2.AsVector(), 16).AsVector128(); b8 = AdvSimd.Add(b8, b13); b7 = Sve2.XorRotateRight(b7.AsVector(), b8.AsVector(), 12).AsVector128();
                b2 = AdvSimd.Add(AdvSimd.Add(b2, mb[1]), b7); b13 = Sve2.XorRotateRight(b13.AsVector(), b2.AsVector(), 8).AsVector128(); b8 = AdvSimd.Add(b8, b13); b7 = Sve2.XorRotateRight(b7.AsVector(), b8.AsVector(), 7).AsVector128();
                b3 = AdvSimd.Add(AdvSimd.Add(b3, mb[6]), b4); b14 = Sve2.XorRotateRight(b14.AsVector(), b3.AsVector(), 16).AsVector128(); b9 = AdvSimd.Add(b9, b14); b4 = Sve2.XorRotateRight(b4.AsVector(), b9.AsVector(), 12).AsVector128();
                b3 = AdvSimd.Add(AdvSimd.Add(b3, mb[4]), b4); b14 = Sve2.XorRotateRight(b14.AsVector(), b3.AsVector(), 8).AsVector128(); b9 = AdvSimd.Add(b9, b14); b4 = Sve2.XorRotateRight(b4.AsVector(), b9.AsVector(), 7).AsVector128();
                // Round 5
                a0 = AdvSimd.Add(AdvSimd.Add(a0, mb[9]), a4); a12 = Sve2.XorRotateRight(a12.AsVector(), a0.AsVector(), 16).AsVector128(); a8 = AdvSimd.Add(a8, a12); a4 = Sve2.XorRotateRight(a4.AsVector(), a8.AsVector(), 12).AsVector128();
                a0 = AdvSimd.Add(AdvSimd.Add(a0, mb[14]), a4); a12 = Sve2.XorRotateRight(a12.AsVector(), a0.AsVector(), 8).AsVector128(); a8 = AdvSimd.Add(a8, a12); a4 = Sve2.XorRotateRight(a4.AsVector(), a8.AsVector(), 7).AsVector128();
                a1 = AdvSimd.Add(AdvSimd.Add(a1, mb[11]), a5); a13 = Sve2.XorRotateRight(a13.AsVector(), a1.AsVector(), 16).AsVector128(); a9 = AdvSimd.Add(a9, a13); a5 = Sve2.XorRotateRight(a5.AsVector(), a9.AsVector(), 12).AsVector128();
                a1 = AdvSimd.Add(AdvSimd.Add(a1, mb[5]), a5); a13 = Sve2.XorRotateRight(a13.AsVector(), a1.AsVector(), 8).AsVector128(); a9 = AdvSimd.Add(a9, a13); a5 = Sve2.XorRotateRight(a5.AsVector(), a9.AsVector(), 7).AsVector128();
                a2 = AdvSimd.Add(AdvSimd.Add(a2, mb[8]), a6); a14 = Sve2.XorRotateRight(a14.AsVector(), a2.AsVector(), 16).AsVector128(); a10 = AdvSimd.Add(a10, a14); a6 = Sve2.XorRotateRight(a6.AsVector(), a10.AsVector(), 12).AsVector128();
                a2 = AdvSimd.Add(AdvSimd.Add(a2, mb[12]), a6); a14 = Sve2.XorRotateRight(a14.AsVector(), a2.AsVector(), 8).AsVector128(); a10 = AdvSimd.Add(a10, a14); a6 = Sve2.XorRotateRight(a6.AsVector(), a10.AsVector(), 7).AsVector128();
                a3 = AdvSimd.Add(AdvSimd.Add(a3, mb[15]), a7); a15 = Sve2.XorRotateRight(a15.AsVector(), a3.AsVector(), 16).AsVector128(); a11 = AdvSimd.Add(a11, a15); a7 = Sve2.XorRotateRight(a7.AsVector(), a11.AsVector(), 12).AsVector128();
                a3 = AdvSimd.Add(AdvSimd.Add(a3, mb[1]), a7); a15 = Sve2.XorRotateRight(a15.AsVector(), a3.AsVector(), 8).AsVector128(); a11 = AdvSimd.Add(a11, a15); a7 = Sve2.XorRotateRight(a7.AsVector(), a11.AsVector(), 7).AsVector128();
                b0 = AdvSimd.Add(AdvSimd.Add(b0, mb[9]), b4); b12 = Sve2.XorRotateRight(b12.AsVector(), b0.AsVector(), 16).AsVector128(); b8 = AdvSimd.Add(b8, b12); b4 = Sve2.XorRotateRight(b4.AsVector(), b8.AsVector(), 12).AsVector128();
                b0 = AdvSimd.Add(AdvSimd.Add(b0, mb[14]), b4); b12 = Sve2.XorRotateRight(b12.AsVector(), b0.AsVector(), 8).AsVector128(); b8 = AdvSimd.Add(b8, b12); b4 = Sve2.XorRotateRight(b4.AsVector(), b8.AsVector(), 7).AsVector128();
                b1 = AdvSimd.Add(AdvSimd.Add(b1, mb[11]), b5); b13 = Sve2.XorRotateRight(b13.AsVector(), b1.AsVector(), 16).AsVector128(); b9 = AdvSimd.Add(b9, b13); b5 = Sve2.XorRotateRight(b5.AsVector(), b9.AsVector(), 12).AsVector128();
                b1 = AdvSimd.Add(AdvSimd.Add(b1, mb[5]), b5); b13 = Sve2.XorRotateRight(b13.AsVector(), b1.AsVector(), 8).AsVector128(); b9 = AdvSimd.Add(b9, b13); b5 = Sve2.XorRotateRight(b5.AsVector(), b9.AsVector(), 7).AsVector128();
                b2 = AdvSimd.Add(AdvSimd.Add(b2, mb[8]), b6); b14 = Sve2.XorRotateRight(b14.AsVector(), b2.AsVector(), 16).AsVector128(); b10 = AdvSimd.Add(b10, b14); b6 = Sve2.XorRotateRight(b6.AsVector(), b10.AsVector(), 12).AsVector128();
                b2 = AdvSimd.Add(AdvSimd.Add(b2, mb[12]), b6); b14 = Sve2.XorRotateRight(b14.AsVector(), b2.AsVector(), 8).AsVector128(); b10 = AdvSimd.Add(b10, b14); b6 = Sve2.XorRotateRight(b6.AsVector(), b10.AsVector(), 7).AsVector128();
                b3 = AdvSimd.Add(AdvSimd.Add(b3, mb[15]), b7); b15 = Sve2.XorRotateRight(b15.AsVector(), b3.AsVector(), 16).AsVector128(); b11 = AdvSimd.Add(b11, b15); b7 = Sve2.XorRotateRight(b7.AsVector(), b11.AsVector(), 12).AsVector128();
                b3 = AdvSimd.Add(AdvSimd.Add(b3, mb[1]), b7); b15 = Sve2.XorRotateRight(b15.AsVector(), b3.AsVector(), 8).AsVector128(); b11 = AdvSimd.Add(b11, b15); b7 = Sve2.XorRotateRight(b7.AsVector(), b11.AsVector(), 7).AsVector128();
                a0 = AdvSimd.Add(AdvSimd.Add(a0, mb[13]), a5); a15 = Sve2.XorRotateRight(a15.AsVector(), a0.AsVector(), 16).AsVector128(); a10 = AdvSimd.Add(a10, a15); a5 = Sve2.XorRotateRight(a5.AsVector(), a10.AsVector(), 12).AsVector128();
                a0 = AdvSimd.Add(AdvSimd.Add(a0, mb[3]), a5); a15 = Sve2.XorRotateRight(a15.AsVector(), a0.AsVector(), 8).AsVector128(); a10 = AdvSimd.Add(a10, a15); a5 = Sve2.XorRotateRight(a5.AsVector(), a10.AsVector(), 7).AsVector128();
                a1 = AdvSimd.Add(AdvSimd.Add(a1, mb[0]), a6); a12 = Sve2.XorRotateRight(a12.AsVector(), a1.AsVector(), 16).AsVector128(); a11 = AdvSimd.Add(a11, a12); a6 = Sve2.XorRotateRight(a6.AsVector(), a11.AsVector(), 12).AsVector128();
                a1 = AdvSimd.Add(AdvSimd.Add(a1, mb[10]), a6); a12 = Sve2.XorRotateRight(a12.AsVector(), a1.AsVector(), 8).AsVector128(); a11 = AdvSimd.Add(a11, a12); a6 = Sve2.XorRotateRight(a6.AsVector(), a11.AsVector(), 7).AsVector128();
                a2 = AdvSimd.Add(AdvSimd.Add(a2, mb[2]), a7); a13 = Sve2.XorRotateRight(a13.AsVector(), a2.AsVector(), 16).AsVector128(); a8 = AdvSimd.Add(a8, a13); a7 = Sve2.XorRotateRight(a7.AsVector(), a8.AsVector(), 12).AsVector128();
                a2 = AdvSimd.Add(AdvSimd.Add(a2, mb[6]), a7); a13 = Sve2.XorRotateRight(a13.AsVector(), a2.AsVector(), 8).AsVector128(); a8 = AdvSimd.Add(a8, a13); a7 = Sve2.XorRotateRight(a7.AsVector(), a8.AsVector(), 7).AsVector128();
                a3 = AdvSimd.Add(AdvSimd.Add(a3, mb[4]), a4); a14 = Sve2.XorRotateRight(a14.AsVector(), a3.AsVector(), 16).AsVector128(); a9 = AdvSimd.Add(a9, a14); a4 = Sve2.XorRotateRight(a4.AsVector(), a9.AsVector(), 12).AsVector128();
                a3 = AdvSimd.Add(AdvSimd.Add(a3, mb[7]), a4); a14 = Sve2.XorRotateRight(a14.AsVector(), a3.AsVector(), 8).AsVector128(); a9 = AdvSimd.Add(a9, a14); a4 = Sve2.XorRotateRight(a4.AsVector(), a9.AsVector(), 7).AsVector128();
                b0 = AdvSimd.Add(AdvSimd.Add(b0, mb[13]), b5); b15 = Sve2.XorRotateRight(b15.AsVector(), b0.AsVector(), 16).AsVector128(); b10 = AdvSimd.Add(b10, b15); b5 = Sve2.XorRotateRight(b5.AsVector(), b10.AsVector(), 12).AsVector128();
                b0 = AdvSimd.Add(AdvSimd.Add(b0, mb[3]), b5); b15 = Sve2.XorRotateRight(b15.AsVector(), b0.AsVector(), 8).AsVector128(); b10 = AdvSimd.Add(b10, b15); b5 = Sve2.XorRotateRight(b5.AsVector(), b10.AsVector(), 7).AsVector128();
                b1 = AdvSimd.Add(AdvSimd.Add(b1, mb[0]), b6); b12 = Sve2.XorRotateRight(b12.AsVector(), b1.AsVector(), 16).AsVector128(); b11 = AdvSimd.Add(b11, b12); b6 = Sve2.XorRotateRight(b6.AsVector(), b11.AsVector(), 12).AsVector128();
                b1 = AdvSimd.Add(AdvSimd.Add(b1, mb[10]), b6); b12 = Sve2.XorRotateRight(b12.AsVector(), b1.AsVector(), 8).AsVector128(); b11 = AdvSimd.Add(b11, b12); b6 = Sve2.XorRotateRight(b6.AsVector(), b11.AsVector(), 7).AsVector128();
                b2 = AdvSimd.Add(AdvSimd.Add(b2, mb[2]), b7); b13 = Sve2.XorRotateRight(b13.AsVector(), b2.AsVector(), 16).AsVector128(); b8 = AdvSimd.Add(b8, b13); b7 = Sve2.XorRotateRight(b7.AsVector(), b8.AsVector(), 12).AsVector128();
                b2 = AdvSimd.Add(AdvSimd.Add(b2, mb[6]), b7); b13 = Sve2.XorRotateRight(b13.AsVector(), b2.AsVector(), 8).AsVector128(); b8 = AdvSimd.Add(b8, b13); b7 = Sve2.XorRotateRight(b7.AsVector(), b8.AsVector(), 7).AsVector128();
                b3 = AdvSimd.Add(AdvSimd.Add(b3, mb[4]), b4); b14 = Sve2.XorRotateRight(b14.AsVector(), b3.AsVector(), 16).AsVector128(); b9 = AdvSimd.Add(b9, b14); b4 = Sve2.XorRotateRight(b4.AsVector(), b9.AsVector(), 12).AsVector128();
                b3 = AdvSimd.Add(AdvSimd.Add(b3, mb[7]), b4); b14 = Sve2.XorRotateRight(b14.AsVector(), b3.AsVector(), 8).AsVector128(); b9 = AdvSimd.Add(b9, b14); b4 = Sve2.XorRotateRight(b4.AsVector(), b9.AsVector(), 7).AsVector128();
                // Round 6
                a0 = AdvSimd.Add(AdvSimd.Add(a0, mb[11]), a4); a12 = Sve2.XorRotateRight(a12.AsVector(), a0.AsVector(), 16).AsVector128(); a8 = AdvSimd.Add(a8, a12); a4 = Sve2.XorRotateRight(a4.AsVector(), a8.AsVector(), 12).AsVector128();
                a0 = AdvSimd.Add(AdvSimd.Add(a0, mb[15]), a4); a12 = Sve2.XorRotateRight(a12.AsVector(), a0.AsVector(), 8).AsVector128(); a8 = AdvSimd.Add(a8, a12); a4 = Sve2.XorRotateRight(a4.AsVector(), a8.AsVector(), 7).AsVector128();
                a1 = AdvSimd.Add(AdvSimd.Add(a1, mb[5]), a5); a13 = Sve2.XorRotateRight(a13.AsVector(), a1.AsVector(), 16).AsVector128(); a9 = AdvSimd.Add(a9, a13); a5 = Sve2.XorRotateRight(a5.AsVector(), a9.AsVector(), 12).AsVector128();
                a1 = AdvSimd.Add(AdvSimd.Add(a1, mb[0]), a5); a13 = Sve2.XorRotateRight(a13.AsVector(), a1.AsVector(), 8).AsVector128(); a9 = AdvSimd.Add(a9, a13); a5 = Sve2.XorRotateRight(a5.AsVector(), a9.AsVector(), 7).AsVector128();
                a2 = AdvSimd.Add(AdvSimd.Add(a2, mb[1]), a6); a14 = Sve2.XorRotateRight(a14.AsVector(), a2.AsVector(), 16).AsVector128(); a10 = AdvSimd.Add(a10, a14); a6 = Sve2.XorRotateRight(a6.AsVector(), a10.AsVector(), 12).AsVector128();
                a2 = AdvSimd.Add(AdvSimd.Add(a2, mb[9]), a6); a14 = Sve2.XorRotateRight(a14.AsVector(), a2.AsVector(), 8).AsVector128(); a10 = AdvSimd.Add(a10, a14); a6 = Sve2.XorRotateRight(a6.AsVector(), a10.AsVector(), 7).AsVector128();
                a3 = AdvSimd.Add(AdvSimd.Add(a3, mb[8]), a7); a15 = Sve2.XorRotateRight(a15.AsVector(), a3.AsVector(), 16).AsVector128(); a11 = AdvSimd.Add(a11, a15); a7 = Sve2.XorRotateRight(a7.AsVector(), a11.AsVector(), 12).AsVector128();
                a3 = AdvSimd.Add(AdvSimd.Add(a3, mb[6]), a7); a15 = Sve2.XorRotateRight(a15.AsVector(), a3.AsVector(), 8).AsVector128(); a11 = AdvSimd.Add(a11, a15); a7 = Sve2.XorRotateRight(a7.AsVector(), a11.AsVector(), 7).AsVector128();
                b0 = AdvSimd.Add(AdvSimd.Add(b0, mb[11]), b4); b12 = Sve2.XorRotateRight(b12.AsVector(), b0.AsVector(), 16).AsVector128(); b8 = AdvSimd.Add(b8, b12); b4 = Sve2.XorRotateRight(b4.AsVector(), b8.AsVector(), 12).AsVector128();
                b0 = AdvSimd.Add(AdvSimd.Add(b0, mb[15]), b4); b12 = Sve2.XorRotateRight(b12.AsVector(), b0.AsVector(), 8).AsVector128(); b8 = AdvSimd.Add(b8, b12); b4 = Sve2.XorRotateRight(b4.AsVector(), b8.AsVector(), 7).AsVector128();
                b1 = AdvSimd.Add(AdvSimd.Add(b1, mb[5]), b5); b13 = Sve2.XorRotateRight(b13.AsVector(), b1.AsVector(), 16).AsVector128(); b9 = AdvSimd.Add(b9, b13); b5 = Sve2.XorRotateRight(b5.AsVector(), b9.AsVector(), 12).AsVector128();
                b1 = AdvSimd.Add(AdvSimd.Add(b1, mb[0]), b5); b13 = Sve2.XorRotateRight(b13.AsVector(), b1.AsVector(), 8).AsVector128(); b9 = AdvSimd.Add(b9, b13); b5 = Sve2.XorRotateRight(b5.AsVector(), b9.AsVector(), 7).AsVector128();
                b2 = AdvSimd.Add(AdvSimd.Add(b2, mb[1]), b6); b14 = Sve2.XorRotateRight(b14.AsVector(), b2.AsVector(), 16).AsVector128(); b10 = AdvSimd.Add(b10, b14); b6 = Sve2.XorRotateRight(b6.AsVector(), b10.AsVector(), 12).AsVector128();
                b2 = AdvSimd.Add(AdvSimd.Add(b2, mb[9]), b6); b14 = Sve2.XorRotateRight(b14.AsVector(), b2.AsVector(), 8).AsVector128(); b10 = AdvSimd.Add(b10, b14); b6 = Sve2.XorRotateRight(b6.AsVector(), b10.AsVector(), 7).AsVector128();
                b3 = AdvSimd.Add(AdvSimd.Add(b3, mb[8]), b7); b15 = Sve2.XorRotateRight(b15.AsVector(), b3.AsVector(), 16).AsVector128(); b11 = AdvSimd.Add(b11, b15); b7 = Sve2.XorRotateRight(b7.AsVector(), b11.AsVector(), 12).AsVector128();
                b3 = AdvSimd.Add(AdvSimd.Add(b3, mb[6]), b7); b15 = Sve2.XorRotateRight(b15.AsVector(), b3.AsVector(), 8).AsVector128(); b11 = AdvSimd.Add(b11, b15); b7 = Sve2.XorRotateRight(b7.AsVector(), b11.AsVector(), 7).AsVector128();
                a0 = AdvSimd.Add(AdvSimd.Add(a0, mb[14]), a5); a15 = Sve2.XorRotateRight(a15.AsVector(), a0.AsVector(), 16).AsVector128(); a10 = AdvSimd.Add(a10, a15); a5 = Sve2.XorRotateRight(a5.AsVector(), a10.AsVector(), 12).AsVector128();
                a0 = AdvSimd.Add(AdvSimd.Add(a0, mb[10]), a5); a15 = Sve2.XorRotateRight(a15.AsVector(), a0.AsVector(), 8).AsVector128(); a10 = AdvSimd.Add(a10, a15); a5 = Sve2.XorRotateRight(a5.AsVector(), a10.AsVector(), 7).AsVector128();
                a1 = AdvSimd.Add(AdvSimd.Add(a1, mb[2]), a6); a12 = Sve2.XorRotateRight(a12.AsVector(), a1.AsVector(), 16).AsVector128(); a11 = AdvSimd.Add(a11, a12); a6 = Sve2.XorRotateRight(a6.AsVector(), a11.AsVector(), 12).AsVector128();
                a1 = AdvSimd.Add(AdvSimd.Add(a1, mb[12]), a6); a12 = Sve2.XorRotateRight(a12.AsVector(), a1.AsVector(), 8).AsVector128(); a11 = AdvSimd.Add(a11, a12); a6 = Sve2.XorRotateRight(a6.AsVector(), a11.AsVector(), 7).AsVector128();
                a2 = AdvSimd.Add(AdvSimd.Add(a2, mb[3]), a7); a13 = Sve2.XorRotateRight(a13.AsVector(), a2.AsVector(), 16).AsVector128(); a8 = AdvSimd.Add(a8, a13); a7 = Sve2.XorRotateRight(a7.AsVector(), a8.AsVector(), 12).AsVector128();
                a2 = AdvSimd.Add(AdvSimd.Add(a2, mb[4]), a7); a13 = Sve2.XorRotateRight(a13.AsVector(), a2.AsVector(), 8).AsVector128(); a8 = AdvSimd.Add(a8, a13); a7 = Sve2.XorRotateRight(a7.AsVector(), a8.AsVector(), 7).AsVector128();
                a3 = AdvSimd.Add(AdvSimd.Add(a3, mb[7]), a4); a14 = Sve2.XorRotateRight(a14.AsVector(), a3.AsVector(), 16).AsVector128(); a9 = AdvSimd.Add(a9, a14); a4 = Sve2.XorRotateRight(a4.AsVector(), a9.AsVector(), 12).AsVector128();
                a3 = AdvSimd.Add(AdvSimd.Add(a3, mb[13]), a4); a14 = Sve2.XorRotateRight(a14.AsVector(), a3.AsVector(), 8).AsVector128(); a9 = AdvSimd.Add(a9, a14); a4 = Sve2.XorRotateRight(a4.AsVector(), a9.AsVector(), 7).AsVector128();
                b0 = AdvSimd.Add(AdvSimd.Add(b0, mb[14]), b5); b15 = Sve2.XorRotateRight(b15.AsVector(), b0.AsVector(), 16).AsVector128(); b10 = AdvSimd.Add(b10, b15); b5 = Sve2.XorRotateRight(b5.AsVector(), b10.AsVector(), 12).AsVector128();
                b0 = AdvSimd.Add(AdvSimd.Add(b0, mb[10]), b5); b15 = Sve2.XorRotateRight(b15.AsVector(), b0.AsVector(), 8).AsVector128(); b10 = AdvSimd.Add(b10, b15); b5 = Sve2.XorRotateRight(b5.AsVector(), b10.AsVector(), 7).AsVector128();
                b1 = AdvSimd.Add(AdvSimd.Add(b1, mb[2]), b6); b12 = Sve2.XorRotateRight(b12.AsVector(), b1.AsVector(), 16).AsVector128(); b11 = AdvSimd.Add(b11, b12); b6 = Sve2.XorRotateRight(b6.AsVector(), b11.AsVector(), 12).AsVector128();
                b1 = AdvSimd.Add(AdvSimd.Add(b1, mb[12]), b6); b12 = Sve2.XorRotateRight(b12.AsVector(), b1.AsVector(), 8).AsVector128(); b11 = AdvSimd.Add(b11, b12); b6 = Sve2.XorRotateRight(b6.AsVector(), b11.AsVector(), 7).AsVector128();
                b2 = AdvSimd.Add(AdvSimd.Add(b2, mb[3]), b7); b13 = Sve2.XorRotateRight(b13.AsVector(), b2.AsVector(), 16).AsVector128(); b8 = AdvSimd.Add(b8, b13); b7 = Sve2.XorRotateRight(b7.AsVector(), b8.AsVector(), 12).AsVector128();
                b2 = AdvSimd.Add(AdvSimd.Add(b2, mb[4]), b7); b13 = Sve2.XorRotateRight(b13.AsVector(), b2.AsVector(), 8).AsVector128(); b8 = AdvSimd.Add(b8, b13); b7 = Sve2.XorRotateRight(b7.AsVector(), b8.AsVector(), 7).AsVector128();
                b3 = AdvSimd.Add(AdvSimd.Add(b3, mb[7]), b4); b14 = Sve2.XorRotateRight(b14.AsVector(), b3.AsVector(), 16).AsVector128(); b9 = AdvSimd.Add(b9, b14); b4 = Sve2.XorRotateRight(b4.AsVector(), b9.AsVector(), 12).AsVector128();
                b3 = AdvSimd.Add(AdvSimd.Add(b3, mb[13]), b4); b14 = Sve2.XorRotateRight(b14.AsVector(), b3.AsVector(), 8).AsVector128(); b9 = AdvSimd.Add(b9, b14); b4 = Sve2.XorRotateRight(b4.AsVector(), b9.AsVector(), 7).AsVector128();

            ref uint blocksA = ref Unsafe.Add(ref outWords, pos / 4);
            ref uint blocksB = ref Unsafe.Add(ref blocksA, 64);
            StoreQuad(AdvSimd.Xor(a0, a8), AdvSimd.Xor(a1, a9), AdvSimd.Xor(a2, a10), AdvSimd.Xor(a3, a11), ref blocksA, 0);
            StoreQuad(AdvSimd.Xor(a4, a12), AdvSimd.Xor(a5, a13), AdvSimd.Xor(a6, a14), AdvSimd.Xor(a7, a15), ref blocksA, 4);
            StoreQuad(AdvSimd.Xor(a8, cv0), AdvSimd.Xor(a9, cv1), AdvSimd.Xor(a10, cv2), AdvSimd.Xor(a11, cv3), ref blocksA, 8);
            StoreQuad(AdvSimd.Xor(a12, cv4), AdvSimd.Xor(a13, cv5), AdvSimd.Xor(a14, cv6), AdvSimd.Xor(a15, cv7), ref blocksA, 12);
            StoreQuad(AdvSimd.Xor(b0, b8), AdvSimd.Xor(b1, b9), AdvSimd.Xor(b2, b10), AdvSimd.Xor(b3, b11), ref blocksB, 0);
            StoreQuad(AdvSimd.Xor(b4, b12), AdvSimd.Xor(b5, b13), AdvSimd.Xor(b6, b14), AdvSimd.Xor(b7, b15), ref blocksB, 4);
            StoreQuad(AdvSimd.Xor(b8, cv0), AdvSimd.Xor(b9, cv1), AdvSimd.Xor(b10, cv2), AdvSimd.Xor(b11, cv3), ref blocksB, 8);
            StoreQuad(AdvSimd.Xor(b12, cv4), AdvSimd.Xor(b13, cv5), AdvSimd.Xor(b14, cv6), AdvSimd.Xor(b15, cv7), ref blocksB, 12);
        }
    }
}
#endif
