using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Blake3.Managed.Internal;

/// <summary>Produces eight independent 64-byte root output blocks at a time.</summary>
internal static class OutputManyAvx2
{
    internal static bool IsSupported => Avx2.IsSupported;

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
        a = Avx2.Add(Avx2.Add(a, mx), b);
        d = RotateRight16(Avx2.Xor(d, a));
        c = Avx2.Add(c, d);
        b = RotateRight12(Avx2.Xor(b, c));
        a = Avx2.Add(Avx2.Add(a, my), b);
        d = RotateRight8(Avx2.Xor(d, a));
        c = Avx2.Add(c, d);
        b = RotateRight7(Avx2.Xor(b, c));
    }


    // Every lane shares the root CV and message; only the output counter differs.
    // The caller supplies a positive multiple of 512 bytes and handles partial blocks.
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void HashOutput(ReadOnlySpan<uint> inputCv, ReadOnlySpan<uint> block,
        ulong counter, uint blockLen, uint flags, Span<byte> output)
    {
        var m0 = Vector256.Create(block[0]);
        var m1 = Vector256.Create(block[1]);
        var m2 = Vector256.Create(block[2]);
        var m3 = Vector256.Create(block[3]);
        var m4 = Vector256.Create(block[4]);
        var m5 = Vector256.Create(block[5]);
        var m6 = Vector256.Create(block[6]);
        var m7 = Vector256.Create(block[7]);
        var m8 = Vector256.Create(block[8]);
        var m9 = Vector256.Create(block[9]);
        var m10 = Vector256.Create(block[10]);
        var m11 = Vector256.Create(block[11]);
        var m12 = Vector256.Create(block[12]);
        var m13 = Vector256.Create(block[13]);
        var m14 = Vector256.Create(block[14]);
        var m15 = Vector256.Create(block[15]);
        for (int pos = 0; pos < output.Length; pos += 512, counter += 8)
        {
            var s0 = Vector256.Create(inputCv[0]);
            var s1 = Vector256.Create(inputCv[1]);
            var s2 = Vector256.Create(inputCv[2]);
            var s3 = Vector256.Create(inputCv[3]);
            var s4 = Vector256.Create(inputCv[4]);
            var s5 = Vector256.Create(inputCv[5]);
            var s6 = Vector256.Create(inputCv[6]);
            var s7 = Vector256.Create(inputCv[7]);
            var s8 = Vector256.Create(Blake3Constants.Iv0);
            var s9 = Vector256.Create(Blake3Constants.Iv1);
            var s10 = Vector256.Create(Blake3Constants.Iv2);
            var s11 = Vector256.Create(Blake3Constants.Iv3);
            var s12 = Vector256.Create((uint)counter, (uint)(counter + 1),
                (uint)(counter + 2), (uint)(counter + 3), (uint)(counter + 4),
                (uint)(counter + 5), (uint)(counter + 6), (uint)(counter + 7));
            var s13 = Vector256.Create((uint)(counter >> 32), (uint)((counter + 1) >> 32),
                (uint)((counter + 2) >> 32), (uint)((counter + 3) >> 32),
                (uint)((counter + 4) >> 32), (uint)((counter + 5) >> 32),
                (uint)((counter + 6) >> 32), (uint)((counter + 7) >> 32));
            var s14 = Vector256.Create(blockLen);
            var s15 = Vector256.Create(flags | Blake3Constants.Root);
            // Round 0: 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
            G256(ref s0, ref s4, ref s8,  ref s12, m0,  m1);
            G256(ref s1, ref s5, ref s9,  ref s13, m2,  m3);
            G256(ref s2, ref s6, ref s10, ref s14, m4,  m5);
            G256(ref s3, ref s7, ref s11, ref s15, m6,  m7);
            G256(ref s0, ref s5, ref s10, ref s15, m8,  m9);
            G256(ref s1, ref s6, ref s11, ref s12, m10, m11);
            G256(ref s2, ref s7, ref s8,  ref s13, m12, m13);
            G256(ref s3, ref s4, ref s9,  ref s14, m14, m15);
            // Round 1: 2,6,3,10,7,0,4,13,1,11,12,5,9,14,15,8
            G256(ref s0, ref s4, ref s8,  ref s12, m2,  m6);
            G256(ref s1, ref s5, ref s9,  ref s13, m3,  m10);
            G256(ref s2, ref s6, ref s10, ref s14, m7,  m0);
            G256(ref s3, ref s7, ref s11, ref s15, m4,  m13);
            G256(ref s0, ref s5, ref s10, ref s15, m1,  m11);
            G256(ref s1, ref s6, ref s11, ref s12, m12, m5);
            G256(ref s2, ref s7, ref s8,  ref s13, m9,  m14);
            G256(ref s3, ref s4, ref s9,  ref s14, m15, m8);
            // Round 2: 3,4,10,12,13,2,7,14,6,5,9,0,11,15,8,1
            G256(ref s0, ref s4, ref s8,  ref s12, m3,  m4);
            G256(ref s1, ref s5, ref s9,  ref s13, m10, m12);
            G256(ref s2, ref s6, ref s10, ref s14, m13, m2);
            G256(ref s3, ref s7, ref s11, ref s15, m7,  m14);
            G256(ref s0, ref s5, ref s10, ref s15, m6,  m5);
            G256(ref s1, ref s6, ref s11, ref s12, m9,  m0);
            G256(ref s2, ref s7, ref s8,  ref s13, m11, m15);
            G256(ref s3, ref s4, ref s9,  ref s14, m8,  m1);
            // Round 3: 10,7,12,9,14,3,13,15,4,0,11,2,5,8,1,6
            G256(ref s0, ref s4, ref s8,  ref s12, m10, m7);
            G256(ref s1, ref s5, ref s9,  ref s13, m12, m9);
            G256(ref s2, ref s6, ref s10, ref s14, m14, m3);
            G256(ref s3, ref s7, ref s11, ref s15, m13, m15);
            G256(ref s0, ref s5, ref s10, ref s15, m4,  m0);
            G256(ref s1, ref s6, ref s11, ref s12, m11, m2);
            G256(ref s2, ref s7, ref s8,  ref s13, m5,  m8);
            G256(ref s3, ref s4, ref s9,  ref s14, m1,  m6);
            // Round 4: 12,13,9,11,15,10,14,8,7,2,5,3,0,1,6,4
            G256(ref s0, ref s4, ref s8,  ref s12, m12, m13);
            G256(ref s1, ref s5, ref s9,  ref s13, m9,  m11);
            G256(ref s2, ref s6, ref s10, ref s14, m15, m10);
            G256(ref s3, ref s7, ref s11, ref s15, m14, m8);
            G256(ref s0, ref s5, ref s10, ref s15, m7,  m2);
            G256(ref s1, ref s6, ref s11, ref s12, m5,  m3);
            G256(ref s2, ref s7, ref s8,  ref s13, m0,  m1);
            G256(ref s3, ref s4, ref s9,  ref s14, m6,  m4);
            // Round 5: 9,14,11,5,8,12,15,1,13,3,0,10,2,6,4,7
            G256(ref s0, ref s4, ref s8,  ref s12, m9,  m14);
            G256(ref s1, ref s5, ref s9,  ref s13, m11, m5);
            G256(ref s2, ref s6, ref s10, ref s14, m8,  m12);
            G256(ref s3, ref s7, ref s11, ref s15, m15, m1);
            G256(ref s0, ref s5, ref s10, ref s15, m13, m3);
            G256(ref s1, ref s6, ref s11, ref s12, m0,  m10);
            G256(ref s2, ref s7, ref s8,  ref s13, m2,  m6);
            G256(ref s3, ref s4, ref s9,  ref s14, m4,  m7);
            // Round 6: 11,15,5,0,1,9,8,6,14,10,2,12,3,4,7,13
            G256(ref s0, ref s4, ref s8,  ref s12, m11, m15);
            G256(ref s1, ref s5, ref s9,  ref s13, m5,  m0);
            G256(ref s2, ref s6, ref s10, ref s14, m1,  m9);
            G256(ref s3, ref s7, ref s11, ref s15, m8,  m6);
            G256(ref s0, ref s5, ref s10, ref s15, m14, m10);
            G256(ref s1, ref s6, ref s11, ref s12, m2,  m12);
            G256(ref s2, ref s7, ref s8,  ref s13, m3,  m4);
            G256(ref s3, ref s4, ref s9,  ref s14, m7,  m13);


            ref uint destination = ref MemoryMarshal.GetReference(MemoryMarshal.Cast<byte, uint>(output.Slice(pos)));
            StoreWords(
                Avx2.Xor(s0, s8),
                Avx2.Xor(s1, s9),
                Avx2.Xor(s2, s10),
                Avx2.Xor(s3, s11),
                Avx2.Xor(s4, s12),
                Avx2.Xor(s5, s13),
                Avx2.Xor(s6, s14),
                Avx2.Xor(s7, s15), ref destination);
            StoreWords(
                Avx2.Xor(s8, Vector256.Create(inputCv[0])),
                Avx2.Xor(s9, Vector256.Create(inputCv[1])),
                Avx2.Xor(s10, Vector256.Create(inputCv[2])),
                Avx2.Xor(s11, Vector256.Create(inputCv[3])),
                Avx2.Xor(s12, Vector256.Create(inputCv[4])),
                Avx2.Xor(s13, Vector256.Create(inputCv[5])),
                Avx2.Xor(s14, Vector256.Create(inputCv[6])),
                Avx2.Xor(s15, Vector256.Create(inputCv[7])), ref Unsafe.Add(ref destination, 8));
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void StoreWords(
        Vector256<uint> cv0, Vector256<uint> cv1, Vector256<uint> cv2, Vector256<uint> cv3,
        Vector256<uint> cv4, Vector256<uint> cv5, Vector256<uint> cv6, Vector256<uint> cv7,
        ref uint outRef)
    {
        // 8x8 transpose: word-major to block-major
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


        VectorCompat.Store(Avx2.Permute2x128(u0, u4, 0x20), ref outRef);       // output block 0
        VectorCompat.Store(Avx2.Permute2x128(u1, u5, 0x20), ref outRef, 16);    // output block 1
        VectorCompat.Store(Avx2.Permute2x128(u2, u6, 0x20), ref outRef, 32);   // output block 2
        VectorCompat.Store(Avx2.Permute2x128(u3, u7, 0x20), ref outRef, 48);   // output block 3
        VectorCompat.Store(Avx2.Permute2x128(u0, u4, 0x31), ref outRef, 64);   // output block 4
        VectorCompat.Store(Avx2.Permute2x128(u1, u5, 0x31), ref outRef, 80);   // output block 5
        VectorCompat.Store(Avx2.Permute2x128(u2, u6, 0x31), ref outRef, 96);   // output block 6
        VectorCompat.Store(Avx2.Permute2x128(u3, u7, 0x31), ref outRef, 112);   // output block 7
    }
}
