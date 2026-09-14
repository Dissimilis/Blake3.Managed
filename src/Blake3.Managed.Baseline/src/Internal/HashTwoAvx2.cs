// Uses the same Samuel Neves shuffle schedule as CompressSse41, adapted from
// Blake2Fast by Clinton Ingram (MIT License), independently in each 128-bit half.

using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Blake3.Managed.Internal;

/// <summary>
/// Hashes two full chunks, one in each 128-bit half of an AVX2 register.
/// Fills the gap below the four- and eight-chunk transposed kernels.
/// </summary>
internal static class HashTwoAvx2
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


    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void HashTwo(ReadOnlySpan<byte> input, ReadOnlySpan<uint> key,
        ulong counter, uint flags, Span<uint> cvs)
    {
        ref byte source = ref MemoryMarshal.GetReference(input);
        ref uint keyRef = ref MemoryMarshal.GetReference(key);
        var key0 = Unsafe.ReadUnaligned<Vector128<uint>>(ref Unsafe.As<uint, byte>(ref keyRef));
        var key1 = Unsafe.ReadUnaligned<Vector128<uint>>(ref Unsafe.As<uint, byte>(ref Unsafe.Add(ref keyRef, 4)));
        var cv0 = Vector256.Create(key0, key0);
        var cv1 = Vector256.Create(key1, key1);
        var iv = Vector128.Create(Blake3Constants.Iv0, Blake3Constants.Iv1,
            Blake3Constants.Iv2, Blake3Constants.Iv3);
        var ivRow = Vector256.Create(iv, iv);

        for (int block = 0; block < 16; block++)
        {
            uint blockFlags = flags | (block == 0 ? Blake3Constants.ChunkStart : 0u)
                                    | (block == 15 ? Blake3Constants.ChunkEnd : 0u);
            var r0 = cv0;
            var r1 = cv1;
            var r2 = ivRow;
            var r3 = Vector256.Create((uint)counter, (uint)(counter >> 32), 64u, blockFlags,
                (uint)(counter + 1), (uint)((counter + 1) >> 32), 64u, blockFlags);
            int offset = block * 64;
            DoRoundsShuffle(ref r0, ref r1, ref r2, ref r3,
                LoadPair(ref source, offset), LoadPair(ref source, offset + 16),
                LoadPair(ref source, offset + 32), LoadPair(ref source, offset + 48));
            cv0 = Avx2.Xor(r0, r2);
            cv1 = Avx2.Xor(r1, r3);
        }

        ref uint destination = ref MemoryMarshal.GetReference(cvs);
        VectorCompat.Store(Avx2.Permute2x128(cv0, cv1, 0x20), ref destination);
        VectorCompat.Store(Avx2.Permute2x128(cv0, cv1, 0x31), ref destination, 8);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<uint> LoadPair(ref byte source, int offset) => Vector256.Create(
        Unsafe.ReadUnaligned<Vector128<uint>>(ref Unsafe.Add(ref source, offset)),
        Unsafe.ReadUnaligned<Vector128<uint>>(ref Unsafe.Add(ref source, offset + 1024)));

    // The single-block shuffle schedule operates independently in each 128-bit half.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void DoRoundsShuffle(
        ref Vector256<uint> row0_ref, ref Vector256<uint> row1_ref,
        ref Vector256<uint> row2_ref, ref Vector256<uint> row3_ref,
        Vector256<uint> m0, Vector256<uint> m1,
        Vector256<uint> m2, Vector256<uint> m3)
    {
        var row0 = row0_ref;
        var row1 = row1_ref;
        var row2 = row2_ref;
        var row3 = row3_ref;
        Vector256<uint> b0, t0, t1, p0, p1, p2;

        // ===== ROUND 1 (identity schedule) =====
        p0 = Avx.Shuffle(m0.AsSingle(), m1.AsSingle(), 0b_10_00_10_00).AsUInt32();
        b0 = p0;
        row0 = Avx2.Add(Avx2.Add(row0, b0), row1);
        row3 = RotateRight16(Avx2.Xor(row3, row0));
        row2 = Avx2.Add(row2, row3);
        row1 = RotateRight12(Avx2.Xor(row1, row2));

        p1 = Avx.Shuffle(m0.AsSingle(), m1.AsSingle(), 0b_11_01_11_01).AsUInt32();
        b0 = p1;
        row0 = Avx2.Add(Avx2.Add(row0, b0), row1);
        row3 = RotateRight8(Avx2.Xor(row3, row0));
        row2 = Avx2.Add(row2, row3);
        row1 = RotateRight7(Avx2.Xor(row1, row2));

        row0 = Avx2.Shuffle(row0, 0b_10_01_00_11);
        row3 = Avx2.Shuffle(row3, 0b_01_00_11_10);
        row2 = Avx2.Shuffle(row2, 0b_00_11_10_01);

        t0 = Avx.Shuffle(m2.AsSingle(), m3.AsSingle(), 0b_10_00_10_00).AsUInt32();
        p2 = Avx2.Shuffle(t0, 0b_10_01_00_11);
        b0 = p2;
        row0 = Avx2.Add(Avx2.Add(row0, b0), row1);
        row3 = RotateRight16(Avx2.Xor(row3, row0));
        row2 = Avx2.Add(row2, row3);
        row1 = RotateRight12(Avx2.Xor(row1, row2));

        t0 = Avx.Shuffle(m2.AsSingle(), m3.AsSingle(), 0b_11_01_11_01).AsUInt32();
        m3 = Avx2.Shuffle(t0, 0b_10_01_00_11);
        m0 = p0;
        m1 = p1;
        m2 = p2;
        b0 = m3;
        row0 = Avx2.Add(Avx2.Add(row0, b0), row1);
        row3 = RotateRight8(Avx2.Xor(row3, row0));
        row2 = Avx2.Add(row2, row3);
        row1 = RotateRight7(Avx2.Xor(row1, row2));

        row0 = Avx2.Shuffle(row0, 0b_00_11_10_01);
        row3 = Avx2.Shuffle(row3, 0b_01_00_11_10);
        row2 = Avx2.Shuffle(row2, 0b_10_01_00_11);

        // ===== ROUND 2 =====
        t0 = Avx.Shuffle(m0.AsSingle(), m1.AsSingle(), 0b_11_01_10_01).AsUInt32();
        p0 = Avx2.Shuffle(t0, 0b_01_11_10_00);
        b0 = p0;
        row0 = Avx2.Add(Avx2.Add(row0, b0), row1);
        row3 = RotateRight16(Avx2.Xor(row3, row0));
        row2 = Avx2.Add(row2, row3);
        row1 = RotateRight12(Avx2.Xor(row1, row2));

        t0 = Avx2.UnpackHigh(m0, m2);
        t1 = Avx.Shuffle(m0.AsDouble(), m3.AsDouble(), 0b_1010).AsUInt32();
        p1 = Avx.Shuffle(t0.AsSingle(), t1.AsSingle(), 0b_11_00_01_10).AsUInt32();
        b0 = p1;
        row0 = Avx2.Add(Avx2.Add(row0, b0), row1);
        row3 = RotateRight8(Avx2.Xor(row3, row0));
        row2 = Avx2.Add(row2, row3);
        row1 = RotateRight7(Avx2.Xor(row1, row2));

        row0 = Avx2.Shuffle(row0, 0b_10_01_00_11);
        row3 = Avx2.Shuffle(row3, 0b_01_00_11_10);
        row2 = Avx2.Shuffle(row2, 0b_00_11_10_01);

        t0 = Avx2.UnpackLow(m3, m1);
        t1 = Avx.Shuffle(m2.AsSingle(), m3.AsSingle(), 0b_10_01_11_01).AsUInt32();
        p2 = Avx.Shuffle(t0.AsSingle(), t1.AsSingle(), 0b_10_01_01_00).AsUInt32();
        b0 = p2;
        row0 = Avx2.Add(Avx2.Add(row0, b0), row1);
        row3 = RotateRight16(Avx2.Xor(row3, row0));
        row2 = Avx2.Add(row2, row3);
        row1 = RotateRight12(Avx2.Xor(row1, row2));

        t0 = Avx.Shuffle(m2.AsDouble(), m1.AsDouble(), 0b_1010).AsUInt32();
        m3 = Avx.Shuffle(t1.AsSingle(), t0.AsSingle(), 0b_00_10_11_00).AsUInt32();
        m0 = p0;
        m1 = p1;
        m2 = p2;
        b0 = m3;
        row0 = Avx2.Add(Avx2.Add(row0, b0), row1);
        row3 = RotateRight8(Avx2.Xor(row3, row0));
        row2 = Avx2.Add(row2, row3);
        row1 = RotateRight7(Avx2.Xor(row1, row2));

        row0 = Avx2.Shuffle(row0, 0b_00_11_10_01);
        row3 = Avx2.Shuffle(row3, 0b_01_00_11_10);
        row2 = Avx2.Shuffle(row2, 0b_10_01_00_11);

        // ===== ROUND 3 =====
        t0 = Avx.Shuffle(m0.AsSingle(), m1.AsSingle(), 0b_11_01_10_01).AsUInt32();
        p0 = Avx2.Shuffle(t0, 0b_01_11_10_00);
        b0 = p0;
        row0 = Avx2.Add(Avx2.Add(row0, b0), row1);
        row3 = RotateRight16(Avx2.Xor(row3, row0));
        row2 = Avx2.Add(row2, row3);
        row1 = RotateRight12(Avx2.Xor(row1, row2));

        t0 = Avx2.UnpackHigh(m0, m2);
        t1 = Avx.Shuffle(m0.AsDouble(), m3.AsDouble(), 0b_1010).AsUInt32();
        p1 = Avx.Shuffle(t0.AsSingle(), t1.AsSingle(), 0b_11_00_01_10).AsUInt32();
        b0 = p1;
        row0 = Avx2.Add(Avx2.Add(row0, b0), row1);
        row3 = RotateRight8(Avx2.Xor(row3, row0));
        row2 = Avx2.Add(row2, row3);
        row1 = RotateRight7(Avx2.Xor(row1, row2));

        row0 = Avx2.Shuffle(row0, 0b_10_01_00_11);
        row3 = Avx2.Shuffle(row3, 0b_01_00_11_10);
        row2 = Avx2.Shuffle(row2, 0b_00_11_10_01);

        t0 = Avx2.UnpackLow(m3, m1);
        t1 = Avx.Shuffle(m2.AsSingle(), m3.AsSingle(), 0b_10_01_11_01).AsUInt32();
        p2 = Avx.Shuffle(t0.AsSingle(), t1.AsSingle(), 0b_10_01_01_00).AsUInt32();
        b0 = p2;
        row0 = Avx2.Add(Avx2.Add(row0, b0), row1);
        row3 = RotateRight16(Avx2.Xor(row3, row0));
        row2 = Avx2.Add(row2, row3);
        row1 = RotateRight12(Avx2.Xor(row1, row2));

        t0 = Avx.Shuffle(m2.AsDouble(), m1.AsDouble(), 0b_1010).AsUInt32();
        m3 = Avx.Shuffle(t1.AsSingle(), t0.AsSingle(), 0b_00_10_11_00).AsUInt32();
        m0 = p0;
        m1 = p1;
        m2 = p2;
        b0 = m3;
        row0 = Avx2.Add(Avx2.Add(row0, b0), row1);
        row3 = RotateRight8(Avx2.Xor(row3, row0));
        row2 = Avx2.Add(row2, row3);
        row1 = RotateRight7(Avx2.Xor(row1, row2));

        row0 = Avx2.Shuffle(row0, 0b_00_11_10_01);
        row3 = Avx2.Shuffle(row3, 0b_01_00_11_10);
        row2 = Avx2.Shuffle(row2, 0b_10_01_00_11);

        // ===== ROUND 4 =====
        t0 = Avx.Shuffle(m0.AsSingle(), m1.AsSingle(), 0b_11_01_10_01).AsUInt32();
        p0 = Avx2.Shuffle(t0, 0b_01_11_10_00);
        b0 = p0;
        row0 = Avx2.Add(Avx2.Add(row0, b0), row1);
        row3 = RotateRight16(Avx2.Xor(row3, row0));
        row2 = Avx2.Add(row2, row3);
        row1 = RotateRight12(Avx2.Xor(row1, row2));

        t0 = Avx2.UnpackHigh(m0, m2);
        t1 = Avx.Shuffle(m0.AsDouble(), m3.AsDouble(), 0b_1010).AsUInt32();
        p1 = Avx.Shuffle(t0.AsSingle(), t1.AsSingle(), 0b_11_00_01_10).AsUInt32();
        b0 = p1;
        row0 = Avx2.Add(Avx2.Add(row0, b0), row1);
        row3 = RotateRight8(Avx2.Xor(row3, row0));
        row2 = Avx2.Add(row2, row3);
        row1 = RotateRight7(Avx2.Xor(row1, row2));

        row0 = Avx2.Shuffle(row0, 0b_10_01_00_11);
        row3 = Avx2.Shuffle(row3, 0b_01_00_11_10);
        row2 = Avx2.Shuffle(row2, 0b_00_11_10_01);

        t0 = Avx2.UnpackLow(m3, m1);
        t1 = Avx.Shuffle(m2.AsSingle(), m3.AsSingle(), 0b_10_01_11_01).AsUInt32();
        p2 = Avx.Shuffle(t0.AsSingle(), t1.AsSingle(), 0b_10_01_01_00).AsUInt32();
        b0 = p2;
        row0 = Avx2.Add(Avx2.Add(row0, b0), row1);
        row3 = RotateRight16(Avx2.Xor(row3, row0));
        row2 = Avx2.Add(row2, row3);
        row1 = RotateRight12(Avx2.Xor(row1, row2));

        t0 = Avx.Shuffle(m2.AsDouble(), m1.AsDouble(), 0b_1010).AsUInt32();
        m3 = Avx.Shuffle(t1.AsSingle(), t0.AsSingle(), 0b_00_10_11_00).AsUInt32();
        m0 = p0;
        m1 = p1;
        m2 = p2;
        b0 = m3;
        row0 = Avx2.Add(Avx2.Add(row0, b0), row1);
        row3 = RotateRight8(Avx2.Xor(row3, row0));
        row2 = Avx2.Add(row2, row3);
        row1 = RotateRight7(Avx2.Xor(row1, row2));

        row0 = Avx2.Shuffle(row0, 0b_00_11_10_01);
        row3 = Avx2.Shuffle(row3, 0b_01_00_11_10);
        row2 = Avx2.Shuffle(row2, 0b_10_01_00_11);

        // ===== ROUND 5 =====
        t0 = Avx.Shuffle(m0.AsSingle(), m1.AsSingle(), 0b_11_01_10_01).AsUInt32();
        p0 = Avx2.Shuffle(t0, 0b_01_11_10_00);
        b0 = p0;
        row0 = Avx2.Add(Avx2.Add(row0, b0), row1);
        row3 = RotateRight16(Avx2.Xor(row3, row0));
        row2 = Avx2.Add(row2, row3);
        row1 = RotateRight12(Avx2.Xor(row1, row2));

        t0 = Avx2.UnpackHigh(m0, m2);
        t1 = Avx.Shuffle(m0.AsDouble(), m3.AsDouble(), 0b_1010).AsUInt32();
        p1 = Avx.Shuffle(t0.AsSingle(), t1.AsSingle(), 0b_11_00_01_10).AsUInt32();
        b0 = p1;
        row0 = Avx2.Add(Avx2.Add(row0, b0), row1);
        row3 = RotateRight8(Avx2.Xor(row3, row0));
        row2 = Avx2.Add(row2, row3);
        row1 = RotateRight7(Avx2.Xor(row1, row2));

        row0 = Avx2.Shuffle(row0, 0b_10_01_00_11);
        row3 = Avx2.Shuffle(row3, 0b_01_00_11_10);
        row2 = Avx2.Shuffle(row2, 0b_00_11_10_01);

        t0 = Avx2.UnpackLow(m3, m1);
        t1 = Avx.Shuffle(m2.AsSingle(), m3.AsSingle(), 0b_10_01_11_01).AsUInt32();
        p2 = Avx.Shuffle(t0.AsSingle(), t1.AsSingle(), 0b_10_01_01_00).AsUInt32();
        b0 = p2;
        row0 = Avx2.Add(Avx2.Add(row0, b0), row1);
        row3 = RotateRight16(Avx2.Xor(row3, row0));
        row2 = Avx2.Add(row2, row3);
        row1 = RotateRight12(Avx2.Xor(row1, row2));

        t0 = Avx.Shuffle(m2.AsDouble(), m1.AsDouble(), 0b_1010).AsUInt32();
        m3 = Avx.Shuffle(t1.AsSingle(), t0.AsSingle(), 0b_00_10_11_00).AsUInt32();
        m0 = p0;
        m1 = p1;
        m2 = p2;
        b0 = m3;
        row0 = Avx2.Add(Avx2.Add(row0, b0), row1);
        row3 = RotateRight8(Avx2.Xor(row3, row0));
        row2 = Avx2.Add(row2, row3);
        row1 = RotateRight7(Avx2.Xor(row1, row2));

        row0 = Avx2.Shuffle(row0, 0b_00_11_10_01);
        row3 = Avx2.Shuffle(row3, 0b_01_00_11_10);
        row2 = Avx2.Shuffle(row2, 0b_10_01_00_11);

        // ===== ROUND 6 =====
        t0 = Avx.Shuffle(m0.AsSingle(), m1.AsSingle(), 0b_11_01_10_01).AsUInt32();
        p0 = Avx2.Shuffle(t0, 0b_01_11_10_00);
        b0 = p0;
        row0 = Avx2.Add(Avx2.Add(row0, b0), row1);
        row3 = RotateRight16(Avx2.Xor(row3, row0));
        row2 = Avx2.Add(row2, row3);
        row1 = RotateRight12(Avx2.Xor(row1, row2));

        t0 = Avx2.UnpackHigh(m0, m2);
        t1 = Avx.Shuffle(m0.AsDouble(), m3.AsDouble(), 0b_1010).AsUInt32();
        p1 = Avx.Shuffle(t0.AsSingle(), t1.AsSingle(), 0b_11_00_01_10).AsUInt32();
        b0 = p1;
        row0 = Avx2.Add(Avx2.Add(row0, b0), row1);
        row3 = RotateRight8(Avx2.Xor(row3, row0));
        row2 = Avx2.Add(row2, row3);
        row1 = RotateRight7(Avx2.Xor(row1, row2));

        row0 = Avx2.Shuffle(row0, 0b_10_01_00_11);
        row3 = Avx2.Shuffle(row3, 0b_01_00_11_10);
        row2 = Avx2.Shuffle(row2, 0b_00_11_10_01);

        t0 = Avx2.UnpackLow(m3, m1);
        t1 = Avx.Shuffle(m2.AsSingle(), m3.AsSingle(), 0b_10_01_11_01).AsUInt32();
        p2 = Avx.Shuffle(t0.AsSingle(), t1.AsSingle(), 0b_10_01_01_00).AsUInt32();
        b0 = p2;
        row0 = Avx2.Add(Avx2.Add(row0, b0), row1);
        row3 = RotateRight16(Avx2.Xor(row3, row0));
        row2 = Avx2.Add(row2, row3);
        row1 = RotateRight12(Avx2.Xor(row1, row2));

        t0 = Avx.Shuffle(m2.AsDouble(), m1.AsDouble(), 0b_1010).AsUInt32();
        m3 = Avx.Shuffle(t1.AsSingle(), t0.AsSingle(), 0b_00_10_11_00).AsUInt32();
        m0 = p0;
        m1 = p1;
        m2 = p2;
        b0 = m3;
        row0 = Avx2.Add(Avx2.Add(row0, b0), row1);
        row3 = RotateRight8(Avx2.Xor(row3, row0));
        row2 = Avx2.Add(row2, row3);
        row1 = RotateRight7(Avx2.Xor(row1, row2));

        row0 = Avx2.Shuffle(row0, 0b_00_11_10_01);
        row3 = Avx2.Shuffle(row3, 0b_01_00_11_10);
        row2 = Avx2.Shuffle(row2, 0b_10_01_00_11);

        // ===== ROUND 7 (last round — skip m0/m1/m2 update) =====
        t0 = Avx.Shuffle(m0.AsSingle(), m1.AsSingle(), 0b_11_01_10_01).AsUInt32();
        p0 = Avx2.Shuffle(t0, 0b_01_11_10_00);
        b0 = p0;
        row0 = Avx2.Add(Avx2.Add(row0, b0), row1);
        row3 = RotateRight16(Avx2.Xor(row3, row0));
        row2 = Avx2.Add(row2, row3);
        row1 = RotateRight12(Avx2.Xor(row1, row2));

        t0 = Avx2.UnpackHigh(m0, m2);
        t1 = Avx.Shuffle(m0.AsDouble(), m3.AsDouble(), 0b_1010).AsUInt32();
        p1 = Avx.Shuffle(t0.AsSingle(), t1.AsSingle(), 0b_11_00_01_10).AsUInt32();
        b0 = p1;
        row0 = Avx2.Add(Avx2.Add(row0, b0), row1);
        row3 = RotateRight8(Avx2.Xor(row3, row0));
        row2 = Avx2.Add(row2, row3);
        row1 = RotateRight7(Avx2.Xor(row1, row2));

        row0 = Avx2.Shuffle(row0, 0b_10_01_00_11);
        row3 = Avx2.Shuffle(row3, 0b_01_00_11_10);
        row2 = Avx2.Shuffle(row2, 0b_00_11_10_01);

        t0 = Avx2.UnpackLow(m3, m1);
        t1 = Avx.Shuffle(m2.AsSingle(), m3.AsSingle(), 0b_10_01_11_01).AsUInt32();
        p2 = Avx.Shuffle(t0.AsSingle(), t1.AsSingle(), 0b_10_01_01_00).AsUInt32();
        b0 = p2;
        row0 = Avx2.Add(Avx2.Add(row0, b0), row1);
        row3 = RotateRight16(Avx2.Xor(row3, row0));
        row2 = Avx2.Add(row2, row3);
        row1 = RotateRight12(Avx2.Xor(row1, row2));

        t0 = Avx.Shuffle(m2.AsDouble(), m1.AsDouble(), 0b_1010).AsUInt32();
        m3 = Avx.Shuffle(t1.AsSingle(), t0.AsSingle(), 0b_00_10_11_00).AsUInt32();
        b0 = m3;
        row0 = Avx2.Add(Avx2.Add(row0, b0), row1);
        row3 = RotateRight8(Avx2.Xor(row3, row0));
        row2 = Avx2.Add(row2, row3);
        row1 = RotateRight7(Avx2.Xor(row1, row2));

        row0 = Avx2.Shuffle(row0, 0b_00_11_10_01);
        row3 = Avx2.Shuffle(row3, 0b_01_00_11_10);
        row2 = Avx2.Shuffle(row2, 0b_10_01_00_11);

        row0_ref = row0;
        row1_ref = row1;
        row2_ref = row2;
        row3_ref = row3;
    }
}
