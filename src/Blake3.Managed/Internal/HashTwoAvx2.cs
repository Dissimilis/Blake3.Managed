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

    // The pshufb masks are written in place in these helpers, in every kernel class. A static
    // readonly field is only a JIT constant once its class is initialized, so a kernel compiled
    // first kept a class-init check and a memory load in its loop; a property instead adds a call
    // level that the inlining budget of these large kernels does not cover, and became a real
    // call per rotate (AVX2 without AVX-512 VL, Haswell, 2026-10-08: 1.4-1.9x slower).
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<uint> RotateRight16(Vector256<uint> v)
    {
#if NET8_0_OR_GREATER
        if (Avx512F.VL.IsSupported)
            return Avx512F.VL.RotateRight(v, 16);
#endif
        return Avx2.Shuffle(v.AsByte(), Vector256.Create((byte)2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13, 2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13)).AsUInt32();
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
        return Avx2.Shuffle(v.AsByte(), Vector256.Create((byte)1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12, 1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12)).AsUInt32();
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


    // NoInlining: with profile data Tier1 otherwise inlines this whole kernel into the tree's
    // entry method, which measured four to five times slower for two-chunk inputs.
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization | MethodImplOptions.NoInlining)]
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

    /// <summary>
    /// Hashes one full chunk and the partial chunk after it (1025..2047 input bytes), one in
    /// each 128-bit half, writing both chaining values to <paramref name="cvs"/>.
    /// </summary>
    /// <remarks>
    /// The partial chunk used to be hashed after the full one, so a 1025-byte input cost 17
    /// sequential compressions before the parent; here it rides in the spare lane and the whole
    /// leaf level costs the 16 compressions of the full chunk. Once the short chunk's final block
    /// is done its chaining value is latched, and the lane keeps recompressing that padded block
    /// until the full chunk finishes; those results are discarded.
    /// </remarks>
    /// <param name="input">One complete chunk followed by a nonempty partial chunk.</param>
    /// <param name="key">The eight key words.</param>
    /// <param name="counter">The first chunk's counter.</param>
    /// <param name="flags">The hashing mode flags.</param>
    /// <param name="cvs">Receives the two eight-word chaining values in chunk order.</param>
    /// <param name="prefixCv">If not empty, receives the partial chunk's chaining value before its
    /// final block -- the state <c>ChunkState</c> reaches on its own after absorbing those bytes,
    /// so an incremental Update can hand it the chunk without compressing it again.</param>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization | MethodImplOptions.NoInlining)]
    internal static void HashOneAndPartial(ReadOnlySpan<byte> input, ReadOnlySpan<uint> key,
        ulong counter, uint flags, Span<uint> cvs, Span<uint> prefixCv = default)
    {
        int partialLen = input.Length - Blake3Constants.ChunkLen;
        int lastBlock = (partialLen - 1) >> 6;                   // 0..15
        int lastLen = partialLen - (lastBlock << 6);              // 1..64

        // The short chunk's final block, zero-padded, so no lane ever reads past the input.
        Span<byte> padded = stackalloc byte[Blake3Constants.BlockLen];
        padded.Clear();
        input.Slice(Blake3Constants.ChunkLen + (lastBlock << 6), lastLen).CopyTo(padded);

        ref byte source = ref MemoryMarshal.GetReference(input);
        ref byte partial = ref Unsafe.Add(ref source, Blake3Constants.ChunkLen);
        ref byte pad = ref MemoryMarshal.GetReference(padded);
        ref uint keyRef = ref MemoryMarshal.GetReference(key);
        var key0 = Unsafe.ReadUnaligned<Vector128<uint>>(ref Unsafe.As<uint, byte>(ref keyRef));
        var key1 = Unsafe.ReadUnaligned<Vector128<uint>>(ref Unsafe.As<uint, byte>(ref Unsafe.Add(ref keyRef, 4)));
        var cv0 = Vector256.Create(key0, key0);
        var cv1 = Vector256.Create(key1, key1);
        var iv = Vector128.Create(Blake3Constants.Iv0, Blake3Constants.Iv1,
            Blake3Constants.Iv2, Blake3Constants.Iv3);
        var ivRow = Vector256.Create(iv, iv);
        Vector128<uint> partialCv0 = default, partialCv1 = default;

        for (int block = 0; block < 16; block++)
        {
            uint fullFlags = flags | (block == 0 ? Blake3Constants.ChunkStart : 0u)
                                   | (block == 15 ? Blake3Constants.ChunkEnd : 0u);
            uint partFlags;
            uint partLen;
            ref byte partSource = ref pad;
            if (block < lastBlock)
            {
                partFlags = flags | (block == 0 ? Blake3Constants.ChunkStart : 0u);
                partLen = 64;
                partSource = ref Unsafe.Add(ref partial, block << 6);
            }
            else
            {
                partFlags = flags | (lastBlock == 0 ? Blake3Constants.ChunkStart : 0u)
                                  | Blake3Constants.ChunkEnd;
                partLen = (uint)lastLen;
            }

            var r0 = cv0;
            var r1 = cv1;
            var r2 = ivRow;
            var r3 = Vector256.Create((uint)counter, (uint)(counter >> 32), 64u, fullFlags,
                (uint)(counter + 1), (uint)((counter + 1) >> 32), partLen, partFlags);
            int offset = block * 64;
            DoRoundsShuffle(ref r0, ref r1, ref r2, ref r3,
                LoadPair(ref source, offset, ref partSource, 0),
                LoadPair(ref source, offset + 16, ref partSource, 16),
                LoadPair(ref source, offset + 32, ref partSource, 32),
                LoadPair(ref source, offset + 48, ref partSource, 48));
            cv0 = Avx2.Xor(r0, r2);
            cv1 = Avx2.Xor(r1, r3);
            if (block == lastBlock)
            {
                partialCv0 = cv0.GetUpper();
                partialCv1 = cv1.GetUpper();
            }
            else if (block == lastBlock - 1 && !prefixCv.IsEmpty)
            {
                ref uint pre = ref MemoryMarshal.GetReference(prefixCv);
                VectorCompat.Store(cv0.GetUpper(), ref pre);
                VectorCompat.Store(cv1.GetUpper(), ref pre, 4);
            }
        }

        ref uint destination = ref MemoryMarshal.GetReference(cvs);
        VectorCompat.Store(cv0.GetLower(), ref destination);
        VectorCompat.Store(cv1.GetLower(), ref destination, 4);
        VectorCompat.Store(partialCv0, ref destination, 8);
        VectorCompat.Store(partialCv1, ref destination, 12);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<uint> LoadPair(ref byte low, int lowOffset, ref byte high, int highOffset) => Vector256.Create(
        Unsafe.ReadUnaligned<Vector128<uint>>(ref Unsafe.Add(ref low, lowOffset)),
        Unsafe.ReadUnaligned<Vector128<uint>>(ref Unsafe.Add(ref high, highOffset)));

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

    /// <summary>Compresses two non-root parents, one in each 128-bit half.</summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization | MethodImplOptions.NoInlining)]
    internal static void HashParents2(ReadOnlySpan<uint> children, ReadOnlySpan<uint> key,
        uint parentFlags, Span<uint> cvs)
    {
        ref uint keyRef = ref MemoryMarshal.GetReference(key);
        var key0 = VectorCompat.Load(ref keyRef);
        var key1 = VectorCompat.Load(ref keyRef, 4);
        var r0 = Vector256.Create(key0, key0);
        var r1 = Vector256.Create(key1, key1);
        var iv = Vector128.Create(Blake3Constants.Iv0, Blake3Constants.Iv1,
            Blake3Constants.Iv2, Blake3Constants.Iv3);
        var r2 = Vector256.Create(iv, iv);
        var state = Vector128.Create(0u, 0u, 64u, parentFlags);
        var r3 = Vector256.Create(state, state);
        ref uint src = ref MemoryMarshal.GetReference(children);
        DoRoundsShuffle(ref r0, ref r1, ref r2, ref r3,
            Vector256.Create(VectorCompat.Load(ref src), VectorCompat.Load(ref src, 16)),
            Vector256.Create(VectorCompat.Load(ref src, 4), VectorCompat.Load(ref src, 20)),
            Vector256.Create(VectorCompat.Load(ref src, 8), VectorCompat.Load(ref src, 24)),
            Vector256.Create(VectorCompat.Load(ref src, 12), VectorCompat.Load(ref src, 28)));
        var cv0 = Avx2.Xor(r0, r2);
        var cv1 = Avx2.Xor(r1, r3);
        ref uint destination = ref MemoryMarshal.GetReference(cvs);
        VectorCompat.Store(Avx2.Permute2x128(cv0, cv1, 0x20), ref destination);
        VectorCompat.Store(Avx2.Permute2x128(cv0, cv1, 0x31), ref destination, 8);
    }

}
