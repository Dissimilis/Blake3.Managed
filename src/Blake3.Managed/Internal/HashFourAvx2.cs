// Uses the same Samuel Neves shuffle schedule as CompressSse41, adapted from
// Blake2Fast by Clinton Ingram (MIT License), independently in each 128-bit half.

using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Blake3.Managed.Internal;

/// <summary>
/// Hashes three or four full chunks as two interleaved copies of the <see cref="HashTwoAvx2"/>
/// schedule, one chunk in each 128-bit half of each register. A single shuffle-schedule chain
/// is latency bound, so two independent chains fill the pipeline that one leaves idle.
/// </summary>
internal static class HashFourAvx2
{
    /// <summary>
    /// Two chains need about twice the registers of one, so with only sixteen 256-bit registers
    /// (AVX2 without AVX-512 VL) they spill -- and still win: on Haswell four chunks took 0.64
    /// of the time of the 128-bit four-way kernel, three chunks 0.90 of a pair plus a single
    /// chunk (2026-10-08). The kernel used to require AVX-512 VL on that expectation, unmeasured.
    /// </summary>
    internal static bool IsSupported => Avx2.IsSupported;

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
    internal static void HashFour(ReadOnlySpan<byte> input, int numChunks, ReadOnlySpan<uint> key,
        ulong counter, uint flags, Span<uint> cvs)
    {
        if (numChunks != 3 && numChunks != 4) throw new ArgumentOutOfRangeException(nameof(numChunks));
        if (input.Length < numChunks * Blake3Constants.ChunkLen) throw new ArgumentException("Input too short.", nameof(input));
        if (cvs.Length < numChunks * 8) throw new ArgumentException("Output too short.", nameof(cvs));

        ref byte source = ref MemoryMarshal.GetReference(input);
        ref uint keyRef = ref MemoryMarshal.GetReference(key);
        var key0 = Unsafe.ReadUnaligned<Vector128<uint>>(ref Unsafe.As<uint, byte>(ref keyRef));
        var key1 = Unsafe.ReadUnaligned<Vector128<uint>>(ref Unsafe.As<uint, byte>(ref Unsafe.Add(ref keyRef, 4)));
        var cv0 = Vector256.Create(key0, key0);
        var cv1 = Vector256.Create(key1, key1);
        var dv0 = cv0;
        var dv1 = cv1;
        var iv = Vector128.Create(Blake3Constants.Iv0, Blake3Constants.Iv1,
            Blake3Constants.Iv2, Blake3Constants.Iv3);
        var ivRow = Vector256.Create(iv, iv);

        // With three chunks the second chain hashes chunk 2 in both halves; the duplicate
        // lane's result is simply not stored.
        int strideB = numChunks == 4 ? Blake3Constants.ChunkLen : 0;
        ulong counterB0 = counter + 2;
        ulong counterB1 = numChunks == 4 ? counter + 3 : counter + 2;
        ref byte sourceB = ref Unsafe.Add(ref source, 2 * Blake3Constants.ChunkLen);

        for (int block = 0; block < 16; block++)
        {
            uint blockFlags = flags | (block == 0 ? Blake3Constants.ChunkStart : 0u)
                                    | (block == 15 ? Blake3Constants.ChunkEnd : 0u);
            var r0 = cv0;
            var r1 = cv1;
            var r2 = ivRow;
            var r3 = Vector256.Create((uint)counter, (uint)(counter >> 32), 64u, blockFlags,
                (uint)(counter + 1), (uint)((counter + 1) >> 32), 64u, blockFlags);
            var s0 = dv0;
            var s1 = dv1;
            var s2 = ivRow;
            var s3 = Vector256.Create((uint)counterB0, (uint)(counterB0 >> 32), 64u, blockFlags,
                (uint)counterB1, (uint)(counterB1 >> 32), 64u, blockFlags);
            int offset = block * 64;
            DoRoundsShuffle2(ref r0, ref r1, ref r2, ref r3,
                LoadPair(ref source, offset, Blake3Constants.ChunkLen),
                LoadPair(ref source, offset + 16, Blake3Constants.ChunkLen),
                LoadPair(ref source, offset + 32, Blake3Constants.ChunkLen),
                LoadPair(ref source, offset + 48, Blake3Constants.ChunkLen),
                ref s0, ref s1, ref s2, ref s3,
                LoadPair(ref sourceB, offset, strideB),
                LoadPair(ref sourceB, offset + 16, strideB),
                LoadPair(ref sourceB, offset + 32, strideB),
                LoadPair(ref sourceB, offset + 48, strideB));
            cv0 = Avx2.Xor(r0, r2);
            cv1 = Avx2.Xor(r1, r3);
            dv0 = Avx2.Xor(s0, s2);
            dv1 = Avx2.Xor(s1, s3);
        }

        ref uint destination = ref MemoryMarshal.GetReference(cvs);
        VectorCompat.Store(Avx2.Permute2x128(cv0, cv1, 0x20), ref destination);
        VectorCompat.Store(Avx2.Permute2x128(cv0, cv1, 0x31), ref destination, 8);
        VectorCompat.Store(Avx2.Permute2x128(dv0, dv1, 0x20), ref destination, 16);
        if (numChunks == 4)
            VectorCompat.Store(Avx2.Permute2x128(dv0, dv1, 0x31), ref destination, 24);
    }

    /// <summary>
    /// Hashes two or three full chunks and the partial chunk after them, writing
    /// <paramref name="fullChunks"/> + 1 chaining values to <paramref name="cvs"/>.
    /// </summary>
    /// <remarks>
    /// The partial chunk rides in the second chain's spare lane (three full chunks) or fills the
    /// second chain (two full chunks) instead of being hashed afterwards, one block at a time.
    /// Once its final block is done its chaining value is latched, and the lane recompresses
    /// that padded block until the full chunks finish; those results are discarded.
    /// </remarks>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization | MethodImplOptions.NoInlining)]
    internal static void HashFourPartial(ReadOnlySpan<byte> input, int fullChunks, ReadOnlySpan<uint> key,
        ulong counter, uint flags, Span<uint> cvs, Span<uint> prefixCv = default)
    {
        if (fullChunks != 2 && fullChunks != 3) throw new ArgumentOutOfRangeException(nameof(fullChunks));
        int partialLen = input.Length - fullChunks * Blake3Constants.ChunkLen;
        if (partialLen <= 0 || partialLen >= Blake3Constants.ChunkLen) throw new ArgumentException("Need a partial final chunk.", nameof(input));
        if (cvs.Length < (fullChunks + 1) * 8) throw new ArgumentException("Output too short.", nameof(cvs));

        int lastBlock = (partialLen - 1) >> 6;
        int lastLen = partialLen - (lastBlock << 6);

        Span<byte> padded = stackalloc byte[Blake3Constants.BlockLen];
        padded.Clear();
        input.Slice(fullChunks * Blake3Constants.ChunkLen + (lastBlock << 6), lastLen).CopyTo(padded);

        ref byte source = ref MemoryMarshal.GetReference(input);
        ref byte partial = ref Unsafe.Add(ref source, fullChunks * Blake3Constants.ChunkLen);
        ref byte pad = ref MemoryMarshal.GetReference(padded);
        ref uint keyRef = ref MemoryMarshal.GetReference(key);
        var key0 = Unsafe.ReadUnaligned<Vector128<uint>>(ref Unsafe.As<uint, byte>(ref keyRef));
        var key1 = Unsafe.ReadUnaligned<Vector128<uint>>(ref Unsafe.As<uint, byte>(ref Unsafe.Add(ref keyRef, 4)));
        var cv0 = Vector256.Create(key0, key0);
        var cv1 = Vector256.Create(key1, key1);
        var dv0 = cv0;
        var dv1 = cv1;
        var iv = Vector128.Create(Blake3Constants.Iv0, Blake3Constants.Iv1,
            Blake3Constants.Iv2, Blake3Constants.Iv3);
        var ivRow = Vector256.Create(iv, iv);
        Vector128<uint> partialCv0 = default, partialCv1 = default;

        // Chain B's low lane: chunk 2 (three full chunks) or the partial chunk itself (two).
        bool lowIsPartial = fullChunks == 2;
        ulong counterB0 = counter + 2;
        ulong counterP = counter + (ulong)fullChunks;
        ref byte sourceB = ref Unsafe.Add(ref source, 2 * Blake3Constants.ChunkLen);

        for (int block = 0; block < 16; block++)
        {
            uint blockFlags = flags | (block == 0 ? Blake3Constants.ChunkStart : 0u)
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
            var r3 = Vector256.Create((uint)counter, (uint)(counter >> 32), 64u, blockFlags,
                (uint)(counter + 1), (uint)((counter + 1) >> 32), 64u, blockFlags);
            var s0 = dv0;
            var s1 = dv1;
            var s2 = ivRow;
            Vector256<uint> s3;
            int offset = block * 64;
            Vector256<uint> n0, n1, n2, n3;
            if (lowIsPartial)
            {
                s3 = Vector256.Create((uint)counterP, (uint)(counterP >> 32), partLen, partFlags,
                    (uint)counterP, (uint)(counterP >> 32), partLen, partFlags);
                n0 = LoadPair(ref partSource, 0, 0);
                n1 = LoadPair(ref partSource, 16, 0);
                n2 = LoadPair(ref partSource, 32, 0);
                n3 = LoadPair(ref partSource, 48, 0);
            }
            else
            {
                s3 = Vector256.Create((uint)counterB0, (uint)(counterB0 >> 32), 64u, blockFlags,
                    (uint)counterP, (uint)(counterP >> 32), partLen, partFlags);
                n0 = LoadPair(ref Unsafe.Add(ref sourceB, offset), ref partSource, 0);
                n1 = LoadPair(ref Unsafe.Add(ref sourceB, offset), ref partSource, 16);
                n2 = LoadPair(ref Unsafe.Add(ref sourceB, offset), ref partSource, 32);
                n3 = LoadPair(ref Unsafe.Add(ref sourceB, offset), ref partSource, 48);
            }

            DoRoundsShuffle2(ref r0, ref r1, ref r2, ref r3,
                LoadPair(ref source, offset, Blake3Constants.ChunkLen),
                LoadPair(ref source, offset + 16, Blake3Constants.ChunkLen),
                LoadPair(ref source, offset + 32, Blake3Constants.ChunkLen),
                LoadPair(ref source, offset + 48, Blake3Constants.ChunkLen),
                ref s0, ref s1, ref s2, ref s3, n0, n1, n2, n3);
            cv0 = Avx2.Xor(r0, r2);
            cv1 = Avx2.Xor(r1, r3);
            dv0 = Avx2.Xor(s0, s2);
            dv1 = Avx2.Xor(s1, s3);
            if (block == lastBlock)
            {
                partialCv0 = dv0.GetUpper();
                partialCv1 = dv1.GetUpper();
            }
            else if (block == lastBlock - 1 && !prefixCv.IsEmpty)
            {
                // The partial chunk's state before its final block, for an incremental Update.
                ref uint pre = ref MemoryMarshal.GetReference(prefixCv);
                VectorCompat.Store(dv0.GetUpper(), ref pre);
                VectorCompat.Store(dv1.GetUpper(), ref pre, 4);
            }
        }

        ref uint destination = ref MemoryMarshal.GetReference(cvs);
        VectorCompat.Store(Avx2.Permute2x128(cv0, cv1, 0x20), ref destination);
        VectorCompat.Store(Avx2.Permute2x128(cv0, cv1, 0x31), ref destination, 8);
        if (!lowIsPartial)
        {
            VectorCompat.Store(dv0.GetLower(), ref destination, 16);
            VectorCompat.Store(dv1.GetLower(), ref destination, 20);
        }
        int p = fullChunks * 8;
        VectorCompat.Store(partialCv0, ref destination, (nuint)p);
        VectorCompat.Store(partialCv1, ref destination, (nuint)(p + 4));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<uint> LoadPair(ref byte low, ref byte high, int offset) => Vector256.Create(
        Unsafe.ReadUnaligned<Vector128<uint>>(ref Unsafe.Add(ref low, offset)),
        Unsafe.ReadUnaligned<Vector128<uint>>(ref Unsafe.Add(ref high, offset)));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<uint> LoadPair(ref byte source, int offset, int secondStride) => Vector256.Create(
        Unsafe.ReadUnaligned<Vector128<uint>>(ref Unsafe.Add(ref source, offset)),
        Unsafe.ReadUnaligned<Vector128<uint>>(ref Unsafe.Add(ref source, offset + secondStride)));

    // Two copies of HashTwoAvx2.DoRoundsShuffle, statement by statement, so the two independent
    // chains are adjacent in program order. Chain B uses s/n/q/u/c names for row/m/p/t/b.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void DoRoundsShuffle2(
        ref Vector256<uint> row0_ref, ref Vector256<uint> row1_ref,
        ref Vector256<uint> row2_ref, ref Vector256<uint> row3_ref,
        Vector256<uint> m0, Vector256<uint> m1,
        Vector256<uint> m2, Vector256<uint> m3,
        ref Vector256<uint> s0_ref, ref Vector256<uint> s1_ref,
        ref Vector256<uint> s2_ref, ref Vector256<uint> s3_ref,
        Vector256<uint> n0, Vector256<uint> n1,
        Vector256<uint> n2, Vector256<uint> n3)
    {
        var row0 = row0_ref;
        var s0 = s0_ref;
        var row1 = row1_ref;
        var s1 = s1_ref;
        var row2 = row2_ref;
        var s2 = s2_ref;
        var row3 = row3_ref;
        var s3 = s3_ref;
        Vector256<uint> b0, t0, t1, p0, p1, p2;
        Vector256<uint> c0, u0, u1, q0, q1, q2;

        // ===== ROUND 1 (identity schedule) =====
        p0 = Avx.Shuffle(m0.AsSingle(), m1.AsSingle(), 0b_10_00_10_00).AsUInt32();
        q0 = Avx.Shuffle(n0.AsSingle(), n1.AsSingle(), 0b_10_00_10_00).AsUInt32();
        b0 = p0;
        c0 = q0;
        row0 = Avx2.Add(Avx2.Add(row0, b0), row1);
        s0 = Avx2.Add(Avx2.Add(s0, c0), s1);
        row3 = RotateRight16(Avx2.Xor(row3, row0));
        s3 = RotateRight16(Avx2.Xor(s3, s0));
        row2 = Avx2.Add(row2, row3);
        s2 = Avx2.Add(s2, s3);
        row1 = RotateRight12(Avx2.Xor(row1, row2));
        s1 = RotateRight12(Avx2.Xor(s1, s2));

        p1 = Avx.Shuffle(m0.AsSingle(), m1.AsSingle(), 0b_11_01_11_01).AsUInt32();
        q1 = Avx.Shuffle(n0.AsSingle(), n1.AsSingle(), 0b_11_01_11_01).AsUInt32();
        b0 = p1;
        c0 = q1;
        row0 = Avx2.Add(Avx2.Add(row0, b0), row1);
        s0 = Avx2.Add(Avx2.Add(s0, c0), s1);
        row3 = RotateRight8(Avx2.Xor(row3, row0));
        s3 = RotateRight8(Avx2.Xor(s3, s0));
        row2 = Avx2.Add(row2, row3);
        s2 = Avx2.Add(s2, s3);
        row1 = RotateRight7(Avx2.Xor(row1, row2));
        s1 = RotateRight7(Avx2.Xor(s1, s2));

        row0 = Avx2.Shuffle(row0, 0b_10_01_00_11);
        s0 = Avx2.Shuffle(s0, 0b_10_01_00_11);
        row3 = Avx2.Shuffle(row3, 0b_01_00_11_10);
        s3 = Avx2.Shuffle(s3, 0b_01_00_11_10);
        row2 = Avx2.Shuffle(row2, 0b_00_11_10_01);
        s2 = Avx2.Shuffle(s2, 0b_00_11_10_01);

        t0 = Avx.Shuffle(m2.AsSingle(), m3.AsSingle(), 0b_10_00_10_00).AsUInt32();
        u0 = Avx.Shuffle(n2.AsSingle(), n3.AsSingle(), 0b_10_00_10_00).AsUInt32();
        p2 = Avx2.Shuffle(t0, 0b_10_01_00_11);
        q2 = Avx2.Shuffle(u0, 0b_10_01_00_11);
        b0 = p2;
        c0 = q2;
        row0 = Avx2.Add(Avx2.Add(row0, b0), row1);
        s0 = Avx2.Add(Avx2.Add(s0, c0), s1);
        row3 = RotateRight16(Avx2.Xor(row3, row0));
        s3 = RotateRight16(Avx2.Xor(s3, s0));
        row2 = Avx2.Add(row2, row3);
        s2 = Avx2.Add(s2, s3);
        row1 = RotateRight12(Avx2.Xor(row1, row2));
        s1 = RotateRight12(Avx2.Xor(s1, s2));

        t0 = Avx.Shuffle(m2.AsSingle(), m3.AsSingle(), 0b_11_01_11_01).AsUInt32();
        u0 = Avx.Shuffle(n2.AsSingle(), n3.AsSingle(), 0b_11_01_11_01).AsUInt32();
        m3 = Avx2.Shuffle(t0, 0b_10_01_00_11);
        n3 = Avx2.Shuffle(u0, 0b_10_01_00_11);
        m0 = p0;
        n0 = q0;
        m1 = p1;
        n1 = q1;
        m2 = p2;
        n2 = q2;
        b0 = m3;
        c0 = n3;
        row0 = Avx2.Add(Avx2.Add(row0, b0), row1);
        s0 = Avx2.Add(Avx2.Add(s0, c0), s1);
        row3 = RotateRight8(Avx2.Xor(row3, row0));
        s3 = RotateRight8(Avx2.Xor(s3, s0));
        row2 = Avx2.Add(row2, row3);
        s2 = Avx2.Add(s2, s3);
        row1 = RotateRight7(Avx2.Xor(row1, row2));
        s1 = RotateRight7(Avx2.Xor(s1, s2));

        row0 = Avx2.Shuffle(row0, 0b_00_11_10_01);
        s0 = Avx2.Shuffle(s0, 0b_00_11_10_01);
        row3 = Avx2.Shuffle(row3, 0b_01_00_11_10);
        s3 = Avx2.Shuffle(s3, 0b_01_00_11_10);
        row2 = Avx2.Shuffle(row2, 0b_10_01_00_11);
        s2 = Avx2.Shuffle(s2, 0b_10_01_00_11);

        // ===== ROUND 2 =====
        t0 = Avx.Shuffle(m0.AsSingle(), m1.AsSingle(), 0b_11_01_10_01).AsUInt32();
        u0 = Avx.Shuffle(n0.AsSingle(), n1.AsSingle(), 0b_11_01_10_01).AsUInt32();
        p0 = Avx2.Shuffle(t0, 0b_01_11_10_00);
        q0 = Avx2.Shuffle(u0, 0b_01_11_10_00);
        b0 = p0;
        c0 = q0;
        row0 = Avx2.Add(Avx2.Add(row0, b0), row1);
        s0 = Avx2.Add(Avx2.Add(s0, c0), s1);
        row3 = RotateRight16(Avx2.Xor(row3, row0));
        s3 = RotateRight16(Avx2.Xor(s3, s0));
        row2 = Avx2.Add(row2, row3);
        s2 = Avx2.Add(s2, s3);
        row1 = RotateRight12(Avx2.Xor(row1, row2));
        s1 = RotateRight12(Avx2.Xor(s1, s2));

        t0 = Avx2.UnpackHigh(m0, m2);
        u0 = Avx2.UnpackHigh(n0, n2);
        t1 = Avx.Shuffle(m0.AsDouble(), m3.AsDouble(), 0b_1010).AsUInt32();
        u1 = Avx.Shuffle(n0.AsDouble(), n3.AsDouble(), 0b_1010).AsUInt32();
        p1 = Avx.Shuffle(t0.AsSingle(), t1.AsSingle(), 0b_11_00_01_10).AsUInt32();
        q1 = Avx.Shuffle(u0.AsSingle(), u1.AsSingle(), 0b_11_00_01_10).AsUInt32();
        b0 = p1;
        c0 = q1;
        row0 = Avx2.Add(Avx2.Add(row0, b0), row1);
        s0 = Avx2.Add(Avx2.Add(s0, c0), s1);
        row3 = RotateRight8(Avx2.Xor(row3, row0));
        s3 = RotateRight8(Avx2.Xor(s3, s0));
        row2 = Avx2.Add(row2, row3);
        s2 = Avx2.Add(s2, s3);
        row1 = RotateRight7(Avx2.Xor(row1, row2));
        s1 = RotateRight7(Avx2.Xor(s1, s2));

        row0 = Avx2.Shuffle(row0, 0b_10_01_00_11);
        s0 = Avx2.Shuffle(s0, 0b_10_01_00_11);
        row3 = Avx2.Shuffle(row3, 0b_01_00_11_10);
        s3 = Avx2.Shuffle(s3, 0b_01_00_11_10);
        row2 = Avx2.Shuffle(row2, 0b_00_11_10_01);
        s2 = Avx2.Shuffle(s2, 0b_00_11_10_01);

        t0 = Avx2.UnpackLow(m3, m1);
        u0 = Avx2.UnpackLow(n3, n1);
        t1 = Avx.Shuffle(m2.AsSingle(), m3.AsSingle(), 0b_10_01_11_01).AsUInt32();
        u1 = Avx.Shuffle(n2.AsSingle(), n3.AsSingle(), 0b_10_01_11_01).AsUInt32();
        p2 = Avx.Shuffle(t0.AsSingle(), t1.AsSingle(), 0b_10_01_01_00).AsUInt32();
        q2 = Avx.Shuffle(u0.AsSingle(), u1.AsSingle(), 0b_10_01_01_00).AsUInt32();
        b0 = p2;
        c0 = q2;
        row0 = Avx2.Add(Avx2.Add(row0, b0), row1);
        s0 = Avx2.Add(Avx2.Add(s0, c0), s1);
        row3 = RotateRight16(Avx2.Xor(row3, row0));
        s3 = RotateRight16(Avx2.Xor(s3, s0));
        row2 = Avx2.Add(row2, row3);
        s2 = Avx2.Add(s2, s3);
        row1 = RotateRight12(Avx2.Xor(row1, row2));
        s1 = RotateRight12(Avx2.Xor(s1, s2));

        t0 = Avx.Shuffle(m2.AsDouble(), m1.AsDouble(), 0b_1010).AsUInt32();
        u0 = Avx.Shuffle(n2.AsDouble(), n1.AsDouble(), 0b_1010).AsUInt32();
        m3 = Avx.Shuffle(t1.AsSingle(), t0.AsSingle(), 0b_00_10_11_00).AsUInt32();
        n3 = Avx.Shuffle(u1.AsSingle(), u0.AsSingle(), 0b_00_10_11_00).AsUInt32();
        m0 = p0;
        n0 = q0;
        m1 = p1;
        n1 = q1;
        m2 = p2;
        n2 = q2;
        b0 = m3;
        c0 = n3;
        row0 = Avx2.Add(Avx2.Add(row0, b0), row1);
        s0 = Avx2.Add(Avx2.Add(s0, c0), s1);
        row3 = RotateRight8(Avx2.Xor(row3, row0));
        s3 = RotateRight8(Avx2.Xor(s3, s0));
        row2 = Avx2.Add(row2, row3);
        s2 = Avx2.Add(s2, s3);
        row1 = RotateRight7(Avx2.Xor(row1, row2));
        s1 = RotateRight7(Avx2.Xor(s1, s2));

        row0 = Avx2.Shuffle(row0, 0b_00_11_10_01);
        s0 = Avx2.Shuffle(s0, 0b_00_11_10_01);
        row3 = Avx2.Shuffle(row3, 0b_01_00_11_10);
        s3 = Avx2.Shuffle(s3, 0b_01_00_11_10);
        row2 = Avx2.Shuffle(row2, 0b_10_01_00_11);
        s2 = Avx2.Shuffle(s2, 0b_10_01_00_11);

        // ===== ROUND 3 =====
        t0 = Avx.Shuffle(m0.AsSingle(), m1.AsSingle(), 0b_11_01_10_01).AsUInt32();
        u0 = Avx.Shuffle(n0.AsSingle(), n1.AsSingle(), 0b_11_01_10_01).AsUInt32();
        p0 = Avx2.Shuffle(t0, 0b_01_11_10_00);
        q0 = Avx2.Shuffle(u0, 0b_01_11_10_00);
        b0 = p0;
        c0 = q0;
        row0 = Avx2.Add(Avx2.Add(row0, b0), row1);
        s0 = Avx2.Add(Avx2.Add(s0, c0), s1);
        row3 = RotateRight16(Avx2.Xor(row3, row0));
        s3 = RotateRight16(Avx2.Xor(s3, s0));
        row2 = Avx2.Add(row2, row3);
        s2 = Avx2.Add(s2, s3);
        row1 = RotateRight12(Avx2.Xor(row1, row2));
        s1 = RotateRight12(Avx2.Xor(s1, s2));

        t0 = Avx2.UnpackHigh(m0, m2);
        u0 = Avx2.UnpackHigh(n0, n2);
        t1 = Avx.Shuffle(m0.AsDouble(), m3.AsDouble(), 0b_1010).AsUInt32();
        u1 = Avx.Shuffle(n0.AsDouble(), n3.AsDouble(), 0b_1010).AsUInt32();
        p1 = Avx.Shuffle(t0.AsSingle(), t1.AsSingle(), 0b_11_00_01_10).AsUInt32();
        q1 = Avx.Shuffle(u0.AsSingle(), u1.AsSingle(), 0b_11_00_01_10).AsUInt32();
        b0 = p1;
        c0 = q1;
        row0 = Avx2.Add(Avx2.Add(row0, b0), row1);
        s0 = Avx2.Add(Avx2.Add(s0, c0), s1);
        row3 = RotateRight8(Avx2.Xor(row3, row0));
        s3 = RotateRight8(Avx2.Xor(s3, s0));
        row2 = Avx2.Add(row2, row3);
        s2 = Avx2.Add(s2, s3);
        row1 = RotateRight7(Avx2.Xor(row1, row2));
        s1 = RotateRight7(Avx2.Xor(s1, s2));

        row0 = Avx2.Shuffle(row0, 0b_10_01_00_11);
        s0 = Avx2.Shuffle(s0, 0b_10_01_00_11);
        row3 = Avx2.Shuffle(row3, 0b_01_00_11_10);
        s3 = Avx2.Shuffle(s3, 0b_01_00_11_10);
        row2 = Avx2.Shuffle(row2, 0b_00_11_10_01);
        s2 = Avx2.Shuffle(s2, 0b_00_11_10_01);

        t0 = Avx2.UnpackLow(m3, m1);
        u0 = Avx2.UnpackLow(n3, n1);
        t1 = Avx.Shuffle(m2.AsSingle(), m3.AsSingle(), 0b_10_01_11_01).AsUInt32();
        u1 = Avx.Shuffle(n2.AsSingle(), n3.AsSingle(), 0b_10_01_11_01).AsUInt32();
        p2 = Avx.Shuffle(t0.AsSingle(), t1.AsSingle(), 0b_10_01_01_00).AsUInt32();
        q2 = Avx.Shuffle(u0.AsSingle(), u1.AsSingle(), 0b_10_01_01_00).AsUInt32();
        b0 = p2;
        c0 = q2;
        row0 = Avx2.Add(Avx2.Add(row0, b0), row1);
        s0 = Avx2.Add(Avx2.Add(s0, c0), s1);
        row3 = RotateRight16(Avx2.Xor(row3, row0));
        s3 = RotateRight16(Avx2.Xor(s3, s0));
        row2 = Avx2.Add(row2, row3);
        s2 = Avx2.Add(s2, s3);
        row1 = RotateRight12(Avx2.Xor(row1, row2));
        s1 = RotateRight12(Avx2.Xor(s1, s2));

        t0 = Avx.Shuffle(m2.AsDouble(), m1.AsDouble(), 0b_1010).AsUInt32();
        u0 = Avx.Shuffle(n2.AsDouble(), n1.AsDouble(), 0b_1010).AsUInt32();
        m3 = Avx.Shuffle(t1.AsSingle(), t0.AsSingle(), 0b_00_10_11_00).AsUInt32();
        n3 = Avx.Shuffle(u1.AsSingle(), u0.AsSingle(), 0b_00_10_11_00).AsUInt32();
        m0 = p0;
        n0 = q0;
        m1 = p1;
        n1 = q1;
        m2 = p2;
        n2 = q2;
        b0 = m3;
        c0 = n3;
        row0 = Avx2.Add(Avx2.Add(row0, b0), row1);
        s0 = Avx2.Add(Avx2.Add(s0, c0), s1);
        row3 = RotateRight8(Avx2.Xor(row3, row0));
        s3 = RotateRight8(Avx2.Xor(s3, s0));
        row2 = Avx2.Add(row2, row3);
        s2 = Avx2.Add(s2, s3);
        row1 = RotateRight7(Avx2.Xor(row1, row2));
        s1 = RotateRight7(Avx2.Xor(s1, s2));

        row0 = Avx2.Shuffle(row0, 0b_00_11_10_01);
        s0 = Avx2.Shuffle(s0, 0b_00_11_10_01);
        row3 = Avx2.Shuffle(row3, 0b_01_00_11_10);
        s3 = Avx2.Shuffle(s3, 0b_01_00_11_10);
        row2 = Avx2.Shuffle(row2, 0b_10_01_00_11);
        s2 = Avx2.Shuffle(s2, 0b_10_01_00_11);

        // ===== ROUND 4 =====
        t0 = Avx.Shuffle(m0.AsSingle(), m1.AsSingle(), 0b_11_01_10_01).AsUInt32();
        u0 = Avx.Shuffle(n0.AsSingle(), n1.AsSingle(), 0b_11_01_10_01).AsUInt32();
        p0 = Avx2.Shuffle(t0, 0b_01_11_10_00);
        q0 = Avx2.Shuffle(u0, 0b_01_11_10_00);
        b0 = p0;
        c0 = q0;
        row0 = Avx2.Add(Avx2.Add(row0, b0), row1);
        s0 = Avx2.Add(Avx2.Add(s0, c0), s1);
        row3 = RotateRight16(Avx2.Xor(row3, row0));
        s3 = RotateRight16(Avx2.Xor(s3, s0));
        row2 = Avx2.Add(row2, row3);
        s2 = Avx2.Add(s2, s3);
        row1 = RotateRight12(Avx2.Xor(row1, row2));
        s1 = RotateRight12(Avx2.Xor(s1, s2));

        t0 = Avx2.UnpackHigh(m0, m2);
        u0 = Avx2.UnpackHigh(n0, n2);
        t1 = Avx.Shuffle(m0.AsDouble(), m3.AsDouble(), 0b_1010).AsUInt32();
        u1 = Avx.Shuffle(n0.AsDouble(), n3.AsDouble(), 0b_1010).AsUInt32();
        p1 = Avx.Shuffle(t0.AsSingle(), t1.AsSingle(), 0b_11_00_01_10).AsUInt32();
        q1 = Avx.Shuffle(u0.AsSingle(), u1.AsSingle(), 0b_11_00_01_10).AsUInt32();
        b0 = p1;
        c0 = q1;
        row0 = Avx2.Add(Avx2.Add(row0, b0), row1);
        s0 = Avx2.Add(Avx2.Add(s0, c0), s1);
        row3 = RotateRight8(Avx2.Xor(row3, row0));
        s3 = RotateRight8(Avx2.Xor(s3, s0));
        row2 = Avx2.Add(row2, row3);
        s2 = Avx2.Add(s2, s3);
        row1 = RotateRight7(Avx2.Xor(row1, row2));
        s1 = RotateRight7(Avx2.Xor(s1, s2));

        row0 = Avx2.Shuffle(row0, 0b_10_01_00_11);
        s0 = Avx2.Shuffle(s0, 0b_10_01_00_11);
        row3 = Avx2.Shuffle(row3, 0b_01_00_11_10);
        s3 = Avx2.Shuffle(s3, 0b_01_00_11_10);
        row2 = Avx2.Shuffle(row2, 0b_00_11_10_01);
        s2 = Avx2.Shuffle(s2, 0b_00_11_10_01);

        t0 = Avx2.UnpackLow(m3, m1);
        u0 = Avx2.UnpackLow(n3, n1);
        t1 = Avx.Shuffle(m2.AsSingle(), m3.AsSingle(), 0b_10_01_11_01).AsUInt32();
        u1 = Avx.Shuffle(n2.AsSingle(), n3.AsSingle(), 0b_10_01_11_01).AsUInt32();
        p2 = Avx.Shuffle(t0.AsSingle(), t1.AsSingle(), 0b_10_01_01_00).AsUInt32();
        q2 = Avx.Shuffle(u0.AsSingle(), u1.AsSingle(), 0b_10_01_01_00).AsUInt32();
        b0 = p2;
        c0 = q2;
        row0 = Avx2.Add(Avx2.Add(row0, b0), row1);
        s0 = Avx2.Add(Avx2.Add(s0, c0), s1);
        row3 = RotateRight16(Avx2.Xor(row3, row0));
        s3 = RotateRight16(Avx2.Xor(s3, s0));
        row2 = Avx2.Add(row2, row3);
        s2 = Avx2.Add(s2, s3);
        row1 = RotateRight12(Avx2.Xor(row1, row2));
        s1 = RotateRight12(Avx2.Xor(s1, s2));

        t0 = Avx.Shuffle(m2.AsDouble(), m1.AsDouble(), 0b_1010).AsUInt32();
        u0 = Avx.Shuffle(n2.AsDouble(), n1.AsDouble(), 0b_1010).AsUInt32();
        m3 = Avx.Shuffle(t1.AsSingle(), t0.AsSingle(), 0b_00_10_11_00).AsUInt32();
        n3 = Avx.Shuffle(u1.AsSingle(), u0.AsSingle(), 0b_00_10_11_00).AsUInt32();
        m0 = p0;
        n0 = q0;
        m1 = p1;
        n1 = q1;
        m2 = p2;
        n2 = q2;
        b0 = m3;
        c0 = n3;
        row0 = Avx2.Add(Avx2.Add(row0, b0), row1);
        s0 = Avx2.Add(Avx2.Add(s0, c0), s1);
        row3 = RotateRight8(Avx2.Xor(row3, row0));
        s3 = RotateRight8(Avx2.Xor(s3, s0));
        row2 = Avx2.Add(row2, row3);
        s2 = Avx2.Add(s2, s3);
        row1 = RotateRight7(Avx2.Xor(row1, row2));
        s1 = RotateRight7(Avx2.Xor(s1, s2));

        row0 = Avx2.Shuffle(row0, 0b_00_11_10_01);
        s0 = Avx2.Shuffle(s0, 0b_00_11_10_01);
        row3 = Avx2.Shuffle(row3, 0b_01_00_11_10);
        s3 = Avx2.Shuffle(s3, 0b_01_00_11_10);
        row2 = Avx2.Shuffle(row2, 0b_10_01_00_11);
        s2 = Avx2.Shuffle(s2, 0b_10_01_00_11);

        // ===== ROUND 5 =====
        t0 = Avx.Shuffle(m0.AsSingle(), m1.AsSingle(), 0b_11_01_10_01).AsUInt32();
        u0 = Avx.Shuffle(n0.AsSingle(), n1.AsSingle(), 0b_11_01_10_01).AsUInt32();
        p0 = Avx2.Shuffle(t0, 0b_01_11_10_00);
        q0 = Avx2.Shuffle(u0, 0b_01_11_10_00);
        b0 = p0;
        c0 = q0;
        row0 = Avx2.Add(Avx2.Add(row0, b0), row1);
        s0 = Avx2.Add(Avx2.Add(s0, c0), s1);
        row3 = RotateRight16(Avx2.Xor(row3, row0));
        s3 = RotateRight16(Avx2.Xor(s3, s0));
        row2 = Avx2.Add(row2, row3);
        s2 = Avx2.Add(s2, s3);
        row1 = RotateRight12(Avx2.Xor(row1, row2));
        s1 = RotateRight12(Avx2.Xor(s1, s2));

        t0 = Avx2.UnpackHigh(m0, m2);
        u0 = Avx2.UnpackHigh(n0, n2);
        t1 = Avx.Shuffle(m0.AsDouble(), m3.AsDouble(), 0b_1010).AsUInt32();
        u1 = Avx.Shuffle(n0.AsDouble(), n3.AsDouble(), 0b_1010).AsUInt32();
        p1 = Avx.Shuffle(t0.AsSingle(), t1.AsSingle(), 0b_11_00_01_10).AsUInt32();
        q1 = Avx.Shuffle(u0.AsSingle(), u1.AsSingle(), 0b_11_00_01_10).AsUInt32();
        b0 = p1;
        c0 = q1;
        row0 = Avx2.Add(Avx2.Add(row0, b0), row1);
        s0 = Avx2.Add(Avx2.Add(s0, c0), s1);
        row3 = RotateRight8(Avx2.Xor(row3, row0));
        s3 = RotateRight8(Avx2.Xor(s3, s0));
        row2 = Avx2.Add(row2, row3);
        s2 = Avx2.Add(s2, s3);
        row1 = RotateRight7(Avx2.Xor(row1, row2));
        s1 = RotateRight7(Avx2.Xor(s1, s2));

        row0 = Avx2.Shuffle(row0, 0b_10_01_00_11);
        s0 = Avx2.Shuffle(s0, 0b_10_01_00_11);
        row3 = Avx2.Shuffle(row3, 0b_01_00_11_10);
        s3 = Avx2.Shuffle(s3, 0b_01_00_11_10);
        row2 = Avx2.Shuffle(row2, 0b_00_11_10_01);
        s2 = Avx2.Shuffle(s2, 0b_00_11_10_01);

        t0 = Avx2.UnpackLow(m3, m1);
        u0 = Avx2.UnpackLow(n3, n1);
        t1 = Avx.Shuffle(m2.AsSingle(), m3.AsSingle(), 0b_10_01_11_01).AsUInt32();
        u1 = Avx.Shuffle(n2.AsSingle(), n3.AsSingle(), 0b_10_01_11_01).AsUInt32();
        p2 = Avx.Shuffle(t0.AsSingle(), t1.AsSingle(), 0b_10_01_01_00).AsUInt32();
        q2 = Avx.Shuffle(u0.AsSingle(), u1.AsSingle(), 0b_10_01_01_00).AsUInt32();
        b0 = p2;
        c0 = q2;
        row0 = Avx2.Add(Avx2.Add(row0, b0), row1);
        s0 = Avx2.Add(Avx2.Add(s0, c0), s1);
        row3 = RotateRight16(Avx2.Xor(row3, row0));
        s3 = RotateRight16(Avx2.Xor(s3, s0));
        row2 = Avx2.Add(row2, row3);
        s2 = Avx2.Add(s2, s3);
        row1 = RotateRight12(Avx2.Xor(row1, row2));
        s1 = RotateRight12(Avx2.Xor(s1, s2));

        t0 = Avx.Shuffle(m2.AsDouble(), m1.AsDouble(), 0b_1010).AsUInt32();
        u0 = Avx.Shuffle(n2.AsDouble(), n1.AsDouble(), 0b_1010).AsUInt32();
        m3 = Avx.Shuffle(t1.AsSingle(), t0.AsSingle(), 0b_00_10_11_00).AsUInt32();
        n3 = Avx.Shuffle(u1.AsSingle(), u0.AsSingle(), 0b_00_10_11_00).AsUInt32();
        m0 = p0;
        n0 = q0;
        m1 = p1;
        n1 = q1;
        m2 = p2;
        n2 = q2;
        b0 = m3;
        c0 = n3;
        row0 = Avx2.Add(Avx2.Add(row0, b0), row1);
        s0 = Avx2.Add(Avx2.Add(s0, c0), s1);
        row3 = RotateRight8(Avx2.Xor(row3, row0));
        s3 = RotateRight8(Avx2.Xor(s3, s0));
        row2 = Avx2.Add(row2, row3);
        s2 = Avx2.Add(s2, s3);
        row1 = RotateRight7(Avx2.Xor(row1, row2));
        s1 = RotateRight7(Avx2.Xor(s1, s2));

        row0 = Avx2.Shuffle(row0, 0b_00_11_10_01);
        s0 = Avx2.Shuffle(s0, 0b_00_11_10_01);
        row3 = Avx2.Shuffle(row3, 0b_01_00_11_10);
        s3 = Avx2.Shuffle(s3, 0b_01_00_11_10);
        row2 = Avx2.Shuffle(row2, 0b_10_01_00_11);
        s2 = Avx2.Shuffle(s2, 0b_10_01_00_11);

        // ===== ROUND 6 =====
        t0 = Avx.Shuffle(m0.AsSingle(), m1.AsSingle(), 0b_11_01_10_01).AsUInt32();
        u0 = Avx.Shuffle(n0.AsSingle(), n1.AsSingle(), 0b_11_01_10_01).AsUInt32();
        p0 = Avx2.Shuffle(t0, 0b_01_11_10_00);
        q0 = Avx2.Shuffle(u0, 0b_01_11_10_00);
        b0 = p0;
        c0 = q0;
        row0 = Avx2.Add(Avx2.Add(row0, b0), row1);
        s0 = Avx2.Add(Avx2.Add(s0, c0), s1);
        row3 = RotateRight16(Avx2.Xor(row3, row0));
        s3 = RotateRight16(Avx2.Xor(s3, s0));
        row2 = Avx2.Add(row2, row3);
        s2 = Avx2.Add(s2, s3);
        row1 = RotateRight12(Avx2.Xor(row1, row2));
        s1 = RotateRight12(Avx2.Xor(s1, s2));

        t0 = Avx2.UnpackHigh(m0, m2);
        u0 = Avx2.UnpackHigh(n0, n2);
        t1 = Avx.Shuffle(m0.AsDouble(), m3.AsDouble(), 0b_1010).AsUInt32();
        u1 = Avx.Shuffle(n0.AsDouble(), n3.AsDouble(), 0b_1010).AsUInt32();
        p1 = Avx.Shuffle(t0.AsSingle(), t1.AsSingle(), 0b_11_00_01_10).AsUInt32();
        q1 = Avx.Shuffle(u0.AsSingle(), u1.AsSingle(), 0b_11_00_01_10).AsUInt32();
        b0 = p1;
        c0 = q1;
        row0 = Avx2.Add(Avx2.Add(row0, b0), row1);
        s0 = Avx2.Add(Avx2.Add(s0, c0), s1);
        row3 = RotateRight8(Avx2.Xor(row3, row0));
        s3 = RotateRight8(Avx2.Xor(s3, s0));
        row2 = Avx2.Add(row2, row3);
        s2 = Avx2.Add(s2, s3);
        row1 = RotateRight7(Avx2.Xor(row1, row2));
        s1 = RotateRight7(Avx2.Xor(s1, s2));

        row0 = Avx2.Shuffle(row0, 0b_10_01_00_11);
        s0 = Avx2.Shuffle(s0, 0b_10_01_00_11);
        row3 = Avx2.Shuffle(row3, 0b_01_00_11_10);
        s3 = Avx2.Shuffle(s3, 0b_01_00_11_10);
        row2 = Avx2.Shuffle(row2, 0b_00_11_10_01);
        s2 = Avx2.Shuffle(s2, 0b_00_11_10_01);

        t0 = Avx2.UnpackLow(m3, m1);
        u0 = Avx2.UnpackLow(n3, n1);
        t1 = Avx.Shuffle(m2.AsSingle(), m3.AsSingle(), 0b_10_01_11_01).AsUInt32();
        u1 = Avx.Shuffle(n2.AsSingle(), n3.AsSingle(), 0b_10_01_11_01).AsUInt32();
        p2 = Avx.Shuffle(t0.AsSingle(), t1.AsSingle(), 0b_10_01_01_00).AsUInt32();
        q2 = Avx.Shuffle(u0.AsSingle(), u1.AsSingle(), 0b_10_01_01_00).AsUInt32();
        b0 = p2;
        c0 = q2;
        row0 = Avx2.Add(Avx2.Add(row0, b0), row1);
        s0 = Avx2.Add(Avx2.Add(s0, c0), s1);
        row3 = RotateRight16(Avx2.Xor(row3, row0));
        s3 = RotateRight16(Avx2.Xor(s3, s0));
        row2 = Avx2.Add(row2, row3);
        s2 = Avx2.Add(s2, s3);
        row1 = RotateRight12(Avx2.Xor(row1, row2));
        s1 = RotateRight12(Avx2.Xor(s1, s2));

        t0 = Avx.Shuffle(m2.AsDouble(), m1.AsDouble(), 0b_1010).AsUInt32();
        u0 = Avx.Shuffle(n2.AsDouble(), n1.AsDouble(), 0b_1010).AsUInt32();
        m3 = Avx.Shuffle(t1.AsSingle(), t0.AsSingle(), 0b_00_10_11_00).AsUInt32();
        n3 = Avx.Shuffle(u1.AsSingle(), u0.AsSingle(), 0b_00_10_11_00).AsUInt32();
        m0 = p0;
        n0 = q0;
        m1 = p1;
        n1 = q1;
        m2 = p2;
        n2 = q2;
        b0 = m3;
        c0 = n3;
        row0 = Avx2.Add(Avx2.Add(row0, b0), row1);
        s0 = Avx2.Add(Avx2.Add(s0, c0), s1);
        row3 = RotateRight8(Avx2.Xor(row3, row0));
        s3 = RotateRight8(Avx2.Xor(s3, s0));
        row2 = Avx2.Add(row2, row3);
        s2 = Avx2.Add(s2, s3);
        row1 = RotateRight7(Avx2.Xor(row1, row2));
        s1 = RotateRight7(Avx2.Xor(s1, s2));

        row0 = Avx2.Shuffle(row0, 0b_00_11_10_01);
        s0 = Avx2.Shuffle(s0, 0b_00_11_10_01);
        row3 = Avx2.Shuffle(row3, 0b_01_00_11_10);
        s3 = Avx2.Shuffle(s3, 0b_01_00_11_10);
        row2 = Avx2.Shuffle(row2, 0b_10_01_00_11);
        s2 = Avx2.Shuffle(s2, 0b_10_01_00_11);

        // ===== ROUND 7 (last round — skip m0/m1/m2 update) =====
        t0 = Avx.Shuffle(m0.AsSingle(), m1.AsSingle(), 0b_11_01_10_01).AsUInt32();
        u0 = Avx.Shuffle(n0.AsSingle(), n1.AsSingle(), 0b_11_01_10_01).AsUInt32();
        p0 = Avx2.Shuffle(t0, 0b_01_11_10_00);
        q0 = Avx2.Shuffle(u0, 0b_01_11_10_00);
        b0 = p0;
        c0 = q0;
        row0 = Avx2.Add(Avx2.Add(row0, b0), row1);
        s0 = Avx2.Add(Avx2.Add(s0, c0), s1);
        row3 = RotateRight16(Avx2.Xor(row3, row0));
        s3 = RotateRight16(Avx2.Xor(s3, s0));
        row2 = Avx2.Add(row2, row3);
        s2 = Avx2.Add(s2, s3);
        row1 = RotateRight12(Avx2.Xor(row1, row2));
        s1 = RotateRight12(Avx2.Xor(s1, s2));

        t0 = Avx2.UnpackHigh(m0, m2);
        u0 = Avx2.UnpackHigh(n0, n2);
        t1 = Avx.Shuffle(m0.AsDouble(), m3.AsDouble(), 0b_1010).AsUInt32();
        u1 = Avx.Shuffle(n0.AsDouble(), n3.AsDouble(), 0b_1010).AsUInt32();
        p1 = Avx.Shuffle(t0.AsSingle(), t1.AsSingle(), 0b_11_00_01_10).AsUInt32();
        q1 = Avx.Shuffle(u0.AsSingle(), u1.AsSingle(), 0b_11_00_01_10).AsUInt32();
        b0 = p1;
        c0 = q1;
        row0 = Avx2.Add(Avx2.Add(row0, b0), row1);
        s0 = Avx2.Add(Avx2.Add(s0, c0), s1);
        row3 = RotateRight8(Avx2.Xor(row3, row0));
        s3 = RotateRight8(Avx2.Xor(s3, s0));
        row2 = Avx2.Add(row2, row3);
        s2 = Avx2.Add(s2, s3);
        row1 = RotateRight7(Avx2.Xor(row1, row2));
        s1 = RotateRight7(Avx2.Xor(s1, s2));

        row0 = Avx2.Shuffle(row0, 0b_10_01_00_11);
        s0 = Avx2.Shuffle(s0, 0b_10_01_00_11);
        row3 = Avx2.Shuffle(row3, 0b_01_00_11_10);
        s3 = Avx2.Shuffle(s3, 0b_01_00_11_10);
        row2 = Avx2.Shuffle(row2, 0b_00_11_10_01);
        s2 = Avx2.Shuffle(s2, 0b_00_11_10_01);

        t0 = Avx2.UnpackLow(m3, m1);
        u0 = Avx2.UnpackLow(n3, n1);
        t1 = Avx.Shuffle(m2.AsSingle(), m3.AsSingle(), 0b_10_01_11_01).AsUInt32();
        u1 = Avx.Shuffle(n2.AsSingle(), n3.AsSingle(), 0b_10_01_11_01).AsUInt32();
        p2 = Avx.Shuffle(t0.AsSingle(), t1.AsSingle(), 0b_10_01_01_00).AsUInt32();
        q2 = Avx.Shuffle(u0.AsSingle(), u1.AsSingle(), 0b_10_01_01_00).AsUInt32();
        b0 = p2;
        c0 = q2;
        row0 = Avx2.Add(Avx2.Add(row0, b0), row1);
        s0 = Avx2.Add(Avx2.Add(s0, c0), s1);
        row3 = RotateRight16(Avx2.Xor(row3, row0));
        s3 = RotateRight16(Avx2.Xor(s3, s0));
        row2 = Avx2.Add(row2, row3);
        s2 = Avx2.Add(s2, s3);
        row1 = RotateRight12(Avx2.Xor(row1, row2));
        s1 = RotateRight12(Avx2.Xor(s1, s2));

        t0 = Avx.Shuffle(m2.AsDouble(), m1.AsDouble(), 0b_1010).AsUInt32();
        u0 = Avx.Shuffle(n2.AsDouble(), n1.AsDouble(), 0b_1010).AsUInt32();
        m3 = Avx.Shuffle(t1.AsSingle(), t0.AsSingle(), 0b_00_10_11_00).AsUInt32();
        n3 = Avx.Shuffle(u1.AsSingle(), u0.AsSingle(), 0b_00_10_11_00).AsUInt32();
        b0 = m3;
        c0 = n3;
        row0 = Avx2.Add(Avx2.Add(row0, b0), row1);
        s0 = Avx2.Add(Avx2.Add(s0, c0), s1);
        row3 = RotateRight8(Avx2.Xor(row3, row0));
        s3 = RotateRight8(Avx2.Xor(s3, s0));
        row2 = Avx2.Add(row2, row3);
        s2 = Avx2.Add(s2, s3);
        row1 = RotateRight7(Avx2.Xor(row1, row2));
        s1 = RotateRight7(Avx2.Xor(s1, s2));

        row0 = Avx2.Shuffle(row0, 0b_00_11_10_01);
        s0 = Avx2.Shuffle(s0, 0b_00_11_10_01);
        row3 = Avx2.Shuffle(row3, 0b_01_00_11_10);
        s3 = Avx2.Shuffle(s3, 0b_01_00_11_10);
        row2 = Avx2.Shuffle(row2, 0b_10_01_00_11);
        s2 = Avx2.Shuffle(s2, 0b_10_01_00_11);

        row0_ref = row0;
        row1_ref = row1;
        row2_ref = row2;
        row3_ref = row3;
        s0_ref = s0;
        s1_ref = s1;
        s2_ref = s2;
        s3_ref = s3;
    }
}
