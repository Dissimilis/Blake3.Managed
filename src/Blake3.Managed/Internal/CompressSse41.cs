// Shuffle-based message permutation approach based on the Samuel Neves technique
// described in https://eprint.iacr.org/2012/275.pdf
// Adapted from Blake2Fast by Clinton Ingram (MIT License)

using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Blake3.Managed.Internal;

internal static class CompressSse41
{
    public static bool IsSupported => Sse2.IsSupported && Ssse3.IsSupported;

    private static readonly Vector128<byte> Rot16Mask = Vector128.Create(
        (byte)2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13);

    private static readonly Vector128<byte> Rot8Mask = Vector128.Create(
        (byte)1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<uint> RotateRight16(Vector128<uint> v)
    {
#if NET8_0_OR_GREATER
        if (Avx512F.VL.IsSupported)
            return Avx512F.VL.RotateRight(v, 16);
#endif
        return Ssse3.Shuffle(v.AsByte(), Rot16Mask).AsUInt32();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<uint> RotateRight12(Vector128<uint> v)
    {
#if NET8_0_OR_GREATER
        if (Avx512F.VL.IsSupported)
            return Avx512F.VL.RotateRight(v, 12);
#endif
        return Sse2.Or(Sse2.ShiftRightLogical(v, 12), Sse2.ShiftLeftLogical(v, 20));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<uint> RotateRight8(Vector128<uint> v)
    {
#if NET8_0_OR_GREATER
        if (Avx512F.VL.IsSupported)
            return Avx512F.VL.RotateRight(v, 8);
#endif
        return Ssse3.Shuffle(v.AsByte(), Rot8Mask).AsUInt32();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<uint> RotateRight7(Vector128<uint> v)
    {
#if NET8_0_OR_GREATER
        if (Avx512F.VL.IsSupported)
            return Avx512F.VL.RotateRight(v, 7);
#endif
        return Sse2.Or(Sse2.ShiftRightLogical(v, 7), Sse2.ShiftLeftLogical(v, 25));
    }

    /// <summary>
    /// Hashes a whole chunk (0..1024 bytes) in default unkeyed mode, producing the 32-byte digest.
    /// </summary>
    /// <remarks>
    /// The generic path calls the compressor once per 64-byte block, and each call reloads the
    /// chaining value from memory and stores it back afterwards. Fusing the block loop here keeps
    /// the chaining value in two registers for the whole chunk, so a 1 KB hash does 15 fewer calls
    /// and loses roughly 60 loads and stores. Everything else in the state is constant: the
    /// counter is zero, the key is the IV, and only the flags and block length vary.
    /// </remarks>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void HashChunkRoot32Iv(ReadOnlySpan<byte> input, Span<uint> output)
    {
        var ivRow = Vector128.Create(Blake3Constants.Iv0, Blake3Constants.Iv1,
            Blake3Constants.Iv2, Blake3Constants.Iv3);

        var cv0 = ivRow;
        var cv1 = Vector128.Create(Blake3Constants.Iv4, Blake3Constants.Iv5,
            Blake3Constants.Iv6, Blake3Constants.Iv7);

        ref byte src = ref MemoryMarshal.GetReference(input);
        int pos = 0;
        uint startFlag = Blake3Constants.ChunkStart;

        // Every block except the last, which must carry CHUNK_END and ROOT.
        while (input.Length - pos > Blake3Constants.BlockLen)
        {
            var row0 = cv0;
            var row1 = cv1;
            var row2 = ivRow;
            var row3 = Vector128.Create(0u, 0u, (uint)Blake3Constants.BlockLen, startFlag);

            ref byte b = ref Unsafe.Add(ref src, pos);
            DoRoundsShuffle(ref row0, ref row1, ref row2, ref row3,
                Unsafe.ReadUnaligned<Vector128<uint>>(ref b),
                Unsafe.ReadUnaligned<Vector128<uint>>(ref Unsafe.Add(ref b, 16)),
                Unsafe.ReadUnaligned<Vector128<uint>>(ref Unsafe.Add(ref b, 32)),
                Unsafe.ReadUnaligned<Vector128<uint>>(ref Unsafe.Add(ref b, 48)));

            // The next chaining value stays in registers rather than going out to memory.
            cv0 = Sse2.Xor(row0, row2);
            cv1 = Sse2.Xor(row1, row3);

            pos += Blake3Constants.BlockLen;
            startFlag = 0;
        }

        int lastLen = input.Length - pos;
        uint lastFlags = startFlag | Blake3Constants.ChunkEnd | Blake3Constants.Root;

        Vector128<uint> m0, m1, m2, m3;
        if (lastLen == Blake3Constants.BlockLen)
        {
            // Full final block: no padding needed, so read it where it lies.
            ref byte b = ref Unsafe.Add(ref src, pos);
            m0 = Unsafe.ReadUnaligned<Vector128<uint>>(ref b);
            m1 = Unsafe.ReadUnaligned<Vector128<uint>>(ref Unsafe.Add(ref b, 16));
            m2 = Unsafe.ReadUnaligned<Vector128<uint>>(ref Unsafe.Add(ref b, 32));
            m3 = Unsafe.ReadUnaligned<Vector128<uint>>(ref Unsafe.Add(ref b, 48));
        }
        else
        {
            Span<byte> padded = stackalloc byte[Blake3Constants.BlockLen];
            ref byte d = ref MemoryMarshal.GetReference(padded);
            Unsafe.WriteUnaligned(ref d, default(Vector128<byte>));
            Unsafe.WriteUnaligned(ref Unsafe.Add(ref d, 16), default(Vector128<byte>));
            Unsafe.WriteUnaligned(ref Unsafe.Add(ref d, 32), default(Vector128<byte>));
            Unsafe.WriteUnaligned(ref Unsafe.Add(ref d, 48), default(Vector128<byte>));
            input.Slice(pos, lastLen).CopyTo(padded);

            m0 = Unsafe.ReadUnaligned<Vector128<uint>>(ref d);
            m1 = Unsafe.ReadUnaligned<Vector128<uint>>(ref Unsafe.Add(ref d, 16));
            m2 = Unsafe.ReadUnaligned<Vector128<uint>>(ref Unsafe.Add(ref d, 32));
            m3 = Unsafe.ReadUnaligned<Vector128<uint>>(ref Unsafe.Add(ref d, 48));
        }

        var f0 = cv0;
        var f1 = cv1;
        var f2 = ivRow;
        var f3 = Vector128.Create(0u, 0u, (uint)lastLen, lastFlags);

        DoRoundsShuffle(ref f0, ref f1, ref f2, ref f3, m0, m1, m2, m3);

        ref uint outRef = ref MemoryMarshal.GetReference(output);
        VectorCompat.Store(Sse2.Xor(f0, f2), ref outRef);
        VectorCompat.Store(Sse2.Xor(f1, f3), ref outRef, 4);
    }

    /// <summary>
    /// Root compression of a single block in default (unkeyed) mode, producing the 32-byte digest.
    /// </summary>
    /// <remarks>
    /// For an unkeyed hash of 64 bytes or less, the chaining value is still the IV, the counter is
    /// zero and the flags are fixed. Every input to the compression except the message and the
    /// block length is therefore a compile-time constant, so all four state rows are materialised
    /// from the instruction stream instead of being loaded from the IV array. That removes three
    /// memory loads and the surrounding span plumbing from a call that is under 100 ns in total.
    /// </remarks>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void CompressRootIvSingleBlock(ReadOnlySpan<uint> block, uint blockLen,
                                                 Span<uint> chainingValue)
    {
        var row0 = Vector128.Create(Blake3Constants.Iv0, Blake3Constants.Iv1,
            Blake3Constants.Iv2, Blake3Constants.Iv3);
        var row1 = Vector128.Create(Blake3Constants.Iv4, Blake3Constants.Iv5,
            Blake3Constants.Iv6, Blake3Constants.Iv7);
        var row2 = row0;
        var row3 = Vector128.Create(0u, 0u, blockLen,
            Blake3Constants.ChunkStart | Blake3Constants.ChunkEnd | Blake3Constants.Root);

        ref uint mRef = ref MemoryMarshal.GetReference(block);
        var m0 = VectorCompat.Load(ref mRef);
        var m1 = VectorCompat.Load(ref mRef, 4);
        var m2 = VectorCompat.Load(ref mRef, 8);
        var m3 = VectorCompat.Load(ref mRef, 12);

        DoRoundsShuffle(ref row0, ref row1, ref row2, ref row3, m0, m1, m2, m3);

        // Only the low eight words: for a 32-byte digest they are the whole answer.
        ref uint outRef = ref MemoryMarshal.GetReference(chainingValue);
        VectorCompat.Store(Sse2.Xor(row0, row2), ref outRef);
        VectorCompat.Store(Sse2.Xor(row1, row3), ref outRef, 4);
    }

    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void Compress(ReadOnlySpan<uint> cv, ReadOnlySpan<uint> block,
                                ulong counter, uint blockLen, uint flags,
                                Span<uint> output)
    {
        ref uint cvRef = ref MemoryMarshal.GetReference(cv);
        var row0 = VectorCompat.Load(ref cvRef);
        var row1 = VectorCompat.Load(ref cvRef, 4);

        var row2 = Vector128.Create(Blake3Constants.Iv0, Blake3Constants.Iv1,
            Blake3Constants.Iv2, Blake3Constants.Iv3);
        var row3 = Vector128.Create((uint)counter, (uint)(counter >> 32), blockLen, flags);

        ref uint mRef = ref MemoryMarshal.GetReference(block);
        var m0 = VectorCompat.Load(ref mRef);
        var m1 = VectorCompat.Load(ref mRef, 4);
        var m2 = VectorCompat.Load(ref mRef, 8);
        var m3 = VectorCompat.Load(ref mRef, 12);

        DoRoundsShuffle(ref row0, ref row1, ref row2, ref row3, m0, m1, m2, m3);

        // Post-XOR: full 16-word output
        var cv0 = VectorCompat.Load(ref cvRef);
        var cv1 = VectorCompat.Load(ref cvRef, 4);

        ref uint outRef = ref MemoryMarshal.GetReference(output);
        VectorCompat.Store(Sse2.Xor(row0, row2), ref outRef);
        VectorCompat.Store(Sse2.Xor(row1, row3), ref outRef, 4);
        VectorCompat.Store(Sse2.Xor(row2, cv0), ref outRef, 8);
        VectorCompat.Store(Sse2.Xor(row3, cv1), ref outRef, 12);
    }

    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void CompressChainingValue(ReadOnlySpan<uint> cv, ReadOnlySpan<uint> block,
                                             ulong counter, uint blockLen, uint flags,
                                             Span<uint> chainingValue)
    {
        ref uint cvRef = ref MemoryMarshal.GetReference(cv);
        var row0 = VectorCompat.Load(ref cvRef);
        var row1 = VectorCompat.Load(ref cvRef, 4);

        var row2 = Vector128.Create(Blake3Constants.Iv0, Blake3Constants.Iv1,
            Blake3Constants.Iv2, Blake3Constants.Iv3);
        var row3 = Vector128.Create((uint)counter, (uint)(counter >> 32), blockLen, flags);

        ref uint mRef = ref MemoryMarshal.GetReference(block);
        var m0 = VectorCompat.Load(ref mRef);
        var m1 = VectorCompat.Load(ref mRef, 4);
        var m2 = VectorCompat.Load(ref mRef, 8);
        var m3 = VectorCompat.Load(ref mRef, 12);

        DoRoundsShuffle(ref row0, ref row1, ref row2, ref row3, m0, m1, m2, m3);

        // Only compute first 8 words: state[i] ^ state[i+8]
        ref uint outRef = ref MemoryMarshal.GetReference(chainingValue);
        VectorCompat.Store(Sse2.Xor(row0, row2), ref outRef);
        VectorCompat.Store(Sse2.Xor(row1, row3), ref outRef, 4);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void DoRoundsShuffle(
        ref Vector128<uint> row0_ref, ref Vector128<uint> row1_ref,
        ref Vector128<uint> row2_ref, ref Vector128<uint> row3_ref,
        Vector128<uint> m0, Vector128<uint> m1,
        Vector128<uint> m2, Vector128<uint> m3)
    {
        var row0 = row0_ref;
        var row1 = row1_ref;
        var row2 = row2_ref;
        var row3 = row3_ref;
        Vector128<uint> b0, t0, t1, p0, p1, p2;

        // ===== ROUND 1 (identity schedule) =====
        p0 = Sse.Shuffle(m0.AsSingle(), m1.AsSingle(), 0b_10_00_10_00).AsUInt32();
        b0 = p0;
        row0 = Sse2.Add(Sse2.Add(row0, b0), row1);
        row3 = RotateRight16(Sse2.Xor(row3, row0));
        row2 = Sse2.Add(row2, row3);
        row1 = RotateRight12(Sse2.Xor(row1, row2));

        p1 = Sse.Shuffle(m0.AsSingle(), m1.AsSingle(), 0b_11_01_11_01).AsUInt32();
        b0 = p1;
        row0 = Sse2.Add(Sse2.Add(row0, b0), row1);
        row3 = RotateRight8(Sse2.Xor(row3, row0));
        row2 = Sse2.Add(row2, row3);
        row1 = RotateRight7(Sse2.Xor(row1, row2));

        row0 = Sse2.Shuffle(row0, 0b_10_01_00_11);
        row3 = Sse2.Shuffle(row3, 0b_01_00_11_10);
        row2 = Sse2.Shuffle(row2, 0b_00_11_10_01);

        t0 = Sse.Shuffle(m2.AsSingle(), m3.AsSingle(), 0b_10_00_10_00).AsUInt32();
        p2 = Sse2.Shuffle(t0, 0b_10_01_00_11);
        b0 = p2;
        row0 = Sse2.Add(Sse2.Add(row0, b0), row1);
        row3 = RotateRight16(Sse2.Xor(row3, row0));
        row2 = Sse2.Add(row2, row3);
        row1 = RotateRight12(Sse2.Xor(row1, row2));

        t0 = Sse.Shuffle(m2.AsSingle(), m3.AsSingle(), 0b_11_01_11_01).AsUInt32();
        m3 = Sse2.Shuffle(t0, 0b_10_01_00_11);
        m0 = p0;
        m1 = p1;
        m2 = p2;
        b0 = m3;
        row0 = Sse2.Add(Sse2.Add(row0, b0), row1);
        row3 = RotateRight8(Sse2.Xor(row3, row0));
        row2 = Sse2.Add(row2, row3);
        row1 = RotateRight7(Sse2.Xor(row1, row2));

        row0 = Sse2.Shuffle(row0, 0b_00_11_10_01);
        row3 = Sse2.Shuffle(row3, 0b_01_00_11_10);
        row2 = Sse2.Shuffle(row2, 0b_10_01_00_11);

        // ===== ROUND 2 =====
        t0 = Sse.Shuffle(m0.AsSingle(), m1.AsSingle(), 0b_11_01_10_01).AsUInt32();
        p0 = Sse2.Shuffle(t0, 0b_01_11_10_00);
        b0 = p0;
        row0 = Sse2.Add(Sse2.Add(row0, b0), row1);
        row3 = RotateRight16(Sse2.Xor(row3, row0));
        row2 = Sse2.Add(row2, row3);
        row1 = RotateRight12(Sse2.Xor(row1, row2));

        t0 = Sse2.UnpackHigh(m0, m2);
        t1 = Sse2.Shuffle(m0.AsDouble(), m3.AsDouble(), 0b_10).AsUInt32();
        p1 = Sse.Shuffle(t0.AsSingle(), t1.AsSingle(), 0b_11_00_01_10).AsUInt32();
        b0 = p1;
        row0 = Sse2.Add(Sse2.Add(row0, b0), row1);
        row3 = RotateRight8(Sse2.Xor(row3, row0));
        row2 = Sse2.Add(row2, row3);
        row1 = RotateRight7(Sse2.Xor(row1, row2));

        row0 = Sse2.Shuffle(row0, 0b_10_01_00_11);
        row3 = Sse2.Shuffle(row3, 0b_01_00_11_10);
        row2 = Sse2.Shuffle(row2, 0b_00_11_10_01);

        t0 = Sse2.UnpackLow(m3, m1);
        t1 = Sse.Shuffle(m2.AsSingle(), m3.AsSingle(), 0b_10_01_11_01).AsUInt32();
        p2 = Sse.Shuffle(t0.AsSingle(), t1.AsSingle(), 0b_10_01_01_00).AsUInt32();
        b0 = p2;
        row0 = Sse2.Add(Sse2.Add(row0, b0), row1);
        row3 = RotateRight16(Sse2.Xor(row3, row0));
        row2 = Sse2.Add(row2, row3);
        row1 = RotateRight12(Sse2.Xor(row1, row2));

        t0 = Sse2.Shuffle(m2.AsDouble(), m1.AsDouble(), 0b_10).AsUInt32();
        m3 = Sse.Shuffle(t1.AsSingle(), t0.AsSingle(), 0b_00_10_11_00).AsUInt32();
        m0 = p0;
        m1 = p1;
        m2 = p2;
        b0 = m3;
        row0 = Sse2.Add(Sse2.Add(row0, b0), row1);
        row3 = RotateRight8(Sse2.Xor(row3, row0));
        row2 = Sse2.Add(row2, row3);
        row1 = RotateRight7(Sse2.Xor(row1, row2));

        row0 = Sse2.Shuffle(row0, 0b_00_11_10_01);
        row3 = Sse2.Shuffle(row3, 0b_01_00_11_10);
        row2 = Sse2.Shuffle(row2, 0b_10_01_00_11);

        // ===== ROUND 3 =====
        t0 = Sse.Shuffle(m0.AsSingle(), m1.AsSingle(), 0b_11_01_10_01).AsUInt32();
        p0 = Sse2.Shuffle(t0, 0b_01_11_10_00);
        b0 = p0;
        row0 = Sse2.Add(Sse2.Add(row0, b0), row1);
        row3 = RotateRight16(Sse2.Xor(row3, row0));
        row2 = Sse2.Add(row2, row3);
        row1 = RotateRight12(Sse2.Xor(row1, row2));

        t0 = Sse2.UnpackHigh(m0, m2);
        t1 = Sse2.Shuffle(m0.AsDouble(), m3.AsDouble(), 0b_10).AsUInt32();
        p1 = Sse.Shuffle(t0.AsSingle(), t1.AsSingle(), 0b_11_00_01_10).AsUInt32();
        b0 = p1;
        row0 = Sse2.Add(Sse2.Add(row0, b0), row1);
        row3 = RotateRight8(Sse2.Xor(row3, row0));
        row2 = Sse2.Add(row2, row3);
        row1 = RotateRight7(Sse2.Xor(row1, row2));

        row0 = Sse2.Shuffle(row0, 0b_10_01_00_11);
        row3 = Sse2.Shuffle(row3, 0b_01_00_11_10);
        row2 = Sse2.Shuffle(row2, 0b_00_11_10_01);

        t0 = Sse2.UnpackLow(m3, m1);
        t1 = Sse.Shuffle(m2.AsSingle(), m3.AsSingle(), 0b_10_01_11_01).AsUInt32();
        p2 = Sse.Shuffle(t0.AsSingle(), t1.AsSingle(), 0b_10_01_01_00).AsUInt32();
        b0 = p2;
        row0 = Sse2.Add(Sse2.Add(row0, b0), row1);
        row3 = RotateRight16(Sse2.Xor(row3, row0));
        row2 = Sse2.Add(row2, row3);
        row1 = RotateRight12(Sse2.Xor(row1, row2));

        t0 = Sse2.Shuffle(m2.AsDouble(), m1.AsDouble(), 0b_10).AsUInt32();
        m3 = Sse.Shuffle(t1.AsSingle(), t0.AsSingle(), 0b_00_10_11_00).AsUInt32();
        m0 = p0;
        m1 = p1;
        m2 = p2;
        b0 = m3;
        row0 = Sse2.Add(Sse2.Add(row0, b0), row1);
        row3 = RotateRight8(Sse2.Xor(row3, row0));
        row2 = Sse2.Add(row2, row3);
        row1 = RotateRight7(Sse2.Xor(row1, row2));

        row0 = Sse2.Shuffle(row0, 0b_00_11_10_01);
        row3 = Sse2.Shuffle(row3, 0b_01_00_11_10);
        row2 = Sse2.Shuffle(row2, 0b_10_01_00_11);

        // ===== ROUND 4 =====
        t0 = Sse.Shuffle(m0.AsSingle(), m1.AsSingle(), 0b_11_01_10_01).AsUInt32();
        p0 = Sse2.Shuffle(t0, 0b_01_11_10_00);
        b0 = p0;
        row0 = Sse2.Add(Sse2.Add(row0, b0), row1);
        row3 = RotateRight16(Sse2.Xor(row3, row0));
        row2 = Sse2.Add(row2, row3);
        row1 = RotateRight12(Sse2.Xor(row1, row2));

        t0 = Sse2.UnpackHigh(m0, m2);
        t1 = Sse2.Shuffle(m0.AsDouble(), m3.AsDouble(), 0b_10).AsUInt32();
        p1 = Sse.Shuffle(t0.AsSingle(), t1.AsSingle(), 0b_11_00_01_10).AsUInt32();
        b0 = p1;
        row0 = Sse2.Add(Sse2.Add(row0, b0), row1);
        row3 = RotateRight8(Sse2.Xor(row3, row0));
        row2 = Sse2.Add(row2, row3);
        row1 = RotateRight7(Sse2.Xor(row1, row2));

        row0 = Sse2.Shuffle(row0, 0b_10_01_00_11);
        row3 = Sse2.Shuffle(row3, 0b_01_00_11_10);
        row2 = Sse2.Shuffle(row2, 0b_00_11_10_01);

        t0 = Sse2.UnpackLow(m3, m1);
        t1 = Sse.Shuffle(m2.AsSingle(), m3.AsSingle(), 0b_10_01_11_01).AsUInt32();
        p2 = Sse.Shuffle(t0.AsSingle(), t1.AsSingle(), 0b_10_01_01_00).AsUInt32();
        b0 = p2;
        row0 = Sse2.Add(Sse2.Add(row0, b0), row1);
        row3 = RotateRight16(Sse2.Xor(row3, row0));
        row2 = Sse2.Add(row2, row3);
        row1 = RotateRight12(Sse2.Xor(row1, row2));

        t0 = Sse2.Shuffle(m2.AsDouble(), m1.AsDouble(), 0b_10).AsUInt32();
        m3 = Sse.Shuffle(t1.AsSingle(), t0.AsSingle(), 0b_00_10_11_00).AsUInt32();
        m0 = p0;
        m1 = p1;
        m2 = p2;
        b0 = m3;
        row0 = Sse2.Add(Sse2.Add(row0, b0), row1);
        row3 = RotateRight8(Sse2.Xor(row3, row0));
        row2 = Sse2.Add(row2, row3);
        row1 = RotateRight7(Sse2.Xor(row1, row2));

        row0 = Sse2.Shuffle(row0, 0b_00_11_10_01);
        row3 = Sse2.Shuffle(row3, 0b_01_00_11_10);
        row2 = Sse2.Shuffle(row2, 0b_10_01_00_11);

        // ===== ROUND 5 =====
        t0 = Sse.Shuffle(m0.AsSingle(), m1.AsSingle(), 0b_11_01_10_01).AsUInt32();
        p0 = Sse2.Shuffle(t0, 0b_01_11_10_00);
        b0 = p0;
        row0 = Sse2.Add(Sse2.Add(row0, b0), row1);
        row3 = RotateRight16(Sse2.Xor(row3, row0));
        row2 = Sse2.Add(row2, row3);
        row1 = RotateRight12(Sse2.Xor(row1, row2));

        t0 = Sse2.UnpackHigh(m0, m2);
        t1 = Sse2.Shuffle(m0.AsDouble(), m3.AsDouble(), 0b_10).AsUInt32();
        p1 = Sse.Shuffle(t0.AsSingle(), t1.AsSingle(), 0b_11_00_01_10).AsUInt32();
        b0 = p1;
        row0 = Sse2.Add(Sse2.Add(row0, b0), row1);
        row3 = RotateRight8(Sse2.Xor(row3, row0));
        row2 = Sse2.Add(row2, row3);
        row1 = RotateRight7(Sse2.Xor(row1, row2));

        row0 = Sse2.Shuffle(row0, 0b_10_01_00_11);
        row3 = Sse2.Shuffle(row3, 0b_01_00_11_10);
        row2 = Sse2.Shuffle(row2, 0b_00_11_10_01);

        t0 = Sse2.UnpackLow(m3, m1);
        t1 = Sse.Shuffle(m2.AsSingle(), m3.AsSingle(), 0b_10_01_11_01).AsUInt32();
        p2 = Sse.Shuffle(t0.AsSingle(), t1.AsSingle(), 0b_10_01_01_00).AsUInt32();
        b0 = p2;
        row0 = Sse2.Add(Sse2.Add(row0, b0), row1);
        row3 = RotateRight16(Sse2.Xor(row3, row0));
        row2 = Sse2.Add(row2, row3);
        row1 = RotateRight12(Sse2.Xor(row1, row2));

        t0 = Sse2.Shuffle(m2.AsDouble(), m1.AsDouble(), 0b_10).AsUInt32();
        m3 = Sse.Shuffle(t1.AsSingle(), t0.AsSingle(), 0b_00_10_11_00).AsUInt32();
        m0 = p0;
        m1 = p1;
        m2 = p2;
        b0 = m3;
        row0 = Sse2.Add(Sse2.Add(row0, b0), row1);
        row3 = RotateRight8(Sse2.Xor(row3, row0));
        row2 = Sse2.Add(row2, row3);
        row1 = RotateRight7(Sse2.Xor(row1, row2));

        row0 = Sse2.Shuffle(row0, 0b_00_11_10_01);
        row3 = Sse2.Shuffle(row3, 0b_01_00_11_10);
        row2 = Sse2.Shuffle(row2, 0b_10_01_00_11);

        // ===== ROUND 6 =====
        t0 = Sse.Shuffle(m0.AsSingle(), m1.AsSingle(), 0b_11_01_10_01).AsUInt32();
        p0 = Sse2.Shuffle(t0, 0b_01_11_10_00);
        b0 = p0;
        row0 = Sse2.Add(Sse2.Add(row0, b0), row1);
        row3 = RotateRight16(Sse2.Xor(row3, row0));
        row2 = Sse2.Add(row2, row3);
        row1 = RotateRight12(Sse2.Xor(row1, row2));

        t0 = Sse2.UnpackHigh(m0, m2);
        t1 = Sse2.Shuffle(m0.AsDouble(), m3.AsDouble(), 0b_10).AsUInt32();
        p1 = Sse.Shuffle(t0.AsSingle(), t1.AsSingle(), 0b_11_00_01_10).AsUInt32();
        b0 = p1;
        row0 = Sse2.Add(Sse2.Add(row0, b0), row1);
        row3 = RotateRight8(Sse2.Xor(row3, row0));
        row2 = Sse2.Add(row2, row3);
        row1 = RotateRight7(Sse2.Xor(row1, row2));

        row0 = Sse2.Shuffle(row0, 0b_10_01_00_11);
        row3 = Sse2.Shuffle(row3, 0b_01_00_11_10);
        row2 = Sse2.Shuffle(row2, 0b_00_11_10_01);

        t0 = Sse2.UnpackLow(m3, m1);
        t1 = Sse.Shuffle(m2.AsSingle(), m3.AsSingle(), 0b_10_01_11_01).AsUInt32();
        p2 = Sse.Shuffle(t0.AsSingle(), t1.AsSingle(), 0b_10_01_01_00).AsUInt32();
        b0 = p2;
        row0 = Sse2.Add(Sse2.Add(row0, b0), row1);
        row3 = RotateRight16(Sse2.Xor(row3, row0));
        row2 = Sse2.Add(row2, row3);
        row1 = RotateRight12(Sse2.Xor(row1, row2));

        t0 = Sse2.Shuffle(m2.AsDouble(), m1.AsDouble(), 0b_10).AsUInt32();
        m3 = Sse.Shuffle(t1.AsSingle(), t0.AsSingle(), 0b_00_10_11_00).AsUInt32();
        m0 = p0;
        m1 = p1;
        m2 = p2;
        b0 = m3;
        row0 = Sse2.Add(Sse2.Add(row0, b0), row1);
        row3 = RotateRight8(Sse2.Xor(row3, row0));
        row2 = Sse2.Add(row2, row3);
        row1 = RotateRight7(Sse2.Xor(row1, row2));

        row0 = Sse2.Shuffle(row0, 0b_00_11_10_01);
        row3 = Sse2.Shuffle(row3, 0b_01_00_11_10);
        row2 = Sse2.Shuffle(row2, 0b_10_01_00_11);

        // ===== ROUND 7 (last round — skip m0/m1/m2 update) =====
        t0 = Sse.Shuffle(m0.AsSingle(), m1.AsSingle(), 0b_11_01_10_01).AsUInt32();
        p0 = Sse2.Shuffle(t0, 0b_01_11_10_00);
        b0 = p0;
        row0 = Sse2.Add(Sse2.Add(row0, b0), row1);
        row3 = RotateRight16(Sse2.Xor(row3, row0));
        row2 = Sse2.Add(row2, row3);
        row1 = RotateRight12(Sse2.Xor(row1, row2));

        t0 = Sse2.UnpackHigh(m0, m2);
        t1 = Sse2.Shuffle(m0.AsDouble(), m3.AsDouble(), 0b_10).AsUInt32();
        p1 = Sse.Shuffle(t0.AsSingle(), t1.AsSingle(), 0b_11_00_01_10).AsUInt32();
        b0 = p1;
        row0 = Sse2.Add(Sse2.Add(row0, b0), row1);
        row3 = RotateRight8(Sse2.Xor(row3, row0));
        row2 = Sse2.Add(row2, row3);
        row1 = RotateRight7(Sse2.Xor(row1, row2));

        row0 = Sse2.Shuffle(row0, 0b_10_01_00_11);
        row3 = Sse2.Shuffle(row3, 0b_01_00_11_10);
        row2 = Sse2.Shuffle(row2, 0b_00_11_10_01);

        t0 = Sse2.UnpackLow(m3, m1);
        t1 = Sse.Shuffle(m2.AsSingle(), m3.AsSingle(), 0b_10_01_11_01).AsUInt32();
        p2 = Sse.Shuffle(t0.AsSingle(), t1.AsSingle(), 0b_10_01_01_00).AsUInt32();
        b0 = p2;
        row0 = Sse2.Add(Sse2.Add(row0, b0), row1);
        row3 = RotateRight16(Sse2.Xor(row3, row0));
        row2 = Sse2.Add(row2, row3);
        row1 = RotateRight12(Sse2.Xor(row1, row2));

        t0 = Sse2.Shuffle(m2.AsDouble(), m1.AsDouble(), 0b_10).AsUInt32();
        m3 = Sse.Shuffle(t1.AsSingle(), t0.AsSingle(), 0b_00_10_11_00).AsUInt32();
        b0 = m3;
        row0 = Sse2.Add(Sse2.Add(row0, b0), row1);
        row3 = RotateRight8(Sse2.Xor(row3, row0));
        row2 = Sse2.Add(row2, row3);
        row1 = RotateRight7(Sse2.Xor(row1, row2));

        row0 = Sse2.Shuffle(row0, 0b_00_11_10_01);
        row3 = Sse2.Shuffle(row3, 0b_01_00_11_10);
        row2 = Sse2.Shuffle(row2, 0b_10_01_00_11);

        row0_ref = row0;
        row1_ref = row1;
        row2_ref = row2;
        row3_ref = row3;
    }
}
