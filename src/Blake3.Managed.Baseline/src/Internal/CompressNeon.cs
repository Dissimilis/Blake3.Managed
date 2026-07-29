using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;

namespace Blake3.Managed.Internal;

internal static class CompressNeon
{
    public static bool IsSupported => AdvSimd.Arm64.IsSupported;

    private static readonly Vector128<byte> Rot8Mask = Vector128.Create(
        (byte)1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<uint> Rot16(Vector128<uint> v)
        => AdvSimd.ReverseElement16(v.AsInt32()).AsUInt32();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<uint> Rot12(Vector128<uint> v)
        => AdvSimd.Or(AdvSimd.ShiftRightLogical(v, 12), AdvSimd.ShiftLeftLogical(v, 20));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<uint> Rot8(Vector128<uint> v)
        => AdvSimd.Arm64.VectorTableLookup(v.AsByte(), Rot8Mask).AsUInt32();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<uint> Rot7(Vector128<uint> v)
        => AdvSimd.Or(AdvSimd.ShiftRightLogical(v, 7), AdvSimd.ShiftLeftLogical(v, 25));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<uint> RL1(Vector128<uint> v)
        => AdvSimd.ExtractVector128(v.AsByte(), v.AsByte(), 4).AsUInt32();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<uint> RL2(Vector128<uint> v)
        => AdvSimd.ExtractVector128(v.AsByte(), v.AsByte(), 8).AsUInt32();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<uint> RL3(Vector128<uint> v)
        => AdvSimd.ExtractVector128(v.AsByte(), v.AsByte(), 12).AsUInt32();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void HalfG(ref Vector128<uint> row0, ref Vector128<uint> row1,
                               ref Vector128<uint> row2, ref Vector128<uint> row3,
                               Vector128<uint> mx, Vector128<uint> my)
    {
        row0 = AdvSimd.Add(AdvSimd.Add(row0, mx), row1);
        row3 = Rot16(AdvSimd.Xor(row3, row0));
        row2 = AdvSimd.Add(row2, row3);
        row1 = Rot12(AdvSimd.Xor(row1, row2));

        row0 = AdvSimd.Add(AdvSimd.Add(row0, my), row1);
        row3 = Rot8(AdvSimd.Xor(row3, row0));
        row2 = AdvSimd.Add(row2, row3);
        row1 = Rot7(AdvSimd.Xor(row1, row2));
    }

    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void DoRounds(ref Vector128<uint> row0, ref Vector128<uint> row1,
                                        ref Vector128<uint> row2, ref Vector128<uint> row3,
                                        uint* b)
    {
        // Round 0: schedule [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
        HalfG(ref row0, ref row1, ref row2, ref row3,
            Vector128.Create(b[0], b[2], b[4], b[6]),
            Vector128.Create(b[1], b[3], b[5], b[7]));
        row1 = RL1(row1); row2 = RL2(row2); row3 = RL3(row3);
        HalfG(ref row0, ref row1, ref row2, ref row3,
            Vector128.Create(b[8], b[10], b[12], b[14]),
            Vector128.Create(b[9], b[11], b[13], b[15]));
        row1 = RL3(row1); row2 = RL2(row2); row3 = RL1(row3);

        // Round 1: schedule [2,6,3,10,7,0,4,13,1,11,12,5,9,14,15,8]
        HalfG(ref row0, ref row1, ref row2, ref row3,
            Vector128.Create(b[2], b[3], b[7], b[4]),
            Vector128.Create(b[6], b[10], b[0], b[13]));
        row1 = RL1(row1); row2 = RL2(row2); row3 = RL3(row3);
        HalfG(ref row0, ref row1, ref row2, ref row3,
            Vector128.Create(b[1], b[12], b[9], b[15]),
            Vector128.Create(b[11], b[5], b[14], b[8]));
        row1 = RL3(row1); row2 = RL2(row2); row3 = RL1(row3);

        // Round 2: schedule [3,4,10,12,13,2,7,14,6,5,9,0,11,15,8,1]
        HalfG(ref row0, ref row1, ref row2, ref row3,
            Vector128.Create(b[3], b[10], b[13], b[7]),
            Vector128.Create(b[4], b[12], b[2], b[14]));
        row1 = RL1(row1); row2 = RL2(row2); row3 = RL3(row3);
        HalfG(ref row0, ref row1, ref row2, ref row3,
            Vector128.Create(b[6], b[9], b[11], b[8]),
            Vector128.Create(b[5], b[0], b[15], b[1]));
        row1 = RL3(row1); row2 = RL2(row2); row3 = RL1(row3);

        // Round 3: schedule [10,7,12,9,14,3,13,15,4,0,11,2,5,8,1,6]
        HalfG(ref row0, ref row1, ref row2, ref row3,
            Vector128.Create(b[10], b[12], b[14], b[13]),
            Vector128.Create(b[7], b[9], b[3], b[15]));
        row1 = RL1(row1); row2 = RL2(row2); row3 = RL3(row3);
        HalfG(ref row0, ref row1, ref row2, ref row3,
            Vector128.Create(b[4], b[11], b[5], b[1]),
            Vector128.Create(b[0], b[2], b[8], b[6]));
        row1 = RL3(row1); row2 = RL2(row2); row3 = RL1(row3);

        // Round 4: schedule [12,13,9,11,15,10,14,8,7,2,5,3,0,1,6,4]
        HalfG(ref row0, ref row1, ref row2, ref row3,
            Vector128.Create(b[12], b[9], b[15], b[14]),
            Vector128.Create(b[13], b[11], b[10], b[8]));
        row1 = RL1(row1); row2 = RL2(row2); row3 = RL3(row3);
        HalfG(ref row0, ref row1, ref row2, ref row3,
            Vector128.Create(b[7], b[5], b[0], b[6]),
            Vector128.Create(b[2], b[3], b[1], b[4]));
        row1 = RL3(row1); row2 = RL2(row2); row3 = RL1(row3);

        // Round 5: schedule [9,14,11,5,8,12,15,1,13,3,0,10,2,6,4,7]
        HalfG(ref row0, ref row1, ref row2, ref row3,
            Vector128.Create(b[9], b[11], b[8], b[15]),
            Vector128.Create(b[14], b[5], b[12], b[1]));
        row1 = RL1(row1); row2 = RL2(row2); row3 = RL3(row3);
        HalfG(ref row0, ref row1, ref row2, ref row3,
            Vector128.Create(b[13], b[0], b[2], b[4]),
            Vector128.Create(b[3], b[10], b[6], b[7]));
        row1 = RL3(row1); row2 = RL2(row2); row3 = RL1(row3);

        // Round 6: schedule [11,15,5,0,1,9,8,6,14,10,2,12,3,4,7,13]
        HalfG(ref row0, ref row1, ref row2, ref row3,
            Vector128.Create(b[11], b[5], b[1], b[8]),
            Vector128.Create(b[15], b[0], b[9], b[6]));
        row1 = RL1(row1); row2 = RL2(row2); row3 = RL3(row3);
        HalfG(ref row0, ref row1, ref row2, ref row3,
            Vector128.Create(b[14], b[2], b[3], b[7]),
            Vector128.Create(b[10], b[12], b[4], b[13]));
        row1 = RL3(row1); row2 = RL2(row2); row3 = RL1(row3);
    }

    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static unsafe void Compress(ReadOnlySpan<uint> cv, ReadOnlySpan<uint> block,
                                ulong counter, uint blockLen, uint flags,
                                Span<uint> output)
    {
        ref uint cvRef = ref MemoryMarshal.GetReference(cv);
        var row0 = VectorCompat.Load(ref cvRef);
        var row1 = VectorCompat.Load(ref cvRef, 4);
        var row2 = VectorCompat.Load(ref MemoryMarshal.GetReference(Blake3Constants.IV));
        var row3 = Vector128.Create((uint)counter, (uint)(counter >> 32), blockLen, flags);

        fixed (uint* bp = block)
        {
            DoRounds(ref row0, ref row1, ref row2, ref row3, bp);
        }

        var cv0 = VectorCompat.Load(ref cvRef);
        var cv1 = VectorCompat.Load(ref cvRef, 4);

        ref uint outRef = ref MemoryMarshal.GetReference(output);
        VectorCompat.Store(AdvSimd.Xor(row0, row2), ref outRef);
        VectorCompat.Store(AdvSimd.Xor(row1, row3), ref outRef, 4);
        VectorCompat.Store(AdvSimd.Xor(row2, cv0), ref outRef, 8);
        VectorCompat.Store(AdvSimd.Xor(row3, cv1), ref outRef, 12);
    }

    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static unsafe void CompressChainingValue(ReadOnlySpan<uint> cv, ReadOnlySpan<uint> block,
                                             ulong counter, uint blockLen, uint flags,
                                             Span<uint> chainingValue)
    {
        ref uint cvRef = ref MemoryMarshal.GetReference(cv);
        var row0 = VectorCompat.Load(ref cvRef);
        var row1 = VectorCompat.Load(ref cvRef, 4);
        var row2 = VectorCompat.Load(ref MemoryMarshal.GetReference(Blake3Constants.IV));
        var row3 = Vector128.Create((uint)counter, (uint)(counter >> 32), blockLen, flags);

        fixed (uint* bp = block)
        {
            DoRounds(ref row0, ref row1, ref row2, ref row3, bp);
        }

        ref uint outRef = ref MemoryMarshal.GetReference(chainingValue);
        VectorCompat.Store(AdvSimd.Xor(row0, row2), ref outRef);
        VectorCompat.Store(AdvSimd.Xor(row1, row3), ref outRef, 4);
    }
}
