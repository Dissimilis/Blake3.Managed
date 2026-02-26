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
    private static Vector128<uint> RotateRight16(Vector128<uint> v)
    {
        return AdvSimd.ReverseElement16(v.AsInt32()).AsUInt32();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<uint> RotateRight12(Vector128<uint> v)
    {
        return AdvSimd.Or(AdvSimd.ShiftRightLogical(v, 12), AdvSimd.ShiftLeftLogical(v, 20));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<uint> RotateRight8(Vector128<uint> v)
    {
        return AdvSimd.Arm64.VectorTableLookup(v.AsByte(), Rot8Mask).AsUInt32();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<uint> RotateRight7(Vector128<uint> v)
    {
        return AdvSimd.Or(AdvSimd.ShiftRightLogical(v, 7), AdvSimd.ShiftLeftLogical(v, 25));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<uint> RotLeft1(Vector128<uint> v)
    {
        return AdvSimd.ExtractVector128(v.AsByte(), v.AsByte(), 4).AsUInt32();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<uint> RotLeft2(Vector128<uint> v)
    {
        return AdvSimd.ExtractVector128(v.AsByte(), v.AsByte(), 8).AsUInt32();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<uint> RotLeft3(Vector128<uint> v)
    {
        return AdvSimd.ExtractVector128(v.AsByte(), v.AsByte(), 12).AsUInt32();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void RoundNeon(
        ref Vector128<uint> row0, ref Vector128<uint> row1,
        ref Vector128<uint> row2, ref Vector128<uint> row3,
        ReadOnlySpan<uint> m, int scheduleOffset)
    {
        ReadOnlySpan<byte> sch = Blake3Constants.MsgSchedule.Slice(scheduleOffset, 16);

        // Column step
        var mx = Vector128.Create(m[sch[0]], m[sch[2]], m[sch[4]], m[sch[6]]);
        var my = Vector128.Create(m[sch[1]], m[sch[3]], m[sch[5]], m[sch[7]]);

        row0 = AdvSimd.Add(AdvSimd.Add(row0, mx), row1);
        row3 = RotateRight16(AdvSimd.Xor(row3, row0));
        row2 = AdvSimd.Add(row2, row3);
        row1 = RotateRight12(AdvSimd.Xor(row1, row2));

        row0 = AdvSimd.Add(AdvSimd.Add(row0, my), row1);
        row3 = RotateRight8(AdvSimd.Xor(row3, row0));
        row2 = AdvSimd.Add(row2, row3);
        row1 = RotateRight7(AdvSimd.Xor(row1, row2));

        // Column -> diagonal
        row1 = RotLeft1(row1);
        row2 = RotLeft2(row2);
        row3 = RotLeft3(row3);

        // Diagonal step
        mx = Vector128.Create(m[sch[8]], m[sch[10]], m[sch[12]], m[sch[14]]);
        my = Vector128.Create(m[sch[9]], m[sch[11]], m[sch[13]], m[sch[15]]);

        row0 = AdvSimd.Add(AdvSimd.Add(row0, mx), row1);
        row3 = RotateRight16(AdvSimd.Xor(row3, row0));
        row2 = AdvSimd.Add(row2, row3);
        row1 = RotateRight12(AdvSimd.Xor(row1, row2));

        row0 = AdvSimd.Add(AdvSimd.Add(row0, my), row1);
        row3 = RotateRight8(AdvSimd.Xor(row3, row0));
        row2 = AdvSimd.Add(row2, row3);
        row1 = RotateRight7(AdvSimd.Xor(row1, row2));

        // Diagonal -> column
        row1 = RotLeft3(row1);
        row2 = RotLeft2(row2);
        row3 = RotLeft1(row3);
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

        ref uint ivRef = ref MemoryMarshal.GetReference(Blake3Constants.IV);
        var row2 = VectorCompat.Load(ref ivRef);
        var row3 = Vector128.Create((uint)counter, (uint)(counter >> 32), blockLen, flags);

        RoundNeon(ref row0, ref row1, ref row2, ref row3, block, 0 * 16);
        RoundNeon(ref row0, ref row1, ref row2, ref row3, block, 1 * 16);
        RoundNeon(ref row0, ref row1, ref row2, ref row3, block, 2 * 16);
        RoundNeon(ref row0, ref row1, ref row2, ref row3, block, 3 * 16);
        RoundNeon(ref row0, ref row1, ref row2, ref row3, block, 4 * 16);
        RoundNeon(ref row0, ref row1, ref row2, ref row3, block, 5 * 16);
        RoundNeon(ref row0, ref row1, ref row2, ref row3, block, 6 * 16);

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
    public static void CompressChainingValue(ReadOnlySpan<uint> cv, ReadOnlySpan<uint> block,
                                             ulong counter, uint blockLen, uint flags,
                                             Span<uint> chainingValue)
    {
        ref uint cvRef = ref MemoryMarshal.GetReference(cv);
        var row0 = VectorCompat.Load(ref cvRef);
        var row1 = VectorCompat.Load(ref cvRef, 4);

        ref uint ivRef = ref MemoryMarshal.GetReference(Blake3Constants.IV);
        var row2 = VectorCompat.Load(ref ivRef);
        var row3 = Vector128.Create((uint)counter, (uint)(counter >> 32), blockLen, flags);

        RoundNeon(ref row0, ref row1, ref row2, ref row3, block, 0 * 16);
        RoundNeon(ref row0, ref row1, ref row2, ref row3, block, 1 * 16);
        RoundNeon(ref row0, ref row1, ref row2, ref row3, block, 2 * 16);
        RoundNeon(ref row0, ref row1, ref row2, ref row3, block, 3 * 16);
        RoundNeon(ref row0, ref row1, ref row2, ref row3, block, 4 * 16);
        RoundNeon(ref row0, ref row1, ref row2, ref row3, block, 5 * 16);
        RoundNeon(ref row0, ref row1, ref row2, ref row3, block, 6 * 16);

        ref uint outRef = ref MemoryMarshal.GetReference(chainingValue);
        VectorCompat.Store(AdvSimd.Xor(row0, row2), ref outRef);
        VectorCompat.Store(AdvSimd.Xor(row1, row3), ref outRef, 4);
    }
}
