using System.Buffers.Binary;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Blake3.Managed.Internal;

internal static class CompressScalar
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static uint RotateRight(uint x, int n) => (x >> n) | (x << (32 - n));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void G(ref uint a, ref uint b, ref uint c, ref uint d, uint mx, uint my)
    {
        a = a + b + mx;
        d = RotateRight(d ^ a, 16);
        c = c + d;
        b = RotateRight(b ^ c, 12);
        a = a + b + my;
        d = RotateRight(d ^ a, 8);
        c = c + d;
        b = RotateRight(b ^ c, 7);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Round(ref uint s0, ref uint s1, ref uint s2, ref uint s3,
                              ref uint s4, ref uint s5, ref uint s6, ref uint s7,
                              ref uint s8, ref uint s9, ref uint s10, ref uint s11,
                              ref uint s12, ref uint s13, ref uint s14, ref uint s15,
                              ReadOnlySpan<uint> m, int scheduleOffset)
    {
        ReadOnlySpan<byte> schedule = Blake3Constants.MsgSchedule.Slice(scheduleOffset, 16);

        // Column step
        G(ref s0, ref s4, ref s8,  ref s12, m[schedule[0]],  m[schedule[1]]);
        G(ref s1, ref s5, ref s9,  ref s13, m[schedule[2]],  m[schedule[3]]);
        G(ref s2, ref s6, ref s10, ref s14, m[schedule[4]],  m[schedule[5]]);
        G(ref s3, ref s7, ref s11, ref s15, m[schedule[6]],  m[schedule[7]]);

        // Diagonal step
        G(ref s0, ref s5, ref s10, ref s15, m[schedule[8]],  m[schedule[9]]);
        G(ref s1, ref s6, ref s11, ref s12, m[schedule[10]], m[schedule[11]]);
        G(ref s2, ref s7, ref s8,  ref s13, m[schedule[12]], m[schedule[13]]);
        G(ref s3, ref s4, ref s9,  ref s14, m[schedule[14]], m[schedule[15]]);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void Compress(ReadOnlySpan<uint> cv, ReadOnlySpan<uint> block,
                                ulong counter, uint blockLen, uint flags,
                                Span<uint> output)
    {
        uint s0 = cv[0], s1 = cv[1], s2 = cv[2], s3 = cv[3];
        uint s4 = cv[4], s5 = cv[5], s6 = cv[6], s7 = cv[7];
        uint s8 = Blake3Constants.IV[0];
        uint s9 = Blake3Constants.IV[1];
        uint s10 = Blake3Constants.IV[2];
        uint s11 = Blake3Constants.IV[3];
        uint s12 = (uint)counter;
        uint s13 = (uint)(counter >> 32);
        uint s14 = blockLen;
        uint s15 = flags;

        Round(ref s0, ref s1, ref s2, ref s3, ref s4, ref s5, ref s6, ref s7,
              ref s8, ref s9, ref s10, ref s11, ref s12, ref s13, ref s14, ref s15,
              block, 0 * 16);
        Round(ref s0, ref s1, ref s2, ref s3, ref s4, ref s5, ref s6, ref s7,
              ref s8, ref s9, ref s10, ref s11, ref s12, ref s13, ref s14, ref s15,
              block, 1 * 16);
        Round(ref s0, ref s1, ref s2, ref s3, ref s4, ref s5, ref s6, ref s7,
              ref s8, ref s9, ref s10, ref s11, ref s12, ref s13, ref s14, ref s15,
              block, 2 * 16);
        Round(ref s0, ref s1, ref s2, ref s3, ref s4, ref s5, ref s6, ref s7,
              ref s8, ref s9, ref s10, ref s11, ref s12, ref s13, ref s14, ref s15,
              block, 3 * 16);
        Round(ref s0, ref s1, ref s2, ref s3, ref s4, ref s5, ref s6, ref s7,
              ref s8, ref s9, ref s10, ref s11, ref s12, ref s13, ref s14, ref s15,
              block, 4 * 16);
        Round(ref s0, ref s1, ref s2, ref s3, ref s4, ref s5, ref s6, ref s7,
              ref s8, ref s9, ref s10, ref s11, ref s12, ref s13, ref s14, ref s15,
              block, 5 * 16);
        Round(ref s0, ref s1, ref s2, ref s3, ref s4, ref s5, ref s6, ref s7,
              ref s8, ref s9, ref s10, ref s11, ref s12, ref s13, ref s14, ref s15,
              block, 6 * 16);

        // Post-XOR: state[i] ^= state[i+8]; state[i+8] ^= cv[i]
        output[0]  = s0 ^ s8;   output[8]  = s8 ^ cv[0];
        output[1]  = s1 ^ s9;   output[9]  = s9 ^ cv[1];
        output[2]  = s2 ^ s10;  output[10] = s10 ^ cv[2];
        output[3]  = s3 ^ s11;  output[11] = s11 ^ cv[3];
        output[4]  = s4 ^ s12;  output[12] = s12 ^ cv[4];
        output[5]  = s5 ^ s13;  output[13] = s13 ^ cv[5];
        output[6]  = s6 ^ s14;  output[14] = s14 ^ cv[6];
        output[7]  = s7 ^ s15;  output[15] = s15 ^ cv[7];
    }

    [SkipLocalsInit]
    public static void CompressChainingValue(ReadOnlySpan<uint> cv, ReadOnlySpan<uint> block,
                                             ulong counter, uint blockLen, uint flags,
                                             Span<uint> chainingValue)
    {
        Span<uint> full = stackalloc uint[16];
        Compress(cv, block, counter, blockLen, flags, full);
        full[..8].CopyTo(chainingValue);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void BlockWordsFromBytes(ReadOnlySpan<byte> block, Span<uint> words)
    {
        if (BitConverter.IsLittleEndian)
        {
            MemoryMarshal.Cast<byte, uint>(block[..64]).CopyTo(words);
        }
        else
        {
            for (int i = 0; i < 16; i++)
            {
                words[i] = BinaryPrimitives.ReadUInt32LittleEndian(block[(i * 4)..]);
            }
        }
    }
}
