using System.Buffers.Binary;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Blake3.Managed.Internal;

/// <summary>
/// Portable single-block compression, and the ARM64 single-block path (see
/// <see cref="CompressNeon"/> for why ARM does not vectorise this).
/// </summary>
/// <remarks>
/// The seven rounds are written out with constant message indices (generated), after one
/// up-front bounds check per span, so no per-access checks or schedule lookups remain. The
/// message word is added before <c>b</c>, <c>(a + m) + b</c>: <c>b</c> is the newest value in
/// each G, so this keeps one add off the dependency chain. And each half-round is emitted
/// step by step across its four independent G's -- all four <c>a</c> updates, then all four
/// <c>d</c> rotates, and so on -- rather than one whole G after another.
/// <para>
/// Measured per 1 KiB chunk on Graviton4 (2026-09-23), against the previous version, which
/// looked the message up through the schedule table and built the chaining value by
/// compressing into a 16-word scratch buffer and copying: expanded with <c>(a + m) + b</c>
/// 0.837, and step-interleaved as here 0.761. RyuJIT emits statements in source order, so the
/// interleaving is what puts independent work next to each other.
/// </para>
/// </remarks>
internal static class CompressScalar
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Rounds(ref uint s0, ref uint s1, ref uint s2, ref uint s3,
                               ref uint s4, ref uint s5, ref uint s6, ref uint s7,
                               ref uint s8, ref uint s9, ref uint s10, ref uint s11,
                               ref uint s12, ref uint s13, ref uint s14, ref uint s15,
                               ReadOnlySpan<uint> m)
    {
        _ = m[15];
        s0 = s0 + m[0] + s4; s1 = s1 + m[2] + s5; s2 = s2 + m[4] + s6; s3 = s3 + m[6] + s7;
        s12 = BitOperations.RotateRight(s12 ^ s0, 16); s13 = BitOperations.RotateRight(s13 ^ s1, 16); s14 = BitOperations.RotateRight(s14 ^ s2, 16); s15 = BitOperations.RotateRight(s15 ^ s3, 16);
        s8 = s8 + s12; s9 = s9 + s13; s10 = s10 + s14; s11 = s11 + s15;
        s4 = BitOperations.RotateRight(s4 ^ s8, 12); s5 = BitOperations.RotateRight(s5 ^ s9, 12); s6 = BitOperations.RotateRight(s6 ^ s10, 12); s7 = BitOperations.RotateRight(s7 ^ s11, 12);
        s0 = s0 + m[1] + s4; s1 = s1 + m[3] + s5; s2 = s2 + m[5] + s6; s3 = s3 + m[7] + s7;
        s12 = BitOperations.RotateRight(s12 ^ s0, 8); s13 = BitOperations.RotateRight(s13 ^ s1, 8); s14 = BitOperations.RotateRight(s14 ^ s2, 8); s15 = BitOperations.RotateRight(s15 ^ s3, 8);
        s8 = s8 + s12; s9 = s9 + s13; s10 = s10 + s14; s11 = s11 + s15;
        s4 = BitOperations.RotateRight(s4 ^ s8, 7); s5 = BitOperations.RotateRight(s5 ^ s9, 7); s6 = BitOperations.RotateRight(s6 ^ s10, 7); s7 = BitOperations.RotateRight(s7 ^ s11, 7);
        s0 = s0 + m[8] + s5; s1 = s1 + m[10] + s6; s2 = s2 + m[12] + s7; s3 = s3 + m[14] + s4;
        s15 = BitOperations.RotateRight(s15 ^ s0, 16); s12 = BitOperations.RotateRight(s12 ^ s1, 16); s13 = BitOperations.RotateRight(s13 ^ s2, 16); s14 = BitOperations.RotateRight(s14 ^ s3, 16);
        s10 = s10 + s15; s11 = s11 + s12; s8 = s8 + s13; s9 = s9 + s14;
        s5 = BitOperations.RotateRight(s5 ^ s10, 12); s6 = BitOperations.RotateRight(s6 ^ s11, 12); s7 = BitOperations.RotateRight(s7 ^ s8, 12); s4 = BitOperations.RotateRight(s4 ^ s9, 12);
        s0 = s0 + m[9] + s5; s1 = s1 + m[11] + s6; s2 = s2 + m[13] + s7; s3 = s3 + m[15] + s4;
        s15 = BitOperations.RotateRight(s15 ^ s0, 8); s12 = BitOperations.RotateRight(s12 ^ s1, 8); s13 = BitOperations.RotateRight(s13 ^ s2, 8); s14 = BitOperations.RotateRight(s14 ^ s3, 8);
        s10 = s10 + s15; s11 = s11 + s12; s8 = s8 + s13; s9 = s9 + s14;
        s5 = BitOperations.RotateRight(s5 ^ s10, 7); s6 = BitOperations.RotateRight(s6 ^ s11, 7); s7 = BitOperations.RotateRight(s7 ^ s8, 7); s4 = BitOperations.RotateRight(s4 ^ s9, 7);
        s0 = s0 + m[2] + s4; s1 = s1 + m[3] + s5; s2 = s2 + m[7] + s6; s3 = s3 + m[4] + s7;
        s12 = BitOperations.RotateRight(s12 ^ s0, 16); s13 = BitOperations.RotateRight(s13 ^ s1, 16); s14 = BitOperations.RotateRight(s14 ^ s2, 16); s15 = BitOperations.RotateRight(s15 ^ s3, 16);
        s8 = s8 + s12; s9 = s9 + s13; s10 = s10 + s14; s11 = s11 + s15;
        s4 = BitOperations.RotateRight(s4 ^ s8, 12); s5 = BitOperations.RotateRight(s5 ^ s9, 12); s6 = BitOperations.RotateRight(s6 ^ s10, 12); s7 = BitOperations.RotateRight(s7 ^ s11, 12);
        s0 = s0 + m[6] + s4; s1 = s1 + m[10] + s5; s2 = s2 + m[0] + s6; s3 = s3 + m[13] + s7;
        s12 = BitOperations.RotateRight(s12 ^ s0, 8); s13 = BitOperations.RotateRight(s13 ^ s1, 8); s14 = BitOperations.RotateRight(s14 ^ s2, 8); s15 = BitOperations.RotateRight(s15 ^ s3, 8);
        s8 = s8 + s12; s9 = s9 + s13; s10 = s10 + s14; s11 = s11 + s15;
        s4 = BitOperations.RotateRight(s4 ^ s8, 7); s5 = BitOperations.RotateRight(s5 ^ s9, 7); s6 = BitOperations.RotateRight(s6 ^ s10, 7); s7 = BitOperations.RotateRight(s7 ^ s11, 7);
        s0 = s0 + m[1] + s5; s1 = s1 + m[12] + s6; s2 = s2 + m[9] + s7; s3 = s3 + m[15] + s4;
        s15 = BitOperations.RotateRight(s15 ^ s0, 16); s12 = BitOperations.RotateRight(s12 ^ s1, 16); s13 = BitOperations.RotateRight(s13 ^ s2, 16); s14 = BitOperations.RotateRight(s14 ^ s3, 16);
        s10 = s10 + s15; s11 = s11 + s12; s8 = s8 + s13; s9 = s9 + s14;
        s5 = BitOperations.RotateRight(s5 ^ s10, 12); s6 = BitOperations.RotateRight(s6 ^ s11, 12); s7 = BitOperations.RotateRight(s7 ^ s8, 12); s4 = BitOperations.RotateRight(s4 ^ s9, 12);
        s0 = s0 + m[11] + s5; s1 = s1 + m[5] + s6; s2 = s2 + m[14] + s7; s3 = s3 + m[8] + s4;
        s15 = BitOperations.RotateRight(s15 ^ s0, 8); s12 = BitOperations.RotateRight(s12 ^ s1, 8); s13 = BitOperations.RotateRight(s13 ^ s2, 8); s14 = BitOperations.RotateRight(s14 ^ s3, 8);
        s10 = s10 + s15; s11 = s11 + s12; s8 = s8 + s13; s9 = s9 + s14;
        s5 = BitOperations.RotateRight(s5 ^ s10, 7); s6 = BitOperations.RotateRight(s6 ^ s11, 7); s7 = BitOperations.RotateRight(s7 ^ s8, 7); s4 = BitOperations.RotateRight(s4 ^ s9, 7);
        s0 = s0 + m[3] + s4; s1 = s1 + m[10] + s5; s2 = s2 + m[13] + s6; s3 = s3 + m[7] + s7;
        s12 = BitOperations.RotateRight(s12 ^ s0, 16); s13 = BitOperations.RotateRight(s13 ^ s1, 16); s14 = BitOperations.RotateRight(s14 ^ s2, 16); s15 = BitOperations.RotateRight(s15 ^ s3, 16);
        s8 = s8 + s12; s9 = s9 + s13; s10 = s10 + s14; s11 = s11 + s15;
        s4 = BitOperations.RotateRight(s4 ^ s8, 12); s5 = BitOperations.RotateRight(s5 ^ s9, 12); s6 = BitOperations.RotateRight(s6 ^ s10, 12); s7 = BitOperations.RotateRight(s7 ^ s11, 12);
        s0 = s0 + m[4] + s4; s1 = s1 + m[12] + s5; s2 = s2 + m[2] + s6; s3 = s3 + m[14] + s7;
        s12 = BitOperations.RotateRight(s12 ^ s0, 8); s13 = BitOperations.RotateRight(s13 ^ s1, 8); s14 = BitOperations.RotateRight(s14 ^ s2, 8); s15 = BitOperations.RotateRight(s15 ^ s3, 8);
        s8 = s8 + s12; s9 = s9 + s13; s10 = s10 + s14; s11 = s11 + s15;
        s4 = BitOperations.RotateRight(s4 ^ s8, 7); s5 = BitOperations.RotateRight(s5 ^ s9, 7); s6 = BitOperations.RotateRight(s6 ^ s10, 7); s7 = BitOperations.RotateRight(s7 ^ s11, 7);
        s0 = s0 + m[6] + s5; s1 = s1 + m[9] + s6; s2 = s2 + m[11] + s7; s3 = s3 + m[8] + s4;
        s15 = BitOperations.RotateRight(s15 ^ s0, 16); s12 = BitOperations.RotateRight(s12 ^ s1, 16); s13 = BitOperations.RotateRight(s13 ^ s2, 16); s14 = BitOperations.RotateRight(s14 ^ s3, 16);
        s10 = s10 + s15; s11 = s11 + s12; s8 = s8 + s13; s9 = s9 + s14;
        s5 = BitOperations.RotateRight(s5 ^ s10, 12); s6 = BitOperations.RotateRight(s6 ^ s11, 12); s7 = BitOperations.RotateRight(s7 ^ s8, 12); s4 = BitOperations.RotateRight(s4 ^ s9, 12);
        s0 = s0 + m[5] + s5; s1 = s1 + m[0] + s6; s2 = s2 + m[15] + s7; s3 = s3 + m[1] + s4;
        s15 = BitOperations.RotateRight(s15 ^ s0, 8); s12 = BitOperations.RotateRight(s12 ^ s1, 8); s13 = BitOperations.RotateRight(s13 ^ s2, 8); s14 = BitOperations.RotateRight(s14 ^ s3, 8);
        s10 = s10 + s15; s11 = s11 + s12; s8 = s8 + s13; s9 = s9 + s14;
        s5 = BitOperations.RotateRight(s5 ^ s10, 7); s6 = BitOperations.RotateRight(s6 ^ s11, 7); s7 = BitOperations.RotateRight(s7 ^ s8, 7); s4 = BitOperations.RotateRight(s4 ^ s9, 7);
        s0 = s0 + m[10] + s4; s1 = s1 + m[12] + s5; s2 = s2 + m[14] + s6; s3 = s3 + m[13] + s7;
        s12 = BitOperations.RotateRight(s12 ^ s0, 16); s13 = BitOperations.RotateRight(s13 ^ s1, 16); s14 = BitOperations.RotateRight(s14 ^ s2, 16); s15 = BitOperations.RotateRight(s15 ^ s3, 16);
        s8 = s8 + s12; s9 = s9 + s13; s10 = s10 + s14; s11 = s11 + s15;
        s4 = BitOperations.RotateRight(s4 ^ s8, 12); s5 = BitOperations.RotateRight(s5 ^ s9, 12); s6 = BitOperations.RotateRight(s6 ^ s10, 12); s7 = BitOperations.RotateRight(s7 ^ s11, 12);
        s0 = s0 + m[7] + s4; s1 = s1 + m[9] + s5; s2 = s2 + m[3] + s6; s3 = s3 + m[15] + s7;
        s12 = BitOperations.RotateRight(s12 ^ s0, 8); s13 = BitOperations.RotateRight(s13 ^ s1, 8); s14 = BitOperations.RotateRight(s14 ^ s2, 8); s15 = BitOperations.RotateRight(s15 ^ s3, 8);
        s8 = s8 + s12; s9 = s9 + s13; s10 = s10 + s14; s11 = s11 + s15;
        s4 = BitOperations.RotateRight(s4 ^ s8, 7); s5 = BitOperations.RotateRight(s5 ^ s9, 7); s6 = BitOperations.RotateRight(s6 ^ s10, 7); s7 = BitOperations.RotateRight(s7 ^ s11, 7);
        s0 = s0 + m[4] + s5; s1 = s1 + m[11] + s6; s2 = s2 + m[5] + s7; s3 = s3 + m[1] + s4;
        s15 = BitOperations.RotateRight(s15 ^ s0, 16); s12 = BitOperations.RotateRight(s12 ^ s1, 16); s13 = BitOperations.RotateRight(s13 ^ s2, 16); s14 = BitOperations.RotateRight(s14 ^ s3, 16);
        s10 = s10 + s15; s11 = s11 + s12; s8 = s8 + s13; s9 = s9 + s14;
        s5 = BitOperations.RotateRight(s5 ^ s10, 12); s6 = BitOperations.RotateRight(s6 ^ s11, 12); s7 = BitOperations.RotateRight(s7 ^ s8, 12); s4 = BitOperations.RotateRight(s4 ^ s9, 12);
        s0 = s0 + m[0] + s5; s1 = s1 + m[2] + s6; s2 = s2 + m[8] + s7; s3 = s3 + m[6] + s4;
        s15 = BitOperations.RotateRight(s15 ^ s0, 8); s12 = BitOperations.RotateRight(s12 ^ s1, 8); s13 = BitOperations.RotateRight(s13 ^ s2, 8); s14 = BitOperations.RotateRight(s14 ^ s3, 8);
        s10 = s10 + s15; s11 = s11 + s12; s8 = s8 + s13; s9 = s9 + s14;
        s5 = BitOperations.RotateRight(s5 ^ s10, 7); s6 = BitOperations.RotateRight(s6 ^ s11, 7); s7 = BitOperations.RotateRight(s7 ^ s8, 7); s4 = BitOperations.RotateRight(s4 ^ s9, 7);
        s0 = s0 + m[12] + s4; s1 = s1 + m[9] + s5; s2 = s2 + m[15] + s6; s3 = s3 + m[14] + s7;
        s12 = BitOperations.RotateRight(s12 ^ s0, 16); s13 = BitOperations.RotateRight(s13 ^ s1, 16); s14 = BitOperations.RotateRight(s14 ^ s2, 16); s15 = BitOperations.RotateRight(s15 ^ s3, 16);
        s8 = s8 + s12; s9 = s9 + s13; s10 = s10 + s14; s11 = s11 + s15;
        s4 = BitOperations.RotateRight(s4 ^ s8, 12); s5 = BitOperations.RotateRight(s5 ^ s9, 12); s6 = BitOperations.RotateRight(s6 ^ s10, 12); s7 = BitOperations.RotateRight(s7 ^ s11, 12);
        s0 = s0 + m[13] + s4; s1 = s1 + m[11] + s5; s2 = s2 + m[10] + s6; s3 = s3 + m[8] + s7;
        s12 = BitOperations.RotateRight(s12 ^ s0, 8); s13 = BitOperations.RotateRight(s13 ^ s1, 8); s14 = BitOperations.RotateRight(s14 ^ s2, 8); s15 = BitOperations.RotateRight(s15 ^ s3, 8);
        s8 = s8 + s12; s9 = s9 + s13; s10 = s10 + s14; s11 = s11 + s15;
        s4 = BitOperations.RotateRight(s4 ^ s8, 7); s5 = BitOperations.RotateRight(s5 ^ s9, 7); s6 = BitOperations.RotateRight(s6 ^ s10, 7); s7 = BitOperations.RotateRight(s7 ^ s11, 7);
        s0 = s0 + m[7] + s5; s1 = s1 + m[5] + s6; s2 = s2 + m[0] + s7; s3 = s3 + m[6] + s4;
        s15 = BitOperations.RotateRight(s15 ^ s0, 16); s12 = BitOperations.RotateRight(s12 ^ s1, 16); s13 = BitOperations.RotateRight(s13 ^ s2, 16); s14 = BitOperations.RotateRight(s14 ^ s3, 16);
        s10 = s10 + s15; s11 = s11 + s12; s8 = s8 + s13; s9 = s9 + s14;
        s5 = BitOperations.RotateRight(s5 ^ s10, 12); s6 = BitOperations.RotateRight(s6 ^ s11, 12); s7 = BitOperations.RotateRight(s7 ^ s8, 12); s4 = BitOperations.RotateRight(s4 ^ s9, 12);
        s0 = s0 + m[2] + s5; s1 = s1 + m[3] + s6; s2 = s2 + m[1] + s7; s3 = s3 + m[4] + s4;
        s15 = BitOperations.RotateRight(s15 ^ s0, 8); s12 = BitOperations.RotateRight(s12 ^ s1, 8); s13 = BitOperations.RotateRight(s13 ^ s2, 8); s14 = BitOperations.RotateRight(s14 ^ s3, 8);
        s10 = s10 + s15; s11 = s11 + s12; s8 = s8 + s13; s9 = s9 + s14;
        s5 = BitOperations.RotateRight(s5 ^ s10, 7); s6 = BitOperations.RotateRight(s6 ^ s11, 7); s7 = BitOperations.RotateRight(s7 ^ s8, 7); s4 = BitOperations.RotateRight(s4 ^ s9, 7);
        s0 = s0 + m[9] + s4; s1 = s1 + m[11] + s5; s2 = s2 + m[8] + s6; s3 = s3 + m[15] + s7;
        s12 = BitOperations.RotateRight(s12 ^ s0, 16); s13 = BitOperations.RotateRight(s13 ^ s1, 16); s14 = BitOperations.RotateRight(s14 ^ s2, 16); s15 = BitOperations.RotateRight(s15 ^ s3, 16);
        s8 = s8 + s12; s9 = s9 + s13; s10 = s10 + s14; s11 = s11 + s15;
        s4 = BitOperations.RotateRight(s4 ^ s8, 12); s5 = BitOperations.RotateRight(s5 ^ s9, 12); s6 = BitOperations.RotateRight(s6 ^ s10, 12); s7 = BitOperations.RotateRight(s7 ^ s11, 12);
        s0 = s0 + m[14] + s4; s1 = s1 + m[5] + s5; s2 = s2 + m[12] + s6; s3 = s3 + m[1] + s7;
        s12 = BitOperations.RotateRight(s12 ^ s0, 8); s13 = BitOperations.RotateRight(s13 ^ s1, 8); s14 = BitOperations.RotateRight(s14 ^ s2, 8); s15 = BitOperations.RotateRight(s15 ^ s3, 8);
        s8 = s8 + s12; s9 = s9 + s13; s10 = s10 + s14; s11 = s11 + s15;
        s4 = BitOperations.RotateRight(s4 ^ s8, 7); s5 = BitOperations.RotateRight(s5 ^ s9, 7); s6 = BitOperations.RotateRight(s6 ^ s10, 7); s7 = BitOperations.RotateRight(s7 ^ s11, 7);
        s0 = s0 + m[13] + s5; s1 = s1 + m[0] + s6; s2 = s2 + m[2] + s7; s3 = s3 + m[4] + s4;
        s15 = BitOperations.RotateRight(s15 ^ s0, 16); s12 = BitOperations.RotateRight(s12 ^ s1, 16); s13 = BitOperations.RotateRight(s13 ^ s2, 16); s14 = BitOperations.RotateRight(s14 ^ s3, 16);
        s10 = s10 + s15; s11 = s11 + s12; s8 = s8 + s13; s9 = s9 + s14;
        s5 = BitOperations.RotateRight(s5 ^ s10, 12); s6 = BitOperations.RotateRight(s6 ^ s11, 12); s7 = BitOperations.RotateRight(s7 ^ s8, 12); s4 = BitOperations.RotateRight(s4 ^ s9, 12);
        s0 = s0 + m[3] + s5; s1 = s1 + m[10] + s6; s2 = s2 + m[6] + s7; s3 = s3 + m[7] + s4;
        s15 = BitOperations.RotateRight(s15 ^ s0, 8); s12 = BitOperations.RotateRight(s12 ^ s1, 8); s13 = BitOperations.RotateRight(s13 ^ s2, 8); s14 = BitOperations.RotateRight(s14 ^ s3, 8);
        s10 = s10 + s15; s11 = s11 + s12; s8 = s8 + s13; s9 = s9 + s14;
        s5 = BitOperations.RotateRight(s5 ^ s10, 7); s6 = BitOperations.RotateRight(s6 ^ s11, 7); s7 = BitOperations.RotateRight(s7 ^ s8, 7); s4 = BitOperations.RotateRight(s4 ^ s9, 7);
        s0 = s0 + m[11] + s4; s1 = s1 + m[5] + s5; s2 = s2 + m[1] + s6; s3 = s3 + m[8] + s7;
        s12 = BitOperations.RotateRight(s12 ^ s0, 16); s13 = BitOperations.RotateRight(s13 ^ s1, 16); s14 = BitOperations.RotateRight(s14 ^ s2, 16); s15 = BitOperations.RotateRight(s15 ^ s3, 16);
        s8 = s8 + s12; s9 = s9 + s13; s10 = s10 + s14; s11 = s11 + s15;
        s4 = BitOperations.RotateRight(s4 ^ s8, 12); s5 = BitOperations.RotateRight(s5 ^ s9, 12); s6 = BitOperations.RotateRight(s6 ^ s10, 12); s7 = BitOperations.RotateRight(s7 ^ s11, 12);
        s0 = s0 + m[15] + s4; s1 = s1 + m[0] + s5; s2 = s2 + m[9] + s6; s3 = s3 + m[6] + s7;
        s12 = BitOperations.RotateRight(s12 ^ s0, 8); s13 = BitOperations.RotateRight(s13 ^ s1, 8); s14 = BitOperations.RotateRight(s14 ^ s2, 8); s15 = BitOperations.RotateRight(s15 ^ s3, 8);
        s8 = s8 + s12; s9 = s9 + s13; s10 = s10 + s14; s11 = s11 + s15;
        s4 = BitOperations.RotateRight(s4 ^ s8, 7); s5 = BitOperations.RotateRight(s5 ^ s9, 7); s6 = BitOperations.RotateRight(s6 ^ s10, 7); s7 = BitOperations.RotateRight(s7 ^ s11, 7);
        s0 = s0 + m[14] + s5; s1 = s1 + m[2] + s6; s2 = s2 + m[3] + s7; s3 = s3 + m[7] + s4;
        s15 = BitOperations.RotateRight(s15 ^ s0, 16); s12 = BitOperations.RotateRight(s12 ^ s1, 16); s13 = BitOperations.RotateRight(s13 ^ s2, 16); s14 = BitOperations.RotateRight(s14 ^ s3, 16);
        s10 = s10 + s15; s11 = s11 + s12; s8 = s8 + s13; s9 = s9 + s14;
        s5 = BitOperations.RotateRight(s5 ^ s10, 12); s6 = BitOperations.RotateRight(s6 ^ s11, 12); s7 = BitOperations.RotateRight(s7 ^ s8, 12); s4 = BitOperations.RotateRight(s4 ^ s9, 12);
        s0 = s0 + m[10] + s5; s1 = s1 + m[12] + s6; s2 = s2 + m[4] + s7; s3 = s3 + m[13] + s4;
        s15 = BitOperations.RotateRight(s15 ^ s0, 8); s12 = BitOperations.RotateRight(s12 ^ s1, 8); s13 = BitOperations.RotateRight(s13 ^ s2, 8); s14 = BitOperations.RotateRight(s14 ^ s3, 8);
        s10 = s10 + s15; s11 = s11 + s12; s8 = s8 + s13; s9 = s9 + s14;
        s5 = BitOperations.RotateRight(s5 ^ s10, 7); s6 = BitOperations.RotateRight(s6 ^ s11, 7); s7 = BitOperations.RotateRight(s7 ^ s8, 7); s4 = BitOperations.RotateRight(s4 ^ s9, 7);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void Compress(ReadOnlySpan<uint> cv, ReadOnlySpan<uint> block,
                                ulong counter, uint blockLen, uint flags,
                                Span<uint> output)
    {
        _ = cv[7];
        _ = output[15];
        uint s0 = cv[0], s1 = cv[1], s2 = cv[2], s3 = cv[3];
        uint s4 = cv[4], s5 = cv[5], s6 = cv[6], s7 = cv[7];
        uint s8 = Blake3Constants.Iv0;
        uint s9 = Blake3Constants.Iv1;
        uint s10 = Blake3Constants.Iv2;
        uint s11 = Blake3Constants.Iv3;
        uint s12 = (uint)counter;
        uint s13 = (uint)(counter >> 32);
        uint s14 = blockLen;
        uint s15 = flags;

        Rounds(ref s0, ref s1, ref s2, ref s3, ref s4, ref s5, ref s6, ref s7,
               ref s8, ref s9, ref s10, ref s11, ref s12, ref s13, ref s14, ref s15, block);

        // Post-XOR: state[i] ^= state[i+8]; state[i+8] ^= cv[i]. The upper half reads cv again
        // before anything is written, so output may alias cv.
        uint c0 = cv[0], c1 = cv[1], c2 = cv[2], c3 = cv[3];
        uint c4 = cv[4], c5 = cv[5], c6 = cv[6], c7 = cv[7];
        output[0]  = s0 ^ s8;   output[8]  = s8 ^ c0;
        output[1]  = s1 ^ s9;   output[9]  = s9 ^ c1;
        output[2]  = s2 ^ s10;  output[10] = s10 ^ c2;
        output[3]  = s3 ^ s11;  output[11] = s11 ^ c3;
        output[4]  = s4 ^ s12;  output[12] = s12 ^ c4;
        output[5]  = s5 ^ s13;  output[13] = s13 ^ c5;
        output[6]  = s6 ^ s14;  output[14] = s14 ^ c6;
        output[7]  = s7 ^ s15;  output[15] = s15 ^ c7;
    }

    /// <summary>The first eight output words only; <paramref name="chainingValue"/> may alias <paramref name="cv"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void CompressChainingValue(ReadOnlySpan<uint> cv, ReadOnlySpan<uint> block,
                                             ulong counter, uint blockLen, uint flags,
                                             Span<uint> chainingValue)
    {
        _ = cv[7];
        _ = chainingValue[7];
        uint s0 = cv[0], s1 = cv[1], s2 = cv[2], s3 = cv[3];
        uint s4 = cv[4], s5 = cv[5], s6 = cv[6], s7 = cv[7];
        uint s8 = Blake3Constants.Iv0;
        uint s9 = Blake3Constants.Iv1;
        uint s10 = Blake3Constants.Iv2;
        uint s11 = Blake3Constants.Iv3;
        uint s12 = (uint)counter;
        uint s13 = (uint)(counter >> 32);
        uint s14 = blockLen;
        uint s15 = flags;

        Rounds(ref s0, ref s1, ref s2, ref s3, ref s4, ref s5, ref s6, ref s7,
               ref s8, ref s9, ref s10, ref s11, ref s12, ref s13, ref s14, ref s15, block);

        chainingValue[0] = s0 ^ s8;
        chainingValue[1] = s1 ^ s9;
        chainingValue[2] = s2 ^ s10;
        chainingValue[3] = s3 ^ s11;
        chainingValue[4] = s4 ^ s12;
        chainingValue[5] = s5 ^ s13;
        chainingValue[6] = s6 ^ s14;
        chainingValue[7] = s7 ^ s15;
    }

    /// <summary>
    /// Hashes a whole 0..1024-byte chunk to its chaining value -- or, with
    /// <paramref name="root"/>, to its 32-byte root digest -- keeping the chaining value in
    /// registers from block to block. Little-endian only.
    /// </summary>
    /// <remarks>
    /// The per-block path calls <see cref="CompressChainingValue"/> once per block, loading the
    /// chaining value from memory and storing it back each time. The round body is inlined once
    /// here, for every block including the last, so the method stays one copy of the rounds.
    /// </remarks>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void HashChunk(ReadOnlySpan<uint> key, ulong counter, uint flags, bool root,
                                 ReadOnlySpan<byte> input, Span<uint> output)
        => HashChunkCore(key, counter, flags, root, prefixOnly: false, input, output);

    /// <summary>
    /// Compresses every block of a 0..1024-byte chunk except the last into
    /// <paramref name="output"/>, and returns the number of bytes consumed, leaving 0..64 bytes
    /// as the final block (for callers that need its whole <c>Output</c>, as extended output does).
    /// </summary>
    public static int HashChunkPrefix(ReadOnlySpan<uint> key, ulong counter, uint flags,
                                      ReadOnlySpan<byte> input, Span<uint> output)
        => HashChunkCore(key, counter, flags, root: false, prefixOnly: true, input, output);

    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.NoInlining | MethodImplOptions.AggressiveOptimization)]
    private static int HashChunkCore(ReadOnlySpan<uint> key, ulong counter, uint flags, bool root,
                                     bool prefixOnly, ReadOnlySpan<byte> input, Span<uint> output)
    {
        _ = key[7];
        _ = output[7];
        uint c0 = key[0], c1 = key[1], c2 = key[2], c3 = key[3];
        uint c4 = key[4], c5 = key[5], c6 = key[6], c7 = key[7];
        uint counterLo = (uint)counter, counterHi = (uint)(counter >> 32);

        Span<uint> padded = stackalloc uint[16];
        int pos = 0;
        uint blockFlags = flags | Blake3Constants.ChunkStart;
        while (true)
        {
            int remaining = input.Length - pos;
            bool last = remaining <= Blake3Constants.BlockLen;
            scoped ReadOnlySpan<uint> m;
            uint blockLen;
            if (!last)
            {
                m = MemoryMarshal.Cast<byte, uint>(input.Slice(pos, Blake3Constants.BlockLen));
                blockLen = Blake3Constants.BlockLen;
            }
            else
            {
                if (prefixOnly) break;
                blockFlags |= Blake3Constants.ChunkEnd | (root ? Blake3Constants.Root : 0u);
                blockLen = (uint)remaining;
                if (remaining == Blake3Constants.BlockLen)
                {
                    m = MemoryMarshal.Cast<byte, uint>(input.Slice(pos, Blake3Constants.BlockLen));
                }
                else
                {
                    // Inline stores rather than Span.Clear/CopyTo, which are out-of-line calls.
                    Blake3Core.ZeroBlock(MemoryMarshal.AsBytes(padded));
                    Blake3Core.CopyUpTo64(input.Slice(pos), MemoryMarshal.AsBytes(padded));
                    m = padded;
                }
            }

            uint s0 = c0, s1 = c1, s2 = c2, s3 = c3, s4 = c4, s5 = c5, s6 = c6, s7 = c7;
            uint s8 = Blake3Constants.Iv0, s9 = Blake3Constants.Iv1;
            uint s10 = Blake3Constants.Iv2, s11 = Blake3Constants.Iv3;
            uint s12 = counterLo, s13 = counterHi, s14 = blockLen, s15 = blockFlags;

            Rounds(ref s0, ref s1, ref s2, ref s3, ref s4, ref s5, ref s6, ref s7,
                   ref s8, ref s9, ref s10, ref s11, ref s12, ref s13, ref s14, ref s15, m);

            c0 = s0 ^ s8; c1 = s1 ^ s9; c2 = s2 ^ s10; c3 = s3 ^ s11;
            c4 = s4 ^ s12; c5 = s5 ^ s13; c6 = s6 ^ s14; c7 = s7 ^ s15;

            if (last) break;
            pos += Blake3Constants.BlockLen;
            blockFlags = flags;
        }

        output[0] = c0; output[1] = c1; output[2] = c2; output[3] = c3;
        output[4] = c4; output[5] = c5; output[6] = c6; output[7] = c7;
        return pos;
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
