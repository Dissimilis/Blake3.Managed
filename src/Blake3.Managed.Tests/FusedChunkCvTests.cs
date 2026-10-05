using System.Runtime.InteropServices;
using Blake3.Managed.Internal;

namespace Blake3.Managed.Tests;

public class FusedChunkCvTests
{
    [Fact]
    public void EveryTailMatchesScalarWithFlagsAndFullWidthCounters()
    {
        if (!CompressSse41.IsSupported) return;

        var input = new byte[1025];
        new Random(42).NextBytes(input);
        uint[] key = [0x12345678, 1, uint.MaxValue, 3, 4, 5, 6, 0x87654321];
        uint[] modes = [0, Blake3Constants.KeyedHash, Blake3Constants.DeriveKeyContext,
            Blake3Constants.DeriveKeyMaterial];
        ulong[] counters = [0, 1, uint.MaxValue, 1UL << 32, ulong.MaxValue];
        Span<byte> block = stackalloc byte[64];
        Span<uint> expected = stackalloc uint[8];
        Span<uint> guarded = stackalloc uint[10];

        foreach (uint flags in modes)
        foreach (ulong counter in counters)
        for (int length = 1; length <= 1024; length++)
        {
            var chunk = input.AsSpan(1, length); // Deliberately unaligned.
            key.CopyTo(expected);
            for (int pos = 0; pos < length; pos += 64)
            {
                int count = Math.Min(64, length - pos);
                block.Clear();
                chunk.Slice(pos, count).CopyTo(block);
                uint blockFlags = flags | (pos == 0 ? Blake3Constants.ChunkStart : 0u)
                    | (pos + count == length ? Blake3Constants.ChunkEnd : 0u);
                CompressScalar.CompressChainingValue(expected,
                    MemoryMarshal.Cast<byte, uint>(block), counter, (uint)count,
                    blockFlags, expected);
            }

            guarded.Fill(0xDEADBEEF);
            CompressSse41.HashChunkCv(key, chunk, counter, flags, guarded.Slice(1, 8));
            Assert.True(expected.SequenceEqual(guarded.Slice(1, 8)),
                $"length={length}, counter={counter}, flags={flags}");
            Assert.Equal(0xDEADBEEFu, guarded[0]);
            Assert.Equal(0xDEADBEEFu, guarded[9]);
        }
    }
}
