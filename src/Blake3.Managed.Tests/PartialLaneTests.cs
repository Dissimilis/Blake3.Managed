using System.Runtime.InteropServices;
using Blake3.Managed.Internal;

namespace Blake3.Managed.Tests;

/// <summary>
/// The kernels that carry a partial final chunk in a spare SIMD lane, and the fused chunk
/// loops, checked lane by lane against an independent per-block scalar reference.
/// </summary>
public class PartialLaneTests
{
    private static readonly uint[] Key = [0x12345678, 1, uint.MaxValue, 3, 4, 5, 6, 0x87654321];
    private static readonly uint[] Modes = [0, Blake3Constants.KeyedHash,
        Blake3Constants.DeriveKeyContext, Blake3Constants.DeriveKeyMaterial];
    private static readonly ulong[] Counters = [0, uint.MaxValue - 2, 1UL << 40];

    private static uint[] ReferenceCv(ReadOnlySpan<byte> chunk, ulong counter, uint flags, bool root = false)
    {
        var cv = (uint[])Key.Clone();
        Span<byte> block = stackalloc byte[64];
        int length = chunk.Length;
        int pos = 0;
        do
        {
            int count = Math.Min(64, length - pos);
            block.Clear();
            chunk.Slice(pos, count).CopyTo(block);
            bool last = pos + count == length;
            uint blockFlags = flags | (pos == 0 ? Blake3Constants.ChunkStart : 0u)
                | (last ? Blake3Constants.ChunkEnd | (root ? Blake3Constants.Root : 0u) : 0u);
            CompressScalar.CompressChainingValue(cv, MemoryMarshal.Cast<byte, uint>(block),
                counter, (uint)count, blockFlags, cv);
            pos += 64;
        } while (pos < length);
        return cv;
    }

    private static byte[] Input(int length, int seed)
    {
        var input = new byte[length + 1];
        new Random(seed).NextBytes(input);
        return input;
    }

    private static void AssertLanes(ReadOnlySpan<byte> input, int lanes, ulong counter, uint flags,
        uint[] actual, int offset, string what)
    {
        for (int lane = 0; lane < lanes; lane++)
        {
            int start = lane * 1024;
            int length = Math.Min(1024, input.Length - start);
            var expected = ReferenceCv(input.Slice(start, length), counter + (ulong)lane, flags);
            Assert.True(expected.AsSpan().SequenceEqual(actual.AsSpan(offset + lane * 8, 8)),
                $"{what}: lane={lane}, length={input.Length}, counter={counter}, flags={flags}");
        }
    }

    [Fact]
    public void HashOneAndPartialMatchesEveryPartialLength()
    {
        if (!HashTwoAvx2.IsSupported) return;
        byte[] data = Input(2047, 1);
        foreach (uint flags in Modes)
        foreach (ulong counter in Counters)
        for (int partial = 1; partial < 1024; partial++)
        {
            var input = data.AsSpan(1, 1024 + partial);
            uint[] guarded = Enumerable.Repeat(0xDEADBEEFu, 18).ToArray();
            HashTwoAvx2.HashOneAndPartial(input, Key, counter, flags, guarded.AsSpan(1, 16));
            AssertLanes(input, 2, counter, flags, guarded, 1, "HashOneAndPartial");
            Assert.Equal(0xDEADBEEFu, guarded[0]);
            Assert.Equal(0xDEADBEEFu, guarded[17]);
        }
    }

    [Fact]
    public void HashFourPartialMatchesEveryPartialLength()
    {
        if (!HashFourAvx2.IsSupported) return;
        byte[] data = Input(4095, 2);
        foreach (int full in new[] { 2, 3 })
        foreach (uint flags in Modes)
        foreach (ulong counter in Counters)
        for (int partial = 1; partial < 1024; partial++)
        {
            var input = data.AsSpan(1, full * 1024 + partial);
            int words = (full + 1) * 8;
            uint[] guarded = Enumerable.Repeat(0xDEADBEEFu, words + 2).ToArray();
            HashFourAvx2.HashFourPartial(input, full, Key, counter, flags, guarded.AsSpan(1, words));
            AssertLanes(input, full + 1, counter, flags, guarded, 1, "HashFourPartial");
            Assert.Equal(0xDEADBEEFu, guarded[0]);
            Assert.Equal(0xDEADBEEFu, guarded[^1]);
        }
    }

    [Fact]
    public void HashManyPartialTailMatchesEveryPartialLength()
    {
        if (!HashManyAvx2.IsSupported) return;
        byte[] data = Input(8191, 3);
        foreach (int full in new[] { 3, 4, 5, 6, 7 })
        foreach (uint flags in new[] { 0u, Blake3Constants.KeyedHash })
        foreach (ulong counter in Counters)
        for (int partial = 1; partial < 1024; partial += full == 7 ? 1 : 7)
        {
            var input = data.AsSpan(1, full * 1024 + partial);
            uint[] guarded = Enumerable.Repeat(0xDEADBEEFu, 66).ToArray();
            HashManyAvx2.HashManyPartialTail(input, full, Key, counter, flags, guarded.AsSpan(1, 64));
            AssertLanes(input, full + 1, counter, flags, guarded, 1, "HashManyPartialTail");
            Assert.Equal(0xDEADBEEFu, guarded[0]);
            Assert.Equal(0xDEADBEEFu, guarded[^1]);
        }
    }

#if NET8_0_OR_GREATER
    [Fact]
    public void Ragged16MatchesEveryChunkCount()
    {
        if (!HashManyAvx512.IsSupported) return;
        byte[] data = Input(16383, 8);
        foreach (int full in new[] { 8, 9, 12, 15 })
        foreach (uint flags in new[] { 0u, Blake3Constants.KeyedHash })
        foreach (ulong counter in Counters)
        for (int partial = full == 8 ? 1 : 0; partial < 1024; partial += full == 15 ? 1 : 13)
        {
            var input = data.AsSpan(1, full * 1024 + partial);
            uint[] guarded = Enumerable.Repeat(0xDEADBEEFu, 130).ToArray();
            HashManyAvx512.HashMany16Ragged(input, Key, counter, flags, guarded.AsSpan(1, 128));
            AssertLanes(input, full + (partial > 0 ? 1 : 0), counter, flags, guarded, 1, "HashMany16Ragged");
            Assert.Equal(0xDEADBEEFu, guarded[0]);
            Assert.Equal(0xDEADBEEFu, guarded[^1]);
        }
    }
#endif

    [Fact]
    public void NeonPartialTailMatchesEveryPartialLength()
    {
        if (!HashManyNeon.IsSupportedWithoutSve2) return;
        byte[] data = Input(4095, 4);
        foreach (int full in new[] { 2, 3 })
        foreach (uint flags in Modes)
        foreach (ulong counter in Counters)
        for (int partial = 1; partial < 1024; partial++)
        {
            var input = data.AsSpan(1, full * 1024 + partial);
            int words = (full + 1) * 8;
            uint[] guarded = Enumerable.Repeat(0xDEADBEEFu, words + 2).ToArray();
            HashManyNeon.HashManyPartialTail(input, full, Key, counter, flags, guarded.AsSpan(1, words));
            AssertLanes(input, full + 1, counter, flags, guarded, 1, "NeonPartialTail");
            Assert.Equal(0xDEADBEEFu, guarded[0]);
            Assert.Equal(0xDEADBEEFu, guarded[^1]);
        }
    }

    [Fact]
    public void ScalarFusedChunkMatchesReferenceForEveryLength()
    {
        if (!BitConverter.IsLittleEndian) return;
        byte[] data = Input(1024, 5);
        uint[] guarded = new uint[10];
        foreach (uint flags in Modes)
        foreach (ulong counter in Counters)
        foreach (bool root in new[] { false, true })
        for (int length = 0; length <= 1024; length++)
        {
            var chunk = data.AsSpan(1, length);
            Array.Fill(guarded, 0xDEADBEEFu);
            CompressScalar.HashChunk(Key, counter, flags, root, chunk, guarded.AsSpan(1, 8));
            var expected = ReferenceCv(chunk, counter, flags, root);
            Assert.True(expected.AsSpan().SequenceEqual(guarded.AsSpan(1, 8)),
                $"length={length}, counter={counter}, flags={flags}, root={root}");
            Assert.Equal(0xDEADBEEFu, guarded[0]);
            Assert.Equal(0xDEADBEEFu, guarded[9]);
        }
    }

    [Fact]
    public void KeyedSseChunkRootMatchesReferenceForEveryLength()
    {
        if (!CompressSse41.IsSupported) return;
        byte[] data = Input(1024, 6);
        uint[] guarded = new uint[10];
        foreach (uint flags in Modes)
        foreach (ulong counter in Counters)
        for (int length = 0; length <= 1024; length++)
        {
            var chunk = data.AsSpan(1, length);
            Array.Fill(guarded, 0xDEADBEEFu);
            CompressSse41.HashChunkRoot32(Key, counter, flags, chunk, guarded.AsSpan(1, 8));
            var expected = ReferenceCv(chunk, counter, flags, root: true);
            Assert.True(expected.AsSpan().SequenceEqual(guarded.AsSpan(1, 8)),
                $"length={length}, counter={counter}, flags={flags}");
            Assert.Equal(0xDEADBEEFu, guarded[0]);
            Assert.Equal(0xDEADBEEFu, guarded[9]);
        }
    }

    [Fact]
    public void ChunkPrefixesLeaveTheFinalBlock()
    {
        byte[] data = Input(1024, 7);
        uint[] iv = Blake3Constants.IV.ToArray();
        for (int length = 0; length <= 1024; length++)
        {
            var chunk = data.AsSpan(1, length);
            int expectedPos = length == 0 ? 0 : (length - 1) / 64 * 64;
            uint[] expected = iv.ToArray();
            for (int pos = 0; pos < expectedPos; pos += 64)
                CompressScalar.CompressChainingValue(expected,
                    MemoryMarshal.Cast<byte, uint>(chunk.Slice(pos, 64)), 0, 64,
                    pos == 0 ? Blake3Constants.ChunkStart : 0u, expected);

            uint[] scalar = new uint[8];
            Assert.Equal(expectedPos, CompressScalar.HashChunkPrefix(iv, 0, 0, chunk, scalar));
            Assert.Equal(expected, scalar);

            if (CompressSse41.IsSupported)
            {
                uint[] sse = new uint[8];
                Assert.Equal(expectedPos, CompressSse41.HashChunkPrefixIv(chunk, sse));
                Assert.Equal(expected, sse);
            }
        }
    }

    [Fact]
    public void PartialChunkPrefixSurvivesEveryUpdatePattern()
    {
        byte[] a = Input(9000, 9).AsSpan(1).ToArray();
        byte[] b = Input(9000, 10).AsSpan(1).ToArray();
        for (int length = 1025; length < 8192; length += length < 4200 ? 7 : 61)
        {
            var first = a.AsSpan(0, length);
            Hasher.New(out var h);
            try
            {
                h.Update(first);
                Assert.Equal(Hasher.Hash(first), h.Finalize());
                Assert.Equal(Hasher.Hash(first), h.Finalize());

                // More input after the speculative chunk: the stored CV must not be used.
                h.Update(a.AsSpan(length, 1));
                Assert.Equal(Hasher.Hash(a.AsSpan(0, length + 1)), h.Finalize());

                // Same length, different bytes, after Reset.
                h.Reset();
                h.Update(b.AsSpan(0, length));
                Assert.Equal(Hasher.Hash(b.AsSpan(0, length)), h.Finalize());

                // Split so the partial chunk is completed by a second Update.
                h.Reset();
                int cut = length - (length % 1024 == 0 ? 1 : length % 1024) / 2 - 1;
                h.Update(a.AsSpan(0, cut));
                h.Update(a.AsSpan(cut, length - cut));
                Assert.Equal(Hasher.Hash(first), h.Finalize());
            }
            finally
            {
                h.Dispose();
            }

            Hasher.NewKeyed(MemoryMarshal.AsBytes(Key.AsSpan()), out var k);
            try
            {
                k.UpdateWithJoin(first);
                Assert.Equal(Hasher.HashKeyed(MemoryMarshal.AsBytes(Key.AsSpan()), first), k.Finalize());
            }
            finally
            {
                k.Dispose();
            }
        }
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(63)]
    [InlineData(64)]
    [InlineData(65)]
    [InlineData(1000)]
    [InlineData(1024)]
    public void SingleChunkFinalizeMatchesOneShotAfterReset(int length)
    {
        byte[] input = HasherTests.MakeTestInput(length);
        byte[] big = HasherTests.MakeTestInput(5000);
        Hasher.New(out var hasher);
        try
        {
            // Fill the CV stack first, so Reset has slots to clear before the short hash.
            hasher.Update(big);
            hasher.Reset();
            hasher.Update(input);
            Assert.Equal(Hasher.Hash(input), hasher.Finalize());
            var span = new byte[32];
            hasher.Finalize(span);
            Assert.Equal(Hasher.Hash(input).AsSpan().ToArray(), span);
        }
        finally
        {
            hasher.Dispose();
        }
    }
}
