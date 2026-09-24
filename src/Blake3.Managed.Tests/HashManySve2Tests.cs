using Blake3.Managed.Internal;
using System.Runtime.InteropServices;

namespace Blake3.Managed.Tests;

/// <summary>
/// Direct checks of the SVE2 kernels against scalar references. They only run on SVE2 hardware
/// (Graviton4 and similar); elsewhere they return immediately, like the AVX2 kernel tests.
/// </summary>
public class HashManySve2Tests
{
    private static readonly ulong[] Counters = { 0UL, 1UL, (ulong)uint.MaxValue - 6, (ulong)uint.MaxValue, 1UL << 48 };

    private static readonly uint[] FlagSets = { 0u, Blake3Constants.KeyedHash,
        Blake3Constants.DeriveKeyContext, Blake3Constants.DeriveKeyMaterial };

    private static void ExpectedChunkCvs(byte[] input, uint[] key, ulong counter, uint flags, int chunks, uint[] expected)
    {
        for (int lane = 0; lane < chunks; lane++)
            Blake3Core.HashChunkCv(key, input.AsSpan(1 + lane * 1024, 1024),
                counter + (ulong)lane, flags, expected.AsSpan(lane * 8, 8));
    }

    [Theory]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(8)]
    public void KernelsMatchIndependentChunksWithUnalignedInputAndCounterCarry(int numChunks)
    {
        if (!HashManySve2.IsSupported) return;
        byte[] input = new byte[numChunks * 1024 + 3];
        new Random(numChunks).NextBytes(input);
        uint[] key = MemoryMarshal.Cast<byte, uint>(HasherTests.MakeTestInput(32)).ToArray();
        foreach (ulong counter in Counters)
        foreach (uint flags in FlagSets)
        {
            uint[] expected = new uint[numChunks * 8];
            ExpectedChunkCvs(input, key, counter, flags, numChunks, expected);

            uint[] guarded = Enumerable.Repeat(0xDEADBEEFu, numChunks * 8 + 2).ToArray();
            var chunks = input.AsSpan(1, numChunks * 1024);
            var output = guarded.AsSpan(1, numChunks * 8);
            if (numChunks == 8)
                HashManySve2.HashMany8(chunks, key, counter, flags, output);
            else if (numChunks == 4)
                HashManySve2.HashMany(chunks, 4, key, counter, flags, output);
            else
                HashManySve2.HashManyPartial(chunks, numChunks, key, counter, flags, output);

            Assert.Equal(expected, output.ToArray());
            Assert.Equal(0xDEADBEEFu, guarded[0]);
            Assert.Equal(0xDEADBEEFu, guarded[^1]);
        }
    }

    [Fact]
    public void ParentsMatchScalarCompression()
    {
        if (!HashManySve2.IsSupported) return;
        uint[] blocks = MemoryMarshal.Cast<byte, uint>(HasherTests.MakeTestInput(4 * 64)).ToArray();
        uint[] key = MemoryMarshal.Cast<byte, uint>(HasherTests.MakeTestInput(32)).ToArray();
        foreach (uint flags in FlagSets)
        {
            uint[] expected = new uint[32];
            for (int p = 0; p < 4; p++)
                Blake3Core.CompressCv(key, blocks.AsSpan(p * 16, 16), 0, Blake3Constants.BlockLen,
                    flags | Blake3Constants.Parent, expected.AsSpan(p * 8, 8));

            uint[] actual = new uint[32];
            HashManySve2.HashParents4(blocks, key, flags, actual);
            Assert.Equal(expected, actual);
        }
    }

    [Fact]
    public void OutputBlocksMatchScalarRootCompression()
    {
        if (!OutputManySve2.IsSupported) return;
        uint[] cv = MemoryMarshal.Cast<byte, uint>(HasherTests.MakeTestInput(32)).ToArray();
        uint[] block = MemoryMarshal.Cast<byte, uint>(HasherTests.MakeTestInput(64)).ToArray();
        foreach (ulong counter in new[] { 0UL, 3UL, (ulong)uint.MaxValue - 5, 1UL << 40 })
        foreach (uint blockLen in new[] { 64u, 17u })
        foreach (uint flags in FlagSets)
        {
            byte[] expected = new byte[1024];
            uint[] state = new uint[16];
            for (int b = 0; b < 16; b++)
            {
                Blake3Core.CompressInPlace(cv, block, counter + (ulong)b, blockLen,
                    flags | Blake3Constants.Root, state);
                MemoryMarshal.AsBytes(state.AsSpan()).CopyTo(expected.AsSpan(b * 64));
            }

            byte[] actual = new byte[1024];
            OutputManySve2.HashOutput8(cv, block, counter, blockLen, flags, actual);
            Assert.Equal(expected, actual);
        }
    }

    [Theory]
    [InlineData(1, 0)]
    [InlineData(32, 0)]
    [InlineData(129, 0)]
    [InlineData(448, 0)]
    [InlineData(512, 0)]
    [InlineData(1000, 0)]
    [InlineData(4096, 0)]
    [InlineData(700, 13)]
    [InlineData(3000, 511)]
    public void XofMatchesAcrossOffsetsAndSplits(int outputLength, int offset)
    {
        byte[] input = HasherTests.MakeTestInput(1500);
        using var hasher = Hasher.New();
        hasher.Update(input);
        byte[] whole = new byte[offset + outputLength];
        hasher.Finalize(whole);
        byte[] seeked = new byte[outputLength];
        hasher.Finalize((ulong)offset, seeked);
        Assert.Equal(whole.AsSpan(offset).ToArray(), seeked);
    }

    [Theory]
    [InlineData(2 * 1024)]
    [InlineData(2 * 1024 + 1)]
    [InlineData(6 * 1024)]
    [InlineData(8 * 1024)]
    [InlineData(10 * 1024 + 7)]
    [InlineData(16 * 1024)]
    [InlineData(24 * 1024 + 1)]
    [InlineData(64 * 1024)]
    [InlineData(200 * 1024 + 3)]
    public void OneShotAndIncrementalAgreeAroundEightChunkBatches(int length)
    {
        byte[] input = HasherTests.MakeTestInput(length);
        var expected = Hasher.Hash(input);
        using var hasher = Hasher.New();
        hasher.Update(input);
        Assert.Equal(expected, hasher.Finalize());
        using var split = Hasher.New();
        split.Update(input.AsSpan(0, length / 3));
        split.Update(input.AsSpan(length / 3));
        Assert.Equal(expected, split.Finalize());
    }
}
