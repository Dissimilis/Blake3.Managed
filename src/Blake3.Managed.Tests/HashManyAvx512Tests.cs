using Blake3.Managed.Internal;
using System.Runtime.InteropServices;

namespace Blake3.Managed.Tests;

public class HashManyAvx512Tests
{
    [Fact]
    public void SixteenChunksMatchIndependentChunksWithUnalignedInputAndCounterCarry()
    {
        if (!HashManyAvx512.IsSupported) return;
        byte[] input = new byte[16 * 1024 + 3];
        new Random(16).NextBytes(input);
        uint[] key = MemoryMarshal.Cast<byte, uint>(HasherTests.MakeTestInput(32)).ToArray();
        foreach (ulong counter in new[] { 0UL, 1UL, (ulong)uint.MaxValue - 9, (ulong)uint.MaxValue, 1UL << 48 })
        foreach (uint flags in new[] { 0u, Blake3Constants.KeyedHash,
                     Blake3Constants.DeriveKeyContext, Blake3Constants.DeriveKeyMaterial })
        {
            uint[] expected = new uint[16 * 8];
            for (int lane = 0; lane < 16; lane++)
                Blake3Core.HashChunkCv(key, input.AsSpan(1 + lane * 1024, 1024),
                    counter + (ulong)lane, flags, expected.AsSpan(lane * 8, 8));

            uint[] guarded = Enumerable.Repeat(0xDEADBEEFu, 16 * 8 + 2).ToArray();
            HashManyAvx512.HashMany16(input.AsSpan(1, 16 * 1024), key, counter, flags, guarded.AsSpan(1, 16 * 8));
            Assert.Equal(expected, guarded.AsSpan(1, 16 * 8).ToArray());
            Assert.Equal(0xDEADBEEFu, guarded[0]);
            Assert.Equal(0xDEADBEEFu, guarded[^1]);
        }
    }

    [Fact]
    public void OutputBlocksMatchPerBlockRootCompression()
    {
        if (!HashManyAvx512.IsSupported) return;
        uint[] cv = MemoryMarshal.Cast<byte, uint>(HasherTests.MakeTestInput(32)).ToArray();
        uint[] block = MemoryMarshal.Cast<byte, uint>(HasherTests.MakeTestInput(64)).ToArray();
        foreach (ulong counter in new[] { 0UL, 3UL, (ulong)uint.MaxValue - 9, 1UL << 40 })
        foreach (uint blockLen in new[] { 64u, 17u })
        {
            byte[] expected = new byte[2048];
            uint[] state = new uint[16];
            for (int b = 0; b < 32; b++)
            {
                Blake3Core.CompressInPlace(cv, block, counter + (ulong)b, blockLen,
                    Blake3Constants.KeyedHash | Blake3Constants.Root, state);
                MemoryMarshal.AsBytes(state.AsSpan()).CopyTo(expected.AsSpan(b * 64));
            }

            byte[] actual = new byte[2048];
            HashManyAvx512.HashOutput16(cv, block, counter, blockLen, Blake3Constants.KeyedHash, actual);
            Assert.Equal(expected, actual);
        }
    }

    [Theory]
    [InlineData(16 * 1024)]
    [InlineData(16 * 1024 + 1)]
    [InlineData(24 * 1024)]
    [InlineData(32 * 1024)]
    [InlineData(48 * 1024 + 5)]
    [InlineData(64 * 1024)]
    [InlineData(300 * 1024 + 11)]
    public void OneShotAndIncrementalAgreeAroundSixteenChunkSubtrees(int length)
    {
        byte[] input = HasherTests.MakeTestInput(length);
        var expected = Hasher.Hash(input);
        using var hasher = Hasher.New();
        hasher.Update(input);
        Assert.Equal(expected, hasher.Finalize());
        using var keyedWhole = Hasher.NewKeyed(HasherTests.MakeTestInput(32));
        keyedWhole.Update(input);
        using var keyedSplit = Hasher.NewKeyed(HasherTests.MakeTestInput(32));
        keyedSplit.Update(input.AsSpan(0, length / 2));
        keyedSplit.Update(input.AsSpan(length / 2));
        Assert.Equal(keyedWhole.Finalize(), keyedSplit.Finalize());
    }
}

