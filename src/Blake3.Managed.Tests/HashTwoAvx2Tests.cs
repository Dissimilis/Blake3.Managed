using Blake3.Managed.Internal;
using System.Runtime.InteropServices;

namespace Blake3.Managed.Tests;

public class HashTwoAvx2Tests
{
    [Theory]
    [InlineData(2048)]
    [InlineData(4096)]
    [InlineData(5120)]
    [InlineData(6144)]
    [InlineData(7168)]
    [InlineData(8192)]
    public void EmptyJoinedUpdatePreservesDeferredFinalChunk(int length)
    {
        byte[] input = HasherTests.MakeTestInput(length);
        using var hasher = Hasher.New();
        hasher.Update(input);
        hasher.UpdateWithJoin(ReadOnlySpan<byte>.Empty);
        hasher.UpdateWithJoin(ReadOnlySpan<byte>.Empty);
        Assert.Equal(Hasher.Hash(input), hasher.Finalize());

        hasher.Update(input.AsSpan(0, 1));
        Assert.Equal(Hasher.Hash(input.Concat(input.Take(1)).ToArray()), hasher.Finalize());
    }

    [Fact]
    public void PartialEightWayBatchesMatchOnlyTheirActiveInputLanes()
    {
        if (!HashManyAvx2.IsSupported) return;
        uint[] key = MemoryMarshal.Cast<byte, uint>(HasherTests.MakeTestInput(32)).ToArray();
        foreach (int count in Enumerable.Range(1, 8))
        foreach (ulong counter in new[] { 0UL, (ulong)uint.MaxValue - 3 })
        {
            byte[] input = new byte[count * 1024 + 1];
            new Random(count).NextBytes(input);
            uint[] actual = new uint[64];
            uint[] expected = new uint[count * 8];
            for (int lane = 0; lane < count; lane++)
                Blake3Core.HashChunkCv(key, input.AsSpan(1 + lane * 1024, 1024),
                    counter + (ulong)lane, Blake3Constants.KeyedHash, expected.AsSpan(lane * 8, 8));
            HashManyAvx2.HashManyPartial(input.AsSpan(1), count, key, counter, Blake3Constants.KeyedHash, actual);
            Assert.Equal(expected, actual.AsSpan(0, count * 8).ToArray());
        }
    }

    [Fact]
    public void SerialEightWayBatchMatchesIndependentChunksAcrossCounterCarry()
    {
        if (!HashManyAvx2.IsSupported) return;
        byte[] input = new byte[8193];
        new Random(84).NextBytes(input);
        uint[] key = MemoryMarshal.Cast<byte, uint>(HasherTests.MakeTestInput(32)).ToArray();
        foreach (ulong counter in new[] { 0UL, (ulong)uint.MaxValue - 3, 1UL << 48 })
        foreach (uint flags in new[] { 0u, Blake3Constants.KeyedHash,
                     Blake3Constants.DeriveKeyContext, Blake3Constants.DeriveKeyMaterial })
        {
            uint[] expected = new uint[64];
            uint[] guarded = Enumerable.Repeat(0xDEADBEEFu, 66).ToArray();
            for (int lane = 0; lane < 8; lane++)
                Blake3Core.HashChunkCv(key, input.AsSpan(1 + lane * 1024, 1024),
                    counter + (ulong)lane, flags, expected.AsSpan(lane * 8, 8));
            HashManyAvx2.HashManySerial(input.AsSpan(1), key, counter, flags, guarded.AsSpan(1, 64));
            Assert.Equal(expected, guarded.AsSpan(1, 64).ToArray());
            Assert.Equal(0xDEADBEEFu, guarded[0]);
            Assert.Equal(0xDEADBEEFu, guarded[^1]);
        }
    }

    [Fact]
    public void LanesMatchIndependentChunksWithUnalignedInputAndCounterCarry()
    {
        if (!HashTwoAvx2.IsSupported) return;
        byte[] input = new byte[2051];
        new Random(42).NextBytes(input);
        uint[] key = MemoryMarshal.Cast<byte, uint>(HasherTests.MakeTestInput(32)).ToArray();
        foreach (ulong counter in new[] { 0UL, 1UL, (ulong)uint.MaxValue, 1UL << 48 })
        foreach (uint flags in new[] { 0u, Blake3Constants.KeyedHash,
                     Blake3Constants.DeriveKeyContext, Blake3Constants.DeriveKeyMaterial })
        {
            uint[] expected = new uint[16];
            uint[] guarded = Enumerable.Repeat(0xDEADBEEFu, 18).ToArray();
            Blake3Core.HashChunkCv(key, input.AsSpan(1, 1024), counter, flags, expected.AsSpan(0, 8));
            Blake3Core.HashChunkCv(key, input.AsSpan(1025, 1024), counter + 1, flags, expected.AsSpan(8, 8));
            HashTwoAvx2.HashTwo(input.AsSpan(1, 2048), key, counter, flags, guarded.AsSpan(1, 16));
            Assert.Equal(expected, guarded.AsSpan(1, 16).ToArray());
            Assert.Equal(0xDEADBEEFu, guarded[0]);
            Assert.Equal(0xDEADBEEFu, guarded[^1]);
        }
    }
}
