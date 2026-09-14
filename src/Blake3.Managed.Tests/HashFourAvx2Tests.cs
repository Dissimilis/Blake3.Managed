using Blake3.Managed.Internal;
using System.Runtime.InteropServices;

namespace Blake3.Managed.Tests;

public class HashFourAvx2Tests
{
    [Theory]
    [InlineData(3)]
    [InlineData(4)]
    public void ChainsMatchIndependentChunksWithUnalignedInputAndCounterCarry(int numChunks)
    {
        if (!HashFourAvx2.IsSupported) return;
        byte[] input = new byte[numChunks * 1024 + 3];
        new Random(numChunks).NextBytes(input);
        uint[] key = MemoryMarshal.Cast<byte, uint>(HasherTests.MakeTestInput(32)).ToArray();
        foreach (ulong counter in new[] { 0UL, 1UL, (ulong)uint.MaxValue - 2, (ulong)uint.MaxValue, 1UL << 48 })
        foreach (uint flags in new[] { 0u, Blake3Constants.KeyedHash,
                     Blake3Constants.DeriveKeyContext, Blake3Constants.DeriveKeyMaterial })
        {
            uint[] expected = new uint[numChunks * 8];
            uint[] guarded = Enumerable.Repeat(0xDEADBEEFu, numChunks * 8 + 2).ToArray();
            for (int lane = 0; lane < numChunks; lane++)
                Blake3Core.HashChunkCv(key, input.AsSpan(1 + lane * 1024, 1024),
                    counter + (ulong)lane, flags, expected.AsSpan(lane * 8, 8));
            HashFourAvx2.HashFour(input.AsSpan(1, numChunks * 1024), numChunks, key, counter, flags,
                guarded.AsSpan(1, numChunks * 8));
            Assert.Equal(expected, guarded.AsSpan(1, numChunks * 8).ToArray());
            Assert.Equal(0xDEADBEEFu, guarded[0]);
            Assert.Equal(0xDEADBEEFu, guarded[^1]);
        }
    }

    [Theory]
    [InlineData(3 * 1024)]
    [InlineData(3 * 1024 + 1)]
    [InlineData(4 * 1024)]
    [InlineData(4 * 1024 + 1)]
    [InlineData(4 * 1024 + 1023)]
    [InlineData(12 * 1024)]
    [InlineData(20 * 1024 + 5)]
    public void OneShotAndIncrementalAgreeAroundThreeAndFourChunkTails(int length)
    {
        byte[] input = HasherTests.MakeTestInput(length);
        var expected = Hasher.Hash(input);
        Span<byte> span = stackalloc byte[32];
        Hasher.Hash(input, span);
        Assert.Equal(expected.AsSpan().ToArray(), span.ToArray());
        using var hasher = Hasher.New();
        hasher.Update(input);
        Assert.Equal(expected, hasher.Finalize());
        using var keyed = Hasher.NewKeyed(HasherTests.MakeTestInput(32));
        keyed.Update(input.AsSpan(0, 1000));
        keyed.Update(input.AsSpan(1000));
        using var keyed2 = Hasher.NewKeyed(HasherTests.MakeTestInput(32));
        keyed2.Update(input);
        Assert.Equal(keyed2.Finalize(), keyed.Finalize());
    }
}
