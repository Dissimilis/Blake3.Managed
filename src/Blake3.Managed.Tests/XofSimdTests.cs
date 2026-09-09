using Blake3.Managed.Internal;
using System.Runtime.InteropServices;

namespace Blake3.Managed.Tests;

public class XofSimdTests
{
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    public void BatchedOutputMatchesIndividualBlocks(int mode)
    {
        foreach (int inputLength in new[] { 0, 1, 64, 65, 128, 1024, 1025, 2048, 6144, 8192 })
        {
            byte[] input = HasherTests.MakeTestInput(inputLength);
            using var hasher = mode switch
            {
                1 => Hasher.NewKeyed(HasherTests.MakeTestInput(32)),
                2 => Hasher.NewDeriveKey("BLAKE3 output batching test"),
                _ => Hasher.New(),
            };
            hasher.Update(input);

            foreach (ulong offset in new ulong[] { 0, 1, 31, 63, 64, 65, 511, 512,
                         ((ulong)uint.MaxValue - 3) * 64, ulong.MaxValue - 2048 })
            foreach (int length in new[] { 0, 31, 32, 64, 128, 511, 512, 513, 575, 576, 1024, 1089 })
            {
                // Each oracle request fits in the current output block, bypassing SIMD.
                byte[] expected = new byte[length];
                int pos = 0;
                ulong counter = offset / 64;
                int skip = (int)(offset % 64);
                while (pos < length)
                {
                    int take = Math.Min(64 - skip, length - pos);
                    hasher.Finalize(counter * 64 + (ulong)skip, expected.AsSpan(pos, take));
                    pos += take;
                    counter++;
                    skip = 0;
                }

                byte[] guarded = Enumerable.Repeat((byte)0xA5, length + 2).ToArray();
                hasher.Finalize(offset, guarded.AsSpan(1, length));
                Assert.Equal(expected, guarded.AsSpan(1, length).ToArray());
                Assert.Equal(0xA5, guarded[0]);
                Assert.Equal(0xA5, guarded[^1]);
            }
        }
    }

    [Fact]
    public void EightOutputLanesMatchScalarCompressionAcrossCounterCarry()
    {
        if (!OutputManyAvx2.IsSupported) return;
        uint[] cv = MemoryMarshal.Cast<byte, uint>(HasherTests.MakeTestInput(32)).ToArray();
        uint[] block = MemoryMarshal.Cast<byte, uint>(HasherTests.MakeTestInput(64)).ToArray();
        foreach (ulong counter in new[] { 0UL, (ulong)uint.MaxValue - 3, ulong.MaxValue / 64 })
        foreach (uint flags in new[] { Blake3Constants.ChunkStart | Blake3Constants.ChunkEnd,
                     Blake3Constants.Parent, Blake3Constants.Parent | Blake3Constants.KeyedHash,
                     Blake3Constants.ChunkEnd | Blake3Constants.DeriveKeyMaterial })
        foreach (uint blockLength in new uint[] { 0, 1, 63, 64 })
        {
            byte[] actual = new byte[1024];
            byte[] expected = new byte[1024];
            for (int i = 0; i < 16; i++)
                CompressScalar.Compress(cv, block, counter + (ulong)i, blockLength,
                    flags | Blake3Constants.Root,
                    MemoryMarshal.Cast<byte, uint>(expected.AsSpan(i * 64, 64)));
            OutputManyAvx2.HashOutput(cv, block, counter, blockLength, flags, actual);
            Assert.Equal(expected, actual);
        }
    }

    [Fact]
    public void SpanDigestHandlesEveryShortLengthAndOverlappingDestination()
    {
        for (int length = 0; length <= 1025; length++)
        {
            byte[] data = new byte[Math.Max(length, 32) + 2];
            HasherTests.MakeTestInput(length).CopyTo(data, 1);
            using var hasher = Hasher.New();
            hasher.Update(data.AsSpan(1, length));
            byte[] expected = hasher.Finalize().AsSpan().ToArray();
            Hasher.Hash(data.AsSpan(1, length), data.AsSpan(1, 32));
            Assert.Equal(expected, data.AsSpan(1, 32).ToArray());
            Assert.Equal(0, data[0]);
            Assert.Equal(0, data[^1]);
        }
    }
}
