using System;
using System.Threading.Tasks;
using Blake3.Managed.Internal;
using Xunit;

namespace Blake3.Managed.Tests;

public class FanOutGateTests
{
    [Fact]
    public void IsMidSize_CoversExactlyTheGatedBand()
    {
        Assert.False(Blake3Tree.IsMidSize(Blake3Tree.MaxUsefulLength));
        Assert.True(Blake3Tree.IsMidSize(Blake3Tree.MaxUsefulLength + 1));
        Assert.True(Blake3Tree.IsMidSize(Blake3Tree.LoadGatedLength));
        Assert.False(Blake3Tree.IsMidSize(Blake3Tree.LoadGatedLength + 1));
    }

    [Fact]
    public void ConcurrentMidSizeHashesMatchSerialDigest()
    {
        // 48 KB sits in the load-gated band: some of these calls fan out, others find a
        // parallel hash in flight and take the serial tree. Every one must agree.
        var data = new byte[48 * 1024];
        new Random(48).NextBytes(data);
        using var serial = Hasher.New();
        serial.Update(data);
        var expected = serial.Finalize();

        Parallel.For(0, 256, new ParallelOptions { MaxDegreeOfParallelism = 32 }, _ =>
        {
            Assert.Equal(expected, Hasher.Hash(data));
            Span<byte> span = stackalloc byte[32];
            Hasher.Hash(data, span);
            Assert.True(expected.AsSpan().SequenceEqual(span));
        });
    }
}
