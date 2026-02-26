using System;
using System.Linq;

namespace Blake3.Managed.Tests;

public class HashTests
{
    [Fact]
    public unsafe void TestSize()
    {
        Assert.Equal(32, sizeof(Hash));
    }

    [Fact]
    public void TestFromBytes_And_ToString()
    {
        Assert.Equal("000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f", Hash.FromBytes(Enumerable.Range(0, 32).Select(x => (byte)x).ToArray()).ToString());
    }

    [Fact]
    public void TestEquality()
    {
        var hash1 = Hash.FromBytes(Enumerable.Range(0, 32).Select(x => (byte)x).ToArray());
        var hash2 = Hash.FromBytes(Enumerable.Range(0, 32).Select(x => (byte)x).ToArray());
        Assert.Equal(hash1, hash2);
        Assert.True(hash1 == hash2);

        var hash3 = Hash.FromBytes(Enumerable.Range(1, 32).Select(x => (byte)x).ToArray());
        Assert.NotEqual(hash1, hash3);
        Assert.True(hash1 != hash3);
    }

    [Fact]
    public void TestAsSpan_RoundTrip()
    {
        var bytes = Enumerable.Range(0, 32).Select(x => (byte)x).ToArray();
        var hash = Hash.FromBytes(bytes);
        Assert.Equal(32, hash.AsSpan().Length);
        Assert.True(hash.AsSpan().SequenceEqual(bytes));
    }

    [Fact]
    public void TestFromBytes_WrongSize_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Hash.FromBytes(new byte[31]));
        Assert.Throws<ArgumentOutOfRangeException>(() => Hash.FromBytes(new byte[33]));
        Assert.Throws<ArgumentOutOfRangeException>(() => Hash.FromBytes(Array.Empty<byte>()));
    }
}
