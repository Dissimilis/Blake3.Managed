using System;
using System.Linq;

namespace Blake3.Managed.Tests;

public class Blake3HashAlgorithmTests
{
    [Fact]
    public void TestComputeHash()
    {
        using var hashAlgorithm = new Blake3HashAlgorithm();
        var result = hashAlgorithm.ComputeHash(HasherTests.BigData);
        var hash = Hash.FromBytes(result);
        Assert.Equal(HasherTests.BigExpected, hash.ToString());
    }

    [Fact]
    public void TestTryHashFinal_BufferTooSmall()
    {
        using var hashAlgorithm = new Blake3HashAlgorithm();

        Span<byte> destination = stackalloc byte[Hash.Size - 1];
        Assert.False(hashAlgorithm.TryComputeHash(HasherTests.BigData, destination, out var written));
        Assert.Equal(0, written);
    }

    [Fact]
    public void TestChunking_And_InitializeReuse()
    {
        var data = HasherTests.BigData;

        using var hashAlgorithm = new Blake3HashAlgorithm();

        hashAlgorithm.TransformBlock(data, 0, 7, null, 0);
        hashAlgorithm.TransformBlock(data, 7, 13, null, 0);
        hashAlgorithm.TransformBlock(data, 20, data.Length - 20, null, 0);
        hashAlgorithm.TransformFinalBlock(Array.Empty<byte>(), 0, 0);

        var hash1 = Hash.FromBytes(hashAlgorithm.Hash);
        Assert.Equal(HasherTests.BigExpected, hash1.ToString());

        hashAlgorithm.Initialize();
        var result = hashAlgorithm.ComputeHash(data);
        var hash2 = Hash.FromBytes(result);
        Assert.Equal(HasherTests.BigExpected, hash2.ToString());
    }
}
