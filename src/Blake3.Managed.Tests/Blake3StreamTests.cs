using System;
using System.IO;

namespace Blake3.Managed.Tests;

public class Blake3StreamTests
{
    [Fact]
    public void TestHashRead()
    {
        var stream = new MemoryStream(HasherTests.BigData);
        using var blake3Stream = new Blake3Stream(stream);
        _ = blake3Stream.Read(new byte[HasherTests.BigData.Length]);
        Assert.Equal(HasherTests.BigExpected, blake3Stream.ComputeHash().ToString());
    }

    [Fact]
    public void TestHashWrite()
    {
        var stream = new MemoryStream();
        using var blake3Stream = new Blake3Stream(stream);
        blake3Stream.Write(HasherTests.BigData);
        Assert.Equal(HasherTests.BigExpected, blake3Stream.ComputeHash().ToString());
    }

    [Fact]
    public void TestResetHash_ThenRecompute()
    {
        var stream = new MemoryStream();
        using var blake3Stream = new Blake3Stream(stream);
        blake3Stream.Write(HasherTests.BigData);
        blake3Stream.ResetHash();
        blake3Stream.Write(HasherTests.SimpleData);
        Assert.Equal(HasherTests.SimpleExpected, blake3Stream.ComputeHash().ToString());
    }
}
