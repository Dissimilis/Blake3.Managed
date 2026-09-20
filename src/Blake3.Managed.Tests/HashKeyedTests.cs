using Blake3.Managed;

namespace Blake3.Managed.Tests;

/// <summary>
/// The keyed one-shot must agree with the incremental keyed path everywhere, since it is only
/// a faster route to the same digest. The sizes below straddle every dispatch boundary the
/// one-shot ladder has: single block, two blocks, sub-chunk, exact chunk, the serial tree, the
/// load-gated band and the unconditional parallel path.
/// </summary>
[Collection(ParallelismCollection.Name)]
public class HashKeyedTests
{
    private static byte[] MakeKey()
    {
        var key = new byte[32];
        for (int i = 0; i < key.Length; i++) key[i] = (byte)(i * 7 + 1);
        return key;
    }

    public static TheoryData<int> Sizes()
    {
        var data = new TheoryData<int>();
        foreach (var n in new[]
        {
            0, 1, 63, 64, 65, 127, 128, 129, 1023, 1024, 1025, 2048, 3072,
            8192, 32 * 1024, 32 * 1024 + 1, 64 * 1024, 72 * 1024, 72 * 1024 + 1,
            256 * 1024, 256 * 1024 + 1, 1024 * 1024,
        })
        {
            data.Add(n);
        }

        return data;
    }

    [Theory]
    [MemberData(nameof(Sizes))]
    public void HashKeyed_MatchesIncrementalKeyedPath(int length)
    {
        var key = MakeKey();
        var input = HasherTests.MakeTestInput(length);

        using var hasher = Hasher.NewKeyed(key);
        hasher.Update(input);
        var expected = hasher.Finalize();

        var actual = Hasher.HashKeyed(key, input);

        Assert.Equal(expected.ToString(), actual.ToString());
    }

    [Fact]
    public void HashKeyed_DiffersFromUnkeyed()
    {
        var input = HasherTests.MakeTestInput(1024);
        Assert.NotEqual(Hasher.Hash(input).ToString(), Hasher.HashKeyed(MakeKey(), input).ToString());
    }

    [Fact]
    public void HashKeyed_DiffersByKey()
    {
        var input = HasherTests.MakeTestInput(4096);
        var a = MakeKey();
        var b = MakeKey();
        b[0] ^= 0xFF;

        Assert.NotEqual(Hasher.HashKeyed(a, input).ToString(), Hasher.HashKeyed(b, input).ToString());
    }

    [Theory]
    [InlineData(0)]
    [InlineData(31)]
    [InlineData(33)]
    public void HashKeyed_RejectsWrongKeyLength(int keyLength)
    {
        var input = HasherTests.MakeTestInput(64);
        var key = new byte[keyLength];

        Assert.Throws<ArgumentOutOfRangeException>(() => Hasher.HashKeyed(key, input));
    }

    /// <summary>
    /// The single-thread path and the fan-out must agree: every unit the parallel path hashes is
    /// a canonical subtree, so capping parallelism can only change the schedule, never the digest.
    /// </summary>
    [Fact]
    public void HashKeyed_SerialAndParallelAgree()
    {
        var key = MakeKey();
        var input = HasherTests.MakeTestInput(1024 * 1024);

        int previous = Hasher.MaxDegreeOfParallelism;
        try
        {
            Hasher.MaxDegreeOfParallelism = 1;
            var serial = Hasher.HashKeyed(key, input);

            Hasher.MaxDegreeOfParallelism = -1;
            var parallel = Hasher.HashKeyed(key, input);

            Assert.Equal(serial.ToString(), parallel.ToString());
        }
        finally
        {
            Hasher.MaxDegreeOfParallelism = previous;
        }
    }
}
