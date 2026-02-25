using System;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;

namespace Blake3.Managed.Tests;

public class HasherTests : Blake3TestsBase
{
    private const string SimpleInput = "BLAKE3";
    public const string SimpleExpected = "f890484173e516bfd935ef3d22b912dc9738de38743993cfedf2c9473b3216a4";
    public const string SimpleKeyedExpected = "52a1c5369af0590e26ccbb31d052485addcfe2599e858711579fb25aa878c6b8";
    public const string SimpleDeriveKeyExpected = "aed725e67e41969964e90fc83f44e17efab90f159a375d3bd213714df2db5ea4";
    public const string BigExpected = "64479cf7293960210547db8d982359e0c4ce054525ed7086cf93030828fc0533";
    public static readonly byte[] SimpleData = Encoding.UTF8.GetBytes(SimpleInput);
    public static readonly byte[] BigData = Enumerable.Range(0, 1024 * 1024).Select(x => (byte)x).ToArray();

    [Fact]
    public void TestHashSimple()
    {
        AssertTextAreEqual(SimpleExpected, Hasher.Hash(SimpleData).ToString());
    }

    [Fact]
    public void TestUpdateBig()
    {
        using var hasher = Hasher.New();
        hasher.Update(BigData);
        var hash = hasher.Finalize();
        AssertTextAreEqual(BigExpected, hash.ToString());
    }

    [Fact]
    public void TestUpdateJoinBig()
    {
        using var hasher = Hasher.New();
        hasher.UpdateWithJoin(BigData);
        var hash = hasher.Finalize();
        AssertTextAreEqual(BigExpected, hash.ToString());
    }

    [Fact]
    public void TestUpdateSimpleKeyed()
    {
        using var hasher = Hasher.NewKeyed(new ReadOnlySpan<byte>(Enumerable.Range(0, 32).Select(x => (byte)x).ToArray()));
        hasher.Update(SimpleData);
        var hash = hasher.Finalize();
        AssertTextAreEqual(SimpleKeyedExpected, hash.ToString());
    }

    [Fact]
    public void TestUpdateSimpleDeriveKey()
    {
        using var hasher = Hasher.NewDeriveKey(new ReadOnlySpan<byte>(Enumerable.Range(0, 32).Select(x => (byte)x).ToArray()));
        hasher.Update(SimpleData);
        var hash = hasher.Finalize();
        AssertTextAreEqual(SimpleDeriveKeyExpected, hash.ToString());
    }

    internal static byte[] MakeTestInput(int length)
    {
        var data = new byte[length];
        for (int i = 0; i < length; i++)
        {
            data[i] = (byte)(i % 251);
        }
        return data;
    }

    private static TestVectorFile LoadTestVectors()
    {
        var path = Path.Combine(AppContext.BaseDirectory, "TestVectors.json");
        var json = File.ReadAllText(path);
        return JsonSerializer.Deserialize<TestVectorFile>(json)!;
    }

    [Fact]
    public void TestOfficialVectors_Hash()
    {
        var vectors = LoadTestVectors();
        foreach (var tc in vectors.Cases)
        {
            var input = MakeTestInput(tc.InputLen);
            using var hasher = Hasher.New();
            hasher.Update(input);

            var hash32 = hasher.Finalize();
            var expected32 = tc.Hash.Substring(0, 64);
            AssertTextAreEqual(expected32, hash32.ToString());

            var extended = new byte[131];
            hasher.Finalize(extended);
            var expectedExtended = tc.Hash;
            AssertTextAreEqual(expectedExtended, Convert.ToHexString(extended).ToLowerInvariant());
        }
    }

    [Fact]
    public void TestOfficialVectors_KeyedHash()
    {
        var vectors = LoadTestVectors();
        var key = Encoding.ASCII.GetBytes(vectors.Key);
        foreach (var tc in vectors.Cases)
        {
            var input = MakeTestInput(tc.InputLen);
            using var hasher = Hasher.NewKeyed(key);
            hasher.Update(input);

            var hash32 = hasher.Finalize();
            var expected32 = tc.KeyedHash.Substring(0, 64);
            AssertTextAreEqual(expected32, hash32.ToString());

            var extended = new byte[131];
            hasher.Finalize(extended);
            AssertTextAreEqual(tc.KeyedHash, Convert.ToHexString(extended).ToLowerInvariant());
        }
    }

    [Fact]
    public void TestOfficialVectors_DeriveKey()
    {
        var vectors = LoadTestVectors();
        foreach (var tc in vectors.Cases)
        {
            var input = MakeTestInput(tc.InputLen);
            using var hasher = Hasher.NewDeriveKey(vectors.ContextString);
            hasher.Update(input);

            var hash32 = hasher.Finalize();
            var expected32 = tc.DeriveKey.Substring(0, 64);
            AssertTextAreEqual(expected32, hash32.ToString());

            var extended = new byte[131];
            hasher.Finalize(extended);
            AssertTextAreEqual(tc.DeriveKey, Convert.ToHexString(extended).ToLowerInvariant());
        }
    }

    [Fact]
    public void TestFinalizeWithOffset()
    {
        using var hasher = Hasher.New();
        hasher.Update(BigData);
        var bigHash = new byte[1024 * 1024].AsSpan();
        hasher.Finalize(bigHash);

        var loopHash = new byte[1024].AsSpan();
        var reconstructedHash = new byte[1024 * 1024].AsSpan();
        for (var i = bigHash.Length; i > 0; i -= 1024)
        {
            hasher.Finalize(i - 1024, loopHash);
            AssertTextAreEqual(bigHash.Slice(i - 1024, 1024).ToString(), loopHash.ToString());
            loopHash.CopyTo(reconstructedHash.Slice(i - 1024, 1024));
        }

        Assert.True(bigHash.SequenceEqual(reconstructedHash.ToArray()));
    }

    [Fact]
    public void TestXof_ShortOutputIsPrefixOfLong()
    {
        using var hasher = Hasher.New();
        hasher.Update(SimpleData);
        var hash32 = hasher.Finalize();
        var hash64 = new byte[64];
        hasher.Finalize(hash64);
        AssertTextAreEqual(SimpleExpected, hash32.ToString());
        AssertTextAreEqual(SimpleExpected, Convert.ToHexString(hash64.AsSpan(0, 32)).ToLowerInvariant());
    }

    [Fact]
    public void TestReset_ProducesSameHashAsFresh()
    {
        using var hasher = Hasher.New();
        hasher.Update(MakeTestInput(512));
        hasher.Reset();
        hasher.Update(SimpleData);
        AssertTextAreEqual(SimpleExpected, hasher.Finalize().ToString());
    }

    [Fact]
    public void TestMultipleFinalizeCalls_AreNonDestructive()
    {
        using var hasher = Hasher.New();
        hasher.Update(SimpleData);
        var hash1 = hasher.Finalize();
        var hash2 = hasher.Finalize();
        AssertTextAreEqual(hash1.ToString(), hash2.ToString());
        AssertTextAreEqual(SimpleExpected, hash1.ToString());
    }

    [Fact]
    public void TestEmptyUpdate_HasNoEffect()
    {
        using var hasher = Hasher.New();
        hasher.Update(ReadOnlySpan<byte>.Empty);
        hasher.Update(SimpleData);
        hasher.Update(ReadOnlySpan<byte>.Empty);
        AssertTextAreEqual(SimpleExpected, hasher.Finalize().ToString());
    }

    [Fact]
    public void TestModeIsolation_AllThreeDifferent()
    {
        var plain = Hasher.Hash(SimpleData).ToString();

        using var hk = Hasher.NewKeyed(new ReadOnlySpan<byte>(Enumerable.Range(0, 32).Select(x => (byte)x).ToArray()));
        hk.Update(SimpleData);
        var keyed = hk.Finalize().ToString();

        using var hd = Hasher.NewDeriveKey(new ReadOnlySpan<byte>(Enumerable.Range(0, 32).Select(x => (byte)x).ToArray()));
        hd.Update(SimpleData);
        var derived = hd.Finalize().ToString();

        Assert.NotEqual(plain, keyed);
        Assert.NotEqual(plain, derived);
        Assert.NotEqual(keyed, derived);
    }
}
