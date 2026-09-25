using System.Text;
using Blake3.Managed;

namespace Blake3.Managed.Tests;

/// <summary>
/// The <c>out</c> factories build the hasher in the caller's storage instead of returning it by
/// value. They must be indistinguishable from the returning factories in everything but speed:
/// same digests in every mode, same argument checks, and a hasher that disposes and resets like
/// any other.
/// </summary>
[Collection(ParallelismCollection.Name)]
public class NewOutOverloadTests
{
    private static readonly int[] Lengths = { 0, 1, 64, 1023, 1024, 1025, 4096, 8192, 8193, 65536, 100_000 };

    private static readonly byte[] Key = Enumerable.Range(0, 32).Select(i => (byte)(i * 7 + 3)).ToArray();

    private static string Run(ref Hasher hasher, byte[] data, int outputLength)
    {
        try
        {
            // Two writes, so the chunk state and the CV stack are both exercised.
            hasher.Update(data.AsSpan(0, data.Length / 3));
            hasher.Update(data.AsSpan(data.Length / 3));
            var output = new byte[outputLength];
            hasher.Finalize(output);
            return Convert.ToHexString(output);
        }
        finally
        {
            hasher.Dispose();
        }
    }

    [Theory]
    [InlineData(32)]
    [InlineData(200)]
    public void NewOutMatchesNew(int outputLength)
    {
        foreach (int length in Lengths)
        {
            byte[] data = HasherTests.MakeTestInput(length);
            var viaReturn = Hasher.New();
            Hasher.New(out var viaOut);
            Assert.Equal(Run(ref viaReturn, data, outputLength), Run(ref viaOut, data, outputLength));
        }
    }

    [Theory]
    [InlineData(32)]
    [InlineData(200)]
    public void NewKeyedOutMatchesNewKeyed(int outputLength)
    {
        foreach (int length in Lengths)
        {
            byte[] data = HasherTests.MakeTestInput(length);
            var viaReturn = Hasher.NewKeyed(Key);
            Hasher.NewKeyed(Key, out var viaOut);
            Assert.Equal(Run(ref viaReturn, data, outputLength), Run(ref viaOut, data, outputLength));
        }
    }

    [Fact]
    public void NewDeriveKeyOutMatchesNewDeriveKey()
    {
        const string context = "BLAKE3 2019-12-27 16:29:52 test vectors context";
        foreach (int length in Lengths)
        {
            byte[] data = HasherTests.MakeTestInput(length);

            var fromString = Hasher.NewDeriveKey(context);
            Hasher.NewDeriveKey(context, out var fromStringOut);
            Assert.Equal(Run(ref fromString, data, 32), Run(ref fromStringOut, data, 32));

            var fromBytes = Hasher.NewDeriveKey(Encoding.UTF8.GetBytes(context));
            Hasher.NewDeriveKey(Encoding.UTF8.GetBytes(context), out var fromBytesOut);
            Assert.Equal(Run(ref fromBytes, data, 32), Run(ref fromBytesOut, data, 32));
        }
    }

    [Theory]
    [InlineData(0)]
    [InlineData(31)]
    [InlineData(33)]
    public void NewKeyedOutRejectsWrongKeyLength(int keyLength)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Hasher.NewKeyed(new byte[keyLength], out _));
    }

    [Fact]
    public void OutHasherResetsAndDisposesLikeAnyOther()
    {
        byte[] data = HasherTests.MakeTestInput(5000);
        string expected = Hasher.Hash(data).ToString();

        Hasher.New(out var hasher);
        try
        {
            hasher.Update(HasherTests.MakeTestInput(3000));
            hasher.Reset();
            hasher.Update(data);
            Assert.Equal(expected, hasher.Finalize().ToString());
        }
        finally
        {
            hasher.Dispose();
        }

        // Dispose clears the state, so a disposed hasher refuses further use.
        Assert.Throws<InvalidOperationException>(() =>
        {
            var disposed = hasher;
            disposed.Update(data);
        });
    }
}
