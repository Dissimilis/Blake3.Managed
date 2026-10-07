using Blake3.Managed;

namespace Blake3.Managed.Tests;

/// <summary>
/// The one-shot <c>Hash(input, output)</c> produces extended output for an input of at most one
/// chunk from the chunk state alone, without the incremental <c>HasherState</c>. The incremental
/// API is the oracle: its single-chunk finalize returns the same chunk output by construction.
/// </summary>
public class OneChunkXofTests
{
    public static IEnumerable<object[]> Cases()
    {
        foreach (int inputLength in new[] { 0, 1, 63, 64, 65, 127, 128, 129, 1023, 1024 })
        foreach (int outputLength in new[] { 65, 128, 129, 200, 1000, 4100 })
        {
            yield return new object[] { inputLength, outputLength };
        }
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void MatchesIncremental(int inputLength, int outputLength)
    {
        var input = HasherTests.MakeTestInput(inputLength);

        var expected = new byte[outputLength];
        Hasher.New(out var hasher);
        try
        {
            hasher.Update(input);
            hasher.Finalize(expected);
        }
        finally
        {
            hasher.Dispose();
        }

        var actual = new byte[outputLength];
        Hasher.Hash(input, actual);

        Assert.Equal(expected, actual);
    }

    /// <summary>
    /// Output overlapping the input: the input must be fully absorbed before any output is written.
    /// </summary>
    [Fact]
    public void OverlappingOutput()
    {
        var input = HasherTests.MakeTestInput(1024);
        var expected = new byte[200];
        Hasher.Hash(input, expected);

        var buffer = new byte[1024 + 200];
        input.CopyTo(buffer, 0);
        Hasher.Hash(buffer.AsSpan(0, 1024), buffer.AsSpan(100, 200));

        Assert.Equal(expected, buffer.AsSpan(100, 200).ToArray());
    }
}
