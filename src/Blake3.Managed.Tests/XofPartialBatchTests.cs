using Blake3.Managed;

namespace Blake3.Managed.Tests;

/// <summary>
/// Extended output is produced by three different pieces of code stitched together: a scalar
/// prefix when the seek lands mid-block, whole 512-byte batches from the eight-lane AVX2 kernel,
/// then -- since 2026-09-19 -- one more eight-lane batch whose 2..7 block prefix is kept, and a
/// scalar loop for anything still left. Every seam between those is a place for an off-by-one in
/// a block index or an output counter, and a wrong block index still yields plausible-looking
/// random bytes.
///
/// The invariant used throughout: XOF output is one deterministic byte stream, so any read must
/// equal the corresponding window of a single long read. That makes a 1 KiB reference the oracle
/// for every shorter read and every seek into it, with no hard-coded digests.
/// </summary>
public class XofPartialBatchTests
{
    private const int ReferenceLength = 4096;

    private static byte[] Reference(Hasher hasher)
    {
        var reference = new byte[ReferenceLength];
        hasher.Finalize(reference);
        return reference;
    }

    private static Hasher NewSeeded()
    {
        var hasher = Hasher.New();
        hasher.Update(HasherTests.MakeTestInput(3000));
        return hasher;
    }

    /// <summary>
    /// Every output length from nothing to past the first full batch. Covers the one-block
    /// scalar case, the whole 2..7 block partial-batch range, the exact 512-byte batch boundary,
    /// and a batch followed by a partial batch.
    /// </summary>
    [Fact]
    public void EveryOutputLengthMatchesThePrefixOfOneLongRead()
    {
        using var hasher = NewSeeded();
        byte[] reference = Reference(hasher);

        for (int length = 0; length <= 1200; length++)
        {
            var actual = new byte[length];
            hasher.Finalize(actual);

            Assert.Equal(reference.AsSpan(0, length).ToArray(), actual);
        }
    }

    /// <summary>
    /// The lengths that sit exactly on a block or batch edge, called out so a failure names the
    /// boundary it broke rather than "one of 1200".
    /// </summary>
    [Theory]
    [InlineData(63)]
    [InlineData(64)]   // one block: stays scalar
    [InlineData(65)]
    [InlineData(127)]
    [InlineData(128)]  // two blocks: the smallest partial batch
    [InlineData(129)]
    [InlineData(192)]
    [InlineData(256)]
    [InlineData(320)]
    [InlineData(384)]
    [InlineData(447)]
    [InlineData(448)]  // seven blocks: the largest partial batch
    [InlineData(449)]
    [InlineData(511)]
    [InlineData(512)]  // exactly one full batch, no partial batch at all
    [InlineData(513)]
    [InlineData(576)]  // a full batch plus the smallest partial batch
    [InlineData(1023)]
    [InlineData(1024)]
    public void BlockAndBatchBoundaryLengthsMatchTheLongRead(int length)
    {
        using var hasher = NewSeeded();
        byte[] reference = Reference(hasher);

        var actual = new byte[length];
        hasher.Finalize(actual);

        Assert.Equal(reference.AsSpan(0, length).ToArray(), actual);
    }

    /// <summary>
    /// A seek prefix is handled by scalar code and the batch that follows it starts from a
    /// non-zero block counter. That composition is where a block-index error would hide, because
    /// at seek zero the counter happens to match the array offset.
    /// </summary>
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(63)]
    [InlineData(64)]
    [InlineData(65)]
    [InlineData(127)]
    [InlineData(128)]
    [InlineData(511)]
    [InlineData(512)]
    [InlineData(513)]
    [InlineData(1000)]
    public void SeekedReadsMatchTheSameWindowOfTheLongRead(int seek)
    {
        using var hasher = NewSeeded();
        byte[] reference = Reference(hasher);

        foreach (int length in new[] { 1, 64, 128, 192, 448, 512, 576 })
        {
            if (seek + length > ReferenceLength) continue;

            var actual = new byte[length];
            hasher.Finalize((ulong)seek, actual);

            Assert.Equal(reference.AsSpan(seek, length).ToArray(), actual);
        }
    }

    /// <summary>
    /// Reading the stream in pieces must equal reading it in one go, whatever the piece sizes.
    /// A partial batch that advanced its block counter by the batch size rather than by the
    /// bytes actually kept would pass every fixed-length test above and fail here.
    /// </summary>
    [Theory]
    [InlineData(new[] { 128, 128, 128 })]
    [InlineData(new[] { 64, 448, 512 })]
    [InlineData(new[] { 192, 1, 63, 256 })]
    [InlineData(new[] { 511, 1, 512 })]
    [InlineData(new[] { 1, 1, 1, 1, 1 })]
    public void ConsecutiveSeekedReadsConcatenateIntoTheLongRead(int[] pieces)
    {
        using var hasher = NewSeeded();
        byte[] reference = Reference(hasher);

        var assembled = new List<byte>();
        ulong offset = 0;
        foreach (int piece in pieces)
        {
            var chunk = new byte[piece];
            hasher.Finalize(offset, chunk);
            assembled.AddRange(chunk);
            offset += (ulong)piece;
        }

        Assert.Equal(reference.AsSpan(0, assembled.Count).ToArray(), assembled.ToArray());
    }

    /// <summary>
    /// The partial batch is reached from the keyed and derive-key roots too, which carry
    /// different flags into the same kernel.
    /// </summary>
    [Theory]
    [InlineData(128)]
    [InlineData(448)]
    [InlineData(576)]
    public void KeyedAndDeriveKeyRootsAlsoMatchTheirLongReads(int length)
    {
        var key = new byte[32];
        for (int i = 0; i < key.Length; i++) key[i] = (byte)(i * 11 + 3);

        using var keyed = Hasher.NewKeyed(key);
        keyed.Update(HasherTests.MakeTestInput(3000));
        byte[] keyedReference = Reference(keyed);
        var keyedActual = new byte[length];
        keyed.Finalize(keyedActual);
        Assert.Equal(keyedReference.AsSpan(0, length).ToArray(), keyedActual);

        using var derived = Hasher.NewDeriveKey("Blake3.Managed test context");
        derived.Update(HasherTests.MakeTestInput(3000));
        byte[] derivedReference = Reference(derived);
        var derivedActual = new byte[length];
        derived.Finalize(derivedActual);
        Assert.Equal(derivedReference.AsSpan(0, length).ToArray(), derivedActual);
    }
}
