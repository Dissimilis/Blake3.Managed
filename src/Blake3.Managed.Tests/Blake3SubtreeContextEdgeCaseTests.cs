using System;

namespace Blake3.Managed.Tests;

/// <summary>
/// The corners: modes and offsets the main suite exercises only at one size, cases a review
/// pointed out were claimed but not covered, and the arithmetic that only breaks at the extremes.
/// </summary>
public class Blake3SubtreeContextEdgeCaseTests
{
    private const int ChunkLen = 1024;

    private static byte[] Data(int length)
    {
        var data = new byte[length];
        for (int i = 0; i < length; i++) data[i] = (byte)(i * 7 % 251);
        return data;
    }

    private static Blake3Subtree[] HashPieces(Blake3SubtreeContext ctx, byte[] data, int pieceSize)
    {
        int pieceCount = data.Length == 0 ? 0 : (data.Length + pieceSize - 1) / pieceSize;
        var pieces = new Blake3Subtree[pieceCount];

        for (int i = 0; i < pieceCount; i++)
        {
            int offset = i * pieceSize;
            int length = Math.Min(pieceSize, data.Length - offset);
            pieces[i] = ctx.HashSubtree(data.AsSpan(offset, length), i);
        }

        return pieces;
    }

    [Fact]
    public void DeriveKeySinglePieceExtendedOutputMatchesTheHasher()
    {
        const string context = "Blake3.Managed 2026-09-01 subtree tests";
        const int pieceSize = 16 * ChunkLen;
        var data = Data(4096);

        using var ctx = Blake3SubtreeContext.CreateDeriveKey(context, pieceSize, data.Length);
        using var hasher = Hasher.NewDeriveKey(context);
        hasher.Update(data);

        var actual = new byte[97];
        ctx.Finalize(HashPieces(ctx, data, pieceSize), 11L, actual);
        var expected = new byte[97];
        hasher.Finalize(11UL, expected);

        Assert.Equal(expected, actual);
    }

    [Fact]
    public void KeyedExtendedOutputAtAnOffsetMatchesTheKeyedHasher()
    {
        const int pieceSize = 2 * ChunkLen;
        var key = Data(32);
        var data = Data(20_000);

        using var ctx = Blake3SubtreeContext.CreateKeyed(key, pieceSize, data.Length);
        using var hasher = Hasher.NewKeyed(key);
        hasher.Update(data);

        var actual = new byte[150];
        ctx.Finalize(HashPieces(ctx, data, pieceSize), 64L, actual);
        var expected = new byte[150];
        hasher.Finalize(64UL, expected);

        Assert.Equal(expected, actual);
    }

    [Fact]
    public void PiecesFromAnUnknownLengthContextFinalizeThroughAKnownLengthContext()
    {
        // The context tag deliberately excludes the total length, so these contexts are
        // compatible. That is the point of tagging by value, and it is also why the final-length
        // check is the only thing standing between the caller and a wrong digest.
        const int pieceSize = 2 * ChunkLen;
        var data = Data(pieceSize + 100);

        using var unknown = Blake3SubtreeContext.Create(pieceSize);
        using var known = Blake3SubtreeContext.Create(pieceSize, data.Length);
        using var wrongLength = Blake3SubtreeContext.Create(pieceSize, data.Length + 1);

        var pieces = HashPieces(unknown, data, pieceSize);

        Assert.Equal(Hasher.Hash(data).ToString(), known.Finalize(pieces).ToString());
        Assert.Throws<ArgumentException>(() => wrongLength.Finalize(pieces));
    }

    [Fact]
    public void FinalizeDoesNotConsumeThePieces()
    {
        // The fold reduces chaining values in place, so it must work on a copy: doing it to the
        // caller's pieces would leave them corrupt after the first call.
        const int pieceSize = 2 * ChunkLen;
        var data = Data(pieceSize * 5 + 3);
        using var ctx = Blake3SubtreeContext.Create(pieceSize, data.Length);

        var pieces = HashPieces(ctx, data, pieceSize);
        var first = ctx.Finalize(pieces);
        var second = ctx.Finalize(pieces);

        Assert.Equal(Hasher.Hash(data).ToString(), first.ToString());
        Assert.Equal(first.ToString(), second.ToString());
    }

    [Fact]
    public void FoldsAPieceCountLargeEnoughToBatchParentsAndCarryAnOddOne()
    {
        // 1001 pieces drives the 8-wide parent kernel over several levels, with an odd child
        // carried up at more than one of them.
        const int pieceSize = ChunkLen;
        var data = Data(pieceSize * 1000 + 17);
        using var ctx = Blake3SubtreeContext.Create(pieceSize, data.Length);

        Assert.Equal(1001, ctx.PieceCount);
        Assert.Equal(Hasher.Hash(data).ToString(),
            ctx.Finalize(HashPieces(ctx, data, pieceSize)).ToString());
    }

    [Fact]
    public void PieceOffsetsAreCorrectBeyondTheRangeOfAnInt()
    {
        // Offsets are long: at a 1 MiB piece size, piece 5000 already sits past int.MaxValue.
        const int pieceSize = 1024 * 1024;
        const long totalLength = 8L * 1024 * 1024 * 1024;
        using var ctx = Blake3SubtreeContext.Create(pieceSize, totalLength);

        Assert.Equal(8192, ctx.PieceCount);
        Assert.Equal(5000L * pieceSize, ctx.GetPieceOffset(5000));
        Assert.True(ctx.GetPieceOffset(5000) > int.MaxValue);
        Assert.Equal(totalLength - pieceSize, ctx.GetPieceOffset(ctx.PieceCount - 1));
        Assert.Equal(pieceSize, ctx.GetPieceLength(ctx.PieceCount - 1));
    }

    [Fact]
    public void AcceptsTheLargestRepresentablePieceSize()
    {
        // 1 GiB is 2^20 chunks: the largest power-of-two chunk count whose byte size fits an int.
        using var ctx = Blake3SubtreeContext.Create(1 << 30);
        Assert.Equal(1 << 30, ctx.PieceSize);
    }

    [Fact]
    public void RejectsAPieceSizeThatIsNotRepresentable()
    {
        Assert.Throws<ArgumentException>(() => Blake3SubtreeContext.Create(int.MinValue));
        Assert.Throws<ArgumentException>(() => Blake3SubtreeContext.Create(int.MaxValue));
    }

    [Fact]
    public void EmptyInputWorksWithoutATotalLength()
    {
        using var ctx = Blake3SubtreeContext.Create(ChunkLen);

        Assert.Equal(Hasher.Hash(Array.Empty<byte>()).ToString(),
            ctx.Finalize(Array.Empty<Blake3Subtree>()).ToString());
    }

    [Fact]
    public void EmptyInputSupportsExtendedOutput()
    {
        using var ctx = Blake3SubtreeContext.Create(ChunkLen, 0);

        var actual = new byte[80];
        ctx.Finalize(Array.Empty<Blake3Subtree>(), 5L, actual);

        using var hasher = Hasher.New();
        var expected = new byte[80];
        hasher.Finalize(5UL, expected);

        Assert.Equal(expected, actual);
    }

    [Fact]
    public void SinglePieceWorksWithoutATotalLength()
    {
        var data = Data(500);
        using var ctx = Blake3SubtreeContext.Create(ChunkLen);

        var pieces = new[] { ctx.HashSubtree(data, 0) };

        Assert.Equal(Hasher.Hash(data).ToString(), ctx.Finalize(pieces).ToString());
    }

    [Fact]
    public void ADefaultPieceReportsNothing()
    {
        var piece = default(Blake3Subtree);

        Assert.Equal(0, piece.PieceIndex);
        Assert.Equal(0, piece.Length);
    }

    [Fact]
    public void ContextsThatDifferOnlyByModeDoNotSharePieces()
    {
        // Same piece size, same piece count: only the mode differs, and the tag must separate them.
        const int pieceSize = 2 * ChunkLen;
        var data = Data(pieceSize * 2);
        var key = Data(32);

        using var plain = Blake3SubtreeContext.Create(pieceSize, data.Length);
        using var derive = Blake3SubtreeContext.CreateDeriveKey("ctx", pieceSize, data.Length);
        using var keyed = Blake3SubtreeContext.CreateKeyed(key, pieceSize, data.Length);

        Assert.Throws<ArgumentException>(() => derive.Finalize(HashPieces(plain, data, pieceSize)));
        Assert.Throws<ArgumentException>(() => plain.Finalize(HashPieces(keyed, data, pieceSize)));
        Assert.Throws<ArgumentException>(() => keyed.Finalize(HashPieces(derive, data, pieceSize)));
    }

    [Fact]
    public void ContextsWithDifferentKeysDoNotSharePieces()
    {
        const int pieceSize = 2 * ChunkLen;
        var data = Data(pieceSize * 2);

        using var one = Blake3SubtreeContext.CreateKeyed(Data(32), pieceSize, data.Length);
        using var two = Blake3SubtreeContext.CreateKeyed(new byte[32], pieceSize, data.Length);

        Assert.Throws<ArgumentException>(() => two.Finalize(HashPieces(one, data, pieceSize)));
    }

    [Fact]
    public void EveryPieceSizeAgreesWithEveryOtherOnTheSameInput()
    {
        // The digest is a property of the input, not of how the caller happened to cut it up.
        var data = Data(64 * ChunkLen + 77);
        string expected = Hasher.Hash(data).ToString();

        for (int chunks = 1; chunks <= 128; chunks *= 2)
        {
            int pieceSize = chunks * ChunkLen;
            using var ctx = Blake3SubtreeContext.Create(pieceSize, data.Length);

            Assert.Equal(expected, ctx.Finalize(HashPieces(ctx, data, pieceSize)).ToString());
        }
    }

    [Fact]
    public void ZeroLengthOutputIsAccepted()
    {
        var data = Data(4096);
        using var ctx = Blake3SubtreeContext.Create(ChunkLen, data.Length);

        ctx.Finalize(HashPieces(ctx, data, ChunkLen), Array.Empty<byte>());
    }
}
