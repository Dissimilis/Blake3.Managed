using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace Blake3.Managed.Tests;

public class Blake3SubtreeContextTests
{
    private const int ChunkLen = 1024;

    private static byte[] Data(int length)
    {
        var data = new byte[length];
        for (int i = 0; i < length; i++) data[i] = (byte)(i * 7 % 251);
        return data;
    }

    /// <summary>Hashes every piece in order and folds, the plain path used by most tests.</summary>
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

    [Theory]
    [InlineData(1 * ChunkLen)]
    [InlineData(2 * ChunkLen)]
    [InlineData(4 * ChunkLen)]
    [InlineData(8 * ChunkLen)]
    [InlineData(16 * ChunkLen)]
    [InlineData(64 * ChunkLen)]
    [InlineData(1024 * ChunkLen)]
    public void MatchesWholeInputHashAtEveryPieceSize(int pieceSize)
    {
        var data = Data(300_000);
        using var ctx = Blake3SubtreeContext.Create(pieceSize, data.Length);

        var hash = ctx.Finalize(HashPieces(ctx, data, pieceSize));

        Assert.Equal(Hasher.Hash(data).ToString(), hash.ToString());
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(7)]
    [InlineData(8)]
    [InlineData(9)]
    [InlineData(37)]
    [InlineData(64)]
    public void MatchesWholeInputHashAtEveryPieceCount(int pieceCount)
    {
        const int pieceSize = 2 * ChunkLen;
        var data = Data(pieceCount * pieceSize);
        using var ctx = Blake3SubtreeContext.Create(pieceSize, data.Length);

        var hash = ctx.Finalize(HashPieces(ctx, data, pieceSize));

        Assert.Equal(Hasher.Hash(data).ToString(), hash.ToString());
    }

    [Fact]
    public void MatchesWholeInputHashForEveryShortFinalPieceLength()
    {
        const int pieceSize = 2 * ChunkLen;

        // Every possible length of trailing piece, over a two-piece input, so both the fold and
        // the partial-chunk path inside the piece are exercised at every offset.
        for (int tail = 1; tail <= pieceSize; tail++)
        {
            var data = Data(pieceSize + tail);
            using var ctx = Blake3SubtreeContext.Create(pieceSize, data.Length);

            var hash = ctx.Finalize(HashPieces(ctx, data, pieceSize));

            Assert.Equal(Hasher.Hash(data).ToString(), hash.ToString());
        }
    }

    [Fact]
    public void SinglePieceInputNeedsNoFold()
    {
        const int pieceSize = 16 * ChunkLen;

        foreach (int length in new[] { 1, 63, 64, 65, 1023, 1024, 1025, pieceSize })
        {
            var data = Data(length);
            using var ctx = Blake3SubtreeContext.Create(pieceSize, data.Length);

            var hash = ctx.Finalize(HashPieces(ctx, data, pieceSize));

            Assert.Equal(Hasher.Hash(data).ToString(), hash.ToString());
        }
    }

    [Fact]
    public void EmptyInputHasNoPieces()
    {
        using var ctx = Blake3SubtreeContext.Create(ChunkLen, 0);

        Assert.Equal(0, ctx.PieceCount);
        Assert.Equal(Hasher.Hash(Array.Empty<byte>()).ToString(),
            ctx.Finalize(Array.Empty<Blake3Subtree>()).ToString());
    }

    [Fact]
    public void OutOfOrderAndConcurrentHashingGivesTheSameDigest()
    {
        const int pieceSize = 4 * ChunkLen;
        var data = Data(200 * ChunkLen);
        using var ctx = Blake3SubtreeContext.Create(pieceSize, data.Length);

        var pieces = new Blake3Subtree[ctx.PieceCount];
        Parallel.ForEach(Enumerable.Range(0, ctx.PieceCount).Reverse(), i =>
        {
            pieces[i] = ctx.HashSubtree(
                data.AsSpan((int)ctx.GetPieceOffset(i), (int)ctx.GetPieceLength(i)), i);
        });

        Assert.Equal(Hasher.Hash(data).ToString(), ctx.Finalize(pieces).ToString());
    }

    [Fact]
    public void UnknownTotalLengthGivesTheSameDigest()
    {
        const int pieceSize = 8 * ChunkLen;
        var data = Data(100_000);

        using var known = Blake3SubtreeContext.Create(pieceSize, data.Length);
        using var unknown = Blake3SubtreeContext.Create(pieceSize);

        Assert.Equal(known.Finalize(HashPieces(known, data, pieceSize)).ToString(),
            unknown.Finalize(HashPieces(unknown, data, pieceSize)).ToString());
    }

    [Fact]
    public void KeyedMatchesKeyedWholeInputHash()
    {
        const int pieceSize = 4 * ChunkLen;
        var key = Data(32);
        var data = Data(70_000);

        using var ctx = Blake3SubtreeContext.CreateKeyed(key, pieceSize, data.Length);
        using var hasher = Hasher.NewKeyed(key);
        hasher.Update(data);

        Assert.Equal(hasher.Finalize().ToString(),
            ctx.Finalize(HashPieces(ctx, data, pieceSize)).ToString());
    }

    [Fact]
    public void DeriveKeyMatchesDeriveKeyWholeInputHash()
    {
        const int pieceSize = 4 * ChunkLen;
        const string context = "BLAKE3 2019-12-27 16:29:52 test vectors context";
        var data = Data(70_000);

        using var ctx = Blake3SubtreeContext.CreateDeriveKey(context, pieceSize, data.Length);
        using var hasher = Hasher.NewDeriveKey(context);
        hasher.Update(data);

        Assert.Equal(hasher.Finalize().ToString(),
            ctx.Finalize(HashPieces(ctx, data, pieceSize)).ToString());
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(63)]
    [InlineData(64)]
    [InlineData(1000)]
    public void ExtendedOutputMatchesTheIncrementalHasher(int offset)
    {
        const int pieceSize = 2 * ChunkLen;
        var data = Data(50_000);

        using var ctx = Blake3SubtreeContext.Create(pieceSize, data.Length);
        var pieces = HashPieces(ctx, data, pieceSize);

        var actual = new byte[131];
        ctx.Finalize(pieces, (long)offset, actual);

        using var hasher = Hasher.New();
        hasher.Update(data);
        var expected = new byte[131];
        hasher.Finalize((ulong)offset, expected);

        Assert.Equal(expected, actual);
    }

    [Fact]
    public void ExtendedOutputFromASinglePieceMatchesTheIncrementalHasher()
    {
        var data = Data(700);

        using var ctx = Blake3SubtreeContext.Create(ChunkLen, data.Length);
        var pieces = HashPieces(ctx, data, ChunkLen);

        var actual = new byte[200];
        ctx.Finalize(pieces, actual);

        using var hasher = Hasher.New();
        hasher.Update(data);
        var expected = new byte[200];
        hasher.Finalize(expected);

        Assert.Equal(expected, actual);
    }

    [Fact]
    public void PieceOffsetAndLengthDescribeTheInput()
    {
        const int pieceSize = 2 * ChunkLen;
        using var ctx = Blake3SubtreeContext.Create(pieceSize, pieceSize * 3 + 5);

        Assert.Equal(4, ctx.PieceCount);
        Assert.Equal(0, ctx.GetPieceOffset(0));
        Assert.Equal(pieceSize * 3, ctx.GetPieceOffset(3));
        Assert.Equal(pieceSize, ctx.GetPieceLength(2));
        Assert.Equal(5, ctx.GetPieceLength(3));
    }

    [Fact]
    public void LengthDerivedMembersThrowWithoutATotalLength()
    {
        using var ctx = Blake3SubtreeContext.Create(ChunkLen);

        Assert.False(ctx.HasTotalLength);
        Assert.Throws<InvalidOperationException>(() => ctx.TotalLength);
        Assert.Throws<InvalidOperationException>(() => ctx.PieceCount);
        Assert.Throws<InvalidOperationException>(() => ctx.GetPieceOffset(0));
        Assert.Throws<InvalidOperationException>(() => ctx.GetPieceLength(0));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(1023)]
    [InlineData(1025)]
    [InlineData(3 * ChunkLen)]
    [InlineData(5 * ChunkLen)]
    [InlineData(6 * ChunkLen)]
    public void RejectsAPieceSizeThatIsNotACanonicalSubtree(int pieceSize)
    {
        Assert.Throws<ArgumentException>(() => Blake3SubtreeContext.Create(pieceSize));
    }

    [Theory]
    [InlineData(ChunkLen)]
    [InlineData(2 * ChunkLen)]
    [InlineData(4096 * ChunkLen)]
    public void AcceptsAPieceSizeThatIsACanonicalSubtree(int pieceSize)
    {
        using var ctx = Blake3SubtreeContext.Create(pieceSize);
        Assert.Equal(pieceSize, ctx.PieceSize);
    }

    [Theory]
    [InlineData(1, 1)]
    [InlineData(1023, 1023)]
    [InlineData(1025, 1025)]
    [InlineData(3 * ChunkLen, 3 * ChunkLen)]
    [InlineData(5 * ChunkLen + 17, 5 * ChunkLen + 17)]
    [InlineData(5 * ChunkLen + 17, 3 * ChunkLen)]
    [InlineData(5 * ChunkLen + 17, 0)]
    [InlineData(1, 0)]
    public void AcceptsAnyPieceSizeWhenTheWholeInputFitsInOnePiece(int pieceSize, int totalLength)
    {
        // With one piece the piece is the whole input, so alignment never comes into play. This
        // lets a caller use pieceSize == fileLength for a file that is not worth splitting.
        var data = Data(totalLength);
        using var ctx = Blake3SubtreeContext.Create(pieceSize, totalLength);

        Assert.Equal(pieceSize, ctx.PieceSize);
        Assert.Equal(totalLength == 0 ? 0 : 1, ctx.PieceCount);

        var pieces = HashPieces(ctx, data, pieceSize);
        Assert.Equal(Hasher.Hash(data), ctx.Finalize(pieces));

        if (totalLength > 0)
        {
            using var hasher = ctx.CreateSubtreeHasher(0);
            hasher.Update(data.AsSpan(0, totalLength / 2));
            hasher.Update(data.AsSpan(totalLength / 2));
            Assert.Equal(Hasher.Hash(data), ctx.Finalize(new[] { hasher.Finish() }));
        }
    }

    [Theory]
    [InlineData(1023, 1024)]
    [InlineData(3 * ChunkLen, 3 * ChunkLen + 1)]
    [InlineData(3 * ChunkLen, 10 * ChunkLen)]
    public void RejectsANonCanonicalPieceSizeWhenTheInputNeedsMoreThanOnePiece(int pieceSize, int totalLength)
    {
        Assert.Throws<ArgumentException>(() => Blake3SubtreeContext.Create(pieceSize, totalLength));
    }

    [Fact]
    public void RejectsANonPositivePieceSizeEvenForAnEmptyInput()
    {
        Assert.Throws<ArgumentException>(() => Blake3SubtreeContext.Create(0, 0));
        Assert.Throws<ArgumentException>(() => Blake3SubtreeContext.Create(-1, 0));
    }

    [Fact]
    public void RejectsAPieceIndexWhoseChunkCounterWouldWrapWithoutATotalLength()
    {
        // 2^62 bytes per piece is 2^52 chunks; index 2^12 would put the counter at exactly 2^64.
        using var ctx = Blake3SubtreeContext.Create(1L << 62);
        Assert.Throws<ArgumentOutOfRangeException>(() => ctx.HashSubtree(new byte[1], 1 << 12));
        Assert.Throws<ArgumentOutOfRangeException>(() => ctx.CreateSubtreeHasher(1 << 12));
        using var ok = ctx.CreateSubtreeHasher((1 << 12) - 1);
    }

    [Fact]
    public void RejectsANegativeTotalLength()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Blake3SubtreeContext.Create(ChunkLen, -1));
    }

    [Fact]
    public void RejectsAKeyThatIsNotThirtyTwoBytes()
    {
        Assert.Throws<ArgumentException>(() => Blake3SubtreeContext.CreateKeyed(new byte[31], ChunkLen));
        Assert.Throws<ArgumentException>(() => Blake3SubtreeContext.CreateKeyed(new byte[33], ChunkLen));
    }

    [Fact]
    public void RejectsAPieceIndexOutsideTheInput()
    {
        using var ctx = Blake3SubtreeContext.Create(ChunkLen, 2 * ChunkLen);

        Assert.Throws<ArgumentOutOfRangeException>(() => ctx.HashSubtree(Data(ChunkLen), -1));
        Assert.Throws<ArgumentOutOfRangeException>(() => ctx.HashSubtree(Data(ChunkLen), 2));
        Assert.Throws<ArgumentOutOfRangeException>(() => ctx.GetPieceOffset(2));
        Assert.Throws<ArgumentOutOfRangeException>(() => ctx.GetPieceLength(2));
    }

    [Fact]
    public void RejectsAnEmptyPiece()
    {
        using var ctx = Blake3SubtreeContext.Create(ChunkLen, ChunkLen);
        Assert.Throws<ArgumentException>(() => ctx.HashSubtree(ReadOnlySpan<byte>.Empty, 0));
    }

    [Fact]
    public void RejectsAPieceOfTheWrongLength()
    {
        const int pieceSize = 2 * ChunkLen;
        using var ctx = Blake3SubtreeContext.Create(pieceSize, pieceSize * 2 + 10);

        // A non-final piece must be exactly the piece size.
        Assert.Throws<ArgumentException>(() => ctx.HashSubtree(Data(pieceSize - 1), 0));
        // The final piece must be exactly what is left.
        Assert.Throws<ArgumentException>(() => ctx.HashSubtree(Data(11), 2));
    }

    [Fact]
    public void RejectsAPieceLargerThanThePieceSizeWithoutATotalLength()
    {
        using var ctx = Blake3SubtreeContext.Create(ChunkLen);
        Assert.Throws<ArgumentException>(() => ctx.HashSubtree(Data(ChunkLen + 1), 0));
    }

    [Fact]
    public void RejectsTheWrongNumberOfPieces()
    {
        const int pieceSize = 2 * ChunkLen;
        var data = Data(pieceSize * 3);
        using var ctx = Blake3SubtreeContext.Create(pieceSize, data.Length);

        var pieces = HashPieces(ctx, data, pieceSize);

        Assert.Throws<ArgumentException>(() => ctx.Finalize(pieces.AsSpan(0, 2)));
    }

    [Fact]
    public void RejectsAPieceThatWasNeverHashed()
    {
        const int pieceSize = 2 * ChunkLen;
        var data = Data(pieceSize * 3);
        using var ctx = Blake3SubtreeContext.Create(pieceSize, data.Length);

        var pieces = HashPieces(ctx, data, pieceSize);
        pieces[1] = default;

        Assert.Throws<ArgumentException>(() => ctx.Finalize(pieces));
    }

    [Fact]
    public void RejectsAPieceStoredAtTheWrongIndex()
    {
        const int pieceSize = 2 * ChunkLen;
        var data = Data(pieceSize * 3);
        using var ctx = Blake3SubtreeContext.Create(pieceSize, data.Length);

        var pieces = HashPieces(ctx, data, pieceSize);
        (pieces[0], pieces[1]) = (pieces[1], pieces[0]);

        Assert.Throws<ArgumentException>(() => ctx.Finalize(pieces));
    }

    [Fact]
    public void RejectsAPieceFromAnIncompatibleContext()
    {
        const int pieceSize = 2 * ChunkLen;
        var data = Data(pieceSize * 2);

        using var plain = Blake3SubtreeContext.Create(pieceSize, data.Length);
        using var keyed = Blake3SubtreeContext.CreateKeyed(Data(32), pieceSize, data.Length);
        using var otherSize = Blake3SubtreeContext.Create(pieceSize * 2, data.Length);

        var pieces = HashPieces(plain, data, pieceSize);

        Assert.Throws<ArgumentException>(() => keyed.Finalize(pieces));
        Assert.Throws<ArgumentException>(() => otherSize.Finalize(pieces));
    }

    [Fact]
    public void AcceptsPiecesFromAnEquivalentContext()
    {
        // Value-based, not identity-based: pieces hashed by a separately created but equivalent
        // context still combine, which is what allows hashing across processes.
        const int pieceSize = 2 * ChunkLen;
        var data = Data(pieceSize * 3);

        using var one = Blake3SubtreeContext.Create(pieceSize, data.Length);
        using var two = Blake3SubtreeContext.Create(pieceSize, data.Length);

        var pieces = HashPieces(one, data, pieceSize);

        Assert.Equal(Hasher.Hash(data).ToString(), two.Finalize(pieces).ToString());
    }

    [Fact]
    public void RejectsAShortPieceThatIsNotTheLast()
    {
        // Reachable only without a total length, where the piece length cannot be checked as the
        // piece is hashed. A short piece in the middle would silently build a different tree.
        const int pieceSize = 2 * ChunkLen;
        using var ctx = Blake3SubtreeContext.Create(pieceSize);

        var pieces = new[]
        {
            ctx.HashSubtree(Data(pieceSize - 1), 0),
            ctx.HashSubtree(Data(pieceSize), 1),
        };

        Assert.Throws<ArgumentException>(() => ctx.Finalize(pieces));
    }

    [Fact]
    public void ThrowsAfterDispose()
    {
        var ctx = Blake3SubtreeContext.Create(ChunkLen, ChunkLen);
        var piece = ctx.HashSubtree(Data(ChunkLen), 0);
        ctx.Dispose();

        Assert.Throws<ObjectDisposedException>(() => ctx.HashSubtree(Data(ChunkLen), 0));
        Assert.Throws<ObjectDisposedException>(() => ctx.Finalize(new[] { piece }));
    }

    [Fact]
    public void DisposeIsIdempotent()
    {
        var ctx = Blake3SubtreeContext.Create(ChunkLen);
        ctx.Dispose();
        ctx.Dispose();
    }

    [Fact]
    public void RejectsANegativeOutputOffset()
    {
        using var ctx = Blake3SubtreeContext.Create(ChunkLen, ChunkLen);
        var pieces = new[] { ctx.HashSubtree(Data(ChunkLen), 0) };

        Assert.Throws<ArgumentOutOfRangeException>(() => ctx.Finalize(pieces, -1L, new byte[32]));
    }

    [Fact]
    public void PieceReportsItsIndexAndLength()
    {
        const int pieceSize = 2 * ChunkLen;
        using var ctx = Blake3SubtreeContext.Create(pieceSize, pieceSize + 7);

        var last = ctx.HashSubtree(Data(7), 1);

        Assert.Equal(1, last.PieceIndex);
        Assert.Equal(7, last.Length);
    }

    [Fact]
    public void MatchesWholeInputHashAcrossManySizes()
    {
        const int pieceSize = ChunkLen;
        var lengths = new List<int>();
        for (int i = 1; i <= 40; i++) lengths.Add(i * 137);
        lengths.AddRange(new[] { ChunkLen - 1, ChunkLen, ChunkLen + 1, 8 * ChunkLen - 1, 8 * ChunkLen, 8 * ChunkLen + 1 });

        foreach (int length in lengths)
        {
            var data = Data(length);
            using var ctx = Blake3SubtreeContext.Create(pieceSize, length);

            Assert.Equal(Hasher.Hash(data).ToString(),
                ctx.Finalize(HashPieces(ctx, data, pieceSize)).ToString());
        }
    }

    [Theory]
    [InlineData(long.MaxValue)]
    [InlineData(long.MaxValue - 1)]
    [InlineData((long)int.MaxValue * ChunkLen + 1)]
    public void RejectsATotalLengthThatNeedsMorePiecesThanCanBeCounted(long totalLength)
    {
        // The ceiling division must not wrap: a wrapped piece count once let a single 1024-byte
        // piece stand in for a long.MaxValue input.
        Assert.Throws<ArgumentOutOfRangeException>(
            () => Blake3SubtreeContext.Create(ChunkLen, totalLength));
    }

    [Fact]
    public void RejectsPiecesWhoseFinalLengthBelongsToADifferentInput()
    {
        // Same piece size, same piece count, same key: the tag matches, so only the final piece's
        // length distinguishes a 2049-byte input from a 2050-byte one.
        const int pieceSize = 2 * ChunkLen;
        var data = Data(pieceSize + 1);

        using var actual = Blake3SubtreeContext.Create(pieceSize, pieceSize + 1);
        using var longer = Blake3SubtreeContext.Create(pieceSize, pieceSize + 2);

        var pieces = HashPieces(actual, data, pieceSize);

        Assert.Equal(2, longer.PieceCount);
        Assert.Throws<ArgumentException>(() => longer.Finalize(pieces));
    }

    [Fact]
    public void SinglePieceExtendedOutputMatchesAcrossTheMultiChunkBranch()
    {
        // The earlier single-piece XOF test used 700 bytes, which only exercises the one-chunk
        // branch. A multi-chunk single piece finalizes through the parent node instead.
        const int pieceSize = 16 * ChunkLen;
        var data = Data(pieceSize);

        using var ctx = Blake3SubtreeContext.Create(pieceSize, data.Length);
        var pieces = HashPieces(ctx, data, pieceSize);

        var actual = new byte[200];
        ctx.Finalize(pieces, 37L, actual);

        using var hasher = Hasher.New();
        hasher.Update(data);
        var expected = new byte[200];
        hasher.Finalize(37UL, expected);

        Assert.Equal(expected, actual);
    }

    [Fact]
    public void KeyedSinglePieceMatchesTheKeyedHasher()
    {
        var key = Data(32);

        foreach (int length in new[] { 1, 1024, 4096, 16 * ChunkLen })
        {
            var data = Data(length);
            using var ctx = Blake3SubtreeContext.CreateKeyed(key, 16 * ChunkLen, data.Length);
            using var hasher = Hasher.NewKeyed(key);
            hasher.Update(data);

            var actual = new byte[100];
            ctx.Finalize(HashPieces(ctx, data, 16 * ChunkLen), actual);
            var expected = new byte[100];
            hasher.Finalize(expected);

            Assert.Equal(expected, actual);
        }
    }
}
