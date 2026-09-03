using System.Text;
using Xunit;

namespace Blake3.Managed.Tests;

public class Blake3SubtreeHasherTests
{
    private const int ChunkLen = 1024;

    private static byte[] Data(int length, int seed = 7)
    {
        var data = new byte[length];
        new Random(seed).NextBytes(data);
        return data;
    }

    /// <summary>
    /// Feeds a piece in fragments of pseudo-random sizes, from 1 byte to a few chunks, so that
    /// block, chunk and SIMD-batch boundaries all fall inside fragments.
    /// </summary>
    private static Blake3Subtree HashInFragments(Blake3SubtreeContext ctx, ReadOnlySpan<byte> piece,
        int pieceIndex, Random rng)
    {
        using var hasher = ctx.CreateSubtreeHasher(pieceIndex);
        int pos = 0;
        while (pos < piece.Length)
        {
            int take = Math.Min(piece.Length - pos, rng.Next(1, 3 * ChunkLen));
            hasher.Update(piece.Slice(pos, take));
            pos += take;
            Assert.Equal(pos, hasher.Length);
        }

        return hasher.Finish();
    }

    public static IEnumerable<object[]> PieceSizesAndLengths()
    {
        foreach (int pieceSize in new[] { ChunkLen, 2 * ChunkLen, 4 * ChunkLen, 8 * ChunkLen, 16 * ChunkLen, 64 * ChunkLen, 128 * ChunkLen })
        {
            foreach (int length in new[] { 1, 63, 64, 65, ChunkLen - 1, ChunkLen, ChunkLen + 1,
                         pieceSize - 1, pieceSize, pieceSize + 1, 3 * pieceSize, 3 * pieceSize + 500,
                         5 * pieceSize + 1, 200 * ChunkLen + 17 })
            {
                yield return new object[] { pieceSize, length };
            }
        }
    }

    [Theory]
    [MemberData(nameof(PieceSizesAndLengths))]
    public void FragmentedPiecesMatchWholeInputHash(int pieceSize, int length)
    {
        var data = Data(length, seed: pieceSize ^ length);
        using var ctx = Blake3SubtreeContext.Create(pieceSize, data.Length);
        var rng = new Random(length);

        var pieces = new Blake3Subtree[ctx.PieceCount];
        for (int i = 0; i < pieces.Length; i++)
        {
            pieces[i] = HashInFragments(ctx,
                data.AsSpan((int)ctx.GetPieceOffset(i), (int)ctx.GetPieceLength(i)), i, rng);
        }

        Assert.Equal(Hasher.Hash(data).ToString(), ctx.Finalize(pieces).ToString());

        // Extended output must agree too: a single-piece input makes the piece the root.
        var expected = new byte[200];
        Hasher.Hash(data, expected);
        var actual = new byte[200];
        ctx.Finalize(pieces, actual);
        Assert.Equal(expected, actual);
    }

    [Theory]
    [InlineData(4 * ChunkLen, 10 * ChunkLen + 3)]
    [InlineData(64 * ChunkLen, 130 * ChunkLen)]
    public void MixesWithHashSubtreeAndWithoutTotalLength(int pieceSize, int length)
    {
        var data = Data(length);
        using var ctx = Blake3SubtreeContext.Create(pieceSize);
        var rng = new Random(1);

        int pieceCount = (length + pieceSize - 1) / pieceSize;
        var pieces = new Blake3Subtree[pieceCount];
        for (int i = 0; i < pieceCount; i++)
        {
            var piece = data.AsSpan(i * pieceSize, Math.Min(pieceSize, length - i * pieceSize));
            pieces[i] = i % 2 == 0
                ? ctx.HashSubtree(piece, i)
                : HashInFragments(ctx, piece, i, rng);
        }

        Assert.Equal(Hasher.Hash(data).ToString(), ctx.Finalize(pieces).ToString());
    }

    [Fact]
    public void KeyedAndDeriveKeyModesMatch()
    {
        var data = Data(70 * ChunkLen + 9);
        var key = Data(32, seed: 99);
        var rng = new Random(2);

        using var keyed = Blake3SubtreeContext.CreateKeyed(key, 16 * ChunkLen, data.Length);
        var keyedPieces = new Blake3Subtree[keyed.PieceCount];
        for (int i = 0; i < keyedPieces.Length; i++)
        {
            keyedPieces[i] = HashInFragments(keyed,
                data.AsSpan((int)keyed.GetPieceOffset(i), (int)keyed.GetPieceLength(i)), i, rng);
        }

        using var keyedHasher = Hasher.NewKeyed(key);
        keyedHasher.Update(data);
        Assert.Equal(keyedHasher.Finalize().ToString(), keyed.Finalize(keyedPieces).ToString());

        using var derive = Blake3SubtreeContext.CreateDeriveKey("test context", 8 * ChunkLen, data.Length);
        var derivePieces = new Blake3Subtree[derive.PieceCount];
        for (int i = 0; i < derivePieces.Length; i++)
        {
            derivePieces[i] = HashInFragments(derive,
                data.AsSpan((int)derive.GetPieceOffset(i), (int)derive.GetPieceLength(i)), i, rng);
        }

        using var deriveHasher = Hasher.NewDeriveKey("test context");
        deriveHasher.Update(data);
        Assert.Equal(deriveHasher.Finalize().ToString(), derive.Finalize(derivePieces).ToString());
    }

    [Fact]
    public void PiecesCanRunConcurrently()
    {
        var data = Data(300 * ChunkLen + 1);
        using var ctx = Blake3SubtreeContext.Create(8 * ChunkLen, data.Length);
        var pieces = new Blake3Subtree[ctx.PieceCount];

        Parallel.ForEach(Enumerable.Range(0, ctx.PieceCount).Reverse(), i =>
        {
            pieces[i] = HashInFragments(ctx,
                data.AsSpan((int)ctx.GetPieceOffset(i), (int)ctx.GetPieceLength(i)), i, new Random(i));
        });

        Assert.Equal(Hasher.Hash(data).ToString(), ctx.Finalize(pieces).ToString());
    }

    [Fact]
    public void RejectsTooManyBytes()
    {
        using var ctx = Blake3SubtreeContext.Create(ChunkLen, 2 * ChunkLen + 5);
        using var first = ctx.CreateSubtreeHasher(0);
        first.Update(new byte[ChunkLen]);
        Assert.Throws<ArgumentException>(() => first.Update(new byte[1]));

        using var last = ctx.CreateSubtreeHasher(2);
        Assert.Throws<ArgumentException>(() => last.Update(new byte[6]));

        using var unknown = Blake3SubtreeContext.Create(ChunkLen);
        using var hasher = unknown.CreateSubtreeHasher(0);
        Assert.Throws<ArgumentException>(() => hasher.Update(new byte[ChunkLen + 1]));
    }

    [Fact]
    public void RejectsAShortOrEmptyPieceAtFinish()
    {
        using var ctx = Blake3SubtreeContext.Create(ChunkLen, 2 * ChunkLen + 5);

        using var empty = ctx.CreateSubtreeHasher(0);
        Assert.Throws<InvalidOperationException>(() => empty.Finish());

        using var partial = ctx.CreateSubtreeHasher(0);
        partial.Update(new byte[ChunkLen - 1]);
        Assert.Throws<InvalidOperationException>(() => partial.Finish());

        using var unknown = Blake3SubtreeContext.Create(ChunkLen);
        using var emptyUnknown = unknown.CreateSubtreeHasher(0);
        Assert.Throws<InvalidOperationException>(() => emptyUnknown.Finish());
    }

    [Fact]
    public void RejectsUseAfterFinishOrDispose()
    {
        using var ctx = Blake3SubtreeContext.Create(ChunkLen, 5);
        var hasher = ctx.CreateSubtreeHasher(0);
        hasher.Update(new byte[5]);
        hasher.Finish();
        Assert.Throws<InvalidOperationException>(() => hasher.Finish());
        Assert.Throws<InvalidOperationException>(() => hasher.Update(new byte[1]));

        hasher.Dispose();
        Assert.Throws<ObjectDisposedException>(() => hasher.Update(new byte[1]));
        Assert.Throws<ObjectDisposedException>(() => hasher.Finish());
    }

    [Fact]
    public void RejectsABadPieceIndex()
    {
        using var ctx = Blake3SubtreeContext.Create(ChunkLen, 5);
        Assert.Throws<ArgumentOutOfRangeException>(() => ctx.CreateSubtreeHasher(-1));
        Assert.Throws<ArgumentOutOfRangeException>(() => ctx.CreateSubtreeHasher(1));

        ctx.Dispose();
        Assert.Throws<ObjectDisposedException>(() => ctx.CreateSubtreeHasher(0));
    }

    [Fact]
    public void ExactlyOneFragmentPerReadMatches()
    {
        // The shape from the issue: every network read hashed as it lands.
        var data = Encoding.UTF8.GetBytes(string.Concat(Enumerable.Repeat("hello blake3 ", 3000)));
        using var ctx = Blake3SubtreeContext.Create(16 * ChunkLen, data.Length);
        var pieces = new Blake3Subtree[ctx.PieceCount];

        for (int i = 0; i < pieces.Length; i++)
        {
            using var hasher = ctx.CreateSubtreeHasher(i);
            long offset = ctx.GetPieceOffset(i);
            long end = offset + ctx.GetPieceLength(i);
            for (long pos = offset; pos < end; pos += 700)
            {
                hasher.Update(data.AsSpan((int)pos, (int)Math.Min(700, end - pos)));
            }
            pieces[i] = hasher.Finish();
        }

        Assert.Equal(Hasher.Hash(data).ToString(), ctx.Finalize(pieces).ToString());
    }
}
