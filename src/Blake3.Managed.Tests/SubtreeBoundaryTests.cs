using System;

namespace Blake3.Managed.Tests;

/// <summary>
/// Consistency tests around the 64-chunk subtree fast-path boundaries (65536 bytes)
/// and the parallel UpdateWithJoin work splitting. All modes must agree with the
/// sequential block-by-block path regardless of input size or update granularity.
/// Cross-implementation correctness is anchored by the official test vectors
/// (up to 102400 bytes), which cover the subtree and parallel paths.
/// </summary>
public class SubtreeBoundaryTests
{
    public static TheoryData<int> BoundarySizes => new()
    {
        4 * 1024 - 1, 4 * 1024, 4 * 1024 + 1,           // SSE/NEON 4-way boundary
        8 * 1024 - 1, 8 * 1024, 8 * 1024 + 1,           // AVX2 8-way boundary
        64 * 1024 - 1, 64 * 1024, 64 * 1024 + 1,        // subtree boundary
        72 * 1024 + 1,                                  // 1 subtree + 1 tail batch (parallel)
        128 * 1024 - 1, 128 * 1024, 128 * 1024 + 1,     // 2 subtrees (first stack merge of subtree CVs)
        192 * 1024 + 333,                               // odd number of subtrees plus tail
        512 * 1024 + 1,                                 // deeper subtree merges
    };

    private static byte[] MakeData(int size)
    {
        var data = new byte[size];
        new Random(size).NextBytes(data);
        return data;
    }

    /// <summary>Reference: feed input in 64-byte blocks so no batched path triggers.</summary>
    private static string SequentialHash(byte[] data)
    {
        using var hasher = Hasher.New();
        for (int pos = 0; pos < data.Length; pos += 64)
        {
            hasher.Update(data.AsSpan(pos, Math.Min(64, data.Length - pos)));
        }
        return hasher.Finalize().ToString();
    }

    [Theory]
    [MemberData(nameof(BoundarySizes))]
    public void AllModesAgreeAtBoundarySizes(int size)
    {
        var data = MakeData(size);
        var expected = SequentialHash(data);

        Assert.Equal(expected, Hasher.Hash(data).ToString());

        using (var h = Hasher.New())
        {
            h.Update(data);
            Assert.Equal(expected, h.Finalize().ToString());
        }

        using (var h = Hasher.New())
        {
            h.UpdateWithJoin(data);
            Assert.Equal(expected, h.Finalize().ToString());
        }
    }

    [Theory]
    [MemberData(nameof(BoundarySizes))]
    public void RandomSplitUpdatesAgree(int size)
    {
        var data = MakeData(size);
        var expected = Hasher.Hash(data).ToString();

        var rng = new Random(unchecked(size * 31));
        using var h = Hasher.New();
        int pos = 0;
        while (pos < data.Length)
        {
            int take = Math.Min(data.Length - pos, rng.Next(1, 100_000));
            h.Update(data.AsSpan(pos, take));
            pos += take;
        }
        Assert.Equal(expected, h.Finalize().ToString());
    }

    [Fact]
    public void UpdateWithJoinFromUnalignedChunkCounterAgrees()
    {
        var data = MakeData(300 * 1024);
        var expected = Hasher.Hash(data).ToString();

        // First update leaves the chunk counter unaligned (33 chunks), so the second
        // UpdateWithJoin cannot use the parallel subtree path and must still be correct.
        using var h = Hasher.New();
        h.UpdateWithJoin(data.AsSpan(0, 33 * 1024));
        h.UpdateWithJoin(data.AsSpan(33 * 1024));
        Assert.Equal(expected, h.Finalize().ToString());
    }
}
