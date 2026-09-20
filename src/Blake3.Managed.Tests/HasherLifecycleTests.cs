using Blake3.Managed;

namespace Blake3.Managed.Tests;

/// <summary>
/// State-machine and lifetime behaviour rather than digests: what happens when a caller copies a
/// disposable struct, disposes it twice, keeps going after Finalize, resets mid-stream, or mixes
/// the two update entry points. None of these are covered by the vector suites, and each one is
/// a contract the type either honours or silently breaks.
/// </summary>
[Collection(ParallelismCollection.Name)]
public class HasherLifecycleTests
{
    private static byte[] Input(int length) => HasherTests.MakeTestInput(length);

    private static string Digest(byte[] data) => Hasher.Hash(data).ToString();

    /// <summary>
    /// <see cref="Hasher"/> is a disposable struct whose state is entirely inline, so a copy is
    /// an independent hasher rather than a second handle on shared state. That is worth pinning
    /// in both directions: the copy must carry the accumulated prefix forward, and disposing one
    /// must leave the other working. If any part of the state ever moves to the heap -- a rented
    /// buffer is the obvious temptation -- this test starts failing, which is the point.
    /// </summary>
    [Fact]
    public void CopyingAHasherForksIndependentState()
    {
        byte[] prefix = Input(5000);
        byte[] left = Input(700);
        byte[] right = Input(1300);

        using var original = Hasher.New();
        original.Update(prefix);

        var copy = original;

        original.Update(left);
        copy.Update(right);

        Assert.Equal(Digest(prefix.Concat(left).ToArray()), original.Finalize().ToString());
        Assert.Equal(Digest(prefix.Concat(right).ToArray()), copy.Finalize().ToString());

        copy.Dispose();
    }

    [Fact]
    public void DisposingOneCopyLeavesTheOtherUsable()
    {
        byte[] data = Input(4096);

        var original = Hasher.New();
        original.Update(data);
        var copy = original;

        original.Dispose();

        Assert.Equal(Digest(data), copy.Finalize().ToString());
        copy.Dispose();
    }

    [Fact]
    public void DisposeIsIdempotent()
    {
        var hasher = Hasher.New();
        hasher.Update(Input(100));

        hasher.Dispose();
        hasher.Dispose();
    }

    /// <summary>
    /// Every entry point checks initialisation, so a disposed hasher fails loudly rather than
    /// hashing with zeroed key material -- which for a keyed hasher would be a silent downgrade
    /// to a fixed, attacker-known key.
    /// </summary>
    [Fact]
    public void EveryEntryPointThrowsAfterDispose()
    {
        var hasher = Hasher.NewKeyed(new byte[32]);
        hasher.Dispose();

        Assert.ThrowsAny<Exception>(() => hasher.Update(Input(10)));
        Assert.ThrowsAny<Exception>(() => hasher.UpdateWithJoin(Input(10)));
        Assert.ThrowsAny<Exception>(() => hasher.Finalize());
        Assert.ThrowsAny<Exception>(() => hasher.Reset());

        Assert.ThrowsAny<Exception>(() =>
        {
            Span<byte> destination = stackalloc byte[32];
            hasher.Finalize(destination);
        });
    }

    /// <summary>
    /// Finalize does not consume the state, so a caller may keep appending. The digest must then
    /// be that of everything supplied so far, not of the suffix alone and not of a state that
    /// absorbed its own root output.
    /// </summary>
    [Fact]
    public void UpdateAfterFinalizeAppendsRatherThanRestarting()
    {
        byte[] first = Input(3000);
        byte[] second = Input(2000);

        using var hasher = Hasher.New();
        hasher.Update(first);

        Assert.Equal(Digest(first), hasher.Finalize().ToString());

        hasher.Update(second);

        Assert.Equal(Digest(first.Concat(second).ToArray()), hasher.Finalize().ToString());
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(1024)]
    [InlineData(100_000)]
    public void ResetMidStreamMatchesAFreshHasher(int consumedBeforeReset)
    {
        byte[] discarded = Input(consumedBeforeReset);
        byte[] kept = Input(9000);

        using var hasher = Hasher.New();
        hasher.UpdateWithJoin(discarded);
        hasher.Reset();
        hasher.Update(kept);

        Assert.Equal(Digest(kept), hasher.Finalize().ToString());
    }

    [Fact]
    public void ResetAfterFinalizeMatchesAFreshHasher()
    {
        byte[] first = Input(70_000);
        byte[] second = Input(4096);

        using var hasher = Hasher.New();
        hasher.UpdateWithJoin(first);
        _ = hasher.Finalize();
        hasher.Reset();
        hasher.Update(second);

        Assert.Equal(Digest(second), hasher.Finalize().ToString());
    }

    /// <summary>
    /// Update and UpdateWithJoin are different code paths and UpdateWithJoin can defer a
    /// complete chunk's chaining value, because it may turn out to be the final chunk. Switching
    /// between them mid-stream has to flush that deferred state first; not doing so silently
    /// drops or double-counts a 1 KiB chunk, which no fixed-shape test would notice.
    ///
    /// The sizes straddle the point where UpdateWithJoin actually fans out (it needs at least
    /// one 64-chunk subtree plus a second item), so some of these interleavings take the
    /// parallel path and some fall back to the serial one.
    /// </summary>
    [Theory]
    [InlineData(1, 1)]
    [InlineData(1024, 1024)]
    [InlineData(1023, 1025)]
    [InlineData(70_000, 1)]
    [InlineData(1, 70_000)]
    [InlineData(70_000, 70_000)]
    [InlineData(65_536, 65_536)]
    [InlineData(200_000, 3000)]
    public void InterleavingUpdateAndUpdateWithJoinMatchesTheConcatenation(int firstLength, int secondLength)
    {
        byte[] first = Input(firstLength);
        byte[] second = Input(secondLength);
        byte[] whole = first.Concat(second).ToArray();

        using (var joinFirst = Hasher.New())
        {
            joinFirst.UpdateWithJoin(first);
            joinFirst.Update(second);
            Assert.Equal(Digest(whole), joinFirst.Finalize().ToString());
        }

        using (var updateFirst = Hasher.New())
        {
            updateFirst.Update(first);
            updateFirst.UpdateWithJoin(second);
            Assert.Equal(Digest(whole), updateFirst.Finalize().ToString());
        }
    }

    /// <summary>
    /// An empty update as the <em>last</em> call before Finalize must change nothing.
    ///
    /// Both update entry points return on an empty span before calling FlushPendingCv, and that
    /// ordering is load-bearing. A complete chunk's chaining value is deliberately left deferred
    /// because it may turn out to be the final chunk, and Finalize needs it as the root parent's
    /// right child. Flushing it because an empty span arrived commits it to the CV stack, and
    /// Finalize then takes its other leg and produces a <em>wrong digest with no exception</em>.
    ///
    /// <see cref="EmptyUpdatesBetweenJoinsAreInert"/> cannot catch that: it feeds more data
    /// afterwards, and a premature flush there is indistinguishable from the flush the next
    /// update would legitimately do. The bug is only observable when the empty call is last.
    /// The lengths below all end exactly on a SIMD batch boundary, which is what leaves a CV
    /// deferred in the first place.
    /// </summary>
    [Theory]
    [InlineData(2048)]
    [InlineData(4096)]
    [InlineData(8192)]
    [InlineData(65_536)]
    [InlineData(73_728)]
    [InlineData(131_072)]
    public void EmptyUpdateAsTheLastCallBeforeFinalizeIsInert(int length)
    {
        byte[] data = Input(length);
        string expected = Digest(data);

        foreach (bool joinFirst in new[] { false, true })
        foreach (bool emptyViaJoin in new[] { false, true })
        {
            using var hasher = Hasher.New();

            if (joinFirst) hasher.UpdateWithJoin(data);
            else hasher.Update(data);

            if (emptyViaJoin) hasher.UpdateWithJoin(ReadOnlySpan<byte>.Empty);
            else hasher.Update(ReadOnlySpan<byte>.Empty);

            Assert.Equal(expected, hasher.Finalize().ToString());

            // Finalize is non-destructive, so a second one must agree too.
            Assert.Equal(expected, hasher.Finalize().ToString());
        }
    }

    /// <summary>
    /// An empty update between two real ones must be inert on both entry points, including
    /// immediately after a join has deferred a chunk.
    /// </summary>
    [Fact]
    public void EmptyUpdatesBetweenJoinsAreInert()
    {
        byte[] first = Input(70_000);
        byte[] second = Input(5000);

        using var hasher = Hasher.New();
        hasher.UpdateWithJoin(first);
        hasher.Update(ReadOnlySpan<byte>.Empty);
        hasher.UpdateWithJoin(ReadOnlySpan<byte>.Empty);
        hasher.Update(second);

        Assert.Equal(Digest(first.Concat(second).ToArray()), hasher.Finalize().ToString());
    }

    /// <summary>
    /// Sizes chosen so the parallel join actually runs, and so both of its branches do.
    ///
    /// <c>UpdateWithJoin</c> only fans out on a fresh hasher whose chunk buffer is empty and
    /// whose counter is 64-aligned, and only once the input yields a whole 64-chunk subtree plus
    /// a second item. Sizes that look large enough often are not: 70,000 bytes gives one subtree
    /// and no tail batch, so the item count is one and it quietly falls back to <c>Update</c>.
    /// A joined call that follows any partial chunk falls back too, so a sequence of joined
    /// updates can run the fan-out exactly never.
    ///
    /// The tail cases are the point. 73,729 bytes is one subtree and one tail batch, the
    /// smallest input that fans out at all; 131,072 is one subtree and seven tail batches.
    /// Without them the worker's tail branch never executes, and the suite would pass even if
    /// that branch wrote the wrong chaining values entirely.
    /// </summary>
    [Theory]
    [InlineData(73_729)]     // 1 subtree + 1 tail batch (smallest input that fans out)
    [InlineData(81_920)]     // 1 subtree + 2 tail batches
    [InlineData(131_072)]    // 1 subtree + 7 tail batches
    [InlineData(139_265)]    // 2 subtrees + 1 tail batch
    [InlineData(200_000)]    // 3 subtrees, no tail batch
    [InlineData(1_048_576)]  // 15 subtrees + 7 tail batches
    public void JoinedUpdateOnAFreshHasherMatchesTheSerialPath(int length)
    {
        byte[] data = Input(length);

        using (var joined = Hasher.New())
        {
            joined.UpdateWithJoin(data);
            Assert.Equal(Digest(data), joined.Finalize().ToString());
        }

        // The same shapes through the keyed root, which carries different flags into the worker
        // but takes the identical dispatch.
        var key = new byte[32];
        for (int i = 0; i < key.Length; i++) key[i] = (byte)(i * 5 + 2);

        using var keyedJoined = Hasher.NewKeyed(key);
        keyedJoined.UpdateWithJoin(data);

        using var keyedSerial = Hasher.NewKeyed(key);
        keyedSerial.Update(data);

        Assert.Equal(keyedSerial.Finalize().ToString(), keyedJoined.Finalize().ToString());
    }

    /// <summary>
    /// Many small joined updates in a row exercise the deferred-chunk bookkeeping repeatedly,
    /// which is where the JoinJob rewrite could have gone wrong without changing any
    /// single-shot digest.
    /// </summary>
    [Fact]
    public void ManyConsecutiveJoinedUpdatesMatchTheConcatenation()
    {
        var whole = new List<byte>();
        using var hasher = Hasher.New();

        foreach (int length in new[] { 70_000, 1, 1023, 70_000, 64, 130_000, 7 })
        {
            byte[] piece = Input(length);
            hasher.UpdateWithJoin(piece);
            whole.AddRange(piece);
        }

        Assert.Equal(Digest(whole.ToArray()), hasher.Finalize().ToString());
    }
}
