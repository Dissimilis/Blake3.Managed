using System;
using System.Linq;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;
using Blake3.Managed.Internal;
using Xunit;

namespace Blake3.Managed.Tests;

[Collection(ParallelismCollection.Name)]
public class FanOutGateTests
{
    /// <summary>
    /// The in-flight counter must come back to zero, including when the hash throws.
    ///
    /// <c>HashMidSize</c> increments a process-wide counter, dispatches, and decrements in a
    /// <c>finally</c>. If that decrement ever stops running on some path -- moved out of the
    /// <c>finally</c>, or skipped by a new early return -- the count ratchets upward and every
    /// later mid-size hash in the process permanently takes the serial tree.
    ///
    /// Nothing else would notice. Every digest stays correct, so no vector or differential test
    /// fires, and a benchmark sees a gate that appears to be working: a leaked count of four on
    /// a sixteen-thread machine just looks like load. The counter is private, so the assertion
    /// reaches it by reflection rather than leaving the behaviour unpinned.
    /// </summary>
    private static int InFlight() =>
        (int)typeof(Blake3Tree)
            .GetField("s_midSizeInFlight", BindingFlags.NonPublic | BindingFlags.Static)!
            .GetValue(null)!;

    /// <summary>
    /// Waits for the counter to settle at zero before a test starts measuring it. The class
    /// runs alone, but a hash started by an earlier class can still be finishing as this one
    /// begins, and that is a scheduling artefact rather than the leak these tests look for.
    /// </summary>
    private static void WaitForQuiet()
    {
        var deadline = DateTime.UtcNow.AddSeconds(30);
        while (InFlight() != 0 && DateTime.UtcNow < deadline)
        {
            Thread.Sleep(5);
        }

        Assert.Equal(0, InFlight());
    }

    [Fact]
    public void MidSizeGateCounterReturnsToZeroOnTheHappyPath()
    {
        var data = new byte[64 * 1024];
        Assert.True(Blake3Tree.IsMidSize(data.Length), "size must be inside the gated band");

        WaitForQuiet();

        Span<byte> destination = stackalloc byte[32];
        var key = new byte[32];
        for (int i = 0; i < 40; i++)
        {
            _ = Hasher.Hash(data);
            Hasher.Hash(data, destination);
            _ = Hasher.HashKeyed(key, data);
        }

        Assert.Equal(0, InFlight());
    }

    /// <summary>
    /// The path the <c>finally</c> exists for. A key span shorter than eight words faults
    /// inside the tree, after the counter has been taken.
    /// </summary>
    [Fact]
    public void MidSizeGateCounterReturnsToZeroWhenTheHashThrows()
    {
        var data = new byte[64 * 1024];
        var destination = new byte[32];

        WaitForQuiet();

        Assert.ThrowsAny<Exception>(() =>
            Blake3Tree.HashMidSize(data, ReadOnlySpan<uint>.Empty, 0, destination, -1));

        Assert.Equal(0, InFlight());
    }

    /// <summary>
    /// Concurrent callers take and release the same counter; read after the join, so the
    /// assertion itself is not racing anything.
    /// </summary>
    [Fact]
    public async Task MidSizeGateCounterReturnsToZeroAfterConcurrentCallers()
    {
        var data = new byte[64 * 1024];
        WaitForQuiet();

        await Task.WhenAll(Enumerable.Range(0, 64).Select(caller => Task.Run(() =>
        {
            for (int i = 0; i < 8; i++) _ = Hasher.Hash(data);
        }))).ConfigureAwait(false);

        Assert.Equal(0, InFlight());
    }

    [Fact]
    public void IsMidSize_GatesEveryLengthAboveTheSerialTree()
    {
        // The serial tree's own range is never gated.
        Assert.False(Blake3Tree.IsMidSize(Blake3Tree.MaxUsefulLength));

        // Everything above it is, including above LoadGatedLength. That upper bound used to
        // end the gate; it now only selects which slot count applies, because leaving lengths
        // above it to fan out unconditionally measured 0.66 of Rust's serial path at 1 MiB
        // with sixteen callers -- the worst cell in the 2026-09-21 rayon table.
        Assert.True(Blake3Tree.IsMidSize(Blake3Tree.MaxUsefulLength + 1));
        Assert.True(Blake3Tree.IsMidSize(Blake3Tree.LoadGatedLength));
        Assert.True(Blake3Tree.IsMidSize(Blake3Tree.LoadGatedLength + 1));
        Assert.True(Blake3Tree.IsMidSize(int.MaxValue));
    }

    [Fact]
    public void ConcurrentMidSizeHashesMatchSerialDigest()
    {
        // 48 KB sits in the load-gated band: some of these calls fan out, others find a
        // parallel hash in flight and take the serial tree. Every one must agree.
        var data = new byte[48 * 1024];
        new Random(48).NextBytes(data);
        using var serial = Hasher.New();
        serial.Update(data);
        var expected = serial.Finalize();

        Parallel.For(0, 256, new ParallelOptions { MaxDegreeOfParallelism = 32 }, _ =>
        {
            Assert.Equal(expected, Hasher.Hash(data));
            Span<byte> span = stackalloc byte[32];
            Hasher.Hash(data, span);
            Assert.True(expected.AsSpan().SequenceEqual(span));
        });
    }
}
