extern alias XoofxManaged;

using System.Text;
using System.Text.Json;
using ManagedHasher = Blake3.Managed.Hasher;
using NativeHasher = Blake3.Hasher;
using XoofxHasher = XoofxManaged::Blake3.Hasher;
using CryptoHivesBlake3 = CryptoHives.Foundation.Security.Cryptography.Hash.Blake3;

namespace Blake3.Managed.Benchmarks;

/// <summary>
/// Correctness gate that runs before every benchmark session.
///
/// Optimizing a hash is uniquely dangerous: a wrong tree shape or a mis-set flag still produces 32
/// plausible-looking bytes, and a faster wrong answer looks exactly like a win. This aborts the
/// run rather than reporting numbers for a broken hash.
///
/// Two independent anchors, because either alone is insufficient:
///   1. The official BLAKE3 test vectors, which are ground truth and cannot drift.
///   2. A differential comparison against the Rust implementation, which covers far more input
///      lengths than the official vectors and so reaches the scheduler boundaries where batching
///      bugs actually live. On its own this could agree-and-both-be-wrong through shared misuse,
///      which is why the official vectors run first.
/// </summary>
internal static class Correctness
{
    /// <summary>
    /// Byte-fill patterns. The pattern matters more than it looks: a fill whose period divides
    /// 1024 makes every chunk byte-identical, which hides swapped SIMD lanes, duplicated chunks,
    /// and wrong batch offsets -- precisely the bugs a parallel/batched rewrite introduces.
    /// 251 is prime and coprime with every power of two, so no two chunks are ever alike.
    /// </summary>
    private enum Pattern
    {
        /// <summary>Official vector fill: 0,1,2,...,250,0,1,... Period 251, never aligns to a chunk.</summary>
        Official,

        /// <summary>Seeded pseudorandom: catches anything a linear pattern is symmetric under.</summary>
        Random,

        /// <summary>Chunk index encoded into every chunk, so a swapped or duplicated chunk cannot match.</summary>
        ChunkMarked,
    }

    private static readonly int[] Lengths = BuildLengths();

    /// <summary>
    /// Lengths straddling every boundary the scheduler branches on: block (64), chunk (1024), the
    /// SIMD batch widths (4, 8), the parent-batch widths, the 64-chunk subtree, and the point
    /// where the parallel path engages. Powers of two and their neighbours matter because the
    /// BLAKE3 tree splits at the largest power of two below the length, so N-1/N/N+1 take
    /// genuinely different paths through the frontier.
    /// </summary>
    private static int[] BuildLengths()
    {
        var lengths = new SortedSet<int> { 0, 1, 2, 3, 63, 64, 65, 127, 128, 129, 1023 };

        var chunkCounts = new SortedSet<int>();

        // Every small chunk count exhaustively: this is where frontier termination and the
        // "stop at two CVs" rule are most easily got wrong.
        for (int i = 0; i <= 20; i++) chunkCounts.Add(i);

        // Powers of two either side, up to a 1024-chunk (1 MB) tree.
        for (int p = 1; p <= 1024; p *= 2)
        {
            chunkCounts.Add(p - 1);
            chunkCounts.Add(p);
            chunkCounts.Add(p + 1);
        }

        // SIMD and parent-batch widths with their remainders: W-1, W, W+1, 2W-1, 2W, 2W+1.
        foreach (var width in new[] { 4, 8, 16, 64 })
        {
            foreach (var multiple in new[] { 1, 2, 3 })
            {
                chunkCounts.Add(width * multiple - 1);
                chunkCounts.Add(width * multiple);
                chunkCounts.Add(width * multiple + 1);
            }
        }

        // Mixed reductions and awkward non-power-of-two trees that exercise odd-CV carry.
        foreach (var n in new[] { 6, 10, 11, 12, 13, 14, 18, 24, 40, 48, 71, 72, 73, 80, 96, 191, 255, 257 })
        {
            chunkCounts.Add(n);
        }

        foreach (var chunks in chunkCounts)
        {
            if (chunks == 0) continue;
            int exact = chunks * 1024;
            lengths.Add(exact - 1);
            lengths.Add(exact);
            lengths.Add(exact + 1);
        }

        return lengths.ToArray();
    }

    public static void Run()
    {
        int checks = OfficialVectors();

        foreach (var length in Lengths)
        {
            // Rotate patterns by length so the whole sweep stays affordable while every pattern
            // still lands on many different tree shapes.
            var pattern = (Pattern)(length % 3);
            var data = MakeData(length, pattern);

            checks += CheckDefault(data, pattern);
            checks += CheckUpdateWithJoin(data, pattern);
            checks += CheckIncremental(data, pattern);

            // The expensive modes run on a subset: every boundary length still gets covered
            // because the subset is chosen by chunk alignment, not by size.
            if (length % 1024 <= 1 || length < 8192)
            {
                checks += CheckKeyed(data, pattern);
                checks += CheckDeriveKey(data, pattern);
                checks += CheckXof(data, pattern);
                checks += CheckStaticSpanOutput(data, pattern);
            }
        }

        checks += CheckAlignment();
        checks += CheckCompetitorsAgree();

        Console.WriteLine($"Correctness gate passed: {checks:N0} checks over {Lengths.Length} input lengths "
                          + $"(to {Lengths[^1]:N0} bytes), 3 fill patterns, official vectors + Rust differential,");
        Console.WriteLine("covering default / keyed / derive-key / XOF+seek / incremental / parallel / unaligned.");
    }

    /// <summary>
    /// Ground truth. Unlike the differential checks, these cannot drift if a dependency changes.
    /// </summary>
    private static int OfficialVectors()
    {
        var path = Path.Combine(AppContext.BaseDirectory, "TestVectors.json");
        if (!File.Exists(path))
        {
            throw new InvalidOperationException(
                $"Official BLAKE3 test vectors not found at {path}. The gate refuses to run without "
                + "ground truth: a differential-only check can agree with a reference and still be wrong.");
        }

        var file = JsonSerializer.Deserialize<TestVectorFile>(File.ReadAllText(path))
                   ?? throw new InvalidOperationException("TestVectors.json could not be parsed.");

        var key = Encoding.UTF8.GetBytes(file.Key);
        int checks = 0;

        foreach (var testCase in file.Cases)
        {
            // Official vectors are defined over a repeating 0..250 fill.
            var input = MakeData(testCase.InputLen, Pattern.Official);

            // Output length is whatever the vector specifies, which exercises XOF too.
            var hash = new byte[testCase.Hash.Length / 2];
            using (var hasher = ManagedHasher.New())
            {
                hasher.Update(input);
                hasher.Finalize(hash);
            }
            Assert(testCase.Hash, Hex(hash), "official vector: hash", testCase.InputLen);

            var keyed = new byte[testCase.KeyedHash.Length / 2];
            using (var hasher = ManagedHasher.NewKeyed(key))
            {
                hasher.Update(input);
                hasher.Finalize(keyed);
            }
            Assert(testCase.KeyedHash, Hex(keyed), "official vector: keyed_hash", testCase.InputLen);

            var derived = new byte[testCase.DeriveKey.Length / 2];
            using (var hasher = ManagedHasher.NewDeriveKey(file.ContextString))
            {
                hasher.Update(input);
                hasher.Finalize(derived);
            }
            Assert(testCase.DeriveKey, Hex(derived), "official vector: derive_key", testCase.InputLen);

            // The parallel path must reproduce ground truth too, not merely agree with our serial path.
            var joined = new byte[testCase.Hash.Length / 2];
            using (var hasher = ManagedHasher.New())
            {
                hasher.UpdateWithJoin(input);
                hasher.Finalize(joined);
            }
            Assert(testCase.Hash, Hex(joined), "official vector: hash via UpdateWithJoin", testCase.InputLen);

            checks += 4;
        }

        return checks;
    }

    private static byte[] MakeData(int length, Pattern pattern)
    {
        var data = new byte[length];

        switch (pattern)
        {
            case Pattern.Official:
                for (int i = 0; i < length; i++) data[i] = (byte)(i % 251);
                break;

            case Pattern.Random:
                new Random(length * 2654435761u is var _ ? unchecked(length * 31 + 17) : 0).NextBytes(data);
                break;

            case Pattern.ChunkMarked:
                for (int i = 0; i < length; i++)
                {
                    int chunk = i / 1024;
                    // Chunk index folded in, so two chunks are never byte-identical.
                    data[i] = (byte)((i % 251) ^ (chunk * 37 + (chunk >> 8)));
                }
                break;
        }

        return data;
    }

    private static int CheckDefault(byte[] data, Pattern pattern)
    {
        Assert(Hex(NativeHasher.Hash(data).AsSpan()), Hex(ManagedHasher.Hash(data).AsSpan()),
            $"default [{pattern}]", data.Length);
        return 1;
    }

    /// <summary>
    /// The parallel path builds the tree in a different order than the serial one, so it needs its
    /// own check at every length rather than only at large ones.
    /// </summary>
    private static int CheckUpdateWithJoin(byte[] data, Pattern pattern)
    {
        Span<byte> actual = stackalloc byte[32];
        using (var ours = ManagedHasher.New())
        {
            ours.UpdateWithJoin(data);
            ours.Finalize(actual);
        }

        Assert(Hex(NativeHasher.Hash(data).AsSpan()), Hex(actual), $"UpdateWithJoin [{pattern}]", data.Length);
        return 1;
    }

    private static int CheckKeyed(byte[] data, Pattern pattern)
    {
        var key = new byte[32];
        for (int i = 0; i < key.Length; i++) key[i] = (byte)(i * 7 + 1);

        Span<byte> expected = stackalloc byte[32];
        using (var native = NativeHasher.NewKeyed(key))
        {
            native.Update(data);
            native.Finalize(expected);
        }

        Span<byte> serial = stackalloc byte[32];
        using (var ours = ManagedHasher.NewKeyed(key))
        {
            ours.Update(data);
            ours.Finalize(serial);
        }
        Assert(Hex(expected), Hex(serial), $"keyed [{pattern}]", data.Length);

        // Keyed CVs and flags must survive the parallel frontier and every parent compression.
        Span<byte> parallel = stackalloc byte[32];
        using (var ours = ManagedHasher.NewKeyed(key))
        {
            ours.UpdateWithJoin(data);
            ours.Finalize(parallel);
        }
        Assert(Hex(expected), Hex(parallel), $"keyed via UpdateWithJoin [{pattern}]", data.Length);

        return 2;
    }

    private static int CheckDeriveKey(byte[] data, Pattern pattern)
    {
        const string context = "Blake3.Managed benchmark gate — ünïcödé context";

        Span<byte> expected = stackalloc byte[32];
        using (var native = NativeHasher.NewDeriveKey(context))
        {
            native.Update(data);
            native.Finalize(expected);
        }

        Span<byte> serial = stackalloc byte[32];
        using (var ours = ManagedHasher.NewDeriveKey(context))
        {
            ours.Update(data);
            ours.Finalize(serial);
        }
        Assert(Hex(expected), Hex(serial), $"derive-key [{pattern}]", data.Length);

        Span<byte> parallel = stackalloc byte[32];
        using (var ours = ManagedHasher.NewDeriveKey(context))
        {
            ours.UpdateWithJoin(data);
            ours.Finalize(parallel);
        }
        Assert(Hex(expected), Hex(parallel), $"derive-key via UpdateWithJoin [{pattern}]", data.Length);

        return 2;
    }

    /// <summary>
    /// XOF output past 64 bytes exercises the root-output block counter, and seeking exercises it
    /// at an arbitrary offset. A tree-shape bug can leave the first 32 bytes correct and corrupt
    /// everything after, so length-only coverage is not enough.
    /// </summary>
    private static int CheckXof(byte[] data, Pattern pattern)
    {
        int checks = 0;

        foreach (var outputLength in new[] { 0, 1, 31, 32, 33, 63, 64, 65, 131, 1024 })
        {
            var expected = new byte[outputLength];
            using (var native = NativeHasher.New())
            {
                native.Update(data);
                native.Finalize(expected);
            }

            var actual = new byte[outputLength];
            using (var ours = ManagedHasher.New())
            {
                ours.Update(data);
                ours.Finalize(actual);
            }

            Assert(Hex(expected), Hex(actual), $"xof({outputLength}) [{pattern}]", data.Length);
            checks++;
        }

        // Seek: both against the reference, and against slicing one long output, which catches a
        // seek that is self-consistent but disagrees with the unsought stream.
        var full = new byte[512];
        using (var ours = ManagedHasher.New())
        {
            ours.Update(data);
            ours.Finalize(full);
        }

        foreach (var offset in new[] { 1, 31, 32, 63, 64, 65, 127, 128, 255 })
        {
            const int sliceLength = 96; // crosses a 64-byte output block from every offset above

            var expected = new byte[sliceLength];
            using (var native = NativeHasher.New())
            {
                native.Update(data);
                native.Finalize((ulong)offset, expected);
            }

            var actual = new byte[sliceLength];
            using (var oursSeek = ManagedHasher.New())
            {
                oursSeek.Update(data);
                oursSeek.Finalize((ulong)offset, actual);
            }

            Assert(Hex(expected), Hex(actual), $"xof seek({offset}) vs Rust [{pattern}]", data.Length);
            Assert(Hex(full.AsSpan(offset, sliceLength)), Hex(actual),
                $"xof seek({offset}) vs slice of full output [{pattern}]", data.Length);
            checks += 2;
        }

        return checks;
    }

    /// <summary>
    /// The static span-output overload has its own fast path, branching on input &lt;= 1024 and
    /// output &lt;= 64, so it can break independently of the instance API.
    /// </summary>
    private static int CheckStaticSpanOutput(byte[] data, Pattern pattern)
    {
        int checks = 0;

        foreach (var outputLength in new[] { 32, 64, 65, 200 })
        {
            var expected = new byte[outputLength];
            using (var native = NativeHasher.New())
            {
                native.Update(data);
                native.Finalize(expected);
            }

            var actual = new byte[outputLength];
            ManagedHasher.Hash(data, actual);

            Assert(Hex(expected), Hex(actual), $"static Hash(span,{outputLength}) [{pattern}]", data.Length);
            checks++;
        }

        return checks;
    }

    /// <summary>
    /// Feeds the same input in awkward pieces. One-shot and incremental share a tree but not a code
    /// path, so a scheduler change can break only one of them.
    /// </summary>
    private static int CheckIncremental(byte[] data, Pattern pattern)
    {
        int checks = 0;
        var expected = Hex(NativeHasher.Hash(data).AsSpan());

        foreach (var split in new[] { 1, 63, 64, 65, 1023, 1024, 1025, 4096, 8192 })
        {
            if (split >= data.Length) continue;

            Span<byte> actual = stackalloc byte[32];
            using (var ours = ManagedHasher.New())
            {
                ours.Update(data.AsSpan(0, split));
                ours.Update(default);           // an empty update must not disturb state
                ours.Update(data.AsSpan(split));
                ours.Finalize(actual);
            }

            Assert(expected, Hex(actual), $"incremental(split={split}) [{pattern}]", data.Length);
            checks++;
        }

        // Mixed serial/parallel calls, and a parallel update starting from an unaligned chunk
        // counter -- the case the parallel path's alignment guard exists to reject.
        if (data.Length > 2048)
        {
            Span<byte> actual = stackalloc byte[32];
            using (var ours = ManagedHasher.New())
            {
                ours.Update(data.AsSpan(0, 100));
                ours.UpdateWithJoin(data.AsSpan(100, data.Length - 200));
                ours.Update(data.AsSpan(data.Length - 100));
                ours.Finalize(actual);
            }

            Assert(expected, Hex(actual), $"Update + UpdateWithJoin + Update [{pattern}]", data.Length);
            checks++;

            // Repeated parallel runs: an intermittent race shows up as a mismatch on some pass.
            for (int repeat = 0; repeat < 3; repeat++)
            {
                Span<byte> repeated = stackalloc byte[32];
                using var hasher = ManagedHasher.New();
                hasher.UpdateWithJoin(data);
                hasher.Finalize(repeated);
                Assert(expected, Hex(repeated), $"UpdateWithJoin repeat {repeat} [{pattern}]", data.Length);
                checks++;
            }
        }

        return checks;
    }

    /// <summary>
    /// Unaligned input spans. SIMD loads that assume alignment, or pointer arithmetic that drops
    /// the span offset, only fail when the buffer does not start on a convenient boundary.
    /// </summary>
    private static int CheckAlignment()
    {
        int checks = 0;
        var backing = MakeData(64 + 8192, Pattern.ChunkMarked);

        foreach (var offset in new[] { 1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63 })
        {
            var slice = backing.AsSpan(offset, 8192);

            Span<byte> expected = stackalloc byte[32];
            using (var native = NativeHasher.New())
            {
                native.Update(slice);
                native.Finalize(expected);
            }

            Span<byte> actual = stackalloc byte[32];
            using (var ours = ManagedHasher.New())
            {
                ours.UpdateWithJoin(slice);
                ours.Finalize(actual);
            }

            Assert(Hex(expected), Hex(actual), $"unaligned input (offset {offset})", 8192);
            checks++;
        }

        return checks;
    }

    /// <summary>
    /// Guards the comparison itself: a competitor wired to the wrong API or output size makes its
    /// benchmark column noise, and any conclusion drawn from it wrong.
    /// </summary>
    private static int CheckCompetitorsAgree()
    {
        int checks = 0;

        foreach (var length in new[] { 0, 1, 64, 1024, 4096, 8192, 65_536, 131_072 })
        {
            var data = MakeData(length, Pattern.Official);
            var expected = Hex(NativeHasher.Hash(data).AsSpan());

            Assert(expected, Hex(XoofxHasher.Hash(data).AsSpan()), "competitor xoofx", length);
            Assert(expected, Convert.ToHexString(CryptoHivesBlake3.HashData(data)), "competitor CryptoHives", length);
            checks += 2;
        }

        return checks;
    }

    private static string Hex(ReadOnlySpan<byte> value) => Convert.ToHexString(value).ToLowerInvariant();

    private static void Assert(string expected, string actual, string mode, int length)
    {
        if (string.Equals(expected, actual, StringComparison.OrdinalIgnoreCase)) return;

        throw new InvalidOperationException(
            $"CORRECTNESS FAILURE in {mode} at input length {length}.{Environment.NewLine}" +
            $"  expected: {expected}{Environment.NewLine}" +
            $"  actual:   {actual}{Environment.NewLine}" +
            "Benchmark aborted: a faster wrong answer is not an improvement.");
    }
}
