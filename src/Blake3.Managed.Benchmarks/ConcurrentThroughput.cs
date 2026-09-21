extern alias Baseline;

using System.Diagnostics;
using ManagedHasher = Blake3.Managed.Hasher;
using BaselineHasher = Baseline::Blake3.Managed.Hasher;

namespace Blake3.Managed.Benchmarks;

/// <summary>
/// Aggregate-throughput check for the one-shot parallel cutoff, run with <c>--concurrent</c>.
///
/// BenchmarkDotNet measures a single caller's latency, where farming a 32-64 KB input to the
/// thread pool wins because the other cores are idle. A server hashing many such inputs at once
/// already saturates the cores, so the same fan-out buys nothing there and pays for queueing and
/// synchronization. This harness measures the aggregate: <c>callers</c> threads each hash their
/// own buffer in a tight loop for a fixed time, and the total bytes hashed per second is the
/// result. The frozen baseline and the current build alternate in short rounds inside one
/// process so thermal drift hits both sides roughly equally; the median round is reported.
///
/// Not a BenchmarkDotNet job on purpose: BDN's process-per-case model cannot alternate two
/// builds, and its per-invocation timing is not the quantity of interest here.
/// </summary>
internal static class ConcurrentThroughput
{
    private static readonly int[] DefaultSizes = { 32_768, 49_152, 65_536, 131_072 };
    private static readonly int[] DefaultCallers = { 1, 8, 16 };

    public static int Run(string[] args)
    {
        var sizes = ParseList(args, "--sizes=", DefaultSizes);
        var callers = ParseList(args, "--callers=", DefaultCallers);
        int rounds = ParseList(args, "--rounds=", new[] { 7 })[0];
        int roundMs = ParseList(args, "--round-ms=", new[] { 400 })[0];

        // --join measures UpdateWithJoin instead of Hash. That is the path Blake3HashAlgorithm
        // and Blake3Stream take, and it has its own fan-out and its own gate, so a change to it
        // is invisible to the default one-shot workload here.
        bool join = args.Contains("--join", StringComparer.OrdinalIgnoreCase);

        Console.WriteLine($"Workload: {(join ? "UpdateWithJoin (adapter path)" : "Hasher.Hash (one-shot)")}");
        Console.WriteLine($"Concurrent-caller throughput: {rounds} alternating rounds of {roundMs} ms per side, median reported.");
        Console.WriteLine($"Logical CPUs: {Environment.ProcessorCount}. Ratio is after/before on MB/s; above 1.00 is an improvement.");
        Console.WriteLine();
        Console.WriteLine($"{"Size",10}  {"Callers",7}  {"before MB/s",12}  {"after MB/s",12}  {"ratio",6}  verdict");

        int regressions = 0;

        foreach (int size in sizes)
        {
            foreach (int n in callers)
            {
                var before = new List<double>(rounds);
                var after = new List<double>(rounds);

                // Warm both sides before anything is recorded. Tiered compilation needs on the
                // order of a second of steady calls to reach Tier1 with PGO; a shorter warm-up
                // measured Tier0 code and produced a spurious 25% "regression" on the workstation.
                int warmMs = Math.Max(roundMs, 1000);
                Measure(size, n, warmMs, baseline: true, join);
                Measure(size, n, warmMs, baseline: false, join);

                for (int r = 0; r < rounds; r++)
                {
                    // Swap order every round so neither side always runs on the hotter package.
                    bool baselineFirst = (r & 1) == 0;
                    before.Add(Measure(size, n, roundMs, baseline: baselineFirst, join));
                    after.Add(Measure(size, n, roundMs, baseline: !baselineFirst, join));
                    if (!baselineFirst) (before[^1], after[^1]) = (after[^1], before[^1]);
                }

                double b = Median(before);
                double a = Median(after);
                double ratio = a / b;
                double spreadB = Spread(before);
                double spreadA = Spread(after);
                double spread = Math.Max(spreadB, spreadA);

                string verdict;
                if (ratio < 0.95)
                {
                    verdict = $"REGRESSED {(1 - ratio) * 100:F1}%";
                    regressions++;
                }
                else if (ratio > 1.05 && spread < (ratio - 1) * 100)
                {
                    verdict = $"IMPROVED {(ratio - 1) * 100:F1}%";
                }
                else
                {
                    verdict = "within 5%";
                }

                Console.WriteLine($"{size,10:N0}  {n,7}  {b,12:N0}  {a,12:N0}  {ratio,6:F3}  {verdict}  (round spread +/-{spread:F1}%)");
            }
        }

        Console.WriteLine();
        Console.WriteLine(regressions == 0
            ? "No configuration lost more than 5% aggregate throughput."
            : $"{regressions} configuration(s) lost more than 5% aggregate throughput.");
        return regressions == 0 ? 0 : 1;
    }

    /// <summary>
    /// Runs <paramref name="callers"/> dedicated threads hashing for about <paramref name="ms"/>
    /// milliseconds and returns the aggregate MB/s. Dedicated threads rather than pool threads:
    /// the hasher's own fan-out uses the pool, and the callers must compete with it the way
    /// request threads in a server would, not share its queue.
    /// </summary>
    private static double Measure(int size, int callers, int ms, bool baseline, bool join)
    {
        var buffers = new byte[callers][];
        for (int i = 0; i < callers; i++)
        {
            buffers[i] = new byte[size];
            new Random(1000 + i).NextBytes(buffers[i]);
        }

        long totalBytes = 0;
        using var start = new ManualResetEventSlim(false);
        var stop = new Stopwatch();
        var threads = new Thread[callers];
        long deadlineTicks = Stopwatch.Frequency * ms / 1000;

        for (int i = 0; i < callers; i++)
        {
            var data = buffers[i];
            threads[i] = new Thread(() =>
            {
                start.Wait();
                long bytes = 0;
                byte sink = 0;
                Span<byte> hash = stackalloc byte[32];
                while (stop.ElapsedTicks < deadlineTicks)
                {
                    if (join)
                    {
                        // A fresh hasher per iteration, as the adapters effectively do: the
                        // fan-out decision depends on the chunk counter being aligned, which a
                        // reused hasher would not reproduce.
                        if (baseline)
                        {
                            using var h = BaselineHasher.New();
                            h.UpdateWithJoin(data);
                            h.Finalize(hash);
                        }
                        else
                        {
                            using var h = ManagedHasher.New();
                            h.UpdateWithJoin(data);
                            h.Finalize(hash);
                        }
                    }
                    else if (baseline)
                    {
                        BaselineHasher.Hash(data, hash);
                    }
                    else
                    {
                        ManagedHasher.Hash(data, hash);
                    }

                    sink ^= hash[0];
                    bytes += data.Length;
                }
                Interlocked.Add(ref totalBytes, bytes);
                if (sink == 0xFF && bytes == -1) Console.Write(sink);
            })
            { IsBackground = true };
            threads[i].Start();
        }

        stop.Start();
        start.Set();
        foreach (var t in threads) t.Join();
        stop.Stop();

        return totalBytes / stop.Elapsed.TotalSeconds / 1_000_000.0;
    }

    private static double Median(List<double> values)
    {
        var sorted = values.OrderBy(v => v).ToArray();
        int mid = sorted.Length / 2;
        return sorted.Length % 2 == 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
    }

    /// <summary>Half the min-max range as a percentage of the median.</summary>
    private static double Spread(List<double> values)
        => (values.Max() - values.Min()) / 2 / Median(values) * 100;

    private static int[] ParseList(string[] args, string prefix, int[] fallback)
    {
        var arg = args.FirstOrDefault(a => a.StartsWith(prefix, StringComparison.OrdinalIgnoreCase));
        if (arg is null) return fallback;
        return arg[prefix.Length..].Split(',', StringSplitOptions.RemoveEmptyEntries)
            .Select(s => int.Parse(s.Replace("_", ""))).ToArray();
    }
}
