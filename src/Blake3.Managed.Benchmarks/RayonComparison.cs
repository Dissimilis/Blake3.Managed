using System.Diagnostics;
using ManagedHasher = Blake3.Managed.Hasher;
using NativeHasher = Blake3.Hasher;

namespace Blake3.Managed.Benchmarks;

/// <summary>
/// The multi-core comparison the published tables have never made, run with <c>--rayon</c>.
///
/// Every large-input row in the README compares our fan-out against the Rust binding's *single*
/// thread, because the competitive job calls the one-shot <c>Blake3.Hasher.Hash</c>. The Rust
/// binding also exposes <c>UpdateWithJoin</c>, backed by rayon, and until that is measured the
/// claim "fastest on large inputs" rests on a core-count difference rather than on being faster.
///
/// Four contenders at each size, so the two comparisons that matter are both in one run:
/// Rust serial vs ours serial (per-core efficiency) and Rust rayon vs our fan-out (what a caller
/// who opts into threading on both sides actually gets).
///
/// Structured like <see cref="ConcurrentThroughput"/> rather than as a BenchmarkDotNet job:
/// contenders alternate in short rounds inside one process so thermal drift hits them roughly
/// equally, and the median round is reported. One caller measures latency; 8 and 16 callers
/// measure what a saturated machine delivers, where a fan-out can lose badly.
/// </summary>
internal static class RayonComparison
{
    private static readonly int[] DefaultSizes = { 65_536, 131_072, 1_048_576, 10_485_760 };
    private static readonly int[] DefaultCallers = { 1, 8, 16 };

    private enum Contender
    {
        NativeSerial,
        NativeRayon,
        OursOneShot,
        OursJoin,
    }

    private static string Label(Contender c) => c switch
    {
        Contender.NativeSerial => "Rust serial",
        Contender.NativeRayon => "Rust rayon",
        Contender.OursOneShot => "ours Hash()",
        _ => "ours Join",
    };

    public static int Run(string[] args)
    {
        var sizes = ParseList(args, "--sizes=", DefaultSizes);
        var callers = ParseList(args, "--callers=", DefaultCallers);
        int rounds = ParseList(args, "--rounds=", new[] { 7 })[0];
        int roundMs = ParseList(args, "--round-ms=", new[] { 400 })[0];

        var contenders = (Contender[])Enum.GetValues(typeof(Contender));

        Console.WriteLine($"Rayon comparison: {rounds} alternating rounds of {roundMs} ms per contender, median reported.");
        Console.WriteLine($"Logical CPUs: {Environment.ProcessorCount}.");
        Console.WriteLine("Ratios are ours/Rust on MB/s; above 1.00 means we are faster.");
        Console.WriteLine("'serial' compares per-core efficiency; 'threaded' compares fan-out against rayon.");
        Console.WriteLine();
        Console.WriteLine($"{"Size",10}  {"Callers",7}  {"Rust serial",12}  {"Rust rayon",12}  {"ours Hash()",12}  {"ours Join",12}  {"serial",7}  {"threaded",8}  best");

        int lost = 0;

        foreach (int size in sizes)
        {
            foreach (int n in callers)
            {
                var samples = new Dictionary<Contender, List<double>>();
                foreach (var c in contenders) samples[c] = new List<double>(rounds);

                // Allocated once per cell, not per measurement: at the top cell the per-call
                // version churned several GB of fresh pages per session. The fill is outside
                // every timed window, but the GC and page-cache pressure it created was not.
                // Sharing them also means every contender hashes exactly the same bytes.
                var buffers = new byte[n][];
                for (int i = 0; i < n; i++)
                {
                    buffers[i] = new byte[size];
                    new Random(1000 + i).NextBytes(buffers[i]);
                }

                // Warm every contender before recording. Tiered compilation needs ~1 s of steady
                // calls to reach Tier1 with PGO; a shorter warm-up has produced spurious verdicts.
                int warmMs = Math.Max(roundMs, 1000);
                foreach (var c in contenders) Measure(buffers, warmMs, c);

                for (int r = 0; r < rounds; r++)
                {
                    // Rotate the order every round so no contender always runs on the hotter package.
                    for (int i = 0; i < contenders.Length; i++)
                    {
                        var c = contenders[(i + r) % contenders.Length];
                        samples[c].Add(Measure(buffers, roundMs, c));
                    }
                }

                double nativeSerial = Median(samples[Contender.NativeSerial]);
                double nativeRayon = Median(samples[Contender.NativeRayon]);
                double oursOneShot = Median(samples[Contender.OursOneShot]);
                double oursJoin = Median(samples[Contender.OursJoin]);

                double serialRatio = oursOneShot / nativeSerial;
                double bestOurs = Math.Max(oursOneShot, oursJoin);
                double bestNative = Math.Max(nativeSerial, nativeRayon);
                double threadedRatio = bestOurs / bestNative;
                if (threadedRatio < 1.0) lost++;

                var best = contenders.OrderByDescending(c => Median(samples[c])).First();

                Console.WriteLine(
                    $"{size,10:N0}  {n,7}  {nativeSerial,12:N0}  {nativeRayon,12:N0}  {oursOneShot,12:N0}  {oursJoin,12:N0}  " +
                    $"{serialRatio,7:F3}  {threadedRatio,8:F3}  {Label(best)}");
            }
        }

        Console.WriteLine();
        Console.WriteLine(lost == 0
            ? "Our best configuration beat Rust's best at every size and caller count."
            : $"Rust's best configuration won {lost} of {sizes.Length * callers.Length} cells. "
              + "The 'fastest on large inputs' claim needs qualifying until those are closed.");
        return 0;
    }

    /// <summary>
    /// Runs <paramref name="callers"/> dedicated threads hashing for about <paramref name="ms"/>
    /// milliseconds and returns aggregate MB/s. Dedicated threads rather than pool threads: both
    /// fan-outs use their own worker pools and the callers must compete with them the way request
    /// threads in a server would, not share their queues.
    /// </summary>
    private static double Measure(byte[][] buffers, int ms, Contender contender)
    {
        int callers = buffers.Length;
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
                    switch (contender)
                    {
                        case Contender.NativeSerial:
                            NativeHasher.Hash(data, hash);
                            break;

                        case Contender.NativeRayon:
                        {
                            // Rust's rayon path is only reachable through the incremental API, so
                            // the hasher is constructed per iteration on both threaded sides to
                            // keep the comparison like for like.
                            using var h = NativeHasher.New();
                            h.UpdateWithJoin(data);
                            h.Finalize(hash);
                            break;
                        }

                        case Contender.OursOneShot:
                            ManagedHasher.Hash(data, hash);
                            break;

                        default:
                        {
                            using var h = ManagedHasher.New();
                            h.UpdateWithJoin(data);
                            h.Finalize(hash);
                            break;
                        }
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

    private static int[] ParseList(string[] args, string prefix, int[] fallback)
    {
        var arg = args.FirstOrDefault(a => a.StartsWith(prefix, StringComparison.OrdinalIgnoreCase));
        if (arg is null) return fallback;

        var fields = arg.Substring(prefix.Length)
            .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);

        var parsed = new List<int>(fields.Length);
        foreach (var field in fields)
        {
            if (!int.TryParse(field, out int value) || value <= 0)
            {
                Console.WriteLine($"Ignoring '{field}' in {prefix}: expected a positive integer.");
                continue;
            }

            parsed.Add(value);
        }

        return parsed.Count == 0 ? fallback : parsed.ToArray();
    }
}
