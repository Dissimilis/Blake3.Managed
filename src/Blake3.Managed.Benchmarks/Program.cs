extern alias Baseline;

using System.Diagnostics;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Diagnosers;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Reports;
using BenchmarkDotNet.Running;
using Perfolizer.Mathematics.OutlierDetection;

namespace Blake3.Managed.Benchmarks;

/// <summary>
/// Entry point for the optimization loop.
///
/// Usage:
///   dotnet run -c Release                    A/B: frozen baseline vs current. The decision run.
///   dotnet run -c Release -- --competitive   where we stand against native / xoofx / CryptoHives
///   dotnet run -c Release -- --api           our own API surfaces against each other
///   dotnet run -c Release -- --all           everything
///   dotnet run -c Release -- --report        breadth for reporting (README numbers)
///   dotnet run -c Release -- --quick         smoke test only -- see the warning below
///   dotnet run -c Release -- --filter "*8192*"   any BenchmarkDotNet filter still works
///
/// HOW TO READ A RESULT. The default run compares the frozen pre-optimization snapshot against
/// the current build. Read the per-size verdict printed after the table, not the table alone.
/// Do NOT compare a Mean from one session against a Mean from another -- this machine throttles
/// roughly 2x under sustained load.
///
/// BenchmarkDotNet runs every case in its own process, sequentially; the before/after pair is
/// adjacent but not interleaved, so the earlier row runs on a slightly cooler package. That biases
/// against 'after', which understates wins and overstates regressions. Pin clocks (turbo off, on
/// AC, max processor state 99%) for any run you intend to act on.
///
/// An earlier version of this harness claimed that ratios against the Rust reference cancelled
/// thermal state out. That was wrong, and measurement showed it: across two sessions with no code
/// change, single-threaded sizes held their native ratio (8 KB: 2.21 then 2.14) while multithreaded
/// sizes drifted (128 KB: 1.22 then 1.51). Our parallel path changes package power and boost
/// behaviour in a way the single-threaded reference does not, so throttling is not a common factor
/// that divides out. Hence the in-process A/B against our own frozen build.
/// </summary>
public static class Program
{
    public static int Main(string[] args)
    {
        var flags = new HashSet<string>(args.Where(a => a.StartsWith("--")), StringComparer.OrdinalIgnoreCase);
        var passthrough = args.Where(a => !IsOurFlag(a)).ToArray();
        bool quick = flags.Contains("--quick");
        bool report = flags.Contains("--report");

        PrintProvenance();

        // A faster wrong answer is not an improvement. Gate every session on correctness, and
        // fail loudly rather than reporting numbers for a broken hash.
        try
        {
            Correctness.Run();
        }
        catch (InvalidOperationException ex)
        {
            Console.Error.WriteLine();
            Console.Error.WriteLine(ex.Message);
            return 1;
        }

        if (quick)
        {
            Console.WriteLine();
            Console.WriteLine("WARNING: --quick is a smoke test. Its error margins are far too wide to");
            Console.WriteLine("decide whether an optimization helped. Never quote a --quick number.");
        }

        Console.WriteLine();

        var config = BuildConfig(quick, report);

        Summary[] summaries = flags.Contains("--all")
            ? BenchmarkRunner.Run(
                new[] { typeof(OptimizationBenchmarks), typeof(CompetitiveBenchmarks), typeof(ApiSurfaceBenchmarks) },
                config, passthrough)
            : flags.Contains("--kernel")
                ? new[] { BenchmarkRunner.Run<KernelBenchmarks>(config, passthrough) }
                : flags.Contains("--dispatch")
                ? new[] { BenchmarkRunner.Run<DispatchBenchmarks>(config, passthrough) }
                : flags.Contains("--competitive")
                ? new[] { BenchmarkRunner.Run<CompetitiveBenchmarks>(config, passthrough) }
                : flags.Contains("--api")
                    ? new[] { BenchmarkRunner.Run<ApiSurfaceBenchmarks>(config, passthrough) }
                    : new[] { BenchmarkRunner.Run<OptimizationBenchmarks>(config, passthrough) };

        foreach (var summary in summaries)
        {
            PrintAbVerdicts(summary);
        }

        return ReportOutcome(summaries);
    }

    /// <summary>
    /// Turns the A/B table into an explicit answer. Without this, a run that never reached its
    /// precision target still prints a crisp-looking Ratio, and a tired reader accepts a 3% "win"
    /// that sits well inside a 6% error margin.
    /// </summary>
    private static void PrintAbVerdicts(Summary summary)
    {
        var pairs = summary.Reports
            .Where(r => r.ResultStatistics is not null)
            .Select(r => new
            {
                Report = r,
                Size = r.BenchmarkCase.Parameters.Items.FirstOrDefault(p => p.Name == "Data_Size")?.Value,
                // Match on the method name, not the [Benchmark(Description)] text: the display
                // info is not guaranteed to carry the description, and a silently unmatched pair
                // means no verdict prints at all.
                Display = r.BenchmarkCase.Descriptor.WorkloadMethod.Name,
            })
            .Where(x => x.Size is not null
                        && (x.Display.StartsWith("Before", StringComparison.Ordinal)
                            || x.Display.StartsWith("After", StringComparison.Ordinal)))
            .GroupBy(x => (
                Size: Convert.ToInt64(x.Size),
                Variant: x.Display.StartsWith("Before", StringComparison.Ordinal)
                    ? x.Display["Before".Length..]
                    : x.Display["After".Length..]))
            .Where(g => g.Count() == 2)
            .OrderBy(g => g.Key.Variant)
            .ThenBy(g => g.Key.Size)
            .ToList();

        if (pairs.Count == 0) return;

        Console.WriteLine();
        Console.WriteLine("A/B verdict (before = frozen baseline, after = current build)");
        Console.WriteLine("-------------------------------------------------------------");

        foreach (var pair in pairs)
        {
            var before = pair.First(x => x.Display.StartsWith("Before", StringComparison.Ordinal)).Report;
            var after = pair.First(x => x.Display.StartsWith("After", StringComparison.Ordinal)).Report;

            var beforeStats = before.ResultStatistics!;
            var afterStats = after.ResultStatistics!;

            double ratio = afterStats.Mean / beforeStats.Mean;
            double changePercent = (ratio - 1.0) * 100.0;

            // Achieved precision, not the precision we asked for. A run capped at MaxIterationCount
            // can finish far from the 1% target while the summary table still looks authoritative.
            double beforeError = beforeStats.StandardError / beforeStats.Mean * 100.0;
            double afterError = afterStats.StandardError / afterStats.Mean * 100.0;
            double worstError = Math.Max(beforeError, afterError);

            bool intervalsOverlap =
                beforeStats.ConfidenceInterval.Lower <= afterStats.ConfidenceInterval.Upper
                && afterStats.ConfidenceInterval.Lower <= beforeStats.ConfidenceInterval.Upper;

            string verdict;
            if (worstError > 2.0)
            {
                verdict = $"NO RESULT (noisy: +/-{worstError:F1}%)";
            }
            else if (intervalsOverlap)
            {
                verdict = "NO RESULT (confidence intervals overlap)";
            }
            else if (changePercent < 0)
            {
                verdict = $"IMPROVED {-changePercent:F1}%";
            }
            else
            {
                verdict = $"REGRESSED {changePercent:F1}%";
            }

            Console.WriteLine($"  {pair.Key.Variant,-28} {pair.Key.Size,10:N0} B  ratio {ratio:F3}  {verdict}");
        }

        Console.WriteLine();
        Console.WriteLine("  NO RESULT means the run cannot support a claim either way -- it does not mean");
        Console.WriteLine("  'no change'. Re-run with clocks pinned, or accept that the effect is below");
        Console.WriteLine("  this machine's resolution.");
        Console.WriteLine();
    }

    private static bool IsOurFlag(string arg) =>
        arg.Equals("--competitive", StringComparison.OrdinalIgnoreCase)
        || arg.Equals("--dispatch", StringComparison.OrdinalIgnoreCase)
        || arg.Equals("--kernel", StringComparison.OrdinalIgnoreCase)
        || arg.Equals("--api", StringComparison.OrdinalIgnoreCase)
        || arg.Equals("--all", StringComparison.OrdinalIgnoreCase)
        || arg.Equals("--quick", StringComparison.OrdinalIgnoreCase)
        || arg.Equals("--report", StringComparison.OrdinalIgnoreCase);

    /// <summary>
    /// A benchmark run that silently failed but exited 0 is worse than no run: it looks like a
    /// clean result. Surface validation errors and failed cases as a nonzero exit.
    /// </summary>
    private static int ReportOutcome(Summary[] summaries)
    {
        var problems = 0;

        foreach (var summary in summaries)
        {
            if (summary.HasCriticalValidationErrors)
            {
                Console.Error.WriteLine($"{summary.Title}: critical validation errors.");
                problems++;
            }

            foreach (var report in summary.Reports.Where(r => !r.Success))
            {
                Console.Error.WriteLine($"FAILED: {report.BenchmarkCase.DisplayInfo}");
                problems++;
            }
        }

        if (problems > 0)
        {
            Console.Error.WriteLine($"{problems} benchmark problem(s); results are not trustworthy.");
            return 1;
        }

        return 0;
    }

    private static IConfig BuildConfig(bool quick, bool report = false)
    {
        // Both jobs keep BenchmarkDotNet's pilot stage, which chooses the invocation count.
        // Job.Dry skips it and measures a single invocation, reporting error margins over 1000%.
        Job job;

        if (quick)
        {
            job = Job.ShortRun.WithWarmupCount(1).WithIterationCount(3).WithId("Smoke");
        }
        else if (report)
        {
            // For publishing a table across many sizes, where the useful precision is "which
            // implementation is faster and roughly by how much" rather than "did this commit move
            // it 5%". The Decide job below chases 1% relative error and needs about an hour for a
            // 50-case sweep; this settles in minutes and reproduced the same ratios across
            // sessions during development.
            job = Job.ShortRun.WithWarmupCount(3).WithIterationCount(5).WithId("Report");
        }
        else
        {
            // Deliberately not a fixed IterationCount: pinning it disables the adaptive stopping
            // rule, and 5 iterations cannot separate a 5% change from noise on a throttling
            // laptop. MaxRelativeError lets BenchmarkDotNet keep iterating until the measurement
            // is precise enough to support the claim.
            job = Job.Default
                .WithMinWarmupCount(6)
                .WithMinIterationCount(15)
                .WithMaxIterationCount(60)
                .WithMaxRelativeError(0.01)
                .WithLaunchCount(2)
                // A run that slowed because the package got hot is not an accidental outlier; it
                // is evidence the conditions changed. Keep them visible rather than trimmed.
                .WithOutlierMode(OutlierMode.DontRemove)
                .WithId("Decide");
        }

        // No DisableOptimizationsValidator here on purpose: during an optimization pass, silently
        // benchmarking an unoptimized assembly is exactly the mistake this harness must refuse.
        return ManualConfig.Create(DefaultConfig.Instance)
            .AddJob(job)
            .AddDiagnoser(MemoryDiagnoser.Default)
            .AddDiagnoser(ThreadingDiagnoser.Default);
    }

    /// <summary>
    /// Stamps the commit under test. On a machine whose speed depends on its temperature, an
    /// unattributable number is worse than no number at all.
    /// </summary>
    private static void PrintProvenance()
    {
        Console.WriteLine($"Blake3.Managed benchmark suite  |  commit {GitDescribe()}  |  {DateTime.Now:yyyy-MM-dd HH:mm}");
        Console.WriteLine("Decision metric: current build vs the frozen baseline build, in this same run.");
        Console.WriteLine("Absolute nanoseconds are NOT comparable across sessions (this machine throttles ~2x).");
        VerifyBaselineIsDistinct();
        Console.WriteLine();
    }

    /// <summary>
    /// Proves the two A/B rows really are two different builds.
    /// </summary>
    /// <remarks>
    /// This is the harness's most plausible silent lie: if a future edit points both rows at the
    /// same hasher, or the frozen snapshot is accidentally refreshed, every ratio pins near 1.00
    /// and reads as "the optimization did nothing" -- indistinguishable from a real null result,
    /// and no test fails. Assert the identity rather than trusting a README not to drift.
    /// </remarks>
    private static void VerifyBaselineIsDistinct()
    {
        var current = typeof(Blake3.Managed.Hasher).Assembly;
        var baseline = typeof(Baseline::Blake3.Managed.Hasher).Assembly;

        var currentName = current.GetName().Name;
        var baselineName = baseline.GetName().Name;

        if (currentName == baselineName || ReferenceEquals(current, baseline))
        {
            throw new InvalidOperationException(
                $"A/B control is broken: both rows resolve to assembly '{currentName}'. "
                + "The comparison would report ~1.00 regardless of any optimization.");
        }

        var baselineFile = baseline.Location.Length > 0 ? Path.GetFileName(baseline.Location) : "in-memory";
        Console.WriteLine($"A/B control: current='{currentName}' vs baseline='{baselineName}' ({baselineFile})");
    }

    private static string GitDescribe()
    {
        try
        {
            var psi = new ProcessStartInfo("git", "describe --always --dirty --tags")
            {
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
            };

            using var process = Process.Start(psi);
            if (process is null) return "unknown";

            var output = process.StandardOutput.ReadToEnd().Trim();
            process.WaitForExit(3000);
            return string.IsNullOrEmpty(output) ? "unknown" : output;
        }
        catch
        {
            return "unknown";
        }
    }
}
