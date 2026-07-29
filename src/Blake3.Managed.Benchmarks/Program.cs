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
///   dotnet run -c Release -- --quick         smoke test only -- see the warning below
///   dotnet run -c Release -- --filter "*8192*"   any BenchmarkDotNet filter still works
///
/// HOW TO READ A RESULT. The default run compares the frozen pre-optimization snapshot against
/// the current build, alternating within one process. Read the Ratio column: below 1.00 is an
/// improvement. Do NOT compare a Mean from one session against a Mean from another -- this
/// machine throttles roughly 2x under sustained load.
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

        var config = BuildConfig(quick);

        Summary[] summaries = flags.Contains("--all")
            ? BenchmarkRunner.Run(
                new[] { typeof(OptimizationBenchmarks), typeof(CompetitiveBenchmarks), typeof(ApiSurfaceBenchmarks) },
                config, passthrough)
            : flags.Contains("--competitive")
                ? new[] { BenchmarkRunner.Run<CompetitiveBenchmarks>(config, passthrough) }
                : flags.Contains("--api")
                    ? new[] { BenchmarkRunner.Run<ApiSurfaceBenchmarks>(config, passthrough) }
                    : new[] { BenchmarkRunner.Run<OptimizationBenchmarks>(config, passthrough) };

        return ReportOutcome(summaries);
    }

    private static bool IsOurFlag(string arg) =>
        arg.Equals("--competitive", StringComparison.OrdinalIgnoreCase)
        || arg.Equals("--api", StringComparison.OrdinalIgnoreCase)
        || arg.Equals("--all", StringComparison.OrdinalIgnoreCase)
        || arg.Equals("--quick", StringComparison.OrdinalIgnoreCase);

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

    private static IConfig BuildConfig(bool quick)
    {
        // Both jobs keep BenchmarkDotNet's pilot stage, which chooses the invocation count.
        // Job.Dry skips it and measures a single invocation, reporting error margins over 1000%.
        Job job;

        if (quick)
        {
            job = Job.ShortRun.WithWarmupCount(1).WithIterationCount(3).WithId("Smoke");
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
        Console.WriteLine("Decision metric: Ratio vs the frozen baseline build, measured in this same process.");
        Console.WriteLine("Absolute nanoseconds are NOT comparable across sessions (this machine throttles ~2x).");
        Console.WriteLine();
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
