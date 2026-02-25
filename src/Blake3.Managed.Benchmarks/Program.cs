using System.Security.Cryptography;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Running;
using ManagedHasher = Blake3.Managed.Hasher;
using NativeHasher = Blake3.Hasher;

namespace Blake3.Managed.Benchmarks;

public class HashBenchmark
{
    private byte[] _data = null!;
    private Blake3HashAlgorithm _blake3HashAlgorithm = null!;

    [Params(4, 100, 1_000, 10_000, 100_000, 1_000_000)]
    public int Data_Size;

    [GlobalSetup]
    public void Setup()
    {
        _data = new byte[Data_Size];
        new Random(2).NextBytes(_data);
        _blake3HashAlgorithm = new Blake3HashAlgorithm();
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _blake3HashAlgorithm.Dispose();
    }

    [Benchmark(Baseline = true, Description = "Blake3 (native)")]
    public Blake3.Hash RunBlake3Native() => NativeHasher.Hash(_data);

    [Benchmark(Description = "Blake3.Managed")]
    public Blake3.Managed.Hash RunBlake3Managed() => ManagedHasher.Hash(_data);

    [Benchmark(Description = "Blake3.Managed (HashAlgorithm)")]
    public byte[] RunBlake3ManagedHashAlgorithm() => _blake3HashAlgorithm.ComputeHash(_data);

    [Benchmark(Description = "SHA256")]
    public int RunSHA256()
    {
        Span<byte> hash = stackalloc byte[32];
        SHA256.HashData(_data, hash);
        return hash[0];
    }

    static void Main(string[] args)
    {
        var job = Job.ShortRun.WithWarmupCount(3).WithIterationCount(5).WithId("ShortRun3W5I");
        var config = ManualConfig.Create(DefaultConfig.Instance)
            .WithOptions(ConfigOptions.DisableOptimizationsValidator)
            .AddJob(job);

        BenchmarkRunner.Run<HashBenchmark>(config, args);
    }
}
