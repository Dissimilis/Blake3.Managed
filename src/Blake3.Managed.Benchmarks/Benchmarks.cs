extern alias XoofxManaged;
extern alias Baseline;

using System.Security.Cryptography;
using BenchmarkDotNet.Attributes;
using ManagedHasher = Blake3.Managed.Hasher;
using BaselineHasher = Baseline::Blake3.Managed.Hasher;
using NativeHasher = Blake3.Hasher;
using XoofxHasher = XoofxManaged::Blake3.Hasher;
using CryptoHivesBlake3 = CryptoHives.Foundation.Security.Cryptography.Hash.Blake3;

namespace Blake3.Managed.Benchmarks;

/// <summary>
/// Input sizes shared across the suite.
/// </summary>
/// <remarks>
/// Sizes sit deliberately on and beside the boundaries the scheduler branches on, so a batching
/// regression spikes at one size instead of averaging away. Note 73_729, not 73_728: at exactly
/// 72 chunks the parallel path reserves the final chunk and bails out because <c>items &lt; 2</c>,
/// so 73_729 is the first input that actually reaches the thread pool.
/// </remarks>
internal static class Sizes
{
    public const string Serial = nameof(Serial);
    public const string Parallel = nameof(Parallel);
}

/// <summary>
/// THE DECISION HARNESS: frozen pre-optimization code versus current code.
///
/// This exists because ratio-against-an-external-reference does not survive thermal throttling.
/// Observed on this machine across two sessions: single-threaded sizes held their ratio to the
/// Rust reference (8 KB: 2.21 then 2.14), but multithreaded sizes drifted badly (128 KB: 1.22 then
/// 1.51) with no code change at all. The cause is that our path is multithreaded and the reference
/// is not, so throttling is not a common factor that divides out. Pairing our own two builds
/// removes that asymmetry: both rows of a pair have identical threading behaviour, so thermal
/// sensitivity becomes common-mode.
///
/// WHAT THIS DOES NOT DO: BenchmarkDotNet runs each case in its own process, sequentially, not as
/// interleaved invocations. With the default orderer a before/after pair is adjacent, but the
/// first block still warms the package for the second. The bias therefore runs one way, mid-session
/// 'after' tends to run hotter than 'before', which understates wins and overstates regressions.
/// That is conservative for accepting an improvement, but for a borderline regression confirm with
/// a swapped-order run before believing it. Pinning clocks (turbo off, AC power, max processor
/// state 99%) removes most of this and is worth doing for any decision run.
///
/// Read the Ratio column against the 'before' row. Below 1.00 is an improvement. The run prints an
/// explicit per-size verdict afterwards; prefer that to eyeballing the table.
/// </summary>
public class OptimizationBenchmarks
{
    private byte[] _data = null!;

    // Includes 1 MB and 10 MB: subtree scheduling and parent-reduction changes land in the
    // multithreaded band, so leaving the decision run at 128 KB would give the largest planned
    // changes no before/after coverage at all.
    [Params(4, 128, 1_024, 1_025, 4_095, 4_096, 4_097, 8_192, 16_384, 65_536, 73_729, 131_072, 1_048_576, 10_485_760)]
    public int Data_Size;

    [GlobalSetup]
    public void Setup()
    {
        _data = new byte[Data_Size];
        new Random(2).NextBytes(_data);
    }

    // Benchmarks return one byte of the digest rather than the Hash struct for two reasons: the
    // byte is observably dependent on every input block, so the JIT cannot elide the hash; and
    // BenchmarkDotNet emits these signatures into a generated project that does not carry our
    // extern aliases, where naming any Blake3.Hash type fails to compile (CS0433).

    [Benchmark(Baseline = true, Description = "before: Hash() [auto-parallel]")]
    public byte BeforeOneShot() => BaselineHasher.Hash(_data).AsSpan()[0];

    [Benchmark(Description = "after:  Hash() [auto-parallel]")]
    public byte AfterOneShot() => ManagedHasher.Hash(_data).AsSpan()[0];

    // Forced-serial pair. Hash() farms subtrees to the thread pool above ~72 KB, which makes the
    // one-shot rows above a wall-clock comparison rather than a kernel comparison. These two
    // isolate single-threaded kernel work, where most of the optimization actually lands.
    [Benchmark(Description = "before: Update() [serial]")]
    public byte BeforeSerial()
    {
        Span<byte> hash = stackalloc byte[32];
        using var hasher = BaselineHasher.New();
        hasher.Update(_data);
        hasher.Finalize(hash);
        return hash[0];
    }

    [Benchmark(Description = "after:  Update() [serial]")]
    public byte AfterSerial()
    {
        Span<byte> hash = stackalloc byte[32];
        using var hasher = ManagedHasher.New();
        hasher.Update(_data);
        hasher.Finalize(hash);
        return hash[0];
    }
}

/// <summary>
/// Competitive context: where we stand against the field. Useful for reporting and for choosing
/// what to work on next, but NOT the signal for whether a given commit helped — use
/// <see cref="OptimizationBenchmarks"/> for that.
///
/// Threading policy is labelled per row because it is not uniform: our one-shot goes parallel
/// above ~72 KB while every rival here is single-threaded, so the large-input rows compare
/// wall-clock latency across differing core counts, not per-core efficiency.
/// </summary>
public class CompetitiveBenchmarks
{
    private byte[] _data = null!;

    [Params(4, 128, 1_024, 4_096, 8_192, 16_384, 65_536, 131_072, 1_048_576, 10_485_760)]
    public int Data_Size;

    [GlobalSetup]
    public void Setup()
    {
        _data = new byte[Data_Size];
        new Random(2).NextBytes(_data);
    }

    [Benchmark(Baseline = true, Description = "Blake3.Native Rust [serial]")]
    public byte Native() => NativeHasher.Hash(_data).AsSpan()[0];

    [Benchmark(Description = "Blake3 3.x xoofx managed [serial]")]
    public byte Xoofx() => XoofxHasher.Hash(_data).AsSpan()[0];

    // Runs their SSSE3 kernel on Zen 4: their detection reports Ssse3 despite AVX2/AVX-512 being
    // available, and their constructor intersects the requested set with the detected one, so the
    // fast path cannot be forced from outside the assembly. This is CryptoHives-at-SSSE3, not
    // CryptoHives-at-best; do not quote a win over this column without that caveat.
    [Benchmark(Description = "CryptoHives SSSE3 here [serial]")]
    public byte CryptoHives()
    {
        Span<byte> hash = stackalloc byte[32];
        CryptoHivesBlake3.TryHashData(_data, hash, out _);
        return hash[0];
    }

    [Benchmark(Description = "ours Hash() [parallel >72KB]")]
    public byte OursOneShot() => ManagedHasher.Hash(_data).AsSpan()[0];

    [Benchmark(Description = "ours Update() [serial]")]
    public byte OursSerial()
    {
        Span<byte> hash = stackalloc byte[32];
        using var hasher = ManagedHasher.New();
        hasher.Update(_data);
        hasher.Finalize(hash);
        return hash[0];
    }

    // Not a BLAKE3 rival, but the reference most readers actually have intuition for.
    [Benchmark(Description = "SHA256 [serial]")]
    public byte Sha256()
    {
        Span<byte> hash = stackalloc byte[32];
        SHA256.HashData(_data, hash);
        return hash[0];
    }
}

/// <summary>
/// Decides between two internal dispatch shapes for one-chunk inputs, in a single run so that
/// thermal drift cannot favour either. Kept because the two candidates measured differently under
/// BenchmarkDotNet than in a tight-loop harness, and only a same-run comparison settles it.
/// </summary>
public class DispatchBenchmarks
{
    private byte[] _data = null!;

    [Params(4, 64, 128, 1_024)]
    public int Data_Size;

    [GlobalSetup]
    public void Setup()
    {
        _data = new byte[Data_Size];
        new Random(2).NextBytes(_data);
    }

    // Route through Blake3Core, as before: one extra call layer.
    [Benchmark(Baseline = true, Description = "via Blake3Core")]
    public byte ViaCore()
    {
        Span<byte> hash = stackalloc byte[32];
        Blake3.Managed.Internal.Blake3Core.HashOneChunkRoot32(
            Blake3.Managed.Internal.Blake3Constants.IV, 0, 0, hash, _data);
        return hash[0];
    }

    // The actual public API, for reference: same work, plus the wrapper and a 32-byte struct
    // return. Included here because it measured far slower than the internal paths and the
    // difference had to be attributable to something.
    [Benchmark(Description = "public Hasher.Hash")]
    public byte PublicApi() => ManagedHasher.Hash(_data).AsSpan()[0];

    // Same, but writing into a caller-provided span instead of returning the Hash struct.
    [Benchmark(Description = "public Hash(span,span)")]
    public byte PublicApiSpan()
    {
        Span<byte> hash = stackalloc byte[32];
        ManagedHasher.Hash(_data, hash);
        return hash[0];
    }

    // Dispatch straight to the compressor, as Hasher.Hash now does.
    [Benchmark(Description = "direct dispatch")]
    public byte Direct()
    {
        Span<byte> hash = stackalloc byte[32];
        Span<uint> words = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, uint>(hash);

        if (_data.Length <= 64)
        {
            Blake3.Managed.Internal.CompressSse41.CompressRootIvSingleBlock(_data, words);
        }
        else
        {
            Blake3.Managed.Internal.CompressSse41.HashChunkRoot32Iv(_data, words);
        }

        return hash[0];
    }
}

/// <summary>
/// Raw kernel throughput, isolated from tree building and parent hashing.
/// </summary>
/// <remarks>
/// The end-to-end numbers mix chunk hashing, parent hashing and scheduling, which makes it
/// impossible to tell which one is behind a gap. These measure one call of each multi-chunk kernel
/// so the per-chunk cost can be compared directly against what the reference implementation
/// achieves.
/// </remarks>
public class KernelBenchmarks
{
    private byte[] _chunks = null!;
    private byte[] _big = null!;
    private uint[] _cvs = null!;
    private uint[] _parents = null!;

    [GlobalSetup]
    public void Setup()
    {
        _chunks = new byte[64 * 1024];
        new Random(3).NextBytes(_chunks);
        _big = new byte[1024 * 1024];
        new Random(5).NextBytes(_big);
        _cvs = new uint[64 * 8];
        _parents = new uint[16 * 8];
        new Random(4).NextBytes(System.Runtime.InteropServices.MemoryMarshal.AsBytes(_parents.AsSpan()));
    }

    // 8 chunks (8 KB) through the AVX2 kernel: 16 eight-way compressions.
    [Benchmark(Description = "AVX2 HashMany 8 chunks")]
    public uint Avx2Eight()
    {
        Blake3.Managed.Internal.HashManyAvx2.HashMany(_chunks.AsSpan(0, 8 * 1024), 8,
            Blake3.Managed.Internal.Blake3Constants.IV, 0, 0, _cvs.AsSpan(0, 64));
        return _cvs[0];
    }

    // 4 chunks through the SSE kernel, which is what a 4 KB input currently uses.
    [Benchmark(Description = "SSE HashMany 4 chunks")]
    public uint SseFour()
    {
        Blake3.Managed.Internal.HashManySse41.HashMany(_chunks.AsSpan(0, 4 * 1024), 4,
            Blake3.Managed.Internal.Blake3Constants.IV, 0, 0, _cvs.AsSpan(0, 32));
        return _cvs[0];
    }

    // The same 4 chunks through the AVX2 kernel with four lanes wasted, for comparison.
    [Benchmark(Description = "AVX2 HashMany 8 chunks (for 4KB compare)")]
    public uint Avx2EightFor4K()
    {
        Blake3.Managed.Internal.HashManyAvx2.HashMany(_chunks.AsSpan(0, 8 * 1024), 8,
            Blake3.Managed.Internal.Blake3Constants.IV, 0, 0, _cvs.AsSpan(0, 64));
        return _cvs[0];
    }

    // Serial tree over 1 MB: the per-unit work with no thread-pool involvement.
    [Benchmark(Description = "serial tree 1MB")]
    public byte SerialTree1M()
    {
        Span<byte> hash = stackalloc byte[32];
        Blake3.Managed.Internal.Blake3Tree.HashAllAtOnce(_big,
            Blake3.Managed.Internal.Blake3Constants.IV, 0, hash);
        return hash[0];
    }

    // Parallel tree over the same 1 MB.
    [Benchmark(Description = "parallel tree 1MB")]
    public byte ParallelTree1M()
    {
        Span<byte> hash = stackalloc byte[32];
        Blake3.Managed.Internal.Blake3Tree.HashAllAtOnceParallel(_big,
            Blake3.Managed.Internal.Blake3Constants.IV, 0, hash);
        return hash[0];
    }

    // The public API on the same 1 MB buffer, which should reach the same parallel tree.
    [Benchmark(Description = "public Hasher.Hash 1MB")]
    public byte PublicHash1M() => ManagedHasher.Hash(_big).AsSpan()[0];

    // Serial tree over one 128 KB unit, i.e. what a single parallel task does at 10 MB.
    [Benchmark(Description = "serial tree 128KB unit")]
    public byte SerialUnit()
    {
        Span<byte> hash = stackalloc byte[32];
        Blake3.Managed.Internal.Blake3Tree.HashAllAtOnce(_big.AsSpan(0, 128 * 1024),
            Blake3.Managed.Internal.Blake3Constants.IV, 0, hash);
        return hash[0];
    }

    // 8 parent nodes in one AVX2 call.
    [Benchmark(Description = "AVX2 HashParents8")]
    public uint Parents8()
    {
        Blake3.Managed.Internal.HashManyAvx2.HashParents8(_parents, 
            Blake3.Managed.Internal.Blake3Constants.IV, Blake3.Managed.Internal.Blake3Constants.Parent,
            _cvs.AsSpan(0, 64));
        return _cvs[0];
    }

    // One scalar/SSE parent compression, for the per-parent cost.
    [Benchmark(Description = "single parent compression")]
    public uint Parent1()
    {
        Blake3.Managed.Internal.Blake3Core.CompressCv(
            Blake3.Managed.Internal.Blake3Constants.IV, _parents.AsSpan(0, 16), 0, 64,
            Blake3.Managed.Internal.Blake3Constants.Parent, _cvs.AsSpan(0, 8));
        return _cvs[0];
    }
}

/// <summary>
/// Our own API surfaces against each other. Worth tracking separately because the adapters do not
/// reach the parallel path today, so <see cref="Blake3HashAlgorithm"/> on a large input is several
/// times slower than <c>Hasher.Hash</c> on the same bytes — and the adapters are what most
/// applications actually consume.
/// </summary>
public class ApiSurfaceBenchmarks
{
    private byte[] _data = null!;
    private Blake3HashAlgorithm _hashAlgorithm = null!;

    [Params(1_024, 65_536, 1_048_576, 10_485_760)]
    public int Data_Size;

    [GlobalSetup]
    public void Setup()
    {
        _data = new byte[Data_Size];
        new Random(2).NextBytes(_data);
        _hashAlgorithm = new Blake3HashAlgorithm();
    }

    [GlobalCleanup]
    public void Cleanup() => _hashAlgorithm.Dispose();

    [Benchmark(Baseline = true, Description = "Hasher.Hash one-shot")]
    public byte OneShot() => ManagedHasher.Hash(_data).AsSpan()[0];

    [Benchmark(Description = "Hasher.Update + Finalize")]
    public byte Incremental()
    {
        Span<byte> hash = stackalloc byte[32];
        using var hasher = ManagedHasher.New();
        hasher.Update(_data);
        hasher.Finalize(hash);
        return hash[0];
    }

    [Benchmark(Description = "Hasher.UpdateWithJoin")]
    public byte UpdateWithJoin()
    {
        Span<byte> hash = stackalloc byte[32];
        using var hasher = ManagedHasher.New();
        hasher.UpdateWithJoin(_data);
        hasher.Finalize(hash);
        return hash[0];
    }

    [Benchmark(Description = "Blake3HashAlgorithm adapter")]
    public byte[] HashAlgorithmAdapter() => _hashAlgorithm.ComputeHash(_data);

    [Benchmark(Description = "Blake3Stream")]
    public byte StreamWrapper()
    {
        using var stream = new Blake3Stream(Stream.Null);
        stream.Write(_data, 0, _data.Length);
        return stream.ComputeHash().AsSpan()[0];
    }

    [Benchmark(Description = "SHA256 reference")]
    public byte Sha256()
    {
        Span<byte> hash = stackalloc byte[32];
        SHA256.HashData(_data, hash);
        return hash[0];
    }
}
