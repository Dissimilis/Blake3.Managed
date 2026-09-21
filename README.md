# Blake3.Managed: BLAKE3 hash function for .NET

A managed C# implementation of the [BLAKE3](https://github.com/BLAKE3-team/BLAKE3) hash function for .NET, with no native dependencies.

One managed DLL, no P/Invoke and no per-platform assets. Use it if you want BLAKE3 without shipping a native library, or if you hash large inputs: its default one-shot `Hash()` spreads large inputs across the thread pool, which the other .NET packages only do through an explicit `UpdateWithJoin` call. On a single thread it is in the same range as the Rust binding, sometimes ahead and sometimes behind. If you only hash small inputs and want the last few percent, or you need an implementation someone else has audited, use [`Blake3.Native`](https://www.nuget.org/packages/Blake3.Native) instead.

[![NuGet](https://img.shields.io/nuget/v/Blake3.Managed.svg)](https://www.nuget.org/packages/Blake3.Managed)
[![NuGet Downloads](https://img.shields.io/nuget/dt/Blake3.Managed.svg)](https://www.nuget.org/packages/Blake3.Managed)
[![CI](https://github.com/Dissimilis/Blake3.Managed/actions/workflows/ci.yml/badge.svg)](https://github.com/Dissimilis/Blake3.Managed/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-donate-yellow.svg)](https://buymeacoffee.com/dissimilis)

## Features

- **Hardware accelerated** - AVX2 8-way parallel hashing, SSE/SSSE3 vectorized compression, ARM NEON 4-way parallel hashing, automatic scalar fallback
- **Multi-threaded** - large inputs are split into subtrees and hashed on the thread pool. One-shot `Hash()` fans out from ~32 KiB, and backs off to a single thread when many hashes are already in flight; `UpdateWithJoin`, used by `Blake3HashAlgorithm` and `Blake3Stream`, fans out from ~32 KiB too on AVX2 hardware, under the same back-off
- **Split hashing** - hash pieces independently and combine them into the whole-input digest with `Blake3SubtreeContext`
- **Zero allocation** for small inputs with `Hasher.Hash()`
- **All BLAKE3 modes** - default hashing, keyed hashing, and key derivation
- **XOF support** - extendable output with a seekable byte stream and AVX2 batching for long output
- **Familiar API** - modeled after [Blake3.NET](https://github.com/xoofx/Blake3.NET)
- **Targets** `net6.0`, `net8.0` and `net10.0`, with Native AOT support on `net8.0` and above

## Other BLAKE3 packages

[Blake3.NET](https://github.com/xoofx/Blake3.NET) by Alexandre Mutel ships two packages in 3.x: [`Blake3`](https://www.nuget.org/packages/Blake3), a managed SIMD implementation, and [`Blake3.Native`](https://www.nuget.org/packages/Blake3.Native), a binding to the Rust implementation.

Both hash on one thread unless you call `UpdateWithJoin` explicitly; here the plain `Hash()` also uses multiple cores for large inputs. It targets .NET 6 as well as 8 and 10. If your application already hashes many inputs in parallel, set `Hasher.MaxDegreeOfParallelism` so the two layers do not compete.

The [benchmarks below](#performance) compare these packages, CryptoHives and SHA256, with separate results for one-shot hashing and extended output.

## Correctness

BLAKE3 has one right answer. CI runs the official test vectors for all three modes. A differential suite against the reference Rust implementation runs before every benchmark session, covering chunk and block boundaries, incremental splits, unaligned spans and extended output at arbitrary offsets.

It has not had an independent security audit. If that matters to you, use a binding to the reference implementation.

## Installation

```bash
dotnet add package Blake3.Managed
```

## Quick Start

```csharp
using Blake3.Managed;

// One-shot hash
var hash = Hasher.Hash("Hello, World!"u8);
Console.WriteLine(hash); // 288a86a79f20a3d6dccdca7713beaed178798296bdfa7913fa2a62d9727bf8f8

// Incremental hashing
using var hasher = Hasher.New();
hasher.Update("Hello, "u8);
hasher.Update("World!"u8);
var result = hasher.Finalize();
```

## Usage examples

### Keyed hashing and key derivation

```csharp
// Keyed hash of an input you already hold, in one call
byte[] key = new byte[32]; // your key here
var mac = Hasher.HashKeyed(key, data);

// Keyed hash, incrementally
using var keyedHasher = Hasher.NewKeyed(key);
keyedHasher.Update(data);
var sameMac = keyedHasher.Finalize();

// Key derivation
using var kdf = Hasher.NewDeriveKey("my-session-key");
kdf.Update(inputKeyMaterial);
var derivedKey = kdf.Finalize();
```

### Extended output

```csharp
using var xof = Hasher.New();
xof.Update(data);

var extendedOutput = new byte[1024];
xof.Finalize(extendedOutput);          // arbitrary length output

var later = new byte[64];
xof.Finalize(1_000_000, later);        // read from any offset in the output stream
```

### Parallel hashing

```csharp
using var parallel = Hasher.New();
parallel.UpdateWithJoin(largeData); // uses thread pool from ~32 KiB
var parallelHash = parallel.Finalize();
```

For one-shot `Hasher.Hash`, the thread limit is process-wide and does not affect `UpdateWithJoin`:

```csharp
Hasher.MaxDegreeOfParallelism = 4;  // use at most 4 threads
Hasher.MaxDegreeOfParallelism = 1;  // run on the calling thread
Hasher.MaxDegreeOfParallelism = -1; // restore the default
```

### Streams and `HashAlgorithm`

`Blake3Stream` hashes data as it passes through, so you can hash a file while copying it:

```csharp
using var source = File.OpenRead(path);
using var hashing = new Blake3Stream(source);
await hashing.CopyToAsync(destination);
var hash = hashing.ComputeHash();
```

`Blake3HashAlgorithm` plugs into anything that takes a `System.Security.Cryptography.HashAlgorithm`:

```csharp
using var algorithm = new Blake3HashAlgorithm();
byte[] digest = algorithm.ComputeHash(stream);
```

Both use `UpdateWithJoin`, so large inputs are hashed on multiple cores where AVX2 is available; elsewhere they fall back to a single thread.

### Hashing independent pieces

`Blake3SubtreeContext` combines independently hashed pieces into the whole-input digest. `ReadPiece` below represents your own file-reading code.

```csharp
using var ctx = Blake3SubtreeContext.Create(pieceSize: 1024 * 1024, totalLength: file.Length);
var pieces = new Blake3Subtree[ctx.PieceCount];
Parallel.For(0, ctx.PieceCount, i =>
{
    var piece = ReadPiece(file, ctx.GetPieceOffset(i), ctx.GetPieceLength(i));
    pieces[i] = ctx.HashSubtree(piece, i);
});
var fileHash = ctx.Finalize(pieces); // equals Hasher.Hash(wholeFile)
```

A piece can also be fed incrementally, without keeping it all in memory. This supports pieces larger than 2 GB:

```csharp
using var pieceHasher = ctx.CreateSubtreeHasher(pieceIndex);
while ((n = await stream.ReadAsync(buffer)) > 0)
    pieceHasher.Update(buffer.AsSpan(0, n));
pieces[pieceIndex] = pieceHasher.Finish();
```

`CreateKeyed` and `CreateDeriveKey` give the same split hashing in the keyed and key-derivation modes.

## Threading and lifetime

`Hasher.Hash` and `Hasher.HashKeyed` are static and safe to call from any number of threads at once. A single `Hasher`, `Blake3Stream` or `Blake3HashAlgorithm` instance is not thread-safe; use one per thread. `Blake3SubtreeContext` is the exception: it exists to be shared, and its pieces can be hashed concurrently and in any order.

`Hasher` is a mutable struct, so copying one copies its state. Pass it by `ref`, and do not store it in a field, a collection or a lambda unless you mean to fork it. `Dispose` zeroes the key material, which matters for keyed and derive-key hashers; the static one-shots need no disposal.

## Public API

| Type | Description |
|------|-------------|
| `Hasher` | Main hasher struct. Factory methods: `New()`, `NewKeyed()`, `NewDeriveKey()`. Static `Hash()` and `HashKeyed()` for one-shot hashing. Incremental via `Update()`/`UpdateWithJoin()`/`Finalize()`. Static `MaxDegreeOfParallelism` caps the fan-out used by `Hash()` and `HashKeyed()`. |
| `Hash` | Fixed 32-byte output struct with constant-time equality and allocation-free `ToString()`. |
| `Blake3Stream` | Stream wrapper that hashes data as it flows through. |
| `Blake3HashAlgorithm` | `System.Security.Cryptography.HashAlgorithm` adapter for interop with existing APIs. |
| `Blake3SubtreeContext` | Hashes fixed-size pieces of one input independently, on any threads and in any order, and folds them into the whole-input digest (32 bytes or XOF). Piece size must be a power-of-two multiple of 1024, unless the known total length fits in a single piece, in which case any size works; it is a `long`, so pieces larger than 2 GB are allowed. |
| `Blake3SubtreeHasher` | Hashes one piece incrementally through `Update` calls and `Finish`, for pieces that arrive in parts or are too large for one span. Created by `Blake3SubtreeContext.CreateSubtreeHasher`. |
| `Blake3Subtree` | Opaque result of hashing one piece, stored at its piece index and passed to `Finalize`. |

## Performance

Above ~32 KiB this library's `Hash()` uses multiple cores while the other columns run on one, so the large-input rows below are a core-count difference rather than a per-core one. Per core it lands within roughly 20% of the Rust binding either way, depending on size.

### Benchmark environment

Measured on **2026-09-21**, on the Linux benchmark host rather than a laptop, so the
numbers are not comparable with the 2026-09-09 tables this replaced.

```text
BenchmarkDotNet 0.15.8, Fedora Linux 44
AMD Ryzen 7 8845HS, 8 physical cores / 16 logical threads
.NET SDK 10.0.111, .NET 10.0.11, X64 RyuJIT with AVX-512 available
Report job: 1 launch, 3 warmup iterations, 5 measured iterations
```

Tables show **mean time ± the 99.9% confidence margin**, with lower times being
better. Each table and chart comes from the same run for that workload. Cases
run sequentially, so clock and thermal changes can affect comparisons; a lower
mean with overlapping intervals is inconclusive. Every session first passed
7,492 correctness checks against official vectors and the Rust implementation.

### One-shot hashing

All BLAKE3 implementations write into a caller-provided 32-byte destination.
This library uses `Hasher.Hash(input, output)`. CryptoHives uses a reused
instance's `TryHashOneShot` with its default SIMD selection; the package reports
SSSE3, AVX2 and AVX-512 support on this machine. SHA256 is a different algorithm,
provided as a reference.

From **64 KiB** upward in this run, this library used multiple cores and the other columns
used one thread, so those rows compare latency at different core counts. KiB and MiB denote
powers of 1,024; the chart's GB/s is decimal.

| Input | Blake3.Native 3.0.2 | Blake3 3.0.2 (xoofx) | CryptoHives 0.6.101 | Blake3.Managed (this library) | SHA256 (.NET) |
|---:|---:|---:|---:|---:|---:|
| 4 B | 65.18 ± 0.27 ns | 61.90 ± 0.10 ns | 64.50 ± 0.07 ns | 40.31 ± 0.05 ns | 287.73 ± 0.56 ns |
| 128 B | 117.84 ± 1.61 ns | 94.23 ± 0.44 ns | 104.49 ± 0.10 ns | 87.75 ± 0.09 ns | 323.89 ± 0.98 ns |
| 1 KiB | 771.00 ± 0.59 ns | 824.70 ± 0.33 ns | 842.49 ± 0.57 ns | 764.66 ± 0.79 ns | 679.29 ± 1.12 ns |
| 2 KiB | 784.19 ± 0.90 ns | 1719.95 ± 2.09 ns | 1229.34 ± 1.38 ns | 786.51 ± 0.61 ns | 1086.81 ± 0.66 ns |
| 4 KiB | 1.07 ± 0.00 us | 3.47 ± 0.00 us | 1.33 ± 0.01 us | 0.99 ± 0.00 us | 1.91 ± 0.00 us |
| 6 KiB | 1.87 ± 0.01 us | 5.19 ± 0.00 us | 1.45 ± 0.00 us | 1.33 ± 0.00 us | 2.73 ± 0.01 us |
| 8 KiB | 1.18 ± 0.00 us | 1.43 ± 0.00 us | 1.78 ± 0.01 us | 1.52 ± 0.01 us | 3.60 ± 0.00 us |
| 16 KiB | 1.99 ± 0.01 us | 2.74 ± 0.01 us | 3.00 ± 0.01 us | 2.57 ± 0.01 us | 6.82 ± 0.00 us |
| 64 KiB | 7.51 ± 0.01 us | 10.20 ± 0.02 us | 10.51 ± 0.06 us | 4.28 ± 0.10 us | 26.60 ± 0.03 us |
| 128 KiB | 14.94 ± 0.01 us | 18.53 ± 0.02 us | 22.21 ± 0.10 us | 6.38 ± 0.46 us | 52.39 ± 0.05 us |
| 1 MiB | 119.69 ± 0.30 us | 174.61 ± 0.34 us | 167.01 ± 0.55 us | 26.42 ± 0.29 us | 418.72 ± 0.57 us |
| 10 MiB | 1.22 ± 0.00 ms | 1.67 ± 0.01 ms | 1.69 ± 0.01 ms | 0.22 ± 0.00 ms | 4.19 ± 0.00 ms |

![One-shot hash throughput with the parallel range shaded](img/benchmark.svg)

Of the single-threaded sizes in this run, this library is fastest at 4 B, 128 B, 1 KiB,
4 KiB and 6 KiB, and native is fastest at 2 KiB, 8 KiB and 16 KiB.

### Extended output (XOF)

Each operation absorbs the **same 1 KiB block twice**, emits the indicated
output length, then resets a reused hasher. All three implementations use one
thread. The chart measures the complete operation: absorption, output and reset.

| Output | Blake3.Native 3.0.2 | CryptoHives 0.6.101 | Blake3.Managed (this library) |
|---:|---:|---:|---:|
| 128 B | 1.55 ± 0.00 us | 1.88 ± 0.00 us | 1.81 ± 0.01 us |
| 1 KiB | 1.62 ± 0.00 us | 2.24 ± 0.00 us | 1.88 ± 0.01 us |
| 8 KiB | 2.39 ± 0.00 us | 3.43 ± 0.00 us | 2.94 ± 0.00 us |
| 128 KiB | 15.51 ± 0.01 us | 24.21 ± 0.03 us | 21.59 ± 0.02 us |

![BLAKE3 absorb, output and reset latency](img/benchmark-xof.svg)

Native is fastest at every output length here; this library is ahead of CryptoHives
throughout. The XOF numbers come from a separate run on the same machine.

### Reproducing the results

```bash
# One-shot table: 5 implementations, 12 input sizes
dotnet run --project src/Blake3.Managed.Benchmarks -c Release -- --competitive --report --filter '*Native*' '*Xoofx*' '*CryptoHives*' '*OursSpan*' '*Sha256*'

# XOF table: 3 implementations, 4 output sizes
dotnet run --project src/Blake3.Managed.Benchmarks -c Release -- --xof --report --filter '*AfterXof*' '*NativeXof*' '*CryptoHivesXof*'

# Regenerate the charts from a BenchmarkDotNet CSV export (Python, no dependencies)
python src/Blake3.Managed.Benchmarks/make_chart.py <oneshot-report.csv> img/benchmark.svg --context "<CPU> · <.NET version> · <date>"
python src/Blake3.Managed.Benchmarks/make_chart.py <xof-report.csv> img/benchmark-xof.svg --xof --context "<CPU> · <.NET version> · <date>"
```

The generator also accepts a BenchmarkDotNet console log. Use `--table table.md`
and `--data results.csv` to produce matching Markdown and chart data from a new
run.

### Hardware Intrinsics Tiering

The implementation automatically selects the best available instruction set at runtime:

| Tier | Instructions | Parallelism |
|------|-------------|-------------|
| **AVX2** | 256-bit vectors; AVX-512 VL rotates when available | 8-chunk batches, partial batches for 5–7 chunks, a 3–4 chunk kernel where AVX-512 VL is present, 2-chunk remainders, 8-way parent hashing and 8-block XOF output |
| **SSE/SSSE3** | 128-bit vectors + shuffle | 4 chunks simultaneously, single-lane SIMD fallback |
| **ARM NEON** | 128-bit vectors | 4 chunks simultaneously; single blocks use the scalar path |
| **Scalar** | Pure C# | Portable fallback |

## Building from Source

```bash
# Requires .NET 10 SDK
dotnet build Blake3.Managed.sln -c Release

# Run tests
dotnet test src/Blake3.Managed.Tests -c Release

# Run benchmarks
dotnet run --project src/Blake3.Managed.Benchmarks -c Release

# Create NuGet package
dotnet pack src/Blake3.Managed -c Release
```

## Acknowledgments

- [Blake3.NET](https://github.com/xoofx/Blake3.NET) by Alexandre Mutel, for API design and test infrastructure
- [Blake2Fast](https://github.com/saucecontrol/Blake2Fast) by Clinton Ingram, for the shuffle-based SSE message permutation
- [BLAKE3 Reference Implementation](https://github.com/BLAKE3-team/BLAKE3), for the algorithm specification and test vectors
