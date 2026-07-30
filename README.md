# Blake3.Managed — BLAKE3 hash function for .NET

A pure managed C# implementation of the [BLAKE3](https://github.com/BLAKE3-team/BLAKE3) cryptographic hash function for .NET. Uses hardware intrinsics (AVX2, SSE/SSSE3, ARM NEON) for high performance with automatic scalar fallback — faster than SHA-256 and MD5, with zero native dependencies and no P/Invoke.

BLAKE3 is a modern, secure hash ideal for file checksums, content addressing, deduplication, data integrity, message authentication (keyed MAC) and key derivation (KDF). This library brings it to C# / .NET as a single fully managed, Native AOT-friendly NuGet package that runs everywhere .NET runs.

[![NuGet](https://img.shields.io/nuget/v/Blake3.Managed.svg)](https://www.nuget.org/packages/Blake3.Managed)
[![NuGet Downloads](https://img.shields.io/nuget/dt/Blake3.Managed.svg)](https://www.nuget.org/packages/Blake3.Managed)
[![CI](https://github.com/Dissimilis/Blake3.Managed/actions/workflows/ci.yml/badge.svg)](https://github.com/Dissimilis/Blake3.Managed/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-donate-yellow.svg)](https://buymeacoffee.com/dissimilis)

## Features

- **Pure managed C#** - no native libraries, no P/Invoke, runs everywhere .NET runs
- **Hardware accelerated** - AVX2 8-way parallel hashing, SSE/SSSE3 vectorized compression, ARM NEON 4-way parallel hashing, automatic scalar fallback
- **Multi-threaded** - one-shot `Hash()`, `Blake3HashAlgorithm` and `Blake3Stream` hash 64-chunk subtrees on the thread pool for inputs above ~72 KiB
- **Zero allocation** for small inputs - one-shot `Hasher.Hash()` uses stack allocation
- **All BLAKE3 modes** - default hashing, keyed hashing, and key derivation
- **XOF support** - extendable output with seekable byte stream
- **API compatible** - similar API to the Blake3 Rust API and mirrors [Blake3.NET](https://github.com/xoofx/Blake3.NET) API surface
- **Targets** `net6.0`, `net8.0` and `net10.0` (with hardware intrinsics)

## How this compares to the other BLAKE3 packages

[Blake3.NET](https://github.com/xoofx/Blake3.NET) by Alexandre Mutel is a very polished BLAKE3 library for .NET, and some ideas and code examples here are borrowed from it. As of 3.x it ships as two packages: `Blake3` is now a pure managed SIMD implementation, and `Blake3.Native` is the P/Invoke binding to the official Rust code. Both are excellent.

So "managed versus native" is no longer the interesting distinction - there are several good managed options. What this library offers instead:

- **A multithreaded one-shot path.** `Hasher.Hash` splits large inputs into 64-chunk subtrees and hashes them on the thread pool, so above roughly 128 KB it finishes sooner than any single-threaded implementation, including the Rust binding. That is wall-clock latency bought with more cores, not better per-core efficiency - worth knowing if you are already saturating every core with other work.
- **`net6.0` support**, which `Blake3` 3.x dropped.
- **No native dependency**, so deployment stays simple and Native AOT and trimming work without extra care.

Below 128 KB, single-threaded, this library is within roughly 10-60% of the Rust implementation depending on size. See the numbers below.

## Installation

```
dotnet add package Blake3.Managed
```

## Quick Start

```csharp
using Blake3.Managed;

// One-shot hash
var hash = Hasher.Hash("Hello, World!"u8);
Console.WriteLine(hash); // 288a86a79f20a3d6dccdca1c47c4c4726cddf1ae8c3ae5bdddf8f57a76a3a02e

// Incremental hashing
using var hasher = Hasher.New();
hasher.Update("Hello, "u8);
hasher.Update("World!"u8);
var result = hasher.Finalize();

// Keyed hash (32-byte key)
byte[] key = new byte[32]; // your key here
using var keyedHasher = Hasher.NewKeyed(key);
keyedHasher.Update(data);
var mac = keyedHasher.Finalize();

// Key derivation
using var kdf = Hasher.NewDeriveKey("my-session-key");
kdf.Update(inputKeyMaterial);
var derivedKey = kdf.Finalize();

// Extended output (XOF)
using var xof = Hasher.New();
xof.Update(data);
var extendedOutput = new byte[1024];
xof.Finalize(extendedOutput); // arbitrary length output

// Parallel hashing for large data
using var parallel = Hasher.New();
parallel.UpdateWithJoin(largeData); // uses thread pool above ~72 KiB
var parallelHash = parallel.Finalize();
```

## Public API

| Type | Description |
|------|-------------|
| `Hasher` | Main hasher struct. Factory methods: `New()`, `NewKeyed()`, `NewDeriveKey()`. Static `Hash()` for one-shot. Incremental via `Update()`/`UpdateWithJoin()`/`Finalize()`. |
| `Hash` | Fixed 32-byte output struct with constant-time equality and allocation-free `ToString()`. |
| `Blake3Stream` | Stream wrapper that hashes data as it flows through. |
| `Blake3HashAlgorithm` | `System.Security.Cryptography.HashAlgorithm` adapter for interop with existing APIs. |

## Performance

### Benchmarks

One-shot hashing (`Hasher.Hash(data)`) against the Rust implementation via P/Invoke
([Blake3.Native](https://www.nuget.org/packages/Blake3.Native)), the managed
[Blake3](https://www.nuget.org/packages/Blake3) 3.x package, and `System.Security.Cryptography.SHA256`.

```
BenchmarkDotNet v0.15.8, Windows 11 (10.0.26200)
AMD Ryzen 7 PRO 7840U w/ Radeon 780M Graphics, 1 CPU, 16 logical and 8 physical cores
.NET SDK 10.0.302, .NET 10.0.10, X64 RyuJIT AVX-512
```

Mean time per hash, and the ratio against the Rust implementation (lower is better):

| Input | Blake3.Native (Rust) | Blake3 3.x (managed) | **Blake3.Managed** | SHA256 |
|------:|---------------------:|---------------------:|-------------------:|-------:|
| 4 B | 74 ns | 74 ns _(1.00)_ | **91 ns _(1.22)_** | 151 ns _(2.02)_ |
| 128 B | 125 ns | 107 ns _(0.85)_ | **141 ns _(1.13)_** | 184 ns _(1.47)_ |
| 1 KB | 832 ns | 894 ns _(1.07)_ | **914 ns _(1.10)_** | 577 ns _(0.69)_ |
| 4 KB | 1.20 us | 3.83 us _(3.19)_ | **1.48 us _(1.23)_** | 1.92 us _(1.60)_ |
| 8 KB | 1.42 us | 1.66 us _(1.17)_ | **1.77 us _(1.25)_** | 3.72 us _(2.61)_ |
| 16 KB | 2.22 us | 2.95 us _(1.33)_ | **3.21 us _(1.45)_** | 7.25 us _(3.27)_ |
| 64 KB | 8.38 us | 11.49 us _(1.37)_ | **13.49 us _(1.61)_** | 28.60 us _(3.41)_ |
| 128 KB | 16.59 us | 20.92 us _(1.26)_ | **24.50 us _(1.48)_** | 57.01 us _(3.44)_ |
| 1 MB | 140.1 us | 181.8 us _(1.30)_ | **74.4 us _(0.53)_** | 460.9 us _(3.29)_ |
| 10 MB | 2.19 ms | 3.38 ms _(1.54)_ | **0.60 ms _(0.27)_** | 4.59 ms _(2.10)_ |

![BLAKE3 throughput on .NET](img/benchmark.svg)

- **Under 1 KB**, all three BLAKE3 implementations are within about 20% of each other. SHA-256 wins at
  exactly 1 KB on this CPU because it has dedicated hardware instructions; BLAKE3 pulls ahead from 4 KB up.
- **4 KB to 128 KB** is single-threaded work, where this library runs 1.2-1.6x the Rust implementation.
- **Above 128 KB**, one-shot `Hash()` is *faster than Rust* - 1.9x at 1 MB and 3.7x at 10 MB - because it
  hashes 64-chunk subtrees on the thread pool while the others are single-threaded. This is wall-clock
  latency from using 8 cores, not per-core efficiency.
- **Incremental `Update()` is slower than one-shot `Hash()`**, by roughly 1.2x at small sizes and more at
  large ones, because a streaming hasher cannot know the input length and so cannot batch parent hashing
  or parallelise. When you have all the bytes, prefer `Hasher.Hash(data)`; when streaming a large buffer,
  prefer `UpdateWithJoin`.
- `Blake3HashAlgorithm` and `Blake3Stream` route bulk writes through the parallel path too, so they land
  within a couple of percent of `Hasher.Hash` on large inputs.
- **Caveat on precision.** These come from a laptop that thermally throttles roughly 2x under sustained
  load. Absolute numbers are not comparable between sessions, and repeat runs moved the 16-64 KB ratios by
  up to 20%. Treat them as indicative. The benchmark project in this repo reproduces them, and gates every
  run on ~7,000 correctness checks first; `--report` produces this table and `img/benchmark.svg` is
  generated from it by `src/Blake3.Managed.Benchmarks/make_chart.py`.

### Hardware Intrinsics Tiering

The implementation automatically selects the best available instruction set at runtime:

| Tier | Instructions | Parallelism |
|------|-------------|-------------|
| **AVX2** | 256-bit vectors | 8 chunks simultaneously, plus 8-way parent-node hashing for the merge tree |
| **SSE/SSSE3** | 128-bit vectors + shuffle | 4 chunks simultaneously, single-lane SIMD fallback |
| **ARM NEON** | 128-bit vectors | 4 chunks simultaneously |
| **Scalar** | Pure C# | Portable fallback |

## Building from Source

```bash
# Requires .NET 10 SDK
dotnet build src -c Release

# Run tests
dotnet test src/Blake3.Managed.Tests -c Release

# Run benchmarks
dotnet run --project src/Blake3.Managed.Benchmarks -c Release

# Create NuGet package
dotnet pack src/Blake3.Managed -c Release
```

## Acknowledgments

- [Blake3.NET](https://github.com/xoofx/Blake3.NET) by Alexandre Mutel — API design and test infrastructure
- [Blake2Fast](https://github.com/saucecontrol/Blake2Fast) by Clinton Ingram — shuffle-based SSE message permutation
- [BLAKE3 Reference Implementation](https://github.com/BLAKE3-team/BLAKE3) — algorithm specification and test vectors
