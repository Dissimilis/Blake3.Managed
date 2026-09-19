# CLAUDE.md

Guidance for AI coding assistants and contributors working in this repository. Machine-specific notes (benchmark hosts, credentials, remote workspaces) belong in the git-ignored `CLAUDE.local.md`, never here.

## Project Overview

`Blake3.Managed` is a pure managed C# implementation of the BLAKE3 cryptographic hash function using hardware intrinsics. The project structure:

- `src/Blake3.Managed/` — The library
- `src/Blake3.Managed.Tests/` — Test suite
- `src/Blake3.Managed.Benchmarks/` — Performance benchmarks and the correctness gate (compares against `Blake3.Native` 3.x for Rust, `Blake3` 3.x for managed, and SHA256)
- `src/Blake3.Managed.Baseline/` — Frozen snapshot of the library under a different assembly name, used as the A/B control

## Build Commands

```bash
# Build (requires .NET 10 SDK)
dotnet build Blake3.Managed.sln -c Release

# Run tests
dotnet test src/Blake3.Managed.Tests -c Release

# Run a single test by name
dotnet test src/Blake3.Managed.Tests -c Release --filter "FullyQualifiedName~TestHashSimple"

# Run benchmarks
dotnet run --project src/Blake3.Managed.Benchmarks -c Release

# Create NuGet package
dotnet pack src/Blake3.Managed -c Release
```

## Architecture

### Public API (7 types)

Namespace: `Blake3.Managed`. The library targets `net6.0`, `net8.0` and `net10.0`; tests and benchmarks target `net10.0`. All projects enable `AllowUnsafeBlocks`.

- **`Hasher`** (struct, disposable) — Factory methods: `New()`, `NewKeyed()`, `NewDeriveKey()`. Static `Hash()` for one-shot. Incremental via `Update()`/`UpdateWithJoin()`/`Finalize()`. State stored inline in `Blake3Core.HasherState`.
- **`Hash`** (struct, 32 bytes) — Fixed-size output with constant-time equality (`CryptographicOperations.FixedTimeEquals`). Allocation-free `ToString()`.
- **`Blake3Stream`** — Stream wrapper that hashes data as it flows through.
- **`Blake3HashAlgorithm`** — `System.Security.Cryptography.HashAlgorithm` adapter.
- **`Blake3SubtreeContext`** (sealed class, disposable) — Hashes fixed-size pieces of one input independently (`HashSubtree(bytes, pieceIndex)`) and folds them with `Finalize` into the whole-input digest or XOF output. Piece size must be a power-of-two multiple of the chunk size so each piece is a canonical subtree, except when the known total length fits in one piece, where any positive size is accepted. Built on `Blake3Tree.SubtreeOutput` and `Blake3Tree.RootFromCvs`.
- **`Blake3SubtreeHasher`** (sealed class, disposable) — Incremental hasher for one piece, created by `Blake3SubtreeContext.CreateSubtreeHasher(pieceIndex)`: `Update` any number of times, then `Finish` yields the same `Blake3Subtree` that `HashSubtree` would. Wraps a `Blake3Core.HasherState` constructed with a start chunk; the state keeps chunk counters absolute but counts merges relative to the piece start. Only route for pieces above 2 GB (piece size is a `long`).
- **`Blake3Subtree`** (readonly struct) — Opaque per-piece result carrying the piece's top `Output` node (not just a CV, so a single piece can be the root and XOF works), plus a context tag, index and length that `Finalize` validates.

### Internal Components (`Internal/`)

- **`Blake3Constants.cs`** — Block size (64B), chunk size (1024B), IV words, domain separation flags, 7-round message permutation schedule.
- **`Blake3Core.cs`** — Core hashing: `HasherState` struct, chunk processing, compression dispatch. Contains nested `ChunkState`, `Output`, `HasherState` structs.
- **`CompressScalar.cs`** — Pure C# fallback compression (G mixing function, 7 rounds).
- **`CompressSse41.cs`** — SSE/SSSE3 vectorized compression using shuffle-based message permutation.
- **`HashManySse41.cs`** — SSE 4-way parallel multi-chunk hashing (covers the 4–7 chunk gap below the AVX2 path).
- **`HashManyAvx2.cs`** — AVX2 8-way parallel multi-chunk hashing, plus `HashParents8` for 8-way parent-node compression.
- Its separate `HashManyPartial` kernel handles 5–7 chunks without padding. Keep variable offsets out of the full-batch loop: combining them regressed 8 KB inputs.
- `HashManySerial` interleaves full eight-chunk batches only in the serial one-shot tree. Preserve the original `HashMany` in parallel workers and incremental full batches. A 2026-09-09 follow-up run measured 7.6% less time at 64 KB in the adaptive job and 21.7% less at 8 KB with AVX-512 disabled; large parallel results were inconclusive.
- **`HashTwoAvx2.cs`** — Two complete chunks in the two 128-bit halves of AVX2 registers, for remainders below the 4-way kernel.
- **`HashFourAvx2.cs`** — Three or four complete chunks as two interleaved copies of the `HashTwoAvx2` schedule (generated statement by statement from it). One chain is latency-bound, so two chains cost about the same time. Requires AVX-512 VL for the 32-register file; dispatched ahead of the 128-bit 4-way kernel in the tree and in `Update`. Measured 18-28% less time at 4 KB on a Zen 4 desktop (2026-09-14).
- `HashTwo` and `HashFour` are `NoInlining`: Tier1 with PGO otherwise inlined the whole two-chunk kernel into `Blake3Tree.HashAllAtOnce`, which ran 4-5x slower at 2 KB. Any large kernel without a `stackalloc` can be inlined this way; keep them marked.
- **`OutputManyAvx2.cs`** — Eight 64-byte XOF output blocks per batch; arbitrary seek prefixes and output tails remain in `Output.RootOutputBytesAt`.
- **`CompressNeon.cs` / `HashManyNeon.cs`** — ARM NEON single-block and 4-way multi-chunk hashing. Only `HashManyNeon` is reachable: the ARM64 single-block path deliberately uses `CompressScalar`, because dispatching to `CompressNeon` measured 2.6x slower up to 4 KB on a Cortex-A73 (see the dead ends, and the remark on the class).
- `HasherState.Update` hashes aligned power-of-two subtrees of 8-64 chunks in `HashAlignedSubtree` (own non-inlined method: inlining it changed the shared loop's register allocation and cost 4% at 8 KB) and reduces them with 8-way parents, including 3-7 parent tails in `ReduceCvs`. Per-chunk CV-stack merging costs seven scalar parent compressions per eight chunks; this measured 10-15% less time for `Update` at 64 KB-10 MB (2026-09-14). The lone 5-8 chunk batch keeps the original `HashMany`; `HashManySerial` there was 6.6% slower at 8 KB.
- **`Blake3Tree.cs`** — All-at-once tree for inputs of known length: wide CV frontier, batched parent hashing, and the balanced thread-pool fan-out used by `Hasher.Hash`. `FlatJob` is the shared fan-out primitive: workers queued up front with `ThreadPool.UnsafeQueueUserWorkItem`, claiming units through one atomic counter. `HasherState.UpdateWithJoin` uses it too (via `Blake3Core.JoinJob`); it was still on `Parallel.For` until 2026-09-19, which measured 4-8.5% slower at 73 KiB-10 MB on the path `Blake3HashAlgorithm` and `Blake3Stream` actually take.
- **`VectorCompat.cs`** — Cross-TFM compatibility layer for vector load/store operations.

### Hardware Intrinsics Tiering

Runtime CPU detection dispatches: AVX2 (8-way parallel) > SSE/SSSE3 4-way > SSE/SSSE3 (vectorized single-lane) > Scalar (portable fallback). Opportunistic AVX-512 VL rotate instructions used when available.

On AVX2 machines, pairs of full chunks left after the wider kernels use `HashTwoAvx2`.
The 32-byte span-output overload now uses the same small-input compressors as the value-returning
overload. CryptoHives' comparison calls the span overload, so keep both in performance coverage.

`Hasher.Hash` dispatches by input length: one block, two blocks, one chunk, `Blake3Tree` serial up to 32 KiB, then the load-gated band to 256 KiB (`Blake3Tree.HashMidSize`), then `Blake3Tree` parallel unconditionally. The band was raised from 72 KiB to 256 KiB on 2026-09-19: the serial tree's aggregate advantage on a saturated machine is a property of the machine, not of the band, so the same gate pays above 72 KiB. Measured on the 16-thread Zen 4 host, 16 concurrent callers gained **25.6% at 128 KiB and 21.1% at 256 KiB** aggregate, single-caller latency was unchanged, and eight callers lost 1.3% at 128 KiB and 5.6% at 256 KiB -- the mixed serial/parallel regime that the gate necessarily passes through at intermediate load. In the band, a hash fans out only while fewer than ProcessorCount/4 other hashes of the band are in flight, counting both paths: a single caller gains 2-2.6x from four 16-chunk units, but sixteen concurrent callers lost 22% aggregate throughput to the same fan-out on a saturated 16-thread Zen 4 machine (2026-09-16). Counting only parallel hashes produced a serial/parallel mix that was worse than either extreme (10% lost at eight callers with one slot, 8-11% at sixteen with four); counting every caller switches the band together and matched the old cutoff within noise at eight and sixteen callers. Each step exists because it measured faster than the more general path below it. The incremental API (`Update`/`Finalize`) cannot use the tree, because it must assume more input may arrive and so cannot keep a wide frontier or stop at two CVs for the root.

The serial tree skips frontier reduction for two-chunk inputs and compresses 32-byte roots directly
into the destination on little-endian machines. Preserve the general Output path for XOF output.

### ARM64

First benchmarked 2026-09-19 on a Cortex-A73/A53 big.LITTLE board. Pin to the big cores and
record which ones: an unpinned run mixes core types and is not reproducible, and the fast cores
are not always the low-numbered ones (`lscpu -e`).

Nothing has made the ARM tier faster yet. Both attempts -- SRI rotates, and wiring in the NEON
single-block compressor -- measured worse; see the dead ends. Those numbers are A73-specific and
may invert on a wider core.

## Test Framework

xUnit. Tests use `TestVectors.json` (official BLAKE3 test vectors, copied to output) for known-answer validation. Internals exposed to test project via `InternalsVisibleTo`.

## Performance work

Everything below exists because a plausible-looking change was wrong at least once.

### The loop

```bash
# 1. Freeze the current build as the A/B control (only when starting a new campaign)
rm -rf src/Blake3.Managed.Baseline/src && mkdir -p src/Blake3.Managed.Baseline/src
cp -r src/Blake3.Managed/*.cs src/Blake3.Managed.Baseline/src/
cp -r src/Blake3.Managed/Internal src/Blake3.Managed.Baseline/src/

# 2. Make one change, then decide whether it helped
dotnet run --project src/Blake3.Managed.Benchmarks -c Release        # A/B vs baseline, adaptive
dotnet run --project src/Blake3.Managed.Benchmarks -c Release -- --competitive   # vs Rust/Blake3 3.x/SHA256
dotnet run --project src/Blake3.Managed.Benchmarks -c Release -- --kernel        # isolated kernels
dotnet run --project src/Blake3.Managed.Benchmarks -c Release -- --dispatch      # candidate shapes, one run
dotnet run --project src/Blake3.Managed.Benchmarks -c Release -- --xof           # two 1 KB absorbs, squeeze, reset
dotnet run --project src/Blake3.Managed.Benchmarks -c Release -- --concurrent    # aggregate MB/s, 1/8/16 callers
```

Flags: no flag runs the A/B decision harness; `--report` is a faster job for publishing tables across
many sizes; `--quick` is a smoke test whose numbers must never be quoted. `--concurrent` measures
aggregate MB/s with 1, 8 and 16 caller threads, baseline and current alternating in one process;
run it for any change to the parallel dispatch, because BenchmarkDotNet only sees one caller on an
idle machine. Add `--filter "*(Data_Size: 128)*"` to narrow any of them.

If a run is wrapped in a lock, launch it with `MSBUILDDISABLENODEREUSE=1
DOTNET_CLI_USE_MSBUILD_SERVER=0`. Otherwise the run's MSBuild worker nodes inherit the lock file
descriptor and keep holding it after the run exits, so the next attempt fails instantly with an
empty log and a non-zero exit. `fuser -v <lockfile>` names the holders; `dotnet build-server
shutdown` does not release them.

### Rules that were learned the hard way

- **A correctness gate runs before every benchmark session** (~7,000 checks: official vectors, keyed,
  derive-key, XOF with seek, incremental splits, unaligned spans, boundary lengths). It aborts the run on
  mismatch. A faster wrong answer is not an improvement.
- **Read the A/B verdict, not the table.** It prints IMPROVED / REGRESSED / NO RESULT per size and refuses
  to call anything whose confidence intervals overlap.
- **Never compare absolute nanoseconds across sessions.** A laptop throttles roughly 2x under sustained
  load, and even a desktop drifts 3-12% between runs. Only ratios measured inside one run mean anything.
- **An isolated kernel benchmark can be actively misleading.** Phase-interleaving the AVX2 round made the
  8-chunk kernel measurably faster on one thread and was a 2.8x regression end-to-end at 10 MB: more live
  vectors means spills, which are cheap on one thread and ruinous when sixteen threads contend for
  bandwidth. Always confirm a kernel change end-to-end at a large size.
- **A single-caller latency win can be an aggregate-throughput loss.** Lowering the parallel cutoff
  to 32 chunks halved 64 KB `Hash()` for one caller and cost 22% with sixteen concurrent callers,
  because the parallel path sustains ~44 GB/s aggregate against ~56 GB/s for the serial tree once
  every core is busy. Check `--concurrent` before moving any parallel threshold.
- **Check whether the code you are measuring actually runs.** Two ARM changes measured at exactly
  1.000 before a grep for callers showed the class they touched had none. When a verdict is a flat
  1.000 across every size, suspect the harness or the dispatch before concluding "no effect".
- **A verdict at a size the change cannot reach measures noise, so use it to calibrate the rest.**
  The `UpdateWithJoin` port only executes at four of the twenty sizes. The +7.7% and -5.8% it
  "scored" at sizes that fall back to `Update` put that session's noise floor near 6-8%, which is
  the right lens for the +4% readings in the same table.
- **Compare candidate shapes in a single run** (`--dispatch`). Cross-run comparison on a throttling laptop
  has reversed the answer more than once.
- **Regenerate the README chart from the same run as the table**, so they cannot drift:
  `python src/Blake3.Managed.Benchmarks/make_chart.py <bdn-output> img/benchmark.svg`

### Known dead ends

- **Replacing the `stackalloc Vector256<uint>[16]` message array in `HashManyAvx2` with 16 named locals.**
  RyuJIT already folds `m[i]` into the ALU memory operand: a disassembly dump shows 121 folded ops
  (`vpaddd ymm0, ymm0, ymmword ptr [rbp-0x130]`), zero separate stack loads and zero spills, with
  ymm16-31 live. Sixteen more live vectors would exhaust the register file and spill. While you are in
  that dump: all four rotates already emit `vprord` via the AVX-512 VL path, so there is no `vpshufb`
  left to replace. The mnemonics are EVEX forms, so grep for `vprord`/`vpxord`/`vmovups`.
- **Reading `s_maxDegreeOfParallelism` below the serial branches in `Hasher.Hash`** instead of once at
  the top. It saves an acquire load on short inputs and measured **6-7.5% worse at 64 KB with eight
  concurrent callers, reproducibly across three runs**: that load is a mixed regime where some callers
  fan out and others do not, and it is sensitive to the shape of this dispatch. Any edit to the
  one-shot dispatch ladder needs `--concurrent`, not just the adaptive job.
- **Wiring `CompressNeon` into the single-block dispatch on ARM64.** It looks like an obvious gap --
  `Blake3Core.CompressInPlace`/`CompressCv` dispatch SSE4.1 then straight to scalar, so ARM runs the
  portable compressor and the NEON one has no callers at all. Dispatching to it measured **2.6x slower
  up to 4 KB** and 10-29% slower to 10 MB on a Cortex-A73. BLAKE3's sixteen state words fit in ARM64's
  thirty-one integer registers with no shuffles, while the NEON kernel re-diagonalises with EXT every
  round on a two-pipe 64-bit NEON unit. Vectorising pays in `HashManyNeon`, which keeps four chunks in
  four lanes and never diagonalises. Re-measure before trying this on a wider ARM core.
- **`AdvSimd.ShiftRightAndInsert` (SRI) for the NEON Rot12/Rot7 rotates.** One instruction fewer than
  shift/shift/or, and **3-5% slower at every size from 4 KB up** on Cortex-A73: SRI's destination is
  also a source, so it serialises behind the shift feeding it, while the two shifts in the original
  form are independent and dual-issue on the A73's two NEON pipes.
- Hoisting the first block out of the fused chunk loop to make the middle-block state row constant: tried
  twice, measured worse both times. The two-block win came from removing the loop entirely for a known
  block count.
- A 16-way AVX-512 `HashMany`: measured ~2.5x slower than AVX2 8-way on Zen 4 (register spilling plus
  double-pumped 512-bit execution). Register pressure is not the AVX2 kernel's problem either; the
  disassembly shows ymm16-31 in use with no spills.
- Recursive parallel splitting with nested `Parallel.Invoke`: balances perfectly but blocks a pool thread
  at every internal node, and measured 2.4x slower than a flat `Parallel.For` fan-out.
- `Parallel.For` itself for the flat fan-out: 18-27% slower than queueing every worker up front with
  `ThreadPool.UnsafeQueueUserWorkItem` and claiming units through one atomic counter (`Blake3Tree.FlatJob`),
  measured in one run at 128 KB, 256 KB, 1 MB and 10 MB. Its replicating tasks wake the pool one hop at a
  time, which is most of a 40 us job. Capping workers below the logical core count (8 or 12 of 16) was
  slower at every size; 4-chunk units were slower than 8 or 16.

## Key Conventions

- Library targets `net6.0;net8.0;net10.0`; tests and benchmarks target `net10.0`.
- The unchanged control build moves up to 12% between short `--report` runs and about 3% between adaptive runs even on a desktop; decide with the default adaptive job and read the verdict. BDN `--filter` names are `BeforeOneShot/AfterOneShot`, `BeforeSpan/AfterSpan`, `BeforeSerial/AfterSerial` (`Update`), so filter with `"*Serial*Data_Size: 8192)*"`, not `*Update*`.
- Centralized package versions in `src/Directory.Packages.props`.
- Versioning via MinVer (derived from git tags).
- SourceLink enabled for debuggable NuGet packages.
- Key material zeroed on `Dispose()`.
