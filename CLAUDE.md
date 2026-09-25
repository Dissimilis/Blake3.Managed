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

- **`Hasher`** (struct, disposable) — Factory methods: `New()`, `NewKeyed()`, `NewDeriveKey()`, each also with an `out` overload that constructs in the caller's storage (no zeroing or copy of the ~1.9 KB struct; added 2026-09-25). Static `Hash()` for one-shot. Incremental via `Update()`/`UpdateWithJoin()`/`Finalize()`. State stored inline in `Blake3Core.HasherState`.
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
- `HashManySerial` steps the four G's of each half-round (`G256x4`) within one eight-chunk batch -- the same lever that won on NEON; it does not interleave two batches. It runs in the serial one-shot tree, in `Update`'s aligned subtrees (`HashAlignedSubtree`) and in `UpdateWithJoin`'s workers (`JoinJob`). The one-shot parallel workers (`SubtreeCvSerial` -> `CompressSubtreeWide`) keep the original `HashMany`, or `HashMany16` for an exact 16-chunk subtree, and so does `Update`'s lone 5-8 chunk batch. A 2026-09-09 follow-up run measured 7.6% less time at 64 KB in the adaptive job and 21.7% less at 8 KB with AVX-512 disabled; large parallel results were inconclusive.
- **`HashTwoAvx2.cs`** — Two complete chunks in the two 128-bit halves of AVX2 registers, for remainders below the 4-way kernel.
- **`HashFourAvx2.cs`** — Three or four complete chunks as two interleaved copies of the `HashTwoAvx2` schedule (generated statement by statement from it). One chain is latency-bound, so two chains cost about the same time. Requires AVX-512 VL for the 32-register file; dispatched ahead of the 128-bit 4-way kernel in the tree and in `Update`. Measured 18-28% less time at 4 KB on a Zen 4 desktop (2026-09-14).
- `HashTwo` and `HashFour` are `NoInlining`: Tier1 with PGO otherwise inlined the whole two-chunk kernel into `Blake3Tree.HashAllAtOnce`, which ran 4-5x slower at 2 KB. Any large kernel without a `stackalloc` can be inlined this way; keep them marked.
- **`OutputManyAvx2.cs`** — Eight 64-byte XOF output blocks per batch; arbitrary seek prefixes and output tails remain in `Output.RootOutputBytesAt`.
- **`CompressNeon.cs` / `HashManyNeon.cs`** — ARM NEON single-block and 4-way multi-chunk hashing. Only `HashManyNeon.HashMany` is reachable, and only ever with `numChunks` of 4. The ARM64 single-block path deliberately uses `CompressScalar`, because dispatching to `CompressNeon` measured 2.6x slower up to 4 KB on a Cortex-A73 (see the dead ends, and the remark on the class). `HashManyNeon.HashManyPartial` covers a 3-chunk remainder (2 chunks is a dead end, see below) and `HashManyNeon.HashParents4` batches parent compression 4-way. Since 2026-09-25 the rounds of those three kernels are written out statement by statement, stepped across the four G's of each half-round, with `(a + m) + b` and a per-round barrier store -- 0.55-0.56 of the old time on Cortex-A73 and 0.68 on Neoverse-V1, and no spills (see "NEON round, 2026-09-25"). `HashManyNeon.HashMany8` has **no callers at all**, deliberately: it is an inlining-budget failure (33 real calls), and even written out an 8-chunk two-chain kernel does not beat two 4-way calls on either core.
- **`HashManySve2.cs`** (net10.0 only) — the `HashManyNeon` kernels with every xor-then-rotate fused into one SVE2 `XAR`, plus `HashMany8`, two 4-chunk batches interleaved per half-round and written out statement by statement. Reached only when `Sve2.IsSupported`: `HashManyNeon`'s entry points forward to it, `Blake3Tree` uses an SVE2 `SimdDegree` of 8, an 8-way loop and 2-chunk padding in `HashChunks`, and `HashAlignedSubtreeNeon` hands off to `HashAlignedSubtreeSve2`. Every one of those sites was placed so the NEON and x86 code compiles **identically** -- see "SVE2" under ARM64.
- **`HashManyAvx512.cs`** (net8.0+) — `HashMany16` (sixteen chunks per call, fully expanded, spill-free: message, CVs and counters live in a 64-byte-aligned scratch area) and `HashOutput16` (sixteen XOF blocks per call). Reached from `CompressSubtreeWide` (any exact 16-chunk subtree -> `CompressSubtree16`), `HashAlignedSubtree` (-> `HashAlignedSubtreeAvx512`), the `UpdateWithJoin` workers and `RootOutputBytesAt` (-> `RootOutputBytesAtAvx512`). Its transpose helper is `NoInlining | AggressiveOptimization` on purpose -- see the tier-0 trap in the x86 round.
- **`OutputManySve2.cs`** (net10.0 only) — eight XOF root blocks per call on SVE2, forwarded to from the top of `Output.RootOutputBytesAt` (`RootOutputBytesAtSve2`).
- **`OutputManyNeon.cs`** — four XOF root blocks per call on NEON without SVE2, with the `HashManyNeon` round body; forwarded to right after the SVE2 forward (`RootOutputBytesAtNeon`). Whole 4-block batches, one batch into scratch when three or more blocks remain, scalar below. Before it (2026-09-25) NEON produced XOF output one scalar block at a time.
- `HasherState.Update` hashes aligned power-of-two subtrees of 8-64 chunks in `HashAlignedSubtree` (own non-inlined method: inlining it changed the shared loop's register allocation and cost 4% at 8 KB) and reduces them with 8-way parents, including 3-7 parent tails in `ReduceCvs`. Per-chunk CV-stack merging costs seven scalar parent compressions per eight chunks; this measured 10-15% less time for `Update` at 64 KB-10 MB (2026-09-14). The lone 5-8 chunk batch keeps the original `HashMany`; `HashManySerial` there was 6.6% slower at 8 KB. That was measured on a full eight-chunk batch before `FinishAlignedSubtree` existed. An aligned 8 KB `Update` now goes through `HashAlignedSubtree`, so this batch only runs for 5-7 chunks or for a chunk counter that is not a multiple of 8.
- **`Blake3Tree.cs`** — All-at-once tree for inputs of known length: wide CV frontier, batched parent hashing, and the balanced thread-pool fan-out used by `Hasher.Hash`. `FlatJob` is the shared fan-out primitive: workers queued up front with `ThreadPool.UnsafeQueueUserWorkItem`, claiming units through one atomic counter. `HasherState.UpdateWithJoin` uses it too (via `Blake3Core.JoinJob`); it was still on `Parallel.For` until 2026-09-19, which measured 4-8.5% slower at 73 KiB-10 MB on the path `Blake3HashAlgorithm` and `Blake3Stream` actually take. Its workers hash their subtrees with the interleaved `HashManySerial`, which is worth 7.5% at 1 MB and 9.3% at 10 MB (2026-09-20): a worker always holds whole eight-chunk batches, the shape interleaving was written for. Batch shape does not explain why the same kernel lost 6.6% on `Update`'s lone batch, though: that was measured on a full eight-chunk batch too (8 KB `Update`, see the comment in `Blake3Core.cs`). The candidate cause is spill traffic -- `HashManySerial` writes 85 values per block to the frame against 49 for `HashMany` (see the instruction-level accounting below).
- **`VectorCompat.cs`** — Cross-TFM compatibility layer for vector load/store operations.

### Hardware Intrinsics Tiering

Runtime CPU detection dispatches: AVX2 (8-way parallel) > SSE/SSSE3 4-way > SSE/SSSE3 (vectorized single-lane) > Scalar (portable fallback). Opportunistic AVX-512 VL rotate instructions used when available. On ARM64: SVE2 (XAR, 8-way as two interleaved 4-way batches, net10.0+) > NEON 4-way > scalar single-block.

On AVX2 machines, pairs of full chunks left after the wider kernels use `HashTwoAvx2`.
The 32-byte span-output overload now uses the same small-input compressors as the value-returning
overload. CryptoHives' comparison calls the span overload, so keep both in performance coverage.

`Hasher.Hash` dispatches by input length: one block, two blocks, one chunk, `Blake3Tree` serial up to 32 KiB, then the load-gated band to 256 KiB (`Blake3Tree.HashMidSize`), then the load gate again, with a wider slot count, at every length above that (2026-09-21 -- it used to fan out unconditionally there). The band was raised from 72 KiB to 256 KiB on 2026-09-19: the serial tree's aggregate advantage on a saturated machine is a property of the machine, not of the band, so the same gate pays above 72 KiB. Measured on the 16-thread Zen 4 host, 16 concurrent callers gained **25.6% at 128 KiB and 21.1% at 256 KiB** aggregate, single-caller latency was unchanged, and eight callers lost 1.3% at 128 KiB and 5.6% at 256 KiB -- the mixed serial/parallel regime that the gate necessarily passes through at intermediate load. In the band, a hash fans out only while fewer than ProcessorCount/4 other hashes of the band are in flight (floor 2, see `s_fanOutSlots`), counting both paths: a single caller gains 2-2.6x from four 16-chunk units, but sixteen concurrent callers lost 22% aggregate throughput to the same fan-out on a saturated 16-thread Zen 4 machine (2026-09-16). Counting only parallel hashes produced a serial/parallel mix that was worse than either extreme (10% lost at eight callers with one slot, 8-11% at sixteen with four); counting every caller switches the band together and matched the old cutoff within noise at eight and sixteen callers. Each step exists because it measured faster than the more general path below it. The incremental API (`Update`/`Finalize`) cannot use the tree, because it must assume more input may arrive and so cannot keep a wide frontier or stop at two CVs for the root.

**The gate now covers every length above the serial tree, not just the 32-256 chunk band
(2026-09-21).** `LoadGatedLength` no longer ends the gate; it only selects which slot count
applies. The case for extending it was the rayon run's worst cell: at 1 MiB with sixteen callers
we sustained 43,476 MB/s against Rust's serial 66,025 (0.66), and above 256 KiB we fanned out
with no gate at all -- exactly the regime the gate exists for. The argument already written down
for raising the band to 256 KiB (the serial tree's aggregate advantage is a property of the
machine, not of the band) does not stop at 256 KiB either.

**It needs its own slot count, and that is the whole result.** Reusing the band's
`ProcessorCount / 4` measured **0.935 / 0.883 / 0.858 at eight callers** for 512 KiB, 1 MiB and
10 MiB while gaining 19.1 / 13.8 / 4.5% at sixteen -- a straight trade, not a win. The reason is
the documented mixed regime: four slots turns eight callers into four fanning out and four
serial, which is worse than either extreme, while sixteen callers become four out and twelve
serial, i.e. mostly serial, which wins. Eight callers wanted *all* parallel and sixteen wanted
*mostly serial*, so the count has to sit between them. `s_largeFanOutSlots = ProcessorCount / 2`
(floor 2) does that:

| size | 1 caller | 8 callers | 16 callers |
|---|---|---|---|
| 512 KiB | 0.995 | 0.977 | **1.161** |
| 1 MiB | 0.993 | 0.981 | **1.134** |
| 10 MiB | 0.984 | 1.002 | 1.029 |

No configuration lost more than 5%, and single-caller latency -- our strongest claim -- is
untouched. 128 KiB and 256 KiB ran in the same job as in-run controls and read 0.987-0.999.
Verified on four Cortex-A73 cores per the rule that a gate tuned on one core count can invert on
another: **neutral there, every cell 0.989-1.011**, since `ProcessorCount / 2` floors to 2 on
that machine.

**`--concurrent`'s default sizes stop at 128 KiB and could not see this change at all.** The
first run of it produced a full table of confident-looking verdicts -- REGRESSED 6.0%, IMPROVED
12.0%, IMPROVED 10.6% -- at sizes that were already gated before and after, i.e. pure layout
artifacts, and they put that run's real between-build noise near **12%** against a printed round
spread of 0.2-0.9%. Pass `--sizes=` (and `--callers=`) to reach the path under test, and keep a
couple of unaffected sizes in the same job as controls. This is the third time a harness that
could not reach the path reported a confident number.

The serial tree skips frontier reduction for two-chunk inputs and compresses 32-byte roots directly
into the destination on little-endian machines. Preserve the general Output path for XOF output.

**The `HashAlgorithm` adapter used to be much slower than `Hasher.Hash`, and it is what third
parties measure.** `Blake3HashAlgorithm` routes through `UpdateWithJoin`, which is incremental by
construction and cannot use `Blake3Tree`. Measured properly on 2026-09-21 (`--api`, Fedora host),
the adapter was **2.90x** the one-shot at 64 KiB -- worse than the 2.4x a rough probe had
suggested -- while the wrapper itself costs only 6% over raw `UpdateWithJoin`. The whole gap was
the fan-out threshold: `UpdateWithJoin` had only a 64-chunk unit and needs two items, so it ran
serial below ~72 KiB while the one-shot fanned out from ~32 KiB.

Fixed the same day by giving it a 16-chunk unit inside the mid-size band, gated on the one-shot's
own in-flight counter: the adapter went **10.954 us -> 6.075 us at 64 KiB (0.555)**, and
adapter/one-shot fell from 2.90x to **1.58x**. `--concurrent --join` showed 1.20-2.03x at one
caller across 32-128 KiB with nothing worse than 0.992 at eight or sixteen callers.

Both adapters now have before/after coverage in `--api`, and `--concurrent` takes `--join` to
measure this path -- it previously only ever ran `Hasher.Hash` and was blind to it.

### ARM64

First benchmarked 2026-09-19 on a Cortex-A73/A53 big.LITTLE board. Pin to the big cores and
record which ones: an unpinned run mixes core types and is not reproducible, and the fast cores
are not always the low-numbered ones (`lscpu -e`).

Updated 2026-09-21: two changes have now made the ARM tier faster -- a 3-chunk NEON remainder
kernel (10%) and 4-way NEON parent compression (2.4% at 16 KiB). Three attempts measured worse
and are dead ends: SRI rotates, wiring in the NEON single-block compressor, and wiring in
`HashMany8`. All of these numbers are A73-specific and may invert on a wider core.

First competitive numbers, 2026-09-20, pinned to the A73s, against the Rust crate single-threaded.
**Superseded -- current standing is in "ARM gaps round, 2026-09-25" item 5 (A73 now 1.10-1.16x
at 4 KiB-1 MiB, at or ahead of Rust below 1 KiB).** Every ARM change below came after this table,
and the scalar rewrite (SVE2 round 2, 2026-09-23) took the A73 to 0.86-1.01x Rust at every size
up to 2 KiB.

| input | ours vs Rust (2026-09-20) |
|---|---|
| 128 B - 2 KiB | 1.52x - 1.68x slower |
| 4 KiB - 16 KiB | **1.94x - 1.98x slower** |
| 64 KiB and up | 0.48x - 0.55x, but we are using four cores and Rust one |

At the time the ARM single-thread gap was about 2x, roughly double the then 1.30x on x86, and the
obvious cause was that ARM had no two-chunk or three-to-four-chunk remainder kernel, so anything
between the 4-way kernel and a single chunk fell back to per-chunk scalar compression. The rows at
64 KiB and above are not a kernel comparison and should not be read as one.

**The scalar rewrite had made NEON lose to scalar on the A73 -- settled 2026-09-25, and fixed
in the kernels rather than the dispatch.** Measured before the fix: four scalar chunks took 0.940
of the NEON 4-way kernel, three took **0.696** of the padded 3-chunk kernel (so that kernel, shipped
as a 10% win, had become a 1.44x loss), four scalar parents 0.911 of `HashParents4`. The rewritten
NEON round body then made the kernels 1.70x faster than scalar on this core. See "NEON round,
2026-09-25".

Both structural gaps previously listed here were closed or refuted on 2026-09-21. Parent
compression is no longer AVX2-only: `HashManyNeon.HashParents4` batches four parents per call
and measured 2.4% at 16 KiB, under 1% at 8 KiB. The follow-on gap this section used to list as
"the remaining piece of work" -- ARM's `Update` never reaching the batched reduce because
`HashAlignedSubtree` was AVX2-gated -- **was closed the same day** by
`HashAlignedSubtreeNeon` (`Blake3Core.cs`, dispatched alongside the AVX2 form), worth 2.8% at
64 KiB. The unreachable `HashManyNeon.HashMany8` turned out to be a dead end rather than a
missed opportunity.

**The first two changes that ever made the ARM tier faster both landed on 2026-09-21**: the
3-chunk remainder kernel (10%) and 4-way parent compression (2.4% at 16 KiB). The remaining
1-2 chunk remainders are best left scalar on this core.

### SVE2 (Graviton4, 2026-09-23)

First measured on an EC2 c8g.large: Graviton4, Neoverse-V2, 2 vCPU (two physical cores, no
SMT), 128-bit SVE vectors, .NET 10.0.12. **.NET 10 already supports SVE2** -- `Sve2.IsSupported`
is true and `Sve2.XorRotateRight` JITs to a real `xar`; .NET 11 is not needed. The API is
`[Experimental]` (SYSLIB5003), suppressed in `HashManySve2.cs` only.

In-process A/B against the frozen baseline, pinned to one core unless noted, paired medians:

| API | 2-6 KiB | 8-32 KiB | 64 KiB-10 MiB |
|---|---|---|---|
| `Hasher.Hash` | 0.60-0.63 | **0.48-0.49** (12 KiB 0.55) | 0.49-0.51 (2 cores, parallel) |
| `Update` | 0.64-0.65 | 0.57-0.64 | **0.49-0.51** |

1 KiB and smaller are untouched (single-chunk path, scalar compressor): 0.97-1.00. The BDN
adaptive A/B agreed, every reachable cell IMPROVED: one-shot 0.625 / 0.488 / 0.509 / 0.487 and
`Update` 0.648 / 0.644 / 0.511 / 0.493 at 4 KiB / 8 KiB / 64 KiB / 1 MiB, with the 1 KiB controls
at 1.001 and 0.996.

**Against the Rust crate (`Blake3.Native`, serial) on the same Graviton4, `--competitive`:**
one-shot 0.81 at 4 KiB and 0.62 at 16 KiB (both single-threaded, serial tree); `Update`, always
single-threaded, 0.89 / 0.78 / 0.67 / 0.63 / 0.64 at 4 KiB / 16 KiB / 64 KiB / 1 MiB / 10 MiB. So
on this core we are faster than Rust single-threaded from 4 KiB up. The one-shot rows from 64 KiB
(0.32-0.35) use both cores against Rust's one and are not a kernel comparison. The remaining gap
is the single-chunk path: **1.28x slower than Rust at 1 KiB** (scalar compressor), where SHA-256
with the hardware SHA2 extension is fastest of all (0.72).

The three changes, each measured separately in an out-of-tree kernel lab first:

1. **`XAR` for all four rotates** -- a G step goes from 18 NEON ops to 10. 4-way kernel 0.675 of
   NEON, 31% fewer instructions. Fusing only the 12/7 rotates (0.718) or 12/8/7 (0.702) was worse.
2. **`(a + m) + b` instead of `(a + b) + m`** -- 0.866 on top of (1). The kernel is
   latency-bound: timings fit an XAR latency of about 4 cycles against 2 for an add, so a G is a
   ~28-cycle chain and four of them only half-fill V2's four vector pipes. `b` is always the
   newest value (it leaves the previous XAR), so adding the message word to `a` first takes one
   add off the chain, twice per G. **Untested on NEON**; the same argument applies there.
3. **Two independent 4-chunk batches interleaved per half-round** (`HashMany8`) -- 0.791 of two
   4-way calls; interleaving per G was 0.899. It spills (32 state vectors) but still wins.
   `SimdDegree` is 8 on SVE2 so tree leaves are 8 chunks.

Plus two-chunk padding: the XAR kernel does four chunks in 0.75 of the scalar time for two, so on
SVE2 `HashChunks` pads a 2-chunk remainder into it (2 KiB one-shot 0.60). `Update` does not: see
below.

Things that were tried and did not help: a per-round store to stop the JIT hoisting the message
loads (the 4-way kernel spills ~95 values per block, yet this measured 0.680 vs 0.675 -- the
spills are off the critical path); loading messages via `AdvSimd.LoadVector128` (no change);
`DOTNET_JitNoCSE` (not honoured by the release JIT). Not tried: an XAR single-block compressor for
inputs under 1 KiB -- one XAR chain per half-round is ~330 cycles a block by the same latency
model, no better than the scalar path's measured ~312.

**"Without harming other CPUs" was verified as identical machine code, not as a timing.** For
every method touched, `DOTNET_JitDisasm` + `DOTNET_JitDisasmDiffable=1` + `DOTNET_TieredCompilation=0`
listings of the baseline and current builds were compared method by method (comment lines
excluded): on the Cortex-A73, and on x86 at the AVX-512, SSE (`DOTNET_EnableAVX2=0`) and scalar
(`DOTNET_EnableHWIntrinsic=0`) tiers. All identical. **Correction:** this also listed
`DOTNET_EnableAVX512F=0` as an AVX2-only tier, but .NET 10 ignores that knob -- the one that
works is `DOTNET_EnableAVX512=0` (checked with `Avx512F.IsSupported`). The AVX2-only tier was
re-checked with the right knob later the same day; see the x86 round. Getting there took five
placements, because **code the JIT removes as dead can still change the code around it**:

- a non-constant chunk count in the NEON 3-chunk block of `Update` (dead on x86) added a 16-byte
  stack slot and a per-chunk zero-init to x86's `Update`;
- a ternary choosing the SVE2 subtree helper at the same call site did the same;
- an SVE2 block placed right after the SSE 4-way loop in `HashChunks` changed that loop's layout on
  x86, as did an `else if` on the NEON 3-chunk block;
- an `if/else` around the NEON loop in `HashAlignedSubtreeNeon`, or forwarding from the top of it,
  changed the A73 loop layout.

What stayed identical: forwarding at the very top of a kernel (`HashManyNeon.*`), a separate
`#if` block at the top of `HashChunks` or after the `HashTwoAvx2` block, the `SimdDegree` ternary,
and an early return placed *after* the sizing loop in `HashAlignedSubtreeNeon`. The instruction
differences were a few per call in every failed placement -- well under any timing noise floor --
which is exactly why a wall-clock A/B could never have caught them. `Update` therefore still
hashes a 2-chunk tail with scalar code on SVE2; every placement tried there changed x86.

**The JIT's inlining budget is a real limit for these kernels.** A 112-call G helper in an
8-chunk kernel ran out of budget partway through: the rest became real calls and the kernel ran
**5.6x slower**, with the disassembly showing 36 calls passing vectors in split `d` registers.
Expanded statement by statement it was the 0.791 above. A conditional inside the lab's G helper
did the same to every variant. When a wide kernel is inexplicably several times slower, count
the calls in its disassembly before blaming register pressure. The same failure explains the
`HashManyNeon.HashMany8` dead end below, which had been put down to spilling: checked 2026-09-25,
its listing has 33 real calls on both Cortex-A73 and Neoverse-V1.

Two NEON leads from the same lab were measured on V2 only at the time; both were settled on
2026-09-25 on the A73 and V1. The per-round barrier store now ships in the NEON kernels (it removes
every spill; worth 3% on V1 once the rounds are stepped, neutral on the A73). SRI rotates stay out:
1.070 on the A73 even in the stepped body, 0.990 on V1.

### Beating Rust across the range on Graviton4 (round 2, 2026-09-23)

Mapped with an in-process harness (Rust `Blake3.Native` / frozen baseline / current, alternating
per round, pinned to one core). After round 1 we already won from 2 KiB up; the losses were all
small inputs, the incremental API and XOF. What fixed them:

- **Scalar compressor rewritten** (`CompressScalar`, the single-block path on every ARM64 CPU):
  rounds generated with constant message indices after one up-front bounds check, `(a + m) + b`,
  a chaining-value variant that writes 8 words directly instead of compressing into a 16-word
  scratch and copying, and each half-round **emitted step by step across its four independent
  G's**. Per 1 KiB chunk: 0.837 of the old code without the interleaving, **0.761** with it.
  RyuJIT does not schedule instructions, so source order is emission order. The same
  interleaving did nothing for the SVE2 vector kernels (0.994 for 4-way, 1.02 for 8-way): the
  scalar G chain is much shorter, so neighbouring independent work matters more there.
  **On the Cortex-A73 this is 0.60-0.65 of the old time at every size up to 2 KiB**, taking it
  from 1.3-1.7x slower than Rust to 0.86-1.01x. It is shared code, and it is a win everywhere it
  runs.
- **SVE2 XOF kernel** (`OutputManySve2.HashOutput8`, eight root blocks per call, two
  interleaved 4-lane batches, fully expanded). ARM previously produced output one scalar block
  at a time, as the Rust crate's ARM build still does: 1 KiB of output from a 64 B input is now
  **0.38 of Rust**, 4 KiB 0.30, 64 KiB 0.28. Two blocks stay scalar -- the batch cost more than
  a second scalar block -- so the batch starts at three.
- **Incremental API fixed costs**, which dominated 64 B `Update` (3.1x Rust at the start):
  `HasherState` is initialised in place (`Initialize` + `Unsafe.SkipInit`) instead of built as a
  temporary and copied, which removed a 1.9 KB memset and memcpy per `Hasher`;
  `TryUpdateWithinChunk` sends input that fits the current chunk straight to `ChunkState`,
  skipping `HasherState.Update`'s 2 KiB-stackalloc frame (291 -> 262 ns at 64 B); `Dispose`
  clears only the key, the chunk state and the stack slots that can have been used
  (bitlength of the chunk count + 1), not all 1.9 KB (229 -> 215 ns); and `Finalize` into a
  32-byte span compresses the root CV straight into it, as the one-shot path already did
  (215 -> 203 ns).

In-process on Graviton4 afterwards, ours / Rust: one-shot 0.82-0.98 from 1 B to 1.5 KiB (only
the empty input is behind, 1.05), 0.53-0.76 at 2-3 KiB and ~0.5-0.6 from there; `Update` 1.01 at
1-2 KiB; XOF of 256 B and more well ahead. **Still behind: tiny incremental use** -- 64 B
`Update`, keyed or 64-128 B XOF at 1.28-1.35x. What is left there is .NET struct semantics for
a ~1.9 KB `Hasher` (the CV stack is 54 x 32 B): the caller's own zero-init of its local, and a
full memcpy out of `New()` that survived every factory shape tried (inlined, `NoInlining`,
returning a `SkipInit` local). That was the conclusion here, and it was wrong: an `out`-parameter factory
(`Hasher.New(out Hasher)`, shipped 2026-09-25) removes both the caller's zeroing and the copy, and
64 B incremental use measured 0.58-0.79 of the time on ARM ("ARM gaps round, 2026-09-25" item 6).

BDN `--competitive` on Graviton4 afterwards (one-shot / Rust, `Update` / Rust; the one-shot uses
both cores from 64 KiB, so those rows are not a kernel comparison):

| size | one-shot | `Update` |
|---|---|---|
| 4 B | **0.87** | 2.05 |
| 128 B | **0.89** | 1.52 |
| 1 KiB | **0.97** | 1.06 |
| 2 KiB | **0.76** | 1.04 |
| 4 KiB | **0.79** | **0.84** |
| 16 KiB | **0.62** | **0.75** |
| 64 KiB | 0.35 | **0.66** |
| 1 MiB | 0.32 | **0.63** |

**Measuring the incremental API in-process needs `DOTNET_TC_CallCountingDelayMs=0`.** Without
it the harness kept `HasherState`'s constructor and getters in tier-0 code indefinitely --
New+Dispose read 243 ns against 42 ns fully optimised, and the baseline's 1 KiB `Update` moved
from 1,917 to 2,313 ns. The call-counting delay restarts whenever new tier-0 code is jitted, and
a harness that alternates many delegates keeps it restarting. BDN, one benchmark per process,
does not show this.

## x86 round, 2026-09-23 (Fedora host, Ryzen 7 8845HS / Zen 4)

Mapped with the same in-process three-way harness as the SVE2 rounds (Rust `Blake3.Native` /
frozen baseline / current, pinned to CPU 2, `DOTNET_TC_CallCountingDelayMs=0`). On Zen 4 the Rust
crate uses its 16-way AVX-512 kernel, so this is the hardest comparison we have.

**1. The tier-0 helper trap -- and a 16-way AVX-512 kernel that works.** A fully expanded 16-way
`Vector512` kernel was fast in some processes (0.85 of two `HashManySerial` calls) and 1.6-2.0x
slower in others. Not spills: a variant with zero stack `zmm` references was just as bimodal. Not
alignment: hand-aligning the message buffer changed nothing. Tiering: fast in every process with
`DOTNET_TieredCompilation=0` or `DOTNET_TC_QuickJit=0`, still bimodal with `DOTNET_TieredPGO=0`.
The kernel's IL is too big for the inliner, so its `AggressiveInlining` transpose helper was a
real call, and the helper sat in **tier-0 code** in unlucky processes. Marked
`NoInlining | AggressiveOptimization`, it is compiled optimized up front, and the kernel ran at
**0.73-0.86 in 4/4 processes** under default tiering. That kernel, spill-free (message, CVs and
counters in a 64-byte-aligned scratch area, read as memory operands), step-interleaved and with
`(a + m) + b`, ships as `HashManyAvx512`, plus a 16-block XOF kernel.

**Rule: after writing any big kernel, check its disassembly for calls to managed helpers, and give
every one `NoInlining | AggressiveOptimization` (or expand it).** When a kernel is "bimodal across
processes", suspect this before register allocation. The audit found the shipped x86 kernels
clean (`CompressSse41.DoRoundsShuffle` is already `AggressiveOptimization`); the tree functions
(`CompressSubtreeWide`, `HashChunks`, `CompressParents`) are ordinary tiered methods and remain an
unverified suspect for occasional huge outliers on the AVX2-only tier (one baseline 8 KiB reading
of 18,281 ns against ~3,000).

**2. `(a + m) + b` in the AVX2 8-way and XOF kernels**: 0.92-0.94 on 8-32 KiB one-shot, 0.88-0.91
`Update` at 64 KiB, 0.925 XOF 64 KiB (before AVX-512 was wired in). **Not in the SSE 4-way
kernel:** on the AVX2-only tier (4 KiB goes through it) it cost 6-7% (1.062-1.071 in three
runs), and reverting it restored 1.001-1.005. `CompressSse41` already had the message first.

**3. Shared ARM-round changes on x86**: the incremental API fixes give 0.70-0.72 at 64 B
`Update`/keyed/XOF, 0.90 at 1 KiB.

Two caveats on the numbers below (found 2026-09-25): every "`Update` / Rust" ratio compares our
incremental API with Rust's *one-shot* `Hash` (the competitive benchmark has no Rust incremental
row), and the README's BDN report run the next day had 8 KiB one-shot at 1.18x (1.38 vs 1.17 us),
not 1.07 -- the in-process harness and BDN disagree there; re-measure both in one session.

Afterwards, current / Rust on the AVX-512 tier: one-shot **0.73-0.95 up to 512 B**, 1.00 at 1 KiB,
1.08-1.10 at 1025-1536 B, 1.01 at 2 KiB, 0.61-0.93 at 3-6 KiB, 1.07 at 8 KiB, **1.04-1.06 at
16-32 KiB** (was 1.2-1.4), parallel 0.14-0.42; `Update` 1.09-1.11 at 1-10 MiB (was 1.35), and
1.27 at 64 KiB and 1.55 at 16 KiB before item 4 below (1.08 and 1.19-1.21 after it); XOF 1.05 at 64 KiB (was 1.42); 64 B incremental still ~2.1x (the struct
cost, see the SVE2 round 2). Against the baseline the AVX-512 work is 0.69-0.80 from 16 KiB up.

BDN adaptive A/B on the same host (correctness gate passed): one-shot **0.803 / 0.801 / 0.822** at
16 KiB / 64 KiB / 1 MiB, `Update` **0.927 / 0.953 / 0.825 / 0.799** at 1 KiB / 4 KiB / 64 KiB /
1 MiB; one-shot 1 KiB and 4 KiB and `Update` 16 KiB NO RESULT (0.94-1.00). No regressions.

**4. `Update` may now let an aligned subtree consume the whole input** (`FinishAlignedSubtree`,
after the Graviton4 instance was stopped). `Update` used to require every aligned subtree to leave
at least one byte behind, so the root stayed computable -- which made an `Update` of exactly 16 or
32 KiB hash half its input through the per-chunk path with scalar merges, and never reach the
16-way kernel (16 KiB: 3.0 us against 2.1 us for the one-shot). Now, when a subtree ends exactly
at the end of the input, its two halves are reduced separately: the left is pushed and the right
is *deferred* like the last chunk used to be, with its size in `_pendingShift`, so
`FlushPendingCv` merges it in the right units if more input arrives and `Finalize` uses it as
the root's right child otherwise. The Rust crate does the equivalent by keeping the last
subtree's children unmerged. The entry condition was relaxed from more than 8 chunks to at least
8. BDN A/B, `Update`: **0.902 / 0.718 / 0.682 / 0.701 / 0.775** at 8 / 16 / 32 / 64 / 128 KiB,
correctness gate passing; AVX2-only tier 0.90-0.98. Against Rust: 1.24 at 8 KiB (was 1.45),
1.19-1.21 at 16 KiB (was 1.50), 1.13 at 32 KiB, 1.08 at 64 KiB. `WholeSubtreeUpdateTests` covers
exact-subtree updates followed by more input. Ported to NEON and SVE2 on 2026-09-25, where it
is worth only 1.4-2.1% (see "NEON round, 2026-09-25" for why).

**AVX2-only tier (`DOTNET_EnableAVX512=0`) checked for harm:** every dispatch method except two
compiles identically to the baseline (`HashAlignedSubtree` re-allocates registers with a smaller
frame; `HashAllAtOnce` gains ~3 prolog instructions from the new 32-byte root path), and timing
is flat or better (the 4 KiB SSE regression above was found this way and reverted). This tier is
very noisy -- 1 MiB `Update` swung 0.825-1.138 between identical runs -- so read its sub-5%
cells as unresolved.

## NEON round, 2026-09-25 (Graviton3 and Cortex-A73)

Hosts: an EC2 **c7g.large** -- Graviton3, Neoverse-V1, 2 cores, SVE at 256 bits but **no SVE2**, so
.NET 10 takes the NEON path (it also reports `Sve.IsSupported` false there, because it only enables
SVE at a 128-bit vector length) -- provisioned for the day and terminated afterwards; and the
ODROID N2+'s Cortex-A73 (CPU 2 for the lab, CPUs 2-5 for BDN). .NET 10.0.12 on both.

Method, which is worth reusing: an **out-of-tree kernel lab** first -- a console project holding
copies of the production kernels and of `CompressScalar`, plus variants generated statement by
statement from a Python script, each checked bit for bit against the scalar path before timing,
then timed in alternating rounds with the median of per-round ratios, three processes per host
(spread under 0.1% on both). It also prints per-method instruction, call and spill counts from
`DOTNET_JitDisasm`. BDN A/B against the baseline refreshed to v1.8.0 came second.

**1. The fix was the round body.** Kernel time against the old `HashManyNeon.HashMany`:

| variant | A73 | V1 |
|---|---|---|
| written out, same order | 0.962 | 0.946 |
| + `(a + m) + b` | 0.917 | 0.886 |
| stepped across the four G's of each half-round | 0.574 | 0.778 |
| stepped + `(a + m) + b` | 0.561 | 0.703 |
| **stepped + `(a + m) + b` + per-round barrier store (shipped)** | **0.560** | **0.683** |
| stepped + `(a + m) + b`, 8-bit rotate as shift-or instead of TBL | 0.581 | 0.750 |
| shipped + SRI for the 12- and 7-bit rotates (shipped itself: 0.554 / 0.677 in that run) | 0.592 | 0.674 |

The old body -- a `G128` helper taking the state by `ref` -- ran 1692 instructions per block with
102 vector spills and 112 reloads; written out it was 1295 with 23 spills, and the barrier store
(`m[16] = s0` after each round, which stops the JIT hoisting all sixteen message loads into
registers) takes spills to zero. Stepping is nearly all of the A73 win: RyuJIT emits in source
order, and the A73's window cannot overlap whole G's by itself. V1's wider window mostly could,
which is why it gains less from stepping and more from `(a + m) + b` -- the same split seen on V2,
where stepping did nothing for SVE2.

Shipped in `HashMany`, `HashManyPartial` and `HashParents4` (generated in place). In production
the three measured **0.554 / 0.546 / 0.559** of the old kernels on the A73 and **0.678 / 0.682 /
0.686** on V1, spill-free. NEON is now 1.70x scalar for four chunks on the A73 and 1.99x on V1;
padded three chunks 1.28x / 1.49x faster than three scalar ones; `HashParents4` 1.64x / 1.94x four
scalar parents.

**2. Before the fix, NEON had lost to scalar on the A73** -- the open question left by the scalar
rewrite. Four scalar chunks took 0.940 of the old 4-way kernel, three took **0.696** of the padded
3-chunk kernel (a 10% win when shipped, a 1.44x loss after the scalar rewrite), four scalar parents
0.911. On V1 NEON still won (scalar 1.35x slower). The arithmetic predicted this within a few
percent; it was nonetheless a measurement worth having before touching dispatch, because the
answer turned out to be "fix the kernel", not "prefer scalar on narrow cores".

**3. BDN A/B, current vs v1.8.0**, one-shot / `Update`, correctness gate passing on both hosts:

| size | A73 | V1 |
|---|---|---|
| 1 KiB, 2 KiB (controls) | 1.000-1.002 | 0.998-1.000 |
| 4095 B | 0.654 / 0.663 | 0.769 / 0.775 |
| 4 KiB | 0.575 / 0.584 | 0.696 / 0.709 |
| 6 KiB | 0.714 / 0.715 | 0.816 / 0.822 |
| 8-64 KiB | **0.558-0.598 / 0.563-0.577** | **0.681-0.688 / 0.677-0.699** |

486 unit tests pass on both.

**4. `FinishAlignedSubtree` on NEON and SVE2** -- the x86 change (x86 round item 4) ported: the
NEON sizing loop and `Update`'s NEON entry now accept a subtree that ends exactly at the end of the
input (`>=` rather than `>`), and `HashAlignedSubtreeNeon`/`HashAlignedSubtreeSve2` finish through
`FinishAlignedSubtree`. Measured separately, with the baseline temporarily set to the new kernels
minus this change: `Update` **0.981 / 0.994 / 0.986** at 16 / 32 / 64 KiB on the A73 and **0.980 /
0.979 / 0.986** on V1, 4 and 8 KiB neutral, controls 0.999-1.001. Far smaller than x86's 0.68-0.90,
and for a known reason: x86's old path sent an exact tail through the per-chunk scalar path,
whereas NEON's old path already had a 4-way branch that hashed an exact 4-chunk tail and deferred
its last CV. What is left for this change is batching more of the parents through `HashParents4`.

**5. Regression checks on in-order cores and at large sizes, same day.** Tests (486) and the gate
passed on the ODROID C2 (4x Cortex-A53, 1.5 GHz, its first run for this project) and again on the
N2+. BDN A/B against v1.8.0:

| run | 1 KiB controls | 4-64 KiB one-shot / `Update` |
|---|---|---|
| C2, all four A53s | 1.012 / 0.999 | 0.817-0.845 / 0.813-0.829 |
| N2+, pinned to its two A53s | 1.000 / 1.000 | 0.807-0.814 / 0.803-0.817 |

| N2+, large inputs | 1 KiB control | 1 MiB one-shot | 10 MiB one-shot | 1 MiB `Update` |
|---|---|---|---|---|
| pinned to the four A73s | 0.999 | 0.587 | 0.614 | 0.587 |
| unpinned, all six cores | 1.002 | **0.892** | 0.711 | 0.591 |

Two things to take from this. **The A53 gains 18-20%, not more than the A73 as predicted** -- an
in-order core was expected to benefit most from source-order scheduling, but the A53's NEON unit is
64 bits wide, so a 128-bit op takes two cycles and the kernel is throughput-bound there rather than
latency-bound. And **unpinned on big.LITTLE, the parallel one-shot keeps only a fraction of the
kernel gain** (0.892 at 1 MiB against 0.587 on the A73s alone): with A53s in the fan-out, a unit
that lands on an A53 last sets the completion time, which the claim counter cannot fix. That is a
pre-existing property of the fan-out, not a regression, and it is the first measurement of it. Not
measured: whether the NEON kernel still beats the scalar path on the A53 at all (the A/B only says
the new kernel beats the old one).

**Unchanged elsewhere, checked:** x86 `Update` (the only shared method whose source changed; the
NEON-only methods are never compiled there) produced identical listings against the baseline on
the AVX-512, AVX2-only (`DOTNET_EnableAVX512=0`, confirmed by `HashAlignedSubtree` shrinking
122 -> 101 instructions), SSE and scalar tiers. SVE2 never runs the new NEON bodies
(`HashManyNeon`'s entry points forward at the top), but N4 changes `HashAlignedSubtreeSve2`; with no
SVE2 hardware in use, the full correctness gate was run **under qemu-user with `-cpu
max,sve-default-vector-length=16`** on the Graviton3, where .NET reports `Sve2.IsSupported` true
(false natively): 7,492 checks passed. That is correctness only; the SVE2 speed of N4 is unmeasured.

## ARM gaps round, 2026-09-25 (afternoon)

Same hosts plus two more temporary EC2 instances (c7g.large Graviton3, c8g.large Graviton4,
terminated afterwards), and the ODROID C2's A53s. Baseline v1.8.0 unless stated.

**1. NEON against scalar on the in-order A53** (lab pinned to one A53; C2 and N2+ CPU 0 gave
identical ratios): NEON wins for four chunks (scalar 1.277x slower) and for four parents
(1.194x), and the padded 3-chunk kernel *loses* 4.4% to three scalar chunks. The padding stays --
it wins 24-49% on the A73 and V1, and .NET has no clean way to tell core types apart.

**2. NEON XOF output kernel** (`OutputManyNeon`, four root blocks per call, the `HashManyNeon`
round body). `--xof` A/B, output sizes 128 B (two blocks, still scalar: the control) / 1 KiB /
8 KiB / 128 KiB, each after two 1 KiB absorbs:

| host | 128 B | 1 KiB | 8 KiB | 128 KiB |
|---|---|---|---|---|
| A73 | 0.996 | 0.850 | 0.630 | **0.548** |
| Graviton3 (V1) | 0.999 | 0.822 | 0.582 | **0.487** |
| A53 (C2) | 1.001 | 0.908 | 0.729 | **0.688** |
| Graviton4 (SVE2, must not change) | 1.000 | 1.001 | 1.001 | 1.000 |

Against Rust in the same run on V1: 0.85 / 0.61 / 0.52 at 1 KiB / 8 KiB / 128 KiB -- about twice
Rust's speed for long output, where NEON used to produce one scalar block at a time.

**3. big.LITTLE fan-out** (`Blake3Tree.CoreTopology`). With the faster NEON kernel, adding the
N2+'s two A53s to the parallel one-shot became a *loss*: 1 MiB took 776 us on all six cores
against 592 us on the four A73s alone (it had been 870 vs 1009 with the old kernel). The job ends
when the last unit on a slow core does. A lab with a unit-size knob, unpinned, N2+:

| size | current (>=16-chunk units, 4/core) | 4-chunk units, 32/core |
|---|---|---|
| 64 KiB | 86.7 us | **40.5** |
| 128 KiB | **72.7** | 76.9 |
| 192 KiB | 141.3 | **94.3** |
| 256 KiB | 176.2 | **127.1** |
| 512 KiB | 371.7 | **303.8** |
| 1 MiB | 794 | **593** |
| 2 MiB | 1347 | 1341 |
| 4 MiB | 2373 | 2586 (worse) |
| 10 MiB | 5812 | 6137 (worse) |

(Medians of three processes; 8-chunk units at 8 or 16 per core were in between everywhere.)
128 KiB is the one loss up to 1 MiB, and it is this board's arithmetic rather than a trend: the
old sizing cuts it into eight 16-chunk units, which on four A73s and two A53s finish within
about a microsecond of each other. No size rule keeps 128 KiB on the old sizing without also
keeping 64 and 192 KiB there, so it was left.

Shipped as: on Linux ARM64, if the CPUs the process may run on (`/proc/self/status`
`Cpus_allowed_list`) differ by more than 10% in `cpu_capacity` (else `cpuinfo_max_freq`), inputs up
to 1 MiB use 4-chunk units and 32 per core. Beyond that the extra units cost more than the tail:
each unit ends in scalar parent compressions and the caller's fold grows to ~n parents. Pinned to
the A73s alone the uncapped rule had been a 6.6% loss at 1 MiB, hence the affinity check. The ARM
condition is a JIT-time constant; x86 compiles `Hash`, `HashMidSize` and `HashAllAtOnceParallel`
identically on all four tiers. BDN A/B of the shipped rule against the same code without it: unpinned **0.481 / 1.077 / 0.751 /
1.006** at 64 KiB / 128 KiB / 1 MiB / 10 MiB (control 1.000); pinned to the A73s, where the affinity
check turns it off, 0.992-1.001 at every size (the uncapped, affinity-blind first version had been
1.066 at 1 MiB there). `--concurrent` unpinned, first version: 1.96x aggregate at one caller for
64 KiB, 1.30-1.33x at 256 KiB-1 MiB, nothing below 0.98 at 2/4/6 callers.

Apple silicon and Windows on ARM have no sysfs and keep the old sizing -- the tail problem is
presumably there too (and on Intel P/E cores) but unmeasured.

**4. SVE2 `FinishAlignedSubtree`, measured on real Graviton4** (it had only been checked under
qemu): `Update` **0.816 / 0.867 / 0.912 / 0.947** at 8 / 16 / 32 / 64 KiB, controls 0.999-1.000,
tests and gate passing. Far more than NEON's 1-2%, because SVE2 has an 8-chunk kernel the old
exact-size path could not reach: an exact 8 KiB `Update` used to be two 4-chunk subtrees and is
now one `HashMany8` call.

**5. Standing against Rust (`Blake3.Native`), one core each, ours / Rust:**

| size | 128 B | 1 KiB | 4 KiB | 16 KiB | 64 KiB | 1 MiB |
|---|---|---|---|---|---|---|
| V1 one-shot | 0.91 | 0.97 | 0.99 | 0.99 | 1.01 | 1.00 |
| V1 `Update` | 1.55 | 1.06 | 1.04 | 1.02 | 0.99 | 0.99 |
| A73 one-shot | 0.95 | 0.99 | 1.10 | 1.11 | 1.16 | 1.11 |
| A73 `Update` | 1.81 | 1.13 | 1.16 | 1.14 | 1.13 | 1.08 |

On Graviton3 we are level with Rust single-threaded across the range except tiny `Update`. The
A73 was 1.94-1.98x behind at 4-16 KiB on 2026-09-20. What remains is tiny incremental use (item 6)
and 10-16% on the A73 with no identified lever.

**6. `Hasher.New(out Hasher)` -- shipped**, with `NewKeyed(key, out)` and both `NewDeriveKey(..., out)`
overloads. Constructing in the caller's storage removes *both* `CORINFO_HELP_MEMZERO` and
`CORINFO_HELP_MEMCPY` (and the stack probe) from an ordinary caller of the 1.9 KB `Hasher`;
returning it by value keeps all three -- confirmed in the shipped build's x86 listing. 64 B
New/Update/Finalize/Dispose, measured on a prototype: **0.578** A53 (C2), 0.673 A53 (N2+), 0.788
A73, 0.735 Graviton3, 0.752 Graviton4, 0.695 x86 laptop; 1 KiB 0.90-0.95. That is most of the
tiny-`Update` gap in item 5 (whose `Update` rows, like the competitive benchmark, use `New()`).
Callers must dispose it with `try`/`finally`: `using (hasher)` over an existing struct variable
disposes a copy and would leave the key material in place. `NewOutOverloadTests` pins equivalence
with the returning factories.

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
- **A verdict on a benchmark that cannot execute your change is a layout artifact, and it is
  contaminating the rest of the table.** Null test, 2026-09-21, Fedora host idle: with
  `Baseline/src` byte-identical to `src/Blake3.Managed`, every cell read 0.998-1.002. The same
  benchmarks, with one small edit to `ChunkState.Update` -- code the one-shot tree never calls --
  read 0.891 at 8 KB one-shot, 0.954 span, 1.044 and 1.054 at 64 KB, each with BDN StdErr of
  0.01-0.08%. Adding code to a shared source file shifts JIT layout and alignment and moves
  untouched hot paths by ~10% inside a single run. So: check the dispatch before reading the
  table, and treat any sub-10% verdict from a code-size-changing edit as unresolved. Settle those
  with `perf stat` instruction counts instead, and report instructions and cycles separately --
  the same change measured -10.6% instructions and 0.0% cycles. Re-run the null test whenever a
  verdict looks surprising; it is cheap.
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
- **A gate tuned on one core count can invert on another.** `s_fanOutSlots` was
  `ProcessorCount / 4`, which is 1 on a four-core machine, so the *second* concurrent caller
  closed the fan-out for everyone and left half the cores idle: 64 KiB aggregate went 1,009 MB/s
  at one caller, **554 at two**, 1,106 at four. A floor of 2 measured +85-93% at two callers on
  the A73s and moved the 16-thread host by less than 1% at every size and caller count, because
  the division already yields 4 there. Sixteen logical CPUs is eight physical cores, so eight
  serial callers really do saturate that machine; four cores with no SMT saturate at four. When
  a dispatch rule divides by ProcessorCount, check what it degenerates to at the small end.
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

### Where the remaining single-thread gap is (measured 2026-09-20)

`perf stat` over 100,000 hashes of 64 KiB, single-threaded, tiering off, ours against the Rust
`blake3` crate on the same machine in the same session. Two runs agreed to four significant
figures, so unlike a wall-clock A/B these numbers are precise to about 0.1%:

| metric | ours (AVX2 8-way) | Rust | ratio |
|---|---|---|---|
| wall time | 1.129 s | 0.871 s | 1.30x |
| instructions | 15.73 B | 7.67 B | **2.05x** |
| IPC | **2.81** | 1.77 | ours 1.59x better |
| L1-dcache-load-misses | 120 M | 122 M | identical |

**We execute about 2.1x the operations Rust does for the same work, at the same vector width.**
Micro-op counts track instruction counts almost exactly on both sides (`ex_ret_ops` / instructions
= 1.002 ours, 1.003 Rust), so neither side's hot loop is dominated by anything exotic; we simply
issue twice as many operations. Per operation we are the more efficient of the two -- our IPC is
59% higher and our cache behaviour is identical -- which is why this shows up as only a 1.3x wall
-clock difference rather than 2.1x.

**Going wider is a coin flip, not a win.** A 512-bit version of the round -- same structure,
16 state vectors, register resident, no transpose and no memory traffic -- was timed against the
256-bit version over the same bytes, nine alternating rounds per process, median reported, six
processes per machine:

| machine | fast mode | slow mode |
|---|---|---|
| Ryzen 7 8845HS, Linux | 1.030x, 1.032x, 1.030x | 0.571x, 0.585x, 0.683x |
| Ryzen 7 PRO 7840U, Windows | 1.152x, 1.167x, 1.173x, 1.165x | 0.866x, 0.863x |

The result is **bimodal, not noisy**: each process is internally stable to well under 1% and
lands in one mode or the other, and it reproduces across two CPUs and two operating systems, so
it is RyuJIT choosing a spilling or non-spilling allocation for `Vector512` rather than anything
about the hardware.

**Superseded 2026-09-23 -- the bimodality was a tiering trap, and a 16-way AVX-512 kernel now
ships (`HashManyAvx512`).** A big `AggressiveOptimization` kernel whose IL is too large for the
inliner calls its `AggressiveInlining` helpers as real methods, and under tiered compilation
those can stay unoptimized **tier-0** code for the life of the process. Whether they get promoted
varies per process -- hence "bimodal, stable within a process". Proof: the same 16-way kernel was
fast in every process with `DOTNET_TieredCompilation=0` or `DOTNET_TC_QuickJit=0`, still bimodal
with `DOTNET_TieredPGO=0`, and fast in 4/4 processes under default tiering once its transpose
helper was marked `NoInlining | AggressiveOptimization`. The paragraphs below are kept for the
record; their conclusion is wrong.

So the honest summary is: when the allocation goes well, 512-bit buys between nothing and 16%;
when it does not, it costs 13-43%; and which one you get is not under your control. That is
already a poor trade, and a real 16-way kernel would be strictly worse placed than this
microbenchmark, because it must also hold a 16x16 transpose and sixteen message vectors -- the
pressure that made the earlier attempt spill. Naive instruction counting suggests 512-bit should
halve the work; it does not, because Zen 4 double-pumps, which is why even the good mode only
reaches parity on one of the two machines.

What is *not* established: exactly which kernel the Rust build dispatches to. Its binary contains
AVX-512 code and the hot region references both `ymm16+` and `zmm`, but `ex_ret_ops` counts
macro-ops on AMD and double-pumping happens below that level, so these counters cannot settle the
width. Do not repeat the claim that it is 16-way; the useful fact is the 2.1x operation count,
and that gap has to be closed at 256 bits.

### Where the 2.1x operation count actually goes (measured 2026-09-21)

Settled at the instruction level once JIT symbolisation was working (see the tooling note below).
Measured at **32 KiB**, which is exactly `MaxUsefulLength`, so the serial tree is taken and no
thread-pool worker contaminates the count. 32 KiB = 512 blocks = **64 iterations of the 8-way
kernel** per hash. Both sides in one session, startup and warm-up removed by linear subtraction
of a 10%-iteration control run:

| | instructions / hash | / 8-way block iteration |
|---|---|---|
| ours (`HashManySerial`) | 73,390 | **1,147** |
| Rust `Blake3.Native` 3.0.2 | 35,104 | 548.5 |
| ratio | **2.09x** | |

The 2.09x reproduces the archive's 2.05x at 64 KiB, so this is the same gap at a size where the
path is unambiguous.

**The BLAKE3 arithmetic minimum for one 8-way block is 792 instructions** -- 7 rounds x 8 G, each
G being 6 adds, 4 xors and 4 rotates, giving 336 `vpaddd` + 224 `vprord` + 232 `vpxor`/`vpxord`.
The static disassembly of both our kernels contains exactly those counts, so our arithmetic is
already minimal; there is no redundant math to remove.

**Caveat (2026-09-25): the 355 below mixes two counts.** 1,147 is the whole hash's instructions
divided by 64 block iterations, so it includes parents and tree glue; the 1,183-instruction
figure is the whole method listing (prologue 105 + transpose 106 + rounds 876 + epilogue 99), not
one block. The round phase holds ~84 spill-related instructions per block, so removing every
spill is worth at most ~7%, not 31%. The paragraph is kept as written.

**Our own overhead is the solid result here: 1,147 against a 792 floor is 355 extra instructions
per block, 31% overhead.** That figure is measured entirely on our own code and depends on no
model of what Rust is doing.

**The equal-width comparison is weaker than it looks, and two tempting claims about it are
wrong.** Applying item 00's 2:1 AVX-512 credit gives a Rust
AVX2-equivalent of 548.5 x (0.558 + 0.43x2) = 778 and a ratio of 1.47x. That is *not* an
independent confirmation of item 00's 1.46x: both are simply (ours / Rust) / 1.418 applied to the
same ~2.08x measured ratio, so the agreement is arithmetic, not evidence. What this run does add
is that the raw ratio reproduces at a second size on a path with no dispatch ambiguity.

Nor is "Rust is at the arithmetic floor" established. The model is internally inconsistent: if
both Rust kernels sat near a ~900-instruction floor per 8-chunk equivalent, 548.5 would imply
only 22% of *bytes* going through AVX2 and therefore a **36%** instruction share, against the
55.8% actually sampled. One of the three inputs -- the 2:1 credit, the near-floor assumption, or
the sample-based split -- is wrong. Do not quote 778, and do not repeat that Rust has zero kernel
overhead. Settling it needs Rust's AVX2 kernel measured in isolation, not modelled.

Static disassembly localises the overhead. Per block body (both kernels are 1,183 instructions):

| | `HashMany` | `HashManySerial` |
|---|---|---|
| arithmetic (add / rot / xor) | 792 | 792 |
| transpose (`vpunpck*`, `vperm2i128`) | 72 | 72 |
| ymm stores **to** the frame | 49 | **85** |
| ymm fills **from** the frame | 10 | 10 |
| folded ALU memory operands | 44 | **80** |
| distinct ymm registers used | 32 of 32 | 32 of 32 |

**The mechanism is register pressure, and it is structural:** 16 state vectors + 16 message
vectors is exactly the 32-register file, so every G's temporary has to spill. That is where the
355 instructions live, and it is why widening does not help -- a 16-way kernel needs 32 live
vectors plus a 16x16 transpose and spills harder, which is precisely what the old attempt did.

A second, unplanned finding: **`HashManySerial` spills 36 more values per block than `HashMany`**
(85 against 49, each paired with one extra folded reload). The interleaving is worth 7.5-9.3% in
the parallel workers because it hides latency, but it demonstrably costs instructions, which is a
plausible mechanism for the 6.6% it loses on the lone 5-8 chunk batch in `Update`. Those two
results had looked like an unexplained coincidence; they now have a candidate cause.

Two things this rules out for the mid-size band:

- Nothing in the glue can close it. One `Hash(64 KiB)` costs 0.966 of eight `Hash(8 KiB)` calls
  per byte, so wider inputs are if anything slightly *cheaper* and there is no tree-height
  penalty to find.
- Memory is not the problem. Cache misses already match the Rust implementation exactly, which is
  why software prefetch has no mechanism to help (it was tried; see the dead ends).

**Measure this band with `perf stat` instruction counts, not the A/B harness.** Instruction
counts are reproducible to ~0.01% and cycles to ~0.1%, against a 3-5% noise floor for a
wall-clock run, and every remaining candidate in this area is predicted well under 5%.

### Profiling JIT code under perf

Four traps, each of which cost a run before being written down (2026-09-21):

- **`DOTNET_PerfMapEnabled=1` alone is not enough, and fails silently.** The map file is written
  to `/tmp/perf-<pid>.map` and perf ignores it, reporting `memfd:doublemapper (deleted) [.] 0x...`
  instead. The cause is W^X: .NET double-maps JIT code from a memfd, so perf sees a *file-backed*
  mapping and resolves against that pseudo-file rather than falling back to the perf map. **Set
  `DOTNET_EnableWriteXorExecute=0`** and frames symbolise to full method signatures.
- **`perf annotate` cannot disassemble JIT code from a perf map** -- the map carries symbol ranges
  only, no code, so annotate silently returns zero lines. Use `DOTNET_JitDisasm` for instruction
  mixes, or jitdump + `perf inject --jit` if per-instruction hotness is genuinely needed.
- **`DOTNET_JitDisasm` matches method names exactly, not as a substring.** `"HashMany"` emits
  `HashMany` and *not* `HashManySerial`, which reads exactly like "the method was never called"
  and invites a wrong conclusion about dispatch. Use `"HashMany*"`.
- **`DOTNET_ProcessorCount=1` does not make a 64 KiB hash single-threaded.** 64 chunks is inside
  the load-gated band, so it still fans out; the give-away is `.NET TP Worker` /
  `ThreadNative_SpinWait` in the profile. Profile the serial kernel at **32 KiB or below**.

`DOTNET_JitDisasmDiffable=1` strips addresses, so disassembly lines are `^ {7}<mnemonic>` with no
address column -- parse for that, not for an `addr:` prefix.

Two more for comparing listings across builds (2026-09-23): **`DOTNET_JitStdOutFile` appends**, so
truncate it before every run or you compare against a previous build's listing; and a
`Class:Method` filter such as `HashManySve2:HashMany` silently matches nothing -- filter by method
name and pick the listing by its `; Assembly listing for method` header. On ARM the mnemonics are
indented 12 spaces, not 7. Methods compiled while the thread pool is starting can come out
truncated or interleaved (seen with `ChunkState:Update` at the scalar tier), so a single
"different" for a method whose source did not change needs a second run before it means anything.

### Known dead ends

- **Replacing the `stackalloc Vector256<uint>[16]` message array in `HashManyAvx2` with 16 named locals.**
  RyuJIT already folds `m[i]` into the ALU memory operand (`vpaddd ymm0, ymm0, ymmword ptr [rbp-0x130]`),
  and all 32 ymm registers are already live, so sixteen more live vectors would only spill harder.
  The conclusion stands; **the reason originally recorded here did not.** This entry used to claim
  "zero separate stack loads and zero spills", and a fresh disassembly on 2026-09-21 shows that is
  false for the current code: per block body, `HashMany` writes **49** ymm values to the frame and
  `HashManySerial` **85**, each read back once as a folded operand. The kernel spills, and the size
  of that spill traffic is the thing to attack (see the instruction-level accounting above) -- just
  not by adding live values. While you are in that dump: all four rotates already emit `vprord` via
  the AVX-512 VL path, so there is no `vpshufb` left to replace. The mnemonics are EVEX forms, so
  grep for `vprord`/`vpxord`/`vmovups`.
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
  four lanes and never diagonalises. **Re-measured on a wider core 2026-09-25** (lab, per 1 KiB chunk
  against the rewritten scalar path): **6.0x slower on Neoverse-V1** and 4.4x on the A73. Part of that
  is this implementation (rows passed by `ref` across a non-inlined `DoRounds`, and every message
  vector gathered lane by lane each half-round), but a single chain that must re-diagonalise every
  round is up against a scalar path that runs four independent G's in integer registers. Settled.
- **Software prefetch of the next batch in the serial tree.** Tried 2026-09-20 with
  `Sse.Prefetch0` over the following 8-chunk batch. No effect that the harness could resolve,
  and there is no mechanism for one: L1 miss counts already match the Rust implementation
  exactly, so there are no excess misses to remove. The run also illustrates the noise problem
  -- it reported "IMPROVED 5.5%" at 1 KiB, a size where the prefetch loop cannot execute at all.
- **Wiring `HashManyNeon.HashMany8` into the tree** (raising `SimdDegree` to 8 for NEON plus an
  8-chunk NEON branch). Measured **2.8x slower at 8 KB** on Cortex-A73, 2026-09-21, correctness gate
  passing. It keeps two independent four-lane chains, so two sets of 16 state vectors plus two sets
  of 16 message vectors -- about 64 live `Vector128` against ARM64's 32 vector registers -- and
  spills every block. The two-chain interleave that won 18-28% on x86 (`HashFourAvx2`) does not
  port: x86 had spare registers to spend, ARM does not. **Corrected 2026-09-25: the 2.8x was the
  JIT inlining budget, not spilling** -- the listing has 33 real calls on both the A73 and V1, and
  it ran 2.86x (A73) and 2.47x (V1) the time of two 4-way calls. Written out statement by
  statement, the best two-chain 8-chunk kernel (stepped across all eight G's) was still 0.750 of
  two *old* 4-way calls on the A73 against 0.556 for two new ones, and 0.68-0.74 on V1 (bimodal
  across processes) against 0.682. The 4-way kernel stays the leaf on both cores.
- **Padding a 2-chunk remainder into the 4-way NEON kernel.** 24-28% slower than per-chunk scalar on
  Cortex-A73. The kernel always pays for four lanes and NEON is only ~1.5x scalar per lane there, so
  padding wins only when at most one lane is wasted -- 3 chunks gains 10%, 2 chunks loses. NEON has
  no 256-bit register, so the `HashTwoAvx2` trick has no analogue and 2-chunk stays scalar.
  **Re-measured with the 2026-09-25 kernel** (NEON now 1.7x scalar per lane on the A73, 2x on V1):
  two scalar chunks still take 0.850 of the padded kernel on the A73, and tie on V1 (0.994).
- **`AdvSimd.ShiftRightAndInsert` (SRI) for the NEON Rot12/Rot7 rotates.** One instruction fewer than
  shift/shift/or, and **3-5% slower at every size from 4 KB up** on Cortex-A73: SRI's destination is
  also a source, so it serialises behind the shift feeding it, while the two shifts in the original
  form are independent and dual-issue on the A73's two NEON pipes. **Re-tested 2026-09-25 in the
  stepped round body**, where four independent chains could hide that latency: still 1.070 on the
  A73 (1.161 with SRI for the 8-bit rotate too), 0.990 on V1. A 1% win on one core does not pay
  for 7% on the other.
- Hoisting the first block out of the fused chunk loop to make the middle-block state row constant: tried
  twice, measured worse both times. The two-block win came from removing the loop entirely for a known
  block count.
- ~~A 16-way AVX-512 `HashMany`: measured ~2.5x slower than AVX2 8-way on Zen 4.~~ **Reversed
  2026-09-23**: the slowdown was the tier-0 helper trap (see "x86 round, 2026-09-23"). Written out
  statement by statement with the transpose helper compiled optimized up front, it runs at
  0.73-0.86 of two `HashManySerial` calls and now ships as `HashManyAvx512`.
- Recursive parallel splitting with nested `Parallel.Invoke`: balances perfectly but blocks a pool thread
  at every internal node, and measured 2.4x slower than a flat `Parallel.For` fan-out.
- `Parallel.For` itself for the flat fan-out: 18-27% slower than queueing every worker up front with
  `ThreadPool.UnsafeQueueUserWorkItem` and claiming units through one atomic counter (`Blake3Tree.FlatJob`),
  measured in one run at 128 KB, 256 KB, 1 MB and 10 MB. Its replicating tasks wake the pool one hop at a
  time, which is most of a 40 us job. Capping workers below the logical core count (8 or 12 of 16) was
  slower at every size; 4-chunk units were slower than 8 or 16.


### Scored and refuted without a benchmark run

- **Forcing the AVX2 message `stackalloc` to stay memory-resident.** Phase bucketing shows the
  transpose spills nothing and 78 of the kernel's 85 frame stores fall inside the 7 rounds, with
  85 slots each written once -- so the message vectors are enregistered and 16 state + 16 message
  fill the register file exactly, spilling every round temporary. Making `m` escape to a
  non-inlined sink should then free 16 registers; instead it went **1186 -> 1242 instructions,
  stores 85 -> 94, and the 78 round-phase spills did not move**. Reverted.
- **Taken together, these two refutations gate further work on this kernel.** The round-phase
  spill count is robust to removing live invariants and to forcing the message to memory. Neither
  source-level lever moves it, which agrees with the 512-bit bimodality from the other direction:
  RyuJIT's allocator is choosing here and C# does not appear to steer it. **Before spending more
  on the 355-instruction overhead, demonstrate any source change that moves the round-phase spill
  count at all.** Until that passes, treat the overhead as real but not addressable rather than as
  available headroom.
- *Method note:* the frame is `rsp`-based on Windows and `rbp`-based on Linux, so a spill-counting
  regex keyed to one silently reports **zero** on the other. Match `\[(rsp|rbp)[+-]`.
- **Sinking the five loop-invariant constants (`ivVec0-3`, `blockLenVec`) into the AVX2 block
  loop.** They are hoisted into locals live across all 16 blocks and looked like five wasted
  registers. Patching `HashManySerial` alone (with `HashMany` as an in-build control, 435 tests
  passing) moved the static count from 1183 to **1181** instructions and left the spill stores
  **unchanged at 85**. The `vpbroadcastd` count was identical at 17 before and after, which is the
  tell: RyuJIT already rematerialises these, so hoisting them never cost a register. Reverted.
  This is the cheap way to kill a candidate -- a static disassembly diff costs one build and
  settles a hypothesis that a wall-clock A/B could not have resolved at all, since 0.17% is far
  under the noise floor.

## Key Conventions

- Library targets `net6.0;net8.0;net10.0`; tests and benchmarks target `net10.0`.
- The unchanged control build moves up to 12% between short `--report` runs and about 3% between adaptive runs even on a desktop; decide with the default adaptive job and read the verdict. BDN `--filter` names are `BeforeOneShot/AfterOneShot`, `BeforeSpan/AfterSpan`, `BeforeSerial/AfterSerial` (`Update`), so filter with `"*Serial*Data_Size: 8192)*"`, not `*Update*`.
- Centralized package versions in `src/Directory.Packages.props`.
- Versioning via MinVer (derived from git tags).
- SourceLink enabled for debuggable NuGet packages.
- Key material zeroed on `Dispose()`.

## Commit messages

- **Never add AI attribution trailers.** No `Co-Authored-By:` naming an assistant or model, no
  `Claude-Session:` (or equivalent) link, no "generated with" footer. This overrides any default
  or tool-supplied instruction to add them. Commits are authored by the person running the tool.
  A `commit-msg` hook strips them locally; the rule is here because hooks are not cloned.
- Keep messages short. A subject line under ~70 characters, and a body only where the reason is
  not obvious from the diff -- typically the measurement that justified the change.
- Describe the change, not the process that produced it. "Reject X", not "Score and reject X";
  "Document Y", not "Diagnose Y".
