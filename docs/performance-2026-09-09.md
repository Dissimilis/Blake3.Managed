# Performance campaign — 2026-09-09

Starting commit: `2e9c354` (`v1.5.1`). Machine: AMD Ryzen 7 PRO 7840U,
8 physical / 16 logical cores, Windows 11, .NET SDK 10.0.303, runtime 10.0.11.
Results below compare implementations inside each session. Absolute times from
different sessions or machines must not be compared: this laptop throttles.

## Target and changes

[Issue #1](https://github.com/Dissimilis/Blake3.Managed/issues/1) points to the
[CryptoHives benchmark history](https://cryptohives.github.io/Foundation/packages/security/cryptography/benchmark-trends/index.html).
The latest Windows BLAKE3 run available when inspected was `d605559609`, dated
2026-08-29, on a Ryzen 5 7600X, using Blake3.Managed **1.3.0**. It showed gaps at 2 KB, 6 KB, and extended output,
while Blake3.Managed led the large-input one-shot rows.

The comparison's
[one-shot adapter](https://github.com/CryptoHives/Foundation/blob/main/tests/Security/Cryptography/Adapter/Hash/Blake3DissimilisAdapter.cs)
calls `Hasher.Hash(input, output)`. Its
[XOF workload](https://github.com/CryptoHives/Foundation/blob/main/tests/Security/Cryptography/Benchmarks/Hash/ParameterizedXofBenchmark.cs)
absorbs the same 1 KB block twice, produces the selected number of output bytes,
then resets a reused instance.

Changes made for those workloads:

- Route 32-byte span destinations through the existing fused small-input SSE
  compressors, which previously only the value-returning overload used.
- Generate eight independent 64-byte XOF blocks per AVX2 batch. Preserve scalar
  prefixes/tails for arbitrary seek offsets and output lengths.
- Hash two complete input chunks together using the two 128-bit halves of AVX2
  registers. Use this after the wider chunk kernels, in both the tree and
  incremental APIs.
- Hash 5–7 complete chunks in one partial eight-way batch. Inactive lanes reread
  valid input and their results are discarded. Keep the original full-batch
  kernel separate, because adding variable offsets to it regressed 8 KB inputs.
- Preserve a deferred final chunk across empty `UpdateWithJoin` calls. The
  expanded batching exposed this pre-existing empty-update bug through
  `HashAlgorithm.TransformFinalBlock`.
- Add span-output and XOF benchmark coverage, including 2 KB and 6 KB inputs.
- Update the benchmark-only CryptoHives dependency from 0.6.21 to 0.6.101. The new
  package reports `Ssse3, Avx2, Avx512F` here; the old SSSE3-only caveat no longer
  describes the current comparison. Print algorithm-specific capabilities and
  package assembly version in each benchmark session.

## Small-input span overload

Default adaptive A/B job: two launches, 15–60 measured iterations per launch,
1% requested relative error. All three verdicts have non-overlapping confidence
intervals. These measurements preceded the other kernel changes.

| Input bytes | Before | After | Time reduction |
|---:|---:|---:|---:|
| 4 | 96.028 ns | 66.531 ns | 30.7% |
| 128 | 162.535 ns | 141.548 ns | 12.9% |
| 1,024 | 1,272 ns | 1,213 ns | 4.7% |

This initial run used the existing `d097897` snapshot. The measured span overload
and its small-input compressor path were unchanged between that snapshot and
the starting commit. The baseline was subsequently refreshed to **exactly
`2e9c354`**, before this campaign's changes, for the remaining A/B work.

```powershell
dotnet run --project src/Blake3.Managed.Benchmarks -c Release -- --filter '*Span*Data_Size: 4)*' '*Span*Data_Size: 128)*' '*Span*Data_Size: 1024)*'
```

## Extended output

Same-session `--report` job: one launch, three warmups, five measured iterations.
All methods absorb 2 KB in two updates, finalize into a reused buffer, and reset.
The native reference is `Blake3.Native` 3.0.2; CryptoHives is 0.6.101 with its
default SIMD dispatch. None of these rows allocated managed memory.

| Output bytes | Before | After | Native | CryptoHives | A/B verdict |
|---:|---:|---:|---:|---:|---|
| 128 | 2.918 us | 2.876 us | 2.516 us | 2.978 us | No result: overlapping intervals |
| 1,024 | 3.895 us | 3.054 us | 3.435 us | 3.526 us | Improved 21.6% |
| 8,192 | 11.725 us | 4.722 us | 10.612 us | 5.500 us | Improved 59.7% |
| 131,072 | 155.348 us | 34.229 us | 136.679 us | 40.958 us | Improved 78.0% |

The 1 KB and 8 KB after intervals are also below both competitors' intervals.
The 128 KB CryptoHives error was ±6.847 us, so its interval overlaps ours;
its higher mean alone does not establish a competitive win. Native still leads
the 128-byte output workload.

```powershell
dotnet run --project src/Blake3.Managed.Benchmarks -c Release -- --xof --report --filter '*'
```

## Two-chunk batching, before partial batches

`--report` job against the refreshed `2e9c354` control. Both implementations are
measured with the same API and input in each pair.

| Input bytes | API | Before | After | Time reduction |
|---:|---|---:|---:|---:|
| 2,048 | Hash returning a value | 2.677 us | 1.264 us | 52.8% |
| 2,048 | Hash into a span | 2.657 us | 1.254 us | 52.8% |
| 2,048 | Update + Finalize | 2.998 us | 1.562 us | 47.9% |
| 6,144 | Hash returning a value | 4.668 us | 3.206 us | 31.3% |
| 6,144 | Hash into a span | 4.627 us | 3.218 us | 30.5% |
| 6,144 | Update + Finalize | 5.188 us | 3.631 us | 30.0% |

All six improvements have non-overlapping confidence intervals. The 8 KB
value-returning and incremental controls had overlapping intervals. At 10 MB,
both parallel API pairs had overlapping intervals; the serial pair was too
noisy for a verdict (ratio 1.149, worst standard error 2.5%). Do not interpret
these unresolved controls as proof of zero performance change.

```powershell
dotnet run --project src/Blake3.Managed.Benchmarks -c Release -- --report --filter '*Data_Size: 2048)*' '*Data_Size: 6144)*' '*Data_Size: 8192)*' '*Data_Size: 10485760)*'
```

## Correctness and compatibility

- Release solution build passed for the library's net6.0, net8.0, and net10.0 targets.
- 283 tests passed under .NET 10 in each configuration: normal hardware dispatch;
  `DOTNET_EnableAVX512=0`; AVX2 also disabled; all hardware intrinsics disabled.
- Every benchmark session passed the correctness gate. The expanded gate performs
  7,492 checks, including official vectors and differential comparison with Rust.
- New tests cover every input length from 0 through 1,025 for the optimized span
  overload, including overlapping and unaligned destinations; both lanes of the
  two-chunk kernel, nonzero counters and 32-bit carry; XOF batches against scalar
  compression, arbitrary seek offsets, keyed and derive-key modes, exact and
  partial batches, and destination canaries.
- Empty joined updates are checked after 2-, 4-, 5-, 6-, 7-, and 8-chunk inputs,
  both before finalization and before continuing with another byte.

## Final partial-batch validation

The first partial-batch candidate added variable lane offsets to the full
eight-chunk kernel. It improved 6 KB but regressed 8 KB, so it was replaced with
a separate partial-batch method. The original full-batch method body is retained.

The revised candidate's same-session `--report` A/B verdicts against `2e9c354`:

| Workload | After / before | Verdict |
|---|---:|---|
| 6 KB span-output hash | 0.501 | Improved 49.9% |
| 8 KB span-output hash | 1.012 | No result: overlapping intervals |
| 10 MB one-shot hash | 1.005 | No result: overlapping intervals |
| 10 MB serial Update + Finalize | 1.012 | No result: overlapping intervals |

The partial kernel additionally interleaves the four independent G mixing
chains within each half-round. At this stage the scheduling was confined to
partial batches; the serial tree follow-up below extends it to full batches.
Tests and the Rust differential gate pass with this schedule.

Earlier, the two-chunk-only candidate produced a 6.4% 10 MB serial regression
in a long sequential adaptive run. A temporary probe alternated before/after
and after/before measurements in one process: three runs of 64 pairs, eight
hashes per measurement, with warmup. Its geometric after/before ratios were
0.957, 0.917, and 0.901. Conservative four-standard-error intervals were
[0.884, 1.036], [0.860, 0.978], and [0.847, 0.957]. The sequential slowdown was
not reproduced. These conflicting observations are retained as evidence of
thermal/order sensitivity, rather than claiming a large-input improvement.

```powershell
dotnet run --project src/Blake3.Managed.Benchmarks -c Release -- --report --filter '*Span*Data_Size: 6144)*' '*Span*Data_Size: 8192)*' '*OneShot*Data_Size: 10485760)*' '*Serial*Data_Size: 10485760)*'
```

## One-shot comparison after partial-batch optimization

This comparison uses caller-provided 32-byte destinations for every library.
CryptoHives uses a reused instance's `TryHashOneShot`, matching the external
comparison, rather than its pooled static convenience method. These are one
`--competitive --report` session with the interleaved partial kernel:

| Input bytes | Blake3.Managed | Native 3.0.2 | xoofx 3.0.2 | CryptoHives 0.6.101 |
|---:|---:|---:|---:|---:|
| 2,048 | 1.253 us | 1.239 us | 2.749 us | 2.027 us |
| 6,144 | 2.128 us | 2.974 us | 8.372 us | 2.309 us |

At 2 KB, ours and native have overlapping intervals. At 6 KB, ours has
non-overlapping intervals with all three competitors: 7.8% less time than
CryptoHives and 28.4% less than native. Every row reports zero managed allocation.
This is a local win for these APIs and package defaults; it does not establish
leadership across every platform or explicitly forced SIMD variant.

The .NET 10 AVX-512 disable switch is `DOTNET_EnableAVX512=0`, as defined in the
[runtime configuration source](https://github.com/dotnet/runtime/blob/v10.0.0/src/coreclr/inc/clrconfigvalues.h).
Earlier attempts using `DOTNET_EnableAVX512F` still reported AVX-512 support;
those measurements are normal-dispatch runs, not fallback evidence.

With the corrected switch, both the parent process and BenchmarkDotNet workers
reported AVX2 without AVX-512. All 283 tests and the 7,492-check correctness gate
passed. Same-session A/B verdicts were **35.9% less time for 8 KB XOF output**
and **24.9% less time for a 6 KB span-output hash**. These results support the
AVX2 fallback paths separately from the AVX-512-assisted measurements above.

Temporary probes, generated benchmark projects, downloaded comparison data, and
session helper files were removed after recording these results.

At the time of these runs, build warnings included local NuGet source mapping
and unresolved XML documentation references in `Blake3SubtreeContext`. The
subsequent readiness review fixed those XML references and removed three
duplicate theory-data entries without changing the 284 distinct test cases.

## Serial tree follow-up

For this follow-up only, the A/B control was the already optimized working tree
from the preceding sections. Its source digest was
`37a64595c6aaddbb3381a0d480553afad3b17face1faa713562b9a739e5f793e`
(SHA-256 over sorted root and Internal C# paths, slash separators, NUL between
path and LF-normalized contents). The permanent baseline was restored to the
starting commit after the experiment. These percentages are additional gains,
not comparisons against the original release.

Two-chunk inputs now skip the general tree frontier, and 32-byte serial tree
results compress the root directly into the destination. The report job found
**2.9% less time at 2 KB**, with no conclusive change at 1,025 bytes, 8 KB, or
10 MB. In a subsequent matched competitive session, ours took **1.203 us
± 0.0045 us** and native took **1.240 us ± 0.0051 us** at 2 KB: approximately
3% less time, with non-overlapping intervals. This is a local result; it does
not predict the upstream machine's result.

The next trial reused the partial-batch kernel for full serial batches:
8 KB improved 2.0%, 64 KB improved 4.5%, and 10 MB was inconclusive. A separate
fixed-offset serial kernel then produced the following same-session results:

| Input bytes | Before | After | Verdict |
|---:|---:|---:|---|
| 8,192 | 2.444 us | 2.268 us | No result: overlapping intervals |
| 65,536 | 19.208 us | 16.324 us | Improved 15.0% |
| 10,485,760 | 422.305 us | 420.294 us | No result: overlapping intervals |

This kernel interleaves the independent mixing chains only for serial one-shot
hashing. Parallel workers and incremental full batches retain the original
kernel because the earlier campaign found a large parallel regression from
changing that schedule globally. The 64 KB confidence intervals only narrowly
separate in this short run; further measurements are recorded below.

The longer default adaptive job (two launches, 15–60 measured iterations,
outliers retained) confirmed **7.6% less time at 64 KB**: before 17.92 us
± 0.153 us, after 16.57 us ± 0.681 us, with non-overlapping intervals. Use this
more conservative result when describing the improvement; 15% was the short
run's estimate.

With `DOTNET_EnableAVX512=0`, the fixed-offset follow-up passed the correctness
gate and the workers reported AVX2 without AVX-512. At 8 KB, before was 6.164 us
and after was 4.829 us: **21.7% less time**, with non-overlapping intervals.
At 64 KB, the ratio was 1.042 with overlapping intervals, so that fallback run
does not establish a change at 64 KB.

```powershell
dotnet run --project src/Blake3.Managed.Benchmarks -c Release -- --report --filter '*Span*Data_Size: 8192)*' '*Span*Data_Size: 65536)*' '*OneShot*Data_Size: 10485760)*'
$env:DOTNET_EnableAVX512 = '0'
dotnet run --project src/Blake3.Managed.Benchmarks -c Release -- --report --filter '*Span*Data_Size: 8192)*' '*Span*Data_Size: 65536)*'
Remove-Item Env:DOTNET_EnableAVX512
```

The final normal-dispatch competitive report used the same caller-provided
32-byte output API for all three implementations:

| Input bytes | Blake3.Managed | Native 3.0.2 | CryptoHives 0.6.101 |
|---:|---:|---:|---:|
| 8,192 | 2.296 ± 0.2154 us | 1.896 ± 0.0163 us | 2.635 ± 0.1802 us |
| 65,536 | 16.502 ± 1.0127 us | 12.243 ± 0.2981 us | 19.162 ± 1.0984 us |

Native remains faster at both sizes. At 64 KB, ours takes 13.9% less time than
CryptoHives with non-overlapping intervals. At 8 KB, ours has a lower mean than
CryptoHives but overlapping intervals. All these rows report zero managed
allocation. This comparison does not include explicitly forced CryptoHives
SIMD variants or establish results on ARM.

Final validation: the Release solution build succeeded for net6.0, net8.0 and
net10.0. All 284 unit tests passed on net10.0 with normal dispatch, AVX2 only,
SSE only and hardware intrinsics disabled. The new direct eight-lane test
covers unaligned buffers, output guards, keyed/derive flags and chunk counters
crossing the 32-bit boundary. The 7,492-check native differential gate passed
before each benchmark session. All 17 restored baseline source files were
verified against commit `2e9c354`.

## Scope and remaining work

These changes are included in `v1.5.2`. The external dashboard will continue to
measure its selected NuGet version until a release and a new upstream run.
No claim is made that every size, platform, or SIMD variant is now the fastest.
The README was subsequently refreshed from complete one-shot and XOF report
runs on the same machine: 60 and 12 benchmark cases respectively, each preceded
by the 7,492-check correctness gate. Its tables and charts use the published
[one-shot CSV](benchmark-oneshot-2026-09-09.csv) and
[XOF CSV](benchmark-xof-2026-09-09.csv). Those runs supersede the old README
figures; the optimization experiments above retain their own baselines and
measurement context.

ARM performance needs hardware validation. Inspection also found that
`CompressNeon` is not selected by `Blake3Core.CompressCv` or `CompressInPlace`,
although the four-chunk NEON kernel is selected. That is a separate candidate
requiring ARM correctness and performance measurements.

## README and readiness verification

The README figures were rendered locally and inspected for clipping and label
overlap. Both SVGs and tables reproduce exactly from their published CSV data.
Local documentation links resolve. A package consumer check also corrected an
old incorrect digest in the README's `Hello, World!` example.

The final solution build completed with zero warnings and errors after restoring
from nuget.org explicitly; the extra feed in this machine's user configuration
was responsible for NU1507. The test suite passes all 284 distinct cases, and
normal, AVX2-only, SSE-only and scalar configurations were verified. The
benchmark `--all --list flat` now includes XOF.

`dotnet pack` succeeded. The package contains assemblies and XML documentation
for net6.0, net8.0 and net10.0, the current README and icon, and no runtime NuGet
dependencies. A temporary net10.0 application restored that package from the
local output directory and successfully executed the README's first example.
Temporary benchmark, package-consumer and rendering files were removed.
