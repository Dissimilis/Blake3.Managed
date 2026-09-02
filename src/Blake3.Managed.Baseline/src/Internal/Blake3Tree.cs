using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;

namespace Blake3.Managed.Internal;

/// <summary>
/// All-at-once hashing for inputs whose length is known up front.
/// </summary>
/// <remarks>
/// The incremental hasher must assume more input may arrive, so it merges each chunk CV into a
/// stack as it goes and cannot use a wide parent kernel: at 64 KB that means 63 parent nodes
/// compressed one at a time, around a quarter of the total work at that size.
///
/// When the whole input is in hand the tree can instead be built with a wide frontier, keeping
/// many sibling CVs live so parents are compressed 8 at a time, and stopping at exactly two CVs
/// so the final parent can carry the ROOT flag. That last detail is why the incremental path
/// cannot do this: it merges the final pair into one CV without ROOT, and the information needed
/// to produce root output is gone by then.
///
/// This mirrors the reference implementation's compress_subtree_wide / compress_subtree_to_parent_node.
/// It is single-threaded by design; callers use the thread-pool path above the size where that wins.
/// </remarks>
internal static class Blake3Tree
{
    /// <summary>Chunks the best available kernel hashes at once.</summary>
    private static int SimdDegree =>
        HashManyAvx2.IsSupported ? 8
        : HashManyNeon.IsSupported || HashManySse41.IsSupported ? 4
        : 1;

    /// <summary>
    /// Largest input the serial tree handles before the parallel one takes over.
    /// </summary>
    internal const int MaxUsefulLength = 72 * Blake3Constants.ChunkLen;

    /// <summary>
    /// Smallest subtree given its own task. Below this the thread-pool hand-off costs more than
    /// the work saved.
    /// </summary>
    private const int MinParallelSubtree = 16 * Blake3Constants.ChunkLen;

    /// <summary>
    /// Hashes <paramref name="input"/> (which must be longer than one chunk) and writes root
    /// output bytes, of any length, to <paramref name="output"/>.
    /// </summary>
    [SkipLocalsInit]
    internal static void HashAllAtOnce(ReadOnlySpan<byte> input, ReadOnlySpan<uint> key, uint flags,
        Span<byte> output)
    {
        Span<uint> parentBlock = stackalloc uint[16];
        CompressSubtreeToParentBlock(input, key, flags, parentBlock);

        // The root is the final parent node: two CVs as its message, with ROOT applied by Output.
        var rootOutput = new Blake3Core.Output();
        rootOutput.Init(key, parentBlock, 0, Blake3Constants.BlockLen, flags | Blake3Constants.Parent);
        rootOutput.RootOutputBytes(output);
    }


    /// <summary>
    /// Hashes a large input, splitting the tree across the thread pool.
    /// </summary>
    /// <remarks>
    /// Splits at the canonical left-subtree boundary and recurses on both halves concurrently, so
    /// every task gets an equal share. The previous scheme reserved one 64-chunk subtree per task
    /// and handed the remainder out as 8-chunk batches, which at 128 KB produced one task doing
    /// eight times the work of the other seven -- the critical path was a single 64-chunk task, and
    /// the wall clock showed it.
    ///
    /// Splitting at <see cref="LeftSubtreeLength"/> keeps every left side a complete power-of-two
    /// subtree, so its chaining value is canonical and the tree shape is unchanged.
    /// </remarks>
    [SkipLocalsInit]
    internal static unsafe void HashAllAtOnceParallel(ReadOnlySpan<byte> input, ReadOnlySpan<uint> key,
        uint flags, Span<byte> output, int maxDegreeOfParallelism = -1)
    {
        const int chunkLen = Blake3Constants.ChunkLen;

        // A cap has to reach the unit sizing as well as the loop. Capping only the loop would
        // still cut the input into ProcessorCount * 4 units and then push them through fewer
        // threads, paying the hand-off cost for parallelism that was asked not to happen.
        //
        // Clamped to the core count: a cap above it is a ceiling on threads, not a reason to cut
        // the input finer than the machine can actually run. It also keeps the arithmetic below
        // in range -- degree * 4 * 2 overflowed for very large caps, wrapping the loop bound
        // negative, growing units until the whole input was one unit, and then faulting in the
        // two-CV root fold.
        int degree = maxDegreeOfParallelism > 0
            ? Math.Min(maxDegreeOfParallelism, Environment.ProcessorCount)
            : Environment.ProcessorCount;

        // One flat fan-out, not a recursive split. Recursing with Parallel.Invoke blocks a pool
        // thread at every internal node waiting on its children, which starves the pool: it was
        // 2.4x slower than the old scheduler at 1 MB even though it balanced the leaves perfectly.
        int totalChunks = (input.Length + chunkLen - 1) / chunkLen;
        int targetUnits = Math.Max(2, degree * 4);

        // Unit size is a power of two so every unit is a canonical subtree and its chaining value
        // needs no special casing. 16 chunks is the floor: smaller units lose to hand-off cost.
        int unitChunks = 16;
        while (totalChunks / unitChunks > targetUnits * 2 && unitChunks < 4096)
        {
            unitChunks *= 2;
        }

        int unitBytes = unitChunks * chunkLen;
        int fullUnits = input.Length / unitBytes;
        int tailBytes = input.Length - fullUnits * unitBytes;
        int units = fullUnits + (tailBytes > 0 ? 1 : 0);

        var cvBuffer = ArrayPool<uint>.Shared.Rent(units * 8);
        try
        {
            fixed (byte* inputPtr = input)
            fixed (uint* keyPtr = key)
            fixed (uint* cvPtr = cvBuffer)
            {
                // Addresses travel as nint: a pointer local cannot cross a lambda boundary.
                nint inputAddr = (nint)inputPtr;
                nint keyAddr = (nint)keyPtr;
                nint cvAddr = (nint)cvPtr;
                uint f = flags;
                int ub = unitBytes;
                int uc = unitChunks;
                int fu = fullUnits;
                int tb = tailBytes;

                // Uncapped keeps the allocation-free overload; the options object only appears on
                // the path that asked for a limit.
                var options = maxDegreeOfParallelism > 0
                    ? new ParallelOptions { MaxDegreeOfParallelism = maxDegreeOfParallelism }
                    : null;

                Action<int> body = i =>
                {
                    int length = i < fu ? ub : tb;
                    SubtreeCvSerial(
                        new ReadOnlySpan<byte>((void*)(inputAddr + (nint)i * ub), length),
                        new ReadOnlySpan<uint>((void*)keyAddr, 8),
                        (ulong)i * (ulong)uc, f,
                        new Span<uint>((void*)(cvAddr + (nint)i * 8 * sizeof(uint)), 8));
                };

                if (options is null)
                {
                    Parallel.For(0, units, body);
                }
                else
                {
                    Parallel.For(0, units, options, body);
                }
            }

            // Fold the unit chaining values left to right, carrying an odd one up unchanged. That
            // pairing reproduces BLAKE3's canonical tree for any number of units, and stopping at
            // two leaves exactly the root's children so the final parent can carry ROOT.
            Span<uint> cvs = cvBuffer.AsSpan(0, units * 8);
            Span<uint> scratch = stackalloc uint[16 * 8];

            int numCvs = units;
            while (numCvs > 2)
            {
                int produced = CompressParentsGeneral(cvs, numCvs, key, flags, scratch, cvs);
                numCvs = produced;
            }

            Span<uint> parentBlock = stackalloc uint[16];
            cvs.Slice(0, 16).CopyTo(parentBlock);

            var rootOutput = new Blake3Core.Output();
            rootOutput.Init(key, parentBlock, 0, Blake3Constants.BlockLen,
                flags | Blake3Constants.Parent);
            rootOutput.RootOutputBytes(output);
        }
        finally
        {
            ArrayPool<uint>.Shared.Return(cvBuffer);
        }
    }

    /// <summary>
    /// One parent level over an arbitrary number of chaining values, written back in place.
    /// </summary>
    /// <remarks>
    /// <see cref="CompressParents"/> requires 16-CV buffers so it can run the 8-way kernel over a
    /// short tail; here the count can be far larger, so batches of eight are done through the
    /// kernel and whatever remains through scratch or scalar compression.
    /// </remarks>
    private static int CompressParentsGeneral(Span<uint> cvs, int numChildren, ReadOnlySpan<uint> key,
        uint flags, Span<uint> scratch, Span<uint> outCvs)
    {
        uint parentFlags = flags | Blake3Constants.Parent;
        int numParents = numChildren / 2;
        int p = 0;

        if (HashManyAvx2.IsSupported)
        {
            for (; p + 8 <= numParents; p += 8)
            {
                // Copy the children out first: output overlaps input when folding in place.
                cvs.Slice(p * 16, 128).CopyTo(scratch);
                HashManyAvx2.HashParents8(scratch, key, parentFlags, outCvs.Slice(p * 8, 64));
            }
        }

        for (; p < numParents; p++)
        {
            cvs.Slice(p * 16, 16).CopyTo(scratch);
            Blake3Core.CompressCv(key, scratch.Slice(0, 16), 0, Blake3Constants.BlockLen,
                parentFlags, outCvs.Slice(p * 8, 8));
        }

        if ((numChildren & 1) != 0)
        {
            cvs.Slice(numParents * 16, 8).CopyTo(scratch);
            scratch.Slice(0, 8).CopyTo(outCvs.Slice(numParents * 8, 8));
            return numParents + 1;
        }

        return numParents;
    }

    /// <summary>
    /// Reduces a subtree to one chaining value on this thread, using the wide SIMD frontier.
    /// </summary>
    [SkipLocalsInit]
    private static void SubtreeCvSerial(ReadOnlySpan<byte> input, ReadOnlySpan<uint> key,
        ulong chunkCounter, uint flags, Span<uint> outCv)
    {
        Span<uint> a = stackalloc uint[16 * 8];
        Span<uint> b = stackalloc uint[16 * 8];

        int numCvs = CompressSubtreeWide(input, key, chunkCounter, flags, a);

        bool inA = true;
        while (numCvs > 1)
        {
            numCvs = inA
                ? CompressParents(a, numCvs, key, flags, b)
                : CompressParents(b, numCvs, key, flags, a);
            inA = !inA;
        }

        (inA ? a : b).Slice(0, 8).CopyTo(outCv);
    }

    /// <summary>
    /// Reduces the whole input to exactly two chaining values, returned as one 16-word parent block.
    /// </summary>
    [SkipLocalsInit]
    private static void CompressSubtreeToParentBlock(ReadOnlySpan<byte> input, ReadOnlySpan<uint> key,
        uint flags, Span<uint> parentBlock)
    {
        // Two buffers ping-ponged rather than allocating per level. CompressSubtreeWide returns at
        // most 8 CVs, so this halves 8 -> 4 -> 2 and never runs more than twice.
        Span<uint> a = stackalloc uint[16 * 8];
        Span<uint> b = stackalloc uint[16 * 8];

        int numCvs = CompressSubtreeWide(input, key, 0, flags, a);

        bool inA = true;
        while (numCvs > 2)
        {
            numCvs = inA
                ? CompressParents(a, numCvs, key, flags, b)
                : CompressParents(b, numCvs, key, flags, a);
            inA = !inA;
        }

        (inA ? a : b).Slice(0, 16).CopyTo(parentBlock);
    }

    /// <summary>
    /// Hashes a subtree and returns however many chaining values remain at its frontier, rather
    /// than collapsing to one. Keeping the frontier wide is what allows parents to be compressed
    /// in batches instead of individually.
    /// </summary>
    /// <returns>Number of CVs written to <paramref name="outCvs"/>; never more than 8.</returns>
    [SkipLocalsInit]
    private static int CompressSubtreeWide(ReadOnlySpan<byte> input, ReadOnlySpan<uint> key,
        ulong chunkCounter, uint flags, Span<uint> outCvs)
    {
        // Floor of 2: with a scalar-only kernel the leaf case would return a single CV, and a
        // parent node needs two children.
        int degree = Math.Max(SimdDegree, 2);

        if (input.Length <= Blake3Constants.ChunkLen * degree)
        {
            return HashChunks(input, key, chunkCounter, flags, outCvs);
        }

        // BLAKE3 splits a subtree at the largest power-of-two chunk count strictly below the total,
        // so the left side is always a complete subtree and its CV is canonical.
        int leftLen = LeftSubtreeLength(input.Length);
        ulong rightCounter = chunkCounter + (ulong)(leftLen / Blake3Constants.ChunkLen);

        Span<uint> children = stackalloc uint[16 * 8];
        int leftN = CompressSubtreeWide(input.Slice(0, leftLen), key, chunkCounter, flags, children);
        int rightN = CompressSubtreeWide(input.Slice(leftLen), key, rightCounter, flags,
            children.Slice(leftN * 8));

        if (leftN == 1)
        {
            // Scalar kernel: one CV per side. Hand both up so the caller can form a parent.
            children.Slice(0, 16).CopyTo(outCvs);
            return 2;
        }

        return CompressParents(children, leftN + rightN, key, flags, outCvs);
    }

    /// <summary>
    /// Hashes the chunks of a leaf-sized subtree, using the widest kernel each remainder allows.
    /// The final chunk may be partial.
    /// </summary>
    [SkipLocalsInit]
    private static int HashChunks(ReadOnlySpan<byte> input, ReadOnlySpan<uint> key,
        ulong chunkCounter, uint flags, Span<uint> cvs)
    {
        const int chunkLen = Blake3Constants.ChunkLen;

        var remaining = input;
        ulong counter = chunkCounter;
        int n = 0;

        while (remaining.Length >= chunkLen * 8 && HashManyAvx2.IsSupported)
        {
            HashManyAvx2.HashMany(remaining.Slice(0, chunkLen * 8), 8, key, counter, flags,
                cvs.Slice(n * 8, 64));
            remaining = remaining.Slice(chunkLen * 8);
            counter += 8;
            n += 8;
        }

        while (remaining.Length >= chunkLen * 4
               && (HashManyNeon.IsSupported || HashManySse41.IsSupported))
        {
            var batch = remaining.Slice(0, chunkLen * 4);
            var into = cvs.Slice(n * 8, 32);

            if (HashManyNeon.IsSupported)
                HashManyNeon.HashMany(batch, 4, key, counter, flags, into);
            else
                HashManySse41.HashMany(batch, 4, key, counter, flags, into);

            remaining = remaining.Slice(chunkLen * 4);
            counter += 4;
            n += 4;
        }

        while (remaining.Length > 0)
        {
            int take = Math.Min(chunkLen, remaining.Length);
            HashChunkCv(key, remaining.Slice(0, take), counter, flags, cvs.Slice(n * 8, 8));
            remaining = remaining.Slice(take);
            counter++;
            n++;
        }

        return n;
    }

    /// <summary>
    /// Chaining value of one chunk of any length from 1 to 1024 bytes. Unlike
    /// <see cref="Blake3Core.HashChunkCv"/> this tolerates a partial final block.
    /// </summary>
    [SkipLocalsInit]
    private static void HashChunkCv(ReadOnlySpan<uint> key, ReadOnlySpan<byte> chunk,
        ulong chunkCounter, uint flags, Span<uint> cv)
    {
        key[..8].CopyTo(cv);

        int pos = 0;
        int blocksCompressed = 0;

        while (pos + Blake3Constants.BlockLen < chunk.Length)
        {
            ReadOnlySpan<uint> blockWords =
                MemoryMarshal.Cast<byte, uint>(chunk.Slice(pos, Blake3Constants.BlockLen));
            uint blockFlags = flags | (blocksCompressed == 0 ? Blake3Constants.ChunkStart : 0u);
            Blake3Core.CompressCv(cv, blockWords, chunkCounter, Blake3Constants.BlockLen, blockFlags, cv);
            pos += Blake3Constants.BlockLen;
            blocksCompressed++;
        }

        Span<byte> lastBlock = stackalloc byte[Blake3Constants.BlockLen];
        lastBlock.Clear();
        int lastLen = chunk.Length - pos;
        chunk.Slice(pos, lastLen).CopyTo(lastBlock);

        uint lastFlags = flags | Blake3Constants.ChunkEnd
                         | (blocksCompressed == 0 ? Blake3Constants.ChunkStart : 0u);
        Blake3Core.CompressCv(cv, MemoryMarshal.Cast<byte, uint>(lastBlock), chunkCounter,
            (uint)lastLen, lastFlags, cv);
    }

    /// <summary>
    /// Compresses a level of the tree, 8 parents per kernel call where possible. An odd child is
    /// carried up unchanged, as the reference implementation does.
    /// </summary>
    private static int CompressParents(ReadOnlySpan<uint> children, int numChildren,
        ReadOnlySpan<uint> key, uint flags, Span<uint> outCvs)
    {
        uint parentFlags = flags | Blake3Constants.Parent;
        int numParents = numChildren / 2;
        int p = 0;

        if (HashManyAvx2.IsSupported)
        {
            for (; p + 8 <= numParents; p += 8)
            {
                HashManyAvx2.HashParents8(children.Slice(p * 16, 128), key, parentFlags,
                    outCvs.Slice(p * 8, 64));
            }

            // A tail of 3..7 parents still goes through the 8-way kernel, computing lanes we then
            // ignore. One vector compression beats three or more scalar ones even with the waste,
            // and this is the common case: at 8 KB the tree has 8 children, so four parents, which
            // previously fell through to scalar and cost more than the chunk hashing it followed.
            //
            // Safe without staging because every caller's CV buffer holds 16 CVs (128 words) and a
            // tail can only occur at p == 0, since numParents never exceeds 8 here.
            if (numParents - p >= 3 && children.Length >= 128 && outCvs.Length >= 64)
            {
                HashManyAvx2.HashParents8(children.Slice(p * 16, 128), key, parentFlags,
                    outCvs.Slice(p * 8, 64));
                p = numParents;
            }
        }

        for (; p < numParents; p++)
        {
            Blake3Core.CompressCv(key, children.Slice(p * 16, 16), 0, Blake3Constants.BlockLen,
                parentFlags, outCvs.Slice(p * 8, 8));
        }

        if ((numChildren & 1) != 0)
        {
            children.Slice(numParents * 16, 8).CopyTo(outCvs.Slice(numParents * 8, 8));
            return numParents + 1;
        }

        return numParents;
    }

    /// <summary>
    /// Length of the left subtree: the largest power-of-two number of chunks strictly less than
    /// the input's chunk count, in bytes.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int LeftSubtreeLength(int inputLength)
    {
        // -1 so an exact power-of-two chunk count splits in half rather than putting everything left.
        int fullChunks = (inputLength - 1) / Blake3Constants.ChunkLen;
        return RoundDownToPowerOfTwo(fullChunks) * Blake3Constants.ChunkLen;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int RoundDownToPowerOfTwo(int value)
    {
        int result = 1;
        while (result <= value >> 1)
        {
            result <<= 1;
        }
        return result;
    }
}
