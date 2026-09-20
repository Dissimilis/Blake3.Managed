using System.Buffers;
using System.Buffers.Binary;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Threading.Tasks;

namespace Blake3.Managed.Internal;

internal static class Blake3Core
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void CompressInPlace(ReadOnlySpan<uint> cv, ReadOnlySpan<uint> block,
        ulong counter, uint blockLen, uint flags,
        Span<uint> output)
    {
        if (CompressSse41.IsSupported)
        {
            CompressSse41.Compress(cv, block, counter, blockLen, flags, output);
        }
        else
        {
            CompressScalar.Compress(cv, block, counter, blockLen, flags, output);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void CompressCv(ReadOnlySpan<uint> cv, ReadOnlySpan<uint> block,
        ulong counter, uint blockLen, uint flags,
        Span<uint> chainingValue)
    {
        if (CompressSse41.IsSupported)
        {
            CompressSse41.CompressChainingValue(cv, block, counter, blockLen, flags, chainingValue);
        }
        else
        {
            CompressScalar.CompressChainingValue(cv, block, counter, blockLen, flags, chainingValue);
        }
    }

    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void HashChunkCv(ReadOnlySpan<uint> key, ReadOnlySpan<byte> chunk, ulong chunkCounter, uint flags, Span<uint> cv)
    {
        key[..8].CopyTo(cv);

        for (int blockIdx = 0; blockIdx < 15; blockIdx++)
        {
            ReadOnlySpan<uint> blockWords = MemoryMarshal.Cast<byte, uint>(chunk.Slice(blockIdx * 64, 64));
            uint blockFlags = flags | (blockIdx == 0 ? Blake3Constants.ChunkStart : 0u);
            CompressCv(cv, blockWords, chunkCounter, Blake3Constants.BlockLen, blockFlags, cv);
        }

        ReadOnlySpan<uint> lastBlockWords = MemoryMarshal.Cast<byte, uint>(chunk.Slice(15 * 64, 64));
        CompressCv(cv, lastBlockWords, chunkCounter, Blake3Constants.BlockLen,
            flags | Blake3Constants.ChunkEnd, cv);
    }

    /// <summary>
    /// Hashes a single chunk straight into a 32-byte destination.
    /// </summary>
    /// <remarks>
    /// The default one-shot case. The general path computes all 16 root words, copies 32 of the
    /// 64 resulting bytes into a scratch buffer, and copies them again into the returned Hash.
    /// None of that is needed for a 32-byte digest: the root output's first eight words are
    /// exactly the chaining value, so CompressCv produces the answer directly, and it can be
    /// written into the caller's buffer with no intermediate copy at all.
    /// </remarks>
    [SkipLocalsInit]
    internal static void HashOneChunkRoot32(ReadOnlySpan<uint> key, ulong chunkCounter, uint flags,
        Span<byte> output, ReadOnlySpan<byte> input)
    {
        if (input.Length <= Blake3Constants.BlockLen)
        {
            // One block, which is the overwhelmingly common short-input case.
            if (CompressSse41.IsSupported && flags == 0 && chunkCounter == 0)
            {
                // Default unkeyed hash: every compression state row except the message and the
                // block length is a compile-time constant, and the message is assembled in
                // registers, so this touches no scratch memory at all.
                CompressSse41.CompressRootIvSingleBlock(input, MemoryMarshal.Cast<byte, uint>(output));
                return;
            }

            // Keyed, derive-key, or no SSE: the generic compressor needs a padded block. No
            // compression has happened yet, so the chaining value is still the key and is passed
            // straight through rather than copied into a scratch buffer.
            uint singleFlags = flags | Blake3Constants.ChunkStart
                               | Blake3Constants.ChunkEnd | Blake3Constants.Root;

            if (input.Length == Blake3Constants.BlockLen)
            {
                StoreRoot32(key, input, chunkCounter, (uint)input.Length, singleFlags, output);
                return;
            }

            // Scratch is declared here, not at the top of the method: the fast paths above never
            // touch it, and hoisting it made every short hash carry the frame for a buffer it
            // did not use.
            Span<byte> padded = stackalloc byte[Blake3Constants.BlockLen];
            ZeroBlock(padded);
            CopyUpTo64(input, padded);
            StoreRoot32(key, padded, chunkCounter, (uint)input.Length, singleFlags, output);
            return;
        }

        if (CompressSse41.IsSupported && flags == 0 && chunkCounter == 0)
        {
            // Multi-block default unkeyed chunk: the fused loop keeps the chaining value in
            // registers across every block instead of round-tripping it through memory. Placed
            // after the single-block case, which has a cheaper constant-state path of its own.
            CompressSse41.HashChunkRoot32Iv(input, MemoryMarshal.Cast<byte, uint>(output));
            return;
        }

        Span<uint> cv = stackalloc uint[8];
        key[..8].CopyTo(cv);

        int pos = 0;
        int blocksCompressed = 0;

        while (pos + Blake3Constants.BlockLen < input.Length)
        {
            ReadOnlySpan<uint> blockWords = MemoryMarshal.Cast<byte, uint>(input.Slice(pos, Blake3Constants.BlockLen));
            uint blockFlags = flags | (blocksCompressed == 0 ? Blake3Constants.ChunkStart : 0u);
            CompressCv(cv, blockWords, chunkCounter, Blake3Constants.BlockLen, blockFlags, cv);
            pos += Blake3Constants.BlockLen;
            blocksCompressed++;
        }

        // Same again for the tail: an input that is an exact multiple of the block size ends on a
        // full block, which needs neither zeroing nor copying.
        int remaining = input.Length - pos;
        uint tailFlags = flags | Blake3Constants.ChunkEnd | Blake3Constants.Root;

        if (remaining == Blake3Constants.BlockLen)
        {
            StoreRoot32(cv, input.Slice(pos), chunkCounter, (uint)remaining, tailFlags, output);
            return;
        }

        Span<byte> lastBlock = stackalloc byte[Blake3Constants.BlockLen];
        ZeroBlock(lastBlock);
        CopyUpTo64(input.Slice(pos, remaining), lastBlock);
        StoreRoot32(cv, lastBlock, chunkCounter, (uint)remaining, tailFlags, output);
    }

    /// <summary>
    /// Copies 1..63 bytes without leaving the caller's code.
    /// </summary>
    /// <remarks>
    /// Span.CopyTo bottoms out in Buffer.Memmove, an out-of-line call with its own size dispatch.
    /// For a 4-byte hash that call is a visible share of the total. The overlapping-load ladder
    /// below is the standard small-copy shape and compiles to a couple of instructions per case.
    /// The destination is already zeroed, so writing past the source length is not a concern --
    /// but every write here stays within the copied region anyway.
    /// </remarks>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void CopyUpTo64(ReadOnlySpan<byte> src, Span<byte> dst)
    {
        int n = src.Length;
        ref byte s = ref MemoryMarshal.GetReference(src);
        ref byte d = ref MemoryMarshal.GetReference(dst);

        if (n >= 32)
        {
            Unsafe.WriteUnaligned(ref d, Unsafe.ReadUnaligned<Vector128<byte>>(ref s));
            Unsafe.WriteUnaligned(ref Unsafe.Add(ref d, 16), Unsafe.ReadUnaligned<Vector128<byte>>(ref Unsafe.Add(ref s, 16)));
            Unsafe.WriteUnaligned(ref Unsafe.Add(ref d, n - 32), Unsafe.ReadUnaligned<Vector128<byte>>(ref Unsafe.Add(ref s, n - 32)));
            Unsafe.WriteUnaligned(ref Unsafe.Add(ref d, n - 16), Unsafe.ReadUnaligned<Vector128<byte>>(ref Unsafe.Add(ref s, n - 16)));
        }
        else if (n >= 16)
        {
            Unsafe.WriteUnaligned(ref d, Unsafe.ReadUnaligned<Vector128<byte>>(ref s));
            Unsafe.WriteUnaligned(ref Unsafe.Add(ref d, n - 16), Unsafe.ReadUnaligned<Vector128<byte>>(ref Unsafe.Add(ref s, n - 16)));
        }
        else if (n >= 8)
        {
            Unsafe.WriteUnaligned(ref d, Unsafe.ReadUnaligned<ulong>(ref s));
            Unsafe.WriteUnaligned(ref Unsafe.Add(ref d, n - 8), Unsafe.ReadUnaligned<ulong>(ref Unsafe.Add(ref s, n - 8)));
        }
        else if (n >= 4)
        {
            Unsafe.WriteUnaligned(ref d, Unsafe.ReadUnaligned<uint>(ref s));
            Unsafe.WriteUnaligned(ref Unsafe.Add(ref d, n - 4), Unsafe.ReadUnaligned<uint>(ref Unsafe.Add(ref s, n - 4)));
        }
        else
        {
            for (int i = 0; i < n; i++)
            {
                Unsafe.Add(ref d, i) = Unsafe.Add(ref s, i);
            }
        }
    }

    /// <summary>
    /// Zeroes a 64-byte block with inline stores.
    /// </summary>
    /// <remarks>
    /// Span.Clear() is an out-of-line memset call. That is the right choice for large buffers and
    /// the wrong one here: the whole short-input hash is under 100 ns, so a call plus its
    /// size-dispatch costs a visible fraction of it. Four 128-bit stores have no call at all.
    /// </remarks>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ZeroBlock(Span<byte> block)
    {
        ref byte dst = ref MemoryMarshal.GetReference(block);
        Unsafe.WriteUnaligned(ref dst, default(Vector128<byte>));
        Unsafe.WriteUnaligned(ref Unsafe.Add(ref dst, 16), default(Vector128<byte>));
        Unsafe.WriteUnaligned(ref Unsafe.Add(ref dst, 32), default(Vector128<byte>));
        Unsafe.WriteUnaligned(ref Unsafe.Add(ref dst, 48), default(Vector128<byte>));
    }

    /// <summary>
    /// Compresses the final block and writes the 32 root bytes, in little-endian order.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void StoreRoot32(ReadOnlySpan<uint> cv, ReadOnlySpan<byte> block,
        ulong chunkCounter, uint blockLen, uint flags, Span<byte> output)
    {
        ReadOnlySpan<uint> blockWords = MemoryMarshal.Cast<byte, uint>(block);

        if (BitConverter.IsLittleEndian)
        {
            // Straight into the destination; the words are already in the right byte order.
            CompressCv(cv, blockWords, chunkCounter, blockLen, flags,
                MemoryMarshal.Cast<byte, uint>(output));
            return;
        }

        Span<uint> words = stackalloc uint[8];
        CompressCv(cv, blockWords, chunkCounter, blockLen, flags, words);
        for (int i = 0; i < 8; i++)
        {
            BinaryPrimitives.WriteUInt32LittleEndian(output.Slice(i * 4), words[i]);
        }
    }

    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void HashOneChunk(ReadOnlySpan<uint> key, ulong chunkCounter, uint flags, Span<byte> output, ReadOnlySpan<byte> input)
    {
        Span<uint> cv = stackalloc uint[8];
        key[..8].CopyTo(cv);

        int pos = 0;
        int blocksCompressed = 0;

        while (pos + Blake3Constants.BlockLen < input.Length)
        {
            ReadOnlySpan<uint> blockWords = MemoryMarshal.Cast<byte, uint>(input.Slice(pos, Blake3Constants.BlockLen));
            uint blockFlags = flags | (blocksCompressed == 0 ? Blake3Constants.ChunkStart : 0u);
            CompressCv(cv, blockWords, chunkCounter, Blake3Constants.BlockLen, blockFlags, cv);
            pos += Blake3Constants.BlockLen;
            blocksCompressed++;
        }

        Span<byte> lastBlock = stackalloc byte[Blake3Constants.BlockLen];
        lastBlock.Clear();
        int remaining = input.Length - pos;
        if (remaining > 0)
            input.Slice(pos, remaining).CopyTo(lastBlock);

        ReadOnlySpan<uint> lastBlockWords = MemoryMarshal.Cast<byte, uint>(lastBlock);
        uint lastFlags = flags | Blake3Constants.ChunkEnd | Blake3Constants.Root
                         | (blocksCompressed == 0 ? Blake3Constants.ChunkStart : 0u);

        Span<uint> state = stackalloc uint[16];
        CompressInPlace(cv, lastBlockWords, chunkCounter, (uint)remaining, lastFlags, state);

        Span<byte> stateBytes = MemoryMarshal.AsBytes(state);
        int toCopy = Math.Min(output.Length, 64);
        stateBytes.Slice(0, toCopy).CopyTo(output);
    }

    /// <summary>
    /// Entry point for inputs above the serial tree's range.
    /// </summary>
    [MethodImpl(MethodImplOptions.NoInlining)]
    internal static void HashLargeParallel(ReadOnlySpan<byte> input, ReadOnlySpan<uint> key,
        uint flags, Span<byte> output, int maxDegreeOfParallelism = -1)
    {
        Blake3Tree.HashAllAtOnceParallel(input, key, flags, output, maxDegreeOfParallelism);
    }

    /// <summary>
    /// Reduces <paramref name="numCvs"/> chaining values (a power of two, stored
    /// contiguously in <paramref name="cvs"/>) to a single CV at cvs[0..8] by hashing
    /// parent nodes level by level, 8 parents at a time with AVX2 where possible.
    /// </summary>
    [SkipLocalsInit]
    internal static void ReduceCvs(Span<uint> cvs, int numCvs, ReadOnlySpan<uint> key, uint flags)
    {
        uint parentFlags = flags | Blake3Constants.Parent;
        Span<uint> block = stackalloc uint[16];
        while (numCvs > 1)
        {
            int numParents = numCvs >> 1;
            int p = 0;
            if (HashManyAvx2.IsSupported)
            {
                for (; p + 8 <= numParents; p += 8)
                {
                    HashParentsInPlace(cvs, p, key, parentFlags);
                }

                // A tail of 3..7 parents through the 8-way kernel, ignoring the spare lanes, as
                // the one-shot tree does. The kernel loads all its input before storing, and the
                // spare lanes read and write only words beyond the live CVs of this level.
                if (numParents - p >= 3 && cvs.Length >= p * 16 + 128)
                {
                    HashParentsInPlace(cvs, p, key, parentFlags);
                    p = numParents;
                }
            }
            for (; p < numParents; p++)
            {
                // Copy the block out first: the output (cvs[p*8..]) can overlap it.
                cvs.Slice(p * 16, 16).CopyTo(block);
                CompressCv(key, block, 0, Blake3Constants.BlockLen, parentFlags, cvs.Slice(p * 8, 8));
            }
            numCvs = numParents;
        }
    }

    // Non-inlined wrapper so ReduceCvs itself stays small; keeps the AVX2 codegen isolated.
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void HashParentsInPlace(Span<uint> cvs, int p, ReadOnlySpan<uint> key, uint parentFlags)
    {
        HashManyAvx2.HashParents8(cvs.Slice(p * 16, 128), key, parentFlags, cvs.Slice(p * 8, 64));
    }

    public static void WordsFromLeBytes(ReadOnlySpan<byte> bytes, Span<uint> words)
    {
        if (BitConverter.IsLittleEndian)
        {
            MemoryMarshal.Cast<byte, uint>(bytes.Slice(0, 32)).CopyTo(words);
        }
        else
        {
            for (int i = 0; i < 8; i++)
            {
                words[i] = BinaryPrimitives.ReadUInt32LittleEndian(bytes.Slice(i * 4));
            }
        }
    }

    internal struct ChunkState
    {
        private unsafe fixed uint _cv[8];
        internal readonly ulong ChunkCounter;
        private unsafe fixed byte _block[Blake3Constants.BlockLen];
        private byte _blockLen;
        private byte _blocksCompressed;
        private uint _flags;

        [SkipLocalsInit]
        public ChunkState(ReadOnlySpan<uint> key, ulong chunkCounter, uint flags)
        {
            key.Slice(0, 8).CopyTo(CvSpan);
            ChunkCounter = chunkCounter;
            _blockLen = 0;
            _blocksCompressed = 0;
            _flags = flags;
        }

        public int Len => Blake3Constants.BlockLen * _blocksCompressed + _blockLen;

        private unsafe Span<uint> CvSpan
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                fixed (uint* p = _cv) return new Span<uint>(p, 8);
            }
        }

        private unsafe Span<byte> BlockSpan
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                fixed (byte* p = _block) return new Span<byte>(p, Blake3Constants.BlockLen);
            }
        }

        private uint StartFlag => _blocksCompressed == 0 ? Blake3Constants.ChunkStart : 0;

        [SkipLocalsInit]
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public void Update(ReadOnlySpan<byte> input)
        {
            var remaining = input;
            while (remaining.Length > 0)
            {
                if (_blockLen == Blake3Constants.BlockLen)
                {
                    // On LE, reinterpret block bytes as uint words
                    ReadOnlySpan<uint> blockWords = MemoryMarshal.Cast<byte, uint>(BlockSpan);

                    CompressCv(CvSpan, blockWords, ChunkCounter,
                        Blake3Constants.BlockLen, _flags | StartFlag, CvSpan);

                    _blocksCompressed++;
                    _blockLen = 0;
                }

                int want = Blake3Constants.BlockLen - _blockLen;
                int take = Math.Min(want, remaining.Length);
                remaining.Slice(0, take).CopyTo(BlockSpan.Slice(_blockLen));
                _blockLen += (byte)take;
                remaining = remaining.Slice(take);
            }
        }

        [SkipLocalsInit]
        public Output CreateOutput()
        {
            if (_blockLen < Blake3Constants.BlockLen)
                BlockSpan.Slice(_blockLen).Clear();

            // On LE (always true on x86), reinterpret block bytes directly as uint words
            ReadOnlySpan<uint> blockWords = MemoryMarshal.Cast<byte, uint>(BlockSpan);

            uint outputFlags = _flags | StartFlag | Blake3Constants.ChunkEnd;

            var output = new Output();
            output.Init(CvSpan, blockWords, ChunkCounter, (uint)_blockLen, outputFlags);
            return output;
        }
    }

    internal struct Output
    {
        private unsafe fixed uint _inputCv[8];
        private unsafe fixed uint _block[16];
        private ulong _counter;
        private uint _blockLen;
        private uint _flags;

        public unsafe void Init(ReadOnlySpan<uint> inputCv, ReadOnlySpan<uint> block,
            ulong counter, uint blockLen, uint flags)
        {
            fixed (uint* p = _inputCv) inputCv.Slice(0, 8).CopyTo(new Span<uint>(p, 8));
            fixed (uint* p = _block) block.Slice(0, 16).CopyTo(new Span<uint>(p, 16));
            _counter = counter;
            _blockLen = blockLen;
            _flags = flags;
        }

        private unsafe ReadOnlySpan<uint> InputCvSpan
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                fixed (uint* p = _inputCv) return new ReadOnlySpan<uint>(p, 8);
            }
        }

        private unsafe ReadOnlySpan<uint> BlockSpan
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                fixed (uint* p = _block) return new ReadOnlySpan<uint>(p, 16);
            }
        }

        [SkipLocalsInit]
        public void ChainingValue(Span<uint> cv)
        {
            CompressCv(InputCvSpan, BlockSpan, _counter, _blockLen, _flags, cv);
        }

        [SkipLocalsInit]
        public void RootOutputBytes(Span<byte> output)
        {
            RootOutputBytesAt(0, output);
        }

        [SkipLocalsInit]
        public void RootOutputBytesAt(ulong seekOffset, Span<byte> output)
        {
            int outputLen = output.Length;
            int pos = 0;
            ulong blockCounter = seekOffset / 64;
            int byteOffset = (int)(seekOffset % 64);

            Span<uint> state = stackalloc uint[16];

            // Align a seek into the middle of a block before emitting complete SIMD batches.
            if (byteOffset != 0 && outputLen > 0)
            {
                CompressInPlace(InputCvSpan, BlockSpan, blockCounter, _blockLen,
                    _flags | Blake3Constants.Root, state);
                int take = Math.Min(64 - byteOffset, outputLen);
                MemoryMarshal.AsBytes(state).Slice(byteOffset, take).CopyTo(output);
                pos = take;
                blockCounter++;
            }

            if (OutputManyAvx2.IsSupported && outputLen - pos >= 512)
            {
                int batchBytes = (outputLen - pos) & ~511;
                OutputManyAvx2.HashOutput(InputCvSpan, BlockSpan, blockCounter, _blockLen,
                    _flags, output.Slice(pos, batchBytes));
                pos += batchBytes;
                blockCounter += (ulong)(batchBytes / 64);
            }

            // 64 bytes short of a full batch, but more than one block left: run the same
            // eight-lane kernel once into a scratch buffer and keep the prefix. The lanes
            // differ only in the output counter, so a batch costs one compression's worth of
            // latency whatever the block count, while the scalar loop below pays one
            // compression per block. Two blocks is the break-even point -- a single leftover
            // block is cheaper compressed directly than through a wasted eight-lane batch.
            //
            // Out of line, because the 512-byte scratch buffer would otherwise sit in this
            // method's frame on every call, including the large-output path that never reaches
            // this branch. Keeping it here measured 0.3% slower at 8 KB and 128 KB of output.
            if (OutputManyAvx2.IsSupported && outputLen - pos >= 2 * Blake3Constants.BlockLen)
            {
                int take = outputLen - pos;
                EmitPartialBatch(blockCounter, output.Slice(pos, take));
                pos += take;
                blockCounter += (ulong)(take / Blake3Constants.BlockLen);
            }

            while (pos < outputLen)
            {
                CompressInPlace(InputCvSpan, BlockSpan, blockCounter, _blockLen,
                    _flags | Blake3Constants.Root, state);

                Span<byte> stateBytes = MemoryMarshal.AsBytes(state);

                int needed = outputLen - pos;
                int toCopy = Math.Min(64, needed);

                stateBytes.Slice(0, toCopy).CopyTo(output.Slice(pos));
                pos += toCopy;
                blockCounter++;
            }
        }

        /// <summary>
        /// Emits 2..7 root output blocks with one eight-lane batch, keeping the prefix the
        /// caller asked for.
        /// </summary>
        [SkipLocalsInit]
        [MethodImpl(MethodImplOptions.NoInlining)]
        private void EmitPartialBatch(ulong blockCounter, Span<byte> destination)
        {
            Span<byte> batch = stackalloc byte[512];
            OutputManyAvx2.HashOutput(InputCvSpan, BlockSpan, blockCounter, _blockLen,
                _flags, batch);
            batch.Slice(0, destination.Length).CopyTo(destination);
        }
    }

    /// <summary>
    /// Fan-out for <see cref="HasherState.UpdateWithJoin"/>: items below
    /// <c>subtrees</c> are whole 64-chunk subtrees reduced to one CV, the rest are
    /// eight-chunk batches whose CVs are merged by the caller.
    /// </summary>
    private sealed unsafe class JoinJob : Blake3Tree.FlatJob
    {
        private readonly byte* _input;
        private readonly uint* _key;
        private readonly uint[] _cvBuffer;
        private readonly ulong _startCounter;
        private readonly uint _flags;
        private readonly int _subtrees;

        public JoinJob(int items, byte* input, uint* key, uint[] cvBuffer,
            ulong startCounter, uint flags, int subtrees)
            : base(items)
        {
            _input = input;
            _key = key;
            _cvBuffer = cvBuffer;
            _startCounter = startCounter;
            _flags = flags;
            _subtrees = subtrees;
        }

        protected override void Process(int item)
        {
            const int chunkLen = Blake3Constants.ChunkLen;
            const int subtreeChunks = 64;
            const int subtreeLen = subtreeChunks * chunkLen;

            var key = new ReadOnlySpan<uint>(_key, 8);

            if (item < _subtrees)
            {
                byte* subtreeBase = _input + (long)item * subtreeLen;
                ulong counter = _startCounter + (ulong)item * subtreeChunks;
                Span<uint> cvs = stackalloc uint[subtreeChunks * 8];

                for (int b = 0; b < subtreeChunks / 8; b++)
                {
                    HashManyAvx2.HashMany(
                        new ReadOnlySpan<byte>(subtreeBase + b * 8 * chunkLen, 8 * chunkLen),
                        8, key, counter + (ulong)(b * 8), _flags,
                        cvs.Slice(b * 64, 64));
                }

                ReduceCvs(cvs, subtreeChunks, key, _flags);
                cvs.Slice(0, 8).CopyTo(_cvBuffer.AsSpan(item * 8, 8));
            }
            else
            {
                int batch = item - _subtrees;
                long offset = (long)_subtrees * subtreeLen + (long)batch * 8 * chunkLen;
                HashManyAvx2.HashMany(
                    new ReadOnlySpan<byte>(_input + offset, 8 * chunkLen),
                    8, key, _startCounter + (ulong)(_subtrees * subtreeChunks + batch * 8),
                    _flags, _cvBuffer.AsSpan(_subtrees * 8 + batch * 64, 64));
            }
        }
    }

    internal struct HasherState
    {
        private unsafe fixed uint _key[8];
        private unsafe fixed uint _cvStack[Blake3Constants.MaxDepth * 8];
        private byte _cvStackLen;
        private ChunkState _chunkState;
        private readonly uint _flags;

        // Index of the first chunk this state hashes. Zero for a whole input. For one piece of a
        // larger input it is the piece's first chunk, which every chunk counter is offset by,
        // while the merge bookkeeping in AddChunkCv counts chunks from the piece start: the
        // merge rule pairs subtrees by the parity of the relative count, and using the absolute
        // count would try to merge past the top of the stack at the end of an aligned piece.
        private readonly ulong _chunkBase;

        // Set when a complete chunk's CV from a SIMD batch has not yet been merged into the
        // stack, because it may turn out to be the final chunk.
        //
        // Finalize needs the last chunk's chaining value as the right child of the root parent.
        // It used to obtain that by re-running the whole chunk through _chunkState -- 16 further
        // compressions to recompute a value the SIMD kernel had already produced and discarded.
        // Deferring instead: if more input arrives the CV is merged normally, and if not,
        // Finalize consumes it directly.
        //
        // The CV lives in the first unused CV-stack slot rather than its own buffer, and its
        // chunk count is always _chunkState.ChunkCounter. Both details keep HasherState the same
        // size as before: this struct is created per hash, so growing it cost measurably more on
        // small inputs than the deferral saved.
        private bool _hasPendingCv;

        public HasherState(ReadOnlySpan<uint> key, uint flags)
            : this(key, flags, 0)
        {
        }

        /// <summary>
        /// A state whose first chunk is chunk <paramref name="startChunk"/> of a larger input.
        /// </summary>
        /// <remarks>
        /// <see cref="Finalize"/> then describes the top node of the subtree covering the bytes
        /// fed in, which is canonical when those bytes form a power-of-two number of chunks
        /// starting at a multiple of that count, or the short tail of the input.
        /// </remarks>
        public HasherState(ReadOnlySpan<uint> key, uint flags, ulong startChunk)
        {
            // The CV stack needs no clearing: slots are always written by PushCv before
            // PopCv/Finalize read them (Reset() already relies on this).
            _cvStackLen = 0;
            _flags = flags;
            _hasPendingCv = false;
            _chunkBase = startChunk;
            key[..8].CopyTo(KeySpan);
            _chunkState = new ChunkState(key, startChunk, flags);
        }

        // The slot one past the top of the stack: unused until the next PushCv, which cannot
        // happen while a CV is deferred.
        private Span<uint> PendingCvSpan
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => CvStackSpan.Slice(_cvStackLen * 8, 8);
        }

        /// <summary>
        /// Merges a deferred chunk CV into the stack. Called before any new input is consumed:
        /// once more data exists, the deferred chunk is definitely not the last one.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void FlushPendingCv()
        {
            if (!_hasPendingCv) return;

            _hasPendingCv = false;
            AddChunkCv(PendingCvSpan, _chunkState.ChunkCounter - _chunkBase);
        }

        /// <summary>
        /// Defers the CV of the final complete chunk of a SIMD batch instead of re-hashing it.
        /// </summary>
        /// <remarks>
        /// Only called after at least one sibling CV from the same batch has been merged, so
        /// the CV stack is never empty here. That matters: a deferred CV can only serve as the
        /// right child of a root *parent* node. Were the stack empty, the root would have to be
        /// the chunk itself, which needs its pre-final-block state and not just its CV.
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void DeferChunkCv(ReadOnlySpan<uint> cv)
        {
            cv.Slice(0, 8).CopyTo(PendingCvSpan);
            _hasPendingCv = true;
        }

        private unsafe Span<uint> KeySpan
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                fixed (uint* p = _key) return new Span<uint>(p, 8);
            }
        }

        private unsafe Span<uint> CvStackSpan
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                fixed (uint* p = _cvStack) return new Span<uint>(p, Blake3Constants.MaxDepth * 8);
            }
        }

        private void PushCv(ReadOnlySpan<uint> cv)
        {
            cv.Slice(0, 8).CopyTo(CvStackSpan.Slice(_cvStackLen * 8, 8));
            _cvStackLen++;
        }

        private void PopCv(Span<uint> cv)
        {
            _cvStackLen--;
            CvStackSpan.Slice(_cvStackLen * 8, 8).CopyTo(cv);
        }

        /// <summary>
        /// Pushes a new subtree CV onto the CV stack, merging completed sibling pairs.
        /// <paramref name="totalChunks"/> is the total number of *units* processed so far,
        /// where a unit is the subtree size that <paramref name="newCv"/> represents
        /// (1 chunk for ordinary adds; for a 64-chunk subtree CV pass totalChunks >> 6).
        /// </summary>
        [SkipLocalsInit]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void AddChunkCv(ReadOnlySpan<uint> newCv, ulong totalChunks)
        {
            Span<uint> parentBlock = stackalloc uint[16];
            Span<uint> rightCv = stackalloc uint[8];
            Span<uint> leftCv = stackalloc uint[8];

            newCv.Slice(0, 8).CopyTo(rightCv);

            ulong tc = totalChunks;
            while ((tc & 1) == 0)
            {
                PopCv(leftCv);
                leftCv.CopyTo(parentBlock.Slice(0, 8));
                rightCv.Slice(0, 8).CopyTo(parentBlock.Slice(8, 8));

                CompressCv(KeySpan, parentBlock, 0, Blake3Constants.BlockLen,
                    _flags | Blake3Constants.Parent, rightCv);

                tc >>= 1;
            }

            PushCv(rightCv);
        }

        /// <summary>
        /// Hashes the largest aligned power-of-two subtree of 8..64 chunks at the start of
        /// <paramref name="remaining"/> that leaves at least one byte after it, pushes its single
        /// CV, and returns the bytes consumed. The chunk counter must be a multiple of 8 and
        /// <paramref name="remaining"/> longer than 8 chunks. Merging chunk CVs one at a time
        /// costs seven scalar parent compressions per eight chunks; this batches them 8-way.
        /// </summary>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private int HashAlignedSubtree(ReadOnlySpan<byte> remaining, Span<uint> batchCvs)
        {
            const int subtreeChunks = 64;
            ulong startCounter = _chunkState.ChunkCounter;
            int treeChunks = 8;
            while (treeChunks < subtreeChunks
                   && (startCounter & (ulong)(2 * treeChunks - 1)) == 0
                   && remaining.Length > 2 * treeChunks * Blake3Constants.ChunkLen)
            {
                treeChunks *= 2;
            }

            // Single-threaded here, so the interleaved schedule that won in the serial
            // one-shot tree applies; the parallel workers keep the original kernel.
            for (int b = 0; b < treeChunks / 8; b++)
            {
                HashManyAvx2.HashManySerial(
                    remaining.Slice(b * 8 * Blake3Constants.ChunkLen, 8 * Blake3Constants.ChunkLen),
                    KeySpan, startCounter + (ulong)(b * 8), _flags,
                    batchCvs.Slice(b * 64, 64));
            }

            ReduceCvs(batchCvs, treeChunks, KeySpan, _flags);
            AddChunkCv(batchCvs.Slice(0, 8), (startCounter - _chunkBase) / (ulong)treeChunks + 1);

            _chunkState = new ChunkState(KeySpan, startCounter + (ulong)treeChunks, _flags);
            return treeChunks * Blake3Constants.ChunkLen;
        }

        [SkipLocalsInit]
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public void Update(ReadOnlySpan<byte> input)
        {
            const int subtreeChunks = 64;

            if (input.Length == 0) return;

            // New input means any deferred chunk is not the last one after all.
            FlushPendingCv();

            var remaining = input;
            Span<uint> chunkCv = stackalloc uint[8];
            Span<uint> batchCvs = HashManyAvx2.IsSupported
                                ? stackalloc uint[subtreeChunks * 8]
                                : HashManyNeon.IsSupported || HashManySse41.IsSupported
                                    ? stackalloc uint[8 * 8]
                                    : default;

            while (remaining.Length > 0)
            {
                if (_chunkState.Len == Blake3Constants.ChunkLen)
                {
                    var output = _chunkState.CreateOutput();
                    output.ChainingValue(chunkCv);

                    ulong nextChunk = _chunkState.ChunkCounter + 1;
                    AddChunkCv(chunkCv, nextChunk - _chunkBase);

                    _chunkState = new ChunkState(KeySpan, nextChunk, _flags);
                }

                // AVX2 aligned-subtree fast path: 8, 16, 32 or 64 chunks reduced to one CV with
                // 8-way parent hashing. Kept out of line so this loop's register allocation
                // does not change for the paths below.
                if (HashManyAvx2.IsSupported && _chunkState.Len == 0
                    && (_chunkState.ChunkCounter & 7) == 0
                    && remaining.Length > 8 * Blake3Constants.ChunkLen)
                {
                    int consumed = HashAlignedSubtree(remaining, batchCvs);
                    remaining = remaining.Slice(consumed);
                    continue;
                }

                // AVX2 batches of 5..8 complete chunks. Inactive lanes reread valid input,
                // so remainders need neither padding nor a narrower second kernel.
                if (HashManyAvx2.IsSupported && _chunkState.Len == 0 && remaining.Length >= Blake3Constants.ChunkLen * 5)
                {
                    ulong startCounter = _chunkState.ChunkCounter;
                    int chunks = Math.Min(8, remaining.Length / Blake3Constants.ChunkLen);

                    // The interleaved serial kernel measured 6.6% slower than this one for a lone
                    // eight-chunk batch (8 KB Update).
                    if (chunks == 8)
                        HashManyAvx2.HashMany(remaining, 8, KeySpan, startCounter, _flags, batchCvs);
                    else
                        HashManyAvx2.HashManyPartial(remaining, chunks, KeySpan, startCounter, _flags, batchCvs);

                    bool hasMore = remaining.Length > Blake3Constants.ChunkLen * chunks;
                    int cvsToAdd = hasMore ? chunks : chunks - 1;

                    for (int i = 0; i < cvsToAdd; i++)
                    {
                        AddChunkCv(batchCvs.Slice(i * 8, 8), startCounter - _chunkBase + (ulong)i + 1);
                    }

                    _chunkState = new ChunkState(KeySpan, startCounter + (ulong)chunks, _flags);

                    if (!hasMore)
                    {
                        // Preserve the final chunk for root finalization or the next update.
                        DeferChunkCv(batchCvs.Slice((chunks - 1) * 8, 8));
                    }

                    remaining = remaining.Slice(Blake3Constants.ChunkLen * chunks);
                    continue;
                }

                // 4-way fast path (NEON or SSE)
                // Three or four whole chunks as two interleaved two-chunk chains, ahead of the
                // 128-bit four-way kernel; see HashFourAvx2.
                if (HashFourAvx2.IsSupported && _chunkState.Len == 0
                    && remaining.Length >= Blake3Constants.ChunkLen * 3)
                {
                    ulong startCounter = _chunkState.ChunkCounter;
                    int chunks = Math.Min(4, remaining.Length / Blake3Constants.ChunkLen);
                    HashFourAvx2.HashFour(remaining, chunks, KeySpan, startCounter, _flags, batchCvs);

                    bool hasMore = remaining.Length > Blake3Constants.ChunkLen * chunks;
                    int cvsToAdd = hasMore ? chunks : chunks - 1;

                    for (int i = 0; i < cvsToAdd; i++)
                    {
                        AddChunkCv(batchCvs.Slice(i * 8, 8), startCounter - _chunkBase + (ulong)i + 1);
                    }

                    _chunkState = new ChunkState(KeySpan, startCounter + (ulong)chunks, _flags);

                    if (!hasMore)
                    {
                        DeferChunkCv(batchCvs.Slice((chunks - 1) * 8, 8));
                    }

                    remaining = remaining.Slice(Blake3Constants.ChunkLen * chunks);
                    continue;
                }

                if ((HashManyNeon.IsSupported || HashManySse41.IsSupported)
                    && _chunkState.Len == 0 && remaining.Length >= Blake3Constants.ChunkLen * 4)
                {
                    ulong startCounter = _chunkState.ChunkCounter;

                    if (HashManyNeon.IsSupported)
                        HashManyNeon.HashMany(remaining, 4, KeySpan, startCounter, _flags, batchCvs);
                    else
                        HashManySse41.HashMany(remaining, 4, KeySpan, startCounter, _flags, batchCvs);

                    bool hasMore = remaining.Length > Blake3Constants.ChunkLen * 4;
                    int cvsToAdd = hasMore ? 4 : 3;

                    for (int i = 0; i < cvsToAdd; i++)
                    {
                        AddChunkCv(batchCvs.Slice(i * 8, 8), startCounter - _chunkBase + (ulong)i + 1);
                    }

                    _chunkState = new ChunkState(KeySpan, startCounter + 4, _flags);

                    if (!hasMore)
                    {
                        // Input ends exactly on the batch boundary; the 4th chunk's CV is already
                        // in batchCvs. Same deferral as the 8-way path above.
                        DeferChunkCv(batchCvs.Slice(3 * 8, 8));
                    }

                    remaining = remaining.Slice(Blake3Constants.ChunkLen * 4);
                    continue;
                }

                if (HashTwoAvx2.IsSupported && _chunkState.Len == 0
                    && remaining.Length >= Blake3Constants.ChunkLen * 2)
                {
                    ulong startCounter = _chunkState.ChunkCounter;
                    HashTwoAvx2.HashTwo(remaining, KeySpan, startCounter, _flags, batchCvs);
                    AddChunkCv(batchCvs.Slice(0, 8), startCounter - _chunkBase + 1);
                    bool hasMore = remaining.Length > Blake3Constants.ChunkLen * 2;
                    if (hasMore)
                        AddChunkCv(batchCvs.Slice(8, 8), startCounter - _chunkBase + 2);
                    _chunkState = new ChunkState(KeySpan, startCounter + 2, _flags);
                    if (!hasMore)
                        DeferChunkCv(batchCvs.Slice(8, 8));
                    remaining = remaining.Slice(Blake3Constants.ChunkLen * 2);
                    continue;
                }

                if (_chunkState.Len == 0 && remaining.Length > Blake3Constants.ChunkLen)
                {
                    HashChunkCv(KeySpan, remaining.Slice(0, Blake3Constants.ChunkLen),
                        _chunkState.ChunkCounter, _flags, chunkCv);

                    ulong nextChunk = _chunkState.ChunkCounter + 1;
                    AddChunkCv(chunkCv, nextChunk - _chunkBase);
                    _chunkState = new ChunkState(KeySpan, nextChunk, _flags);
                    remaining = remaining.Slice(Blake3Constants.ChunkLen);
                    continue;
                }

                int want = Blake3Constants.ChunkLen - _chunkState.Len;
                int take = Math.Min(want, remaining.Length);
                _chunkState.Update(remaining.Slice(0, take));
                remaining = remaining.Slice(take);
            }
        }

        [SkipLocalsInit]
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public unsafe void UpdateWithJoin(ReadOnlySpan<byte> input)
        {
            const int chunkLen = Blake3Constants.ChunkLen;

            // An empty update must not commit the deferred final CV. HashAlgorithm calls
            // this when TransformFinalBlock receives an empty buffer.
            if (input.IsEmpty) return;

            // As in Update: new input means a deferred chunk is not the last one.
            if (_hasPendingCv) FlushPendingCv();
            const int subtreeChunks = 64;

            // The parallel path needs a 64-aligned chunk counter so each 64-chunk
            // subtree is canonical and can be reduced to a single CV by its worker.
            if (!HashManyAvx2.IsSupported || _chunkState.Len > 0
                || (_chunkState.ChunkCounter & (subtreeChunks - 1)) != 0)
            {
                Update(input);
                return;
            }

            // Chunks we can hash in parallel; the final chunk (possibly partial) must
            // stay in _chunkState so Finalize can produce the root.
            int usableChunks = input.Length % chunkLen == 0
                ? input.Length / chunkLen - 1
                : input.Length / chunkLen;
            int subtrees = usableChunks / subtreeChunks;
            int tailBatches = (usableChunks - subtrees * subtreeChunks) / 8;
            int items = subtrees + tailBatches;

            if (subtrees < 1 || items < 2)
            {
                Update(input);
                return;
            }

            ulong startCounter = _chunkState.ChunkCounter;
            uint flagsCopy = _flags;
            int subtreesLocal = subtrees;

            uint* keyPtr = stackalloc uint[8];
            KeySpan.CopyTo(new Span<uint>(keyPtr, 8));
            nint keyAddr = (nint)keyPtr;

            // Layout: one CV per subtree, then 8 CVs per tail batch.
            var cvBuffer = ArrayPool<uint>.Shared.Rent(subtrees * 8 + tailBatches * 64);

            try
            {
                fixed (byte* inputBase = &Unsafe.AsRef(in MemoryMarshal.GetReference(input)))
                {
                    // The same flat fan-out the one-shot tree uses, rather than Parallel.For.
                    // Parallel.For's replicating tasks wake the pool one hop at a time, which
                    // measured 18-27% slower there; this path was simply never migrated with it.
                    // RunOnPool returns only once every item has settled, so the pinned input
                    // and the stack-allocated key stay valid for exactly as long as before.
                    new JoinJob(items, inputBase, (uint*)keyAddr, cvBuffer, startCounter,
                        flagsCopy, subtreesLocal).RunOnPool(Environment.ProcessorCount);
                }

                Span<uint> tempCv = stackalloc uint[8];

                // Merge subtree CVs (units of 64 chunks; see AddChunkCv).
                for (int i = 0; i < subtrees; i++)
                {
                    cvBuffer.AsSpan(i * 8, 8).CopyTo(tempCv);
                    AddChunkCv(tempCv, ((startCounter - _chunkBase) >> 6) + (ulong)i + 1);
                }

                // Merge tail batch chunk CVs (units of 1 chunk).
                ulong tailStart = startCounter - _chunkBase + (ulong)(subtrees * subtreeChunks);
                for (int j = 0; j < tailBatches * 8; j++)
                {
                    cvBuffer.AsSpan(subtrees * 8 + j * 8, 8).CopyTo(tempCv);
                    AddChunkCv(tempCv, tailStart + (ulong)j + 1);
                }
            }
            finally
            {
                ArrayPool<uint>.Shared.Return(cvBuffer, clearArray: true);
            }

            int consumedChunks = subtrees * subtreeChunks + tailBatches * 8;
            _chunkState = new ChunkState(KeySpan, startCounter + (ulong)consumedChunks, _flags);

            // Remaining < 8 full chunks plus the reserved final chunk; Update handles it.
            Update(input.Slice(consumedChunks * chunkLen));
        }

        [SkipLocalsInit]
        public Output Finalize()
        {
            // Single-chunk input: the chunk itself is the root. Handled before anything else so
            // this path touches no stack buffers and no tree state -- it is the whole cost of
            // hashing a short string, and an extra stackalloc above it was measurable.
            if (!_hasPendingCv && _cvStackLen == 0)
            {
                return _chunkState.CreateOutput();
            }

            Span<uint> chunkCv = stackalloc uint[8];

            if (_hasPendingCv && _chunkState.Len == 0)
            {
                // The deferred chunk really was the last one. Its CV is already known, so the
                // whole chunk (16 compressions) does not have to be hashed a second time.
                // DeferChunkCv only runs with a non-empty stack, so the root is a parent node
                // and the CV alone is sufficient.
                PendingCvSpan.CopyTo(chunkCv);
            }
            else
            {
                FlushPendingCv();

                var output = _chunkState.CreateOutput();

                if (_cvStackLen == 0)
                {
                    return output;
                }

                output.ChainingValue(chunkCv);
            }

            Span<uint> parentBlock = stackalloc uint[16];
            Span<uint> leftCv = stackalloc uint[8];

            int stackIdx = (int)_cvStackLen;
            while (stackIdx > 0)
            {
                stackIdx--;
                CvStackSpan.Slice(stackIdx * 8, 8).CopyTo(leftCv);
                leftCv.CopyTo(parentBlock.Slice(0, 8));
                chunkCv.CopyTo(parentBlock.Slice(8, 8));

                if (stackIdx == 0)
                {
                    var rootOutput = new Output();
                    rootOutput.Init(KeySpan, parentBlock, 0, Blake3Constants.BlockLen,
                        _flags | Blake3Constants.Parent);
                    return rootOutput;
                }

                CompressCv(KeySpan, parentBlock, 0, Blake3Constants.BlockLen,
                    _flags | Blake3Constants.Parent, chunkCv);
            }

            var finalOutput = new Output();
            finalOutput.Init(KeySpan, parentBlock, 0, Blake3Constants.BlockLen,
                _flags | Blake3Constants.Parent);
            return finalOutput;
        }

        public void Reset()
        {
            _cvStackLen = 0;
            _hasPendingCv = false;
            _chunkState = new ChunkState(KeySpan, _chunkBase, _flags);
        }
    }
}
