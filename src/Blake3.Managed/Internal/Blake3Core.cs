using System.Buffers;
using System.Buffers.Binary;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
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

            while (pos < outputLen)
            {
                CompressInPlace(InputCvSpan, BlockSpan, blockCounter, _blockLen,
                    _flags | Blake3Constants.Root, state);

                Span<byte> stateBytes = MemoryMarshal.AsBytes(state);

                int start = pos == 0 ? byteOffset : 0;
                int available = 64 - start;
                int needed = outputLen - pos;
                int toCopy = Math.Min(available, needed);

                stateBytes.Slice(start, toCopy).CopyTo(output.Slice(pos));
                pos += toCopy;
                blockCounter++;
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

        public HasherState(ReadOnlySpan<uint> key, uint flags)
        {
            KeySpan.Clear();
            CvStackSpan.Clear();
            _cvStackLen = 0;
            _flags = flags;
            key[..8].CopyTo(KeySpan);
            _chunkState = new ChunkState(key, 0, flags);
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

        [SkipLocalsInit]
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public void Update(ReadOnlySpan<byte> input)
        {
            var remaining = input;
            Span<uint> chunkCv = stackalloc uint[8];
            Span<uint> batchCvs = HashManyAvx2.IsSupported || HashManyNeon.IsSupported
                                ? stackalloc uint[8 * 8]
                                : default;

            while (remaining.Length > 0)
            {
                if (_chunkState.Len == Blake3Constants.ChunkLen)
                {
                    var output = _chunkState.CreateOutput();
                    output.ChainingValue(chunkCv);

                    ulong totalChunks = _chunkState.ChunkCounter + 1;
                    AddChunkCv(chunkCv, totalChunks);

                    _chunkState = new ChunkState(KeySpan, totalChunks, _flags);
                }

                // AVX2 8-way fast path
                if (HashManyAvx2.IsSupported && _chunkState.Len == 0 && remaining.Length >= Blake3Constants.ChunkLen * 8)
                {
                    ulong startCounter = _chunkState.ChunkCounter;

                    HashManyAvx2.HashMany(remaining, 8, KeySpan, startCounter, _flags, batchCvs);

                    bool hasMore = remaining.Length > Blake3Constants.ChunkLen * 8;
                    int cvsToAdd = hasMore ? 8 : 7;

                    for (int i = 0; i < cvsToAdd; i++)
                    {
                        ulong totalChunks = startCounter + (ulong)i + 1;
                        AddChunkCv(batchCvs.Slice(i * 8, 8), totalChunks);
                    }

                    if (hasMore)
                    {
                        _chunkState = new ChunkState(KeySpan, startCounter + 8, _flags);
                    }
                    else
                    {
                        // Last batch: feed 8th chunk through _chunkState for correct Finalize
                        _chunkState = new ChunkState(KeySpan, startCounter + 7, _flags);
                        _chunkState.Update(remaining.Slice(Blake3Constants.ChunkLen * 7, Blake3Constants.ChunkLen));
                    }

                    remaining = remaining.Slice(Blake3Constants.ChunkLen * 8);
                    continue;
                }

                // NEON 4-way fast path
                if (HashManyNeon.IsSupported && _chunkState.Len == 0 && remaining.Length >= Blake3Constants.ChunkLen * 4)
                {
                    ulong startCounter = _chunkState.ChunkCounter;

                    HashManyNeon.HashMany(remaining, 4, KeySpan, startCounter, _flags, batchCvs);

                    bool hasMore = remaining.Length > Blake3Constants.ChunkLen * 4;
                    int cvsToAdd = hasMore ? 4 : 3;

                    for (int i = 0; i < cvsToAdd; i++)
                    {
                        ulong totalChunks = startCounter + (ulong)i + 1;
                        AddChunkCv(batchCvs.Slice(i * 8, 8), totalChunks);
                    }

                    if (hasMore)
                    {
                        _chunkState = new ChunkState(KeySpan, startCounter + 4, _flags);
                    }
                    else
                    {
                        // Last batch: feed 4th chunk through _chunkState for correct Finalize
                        _chunkState = new ChunkState(KeySpan, startCounter + 3, _flags);
                        _chunkState.Update(remaining.Slice(Blake3Constants.ChunkLen * 3, Blake3Constants.ChunkLen));
                    }

                    remaining = remaining.Slice(Blake3Constants.ChunkLen * 4);
                    continue;
                }

                if (_chunkState.Len == 0 && remaining.Length > Blake3Constants.ChunkLen)
                {
                    HashChunkCv(KeySpan, remaining.Slice(0, Blake3Constants.ChunkLen),
                        _chunkState.ChunkCounter, _flags, chunkCv);

                    ulong totalChunks = _chunkState.ChunkCounter + 1;
                    AddChunkCv(chunkCv, totalChunks);
                    _chunkState = new ChunkState(KeySpan, totalChunks, _flags);
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
            const int parallelThreshold = 64 * Blake3Constants.ChunkLen;

            if (input.Length < parallelThreshold || !HashManyAvx2.IsSupported || _chunkState.Len > 0)
            {
                Update(input);
                return;
            }

            int numFullChunks = input.Length / Blake3Constants.ChunkLen;
            int remainingBytes = input.Length % Blake3Constants.ChunkLen;

            int parallelChunks = remainingBytes > 0 ? numFullChunks : numFullChunks - 1;
            int avx2Batches = parallelChunks / 8;
            int extraChunks = parallelChunks % 8; // 0–7 chunks beyond full AVX2 batches

            if (avx2Batches == 0)
            {
                Update(input);
                return;
            }

            ulong startCounter = _chunkState.ChunkCounter;

            uint* keyPtr = stackalloc uint[8];
            KeySpan.CopyTo(new Span<uint>(keyPtr, 8));
            nint keyAddr = (nint)keyPtr;
            uint flagsCopy = _flags;

            var cvBuffer = ArrayPool<uint>.Shared.Rent(parallelChunks * 8);

            try
            {
                fixed (byte* inputBase = &Unsafe.AsRef(in MemoryMarshal.GetReference(input)))
                {
                    nint inputAddr = (nint)inputBase; // capture as nint; reconstruct inside lambda

                    int workerCount = Math.Min(Environment.ProcessorCount, avx2Batches);
                    Parallel.For(0, workerCount, workerId =>
                    {
                        unsafe
                        {
                            int batchStart = avx2Batches * workerId / workerCount;
                            int batchEnd = avx2Batches * (workerId + 1) / workerCount;
                            for (int batchIdx = batchStart; batchIdx < batchEnd; batchIdx++)
                            {
                                var batchPtr = (byte*)inputAddr + (long)(batchIdx * 8) * Blake3Constants.ChunkLen;
                                var batchInput = new ReadOnlySpan<byte>(batchPtr, 8 * Blake3Constants.ChunkLen);
                                Span<uint> batchCvs = cvBuffer.AsSpan(batchIdx * 64, 64); // 8 chunks × 8 words
                                HashManyAvx2.HashMany(batchInput, 8,
                                    new ReadOnlySpan<uint>((uint*)keyAddr, 8),
                                    startCounter + (ulong)(batchIdx * 8),
                                    flagsCopy, batchCvs);
                            }
                        }
                    });
                }

                for (int i = 0; i < extraChunks; i++)
                {
                    int chunkIdx = avx2Batches * 8 + i;
                    var chunk = input.Slice(chunkIdx * Blake3Constants.ChunkLen, Blake3Constants.ChunkLen);
                    Span<uint> cv = cvBuffer.AsSpan(chunkIdx * 8, 8);
                    HashChunkCv(KeySpan, chunk, startCounter + (ulong)chunkIdx, _flags, cv);
                }

                Span<uint> tempCv = stackalloc uint[8];
                for (int i = 0; i < parallelChunks; i++)
                {
                    cvBuffer.AsSpan(i * 8, 8).CopyTo(tempCv);
                    AddChunkCv(tempCv, startCounter + (ulong)i + 1);
                }
            }
            finally
            {
                ArrayPool<uint>.Shared.Return(cvBuffer, clearArray: true);
            }

            ulong newCounter = startCounter + (ulong)parallelChunks;
            _chunkState = new ChunkState(KeySpan, newCounter, _flags);

            int trailingStart = parallelChunks * Blake3Constants.ChunkLen;
            int trailingLen = remainingBytes > 0 ? remainingBytes : Blake3Constants.ChunkLen;
            _chunkState.Update(input.Slice(trailingStart, trailingLen));
        }

        [SkipLocalsInit]
        public Output Finalize()
        {
            var output = _chunkState.CreateOutput();

            if (_cvStackLen == 0)
            {
                return output;
            }

            Span<uint> chunkCv = stackalloc uint[8];
            output.ChainingValue(chunkCv);

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
            _chunkState = new ChunkState(KeySpan, 0, _flags);
        }
    }
}
