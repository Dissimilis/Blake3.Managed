using System.Buffers;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using System.Text;
using Blake3.Managed.Internal;

namespace Blake3.Managed;

/// <summary>
/// Hashes an input in independent pieces, on any threads and in any order, and folds the results
/// into the digest a single-threaded hash of the whole input would produce.
/// </summary>
/// <remarks>
/// This is for callers who already have their own threads and their own pieces -- a file arriving
/// over several connections, say -- and want to hash each piece as it lands. Callers who simply
/// have all the bytes should use <see cref="Hasher.Hash(ReadOnlySpan{byte})"/>, which parallelises
/// internally and is faster.
///
/// The input is divided into fixed-size pieces, numbered from zero, all of
/// <see cref="PieceSize"/> bytes except the last, which may be short. Hash piece <c>i</c> with
/// <see cref="HashSubtree"/>, keep the result at index <c>i</c>, and pass them all to
/// <see cref="Finalize(ReadOnlySpan{Blake3Subtree})"/>. Pieces may be hashed concurrently and
/// completed out of order.
///
/// <see cref="PieceSize"/> must be a power-of-two multiple of 1024 bytes. That restriction is
/// what makes a piece a canonical BLAKE3 subtree, which is what allows it to be hashed on its own
/// at all. Because the piece size is fixed here and each piece is identified by index, a caller
/// cannot produce a misaligned piece.
///
/// A piece's value depends on where it sits in the input, not only on its bytes. Passing bytes
/// under the wrong index is the one mistake this API cannot detect, and it yields a wrong digest
/// rather than an exception.
///
/// Instances are immutable once created. <see cref="HashSubtree"/> and
/// <see cref="Finalize(ReadOnlySpan{Blake3Subtree})"/> may be called concurrently; neither may
/// run concurrently with <see cref="Dispose"/>.
/// </remarks>
public sealed unsafe class Blake3SubtreeContext : IDisposable
{
    private readonly uint[] _key;
    private readonly uint _flags;
    private readonly int _pieceSize;
    private readonly int _chunksPerPiece;
    private readonly long _totalLength;
    private readonly bool _hasTotalLength;
    private readonly int _pieceCount;
    private readonly ulong _contextTag;
    private bool _disposed;

    private Blake3SubtreeContext(ReadOnlySpan<uint> key, uint flags, int pieceSize,
        long totalLength, bool hasTotalLength)
    {
        ValidatePieceSize(pieceSize);

        if (hasTotalLength)
        {
            if (totalLength < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(totalLength), totalLength,
                    "Total length must be non-negative.");
            }

            // Ceiling division written so it cannot wrap: totalLength + pieceSize - 1 overflows
            // near long.MaxValue, which produced a negative count, slipped past the check below,
            // and truncated to a small positive piece count -- a single piece then stood in for
            // the whole input.
            long pieces = totalLength == 0 ? 0 : (totalLength - 1) / pieceSize + 1;
            if (pieces > int.MaxValue)
            {
                throw new ArgumentOutOfRangeException(nameof(totalLength), totalLength,
                    "Total length divided by the piece size exceeds int.MaxValue pieces. Use a larger piece size.");
            }

            _pieceCount = (int)pieces;
        }

        _key = key.Slice(0, 8).ToArray();
        _flags = flags;
        _pieceSize = pieceSize;
        _chunksPerPiece = pieceSize / Blake3Constants.ChunkLen;
        _totalLength = totalLength;
        _hasTotalLength = hasTotalLength;
        _contextTag = ComputeContextTag(key, flags, pieceSize);
    }

    /// <summary>
    /// A context for the default hash function, for an input whose total length is not known.
    /// </summary>
    /// <param name="pieceSize">
    /// Bytes per piece. Must be a power-of-two multiple of 1024; 1024 * 1024 is a reasonable
    /// default.
    /// </param>
    /// <remarks>
    /// Without a total length the digest is exactly the same, and <c>Finalize</c> still checks
    /// everything it can see across the whole set of pieces. What is lost: the piece count and the
    /// byte range of a piece are unknown, so <see cref="TotalLength"/>, <see cref="PieceCount"/>,
    /// <see cref="GetPieceOffset"/> and <see cref="GetPieceLength"/> all throw, a wrongly sized
    /// piece is caught at <c>Finalize</c> rather than as it is hashed, and -- the one case that
    /// produces a wrong answer silently -- <b>an input truncated on a piece boundary cannot be
    /// detected.</b> It folds to a perfectly valid digest for the shorter input. Pass the total
    /// length whenever it is known.
    /// </remarks>
    /// <exception cref="ArgumentException"><paramref name="pieceSize"/> is not a power-of-two multiple of 1024.</exception>
    public static Blake3SubtreeContext Create(int pieceSize)
        => new(Blake3Constants.IV, 0, pieceSize, 0, hasTotalLength: false);

    /// <summary>
    /// A context for the default hash function, for an input of known length.
    /// </summary>
    /// <param name="pieceSize">
    /// Bytes per piece. Must be a power-of-two multiple of 1024; 1024 * 1024 is a reasonable
    /// default.
    /// </param>
    /// <param name="totalLength">
    /// Total length of the whole input in bytes. Supplying it makes the context able to report
    /// <see cref="PieceCount"/> and each piece's byte range, check every piece's length as it is
    /// hashed, and detect a missing piece at the end of the input.
    /// </param>
    /// <exception cref="ArgumentException"><paramref name="pieceSize"/> is not a power-of-two multiple of 1024.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="totalLength"/> is negative, or implies more than int.MaxValue pieces.</exception>
    public static Blake3SubtreeContext Create(int pieceSize, long totalLength)
        => new(Blake3Constants.IV, 0, pieceSize, totalLength, hasTotalLength: true);

    /// <summary>
    /// A context for the keyed hash function, for an input whose total length is not known.
    /// </summary>
    /// <param name="key">A 32 byte key. The same key is used for every piece and for the fold.</param>
    /// <param name="pieceSize">Bytes per piece. Must be a power-of-two multiple of 1024.</param>
    /// <remarks>See <see cref="Create(int)"/> for what omitting the total length costs.</remarks>
    [SkipLocalsInit]
    public static Blake3SubtreeContext CreateKeyed(ReadOnlySpan<byte> key, int pieceSize)
    {
        Span<uint> keyWords = stackalloc uint[8];
        KeyWords(key, keyWords);
        return new Blake3SubtreeContext(keyWords, Blake3Constants.KeyedHash, pieceSize, 0,
            hasTotalLength: false);
    }

    /// <summary>
    /// A context for the keyed hash function, for an input of known length.
    /// </summary>
    /// <param name="key">A 32 byte key. The same key is used for every piece and for the fold.</param>
    /// <param name="pieceSize">Bytes per piece. Must be a power-of-two multiple of 1024.</param>
    /// <param name="totalLength">Total length of the whole input in bytes.</param>
    [SkipLocalsInit]
    public static Blake3SubtreeContext CreateKeyed(ReadOnlySpan<byte> key, int pieceSize,
        long totalLength)
    {
        Span<uint> keyWords = stackalloc uint[8];
        KeyWords(key, keyWords);
        return new Blake3SubtreeContext(keyWords, Blake3Constants.KeyedHash, pieceSize, totalLength,
            hasTotalLength: true);
    }

    /// <summary>
    /// A context for the key derivation function, for an input whose total length is not known.
    /// </summary>
    /// <remarks>See <see cref="Create(int)"/> for what omitting the total length costs.</remarks>
    public static Blake3SubtreeContext CreateDeriveKey(string context, int pieceSize)
        => CreateDeriveKey(Encoding.UTF8.GetBytes(context), pieceSize);

    /// <summary>
    /// A context for the key derivation function, for an input of known length.
    /// </summary>
    public static Blake3SubtreeContext CreateDeriveKey(string context, int pieceSize, long totalLength)
        => CreateDeriveKey(Encoding.UTF8.GetBytes(context), pieceSize, totalLength);

    /// <summary>
    /// A context for the key derivation function, for an input whose total length is not known.
    /// </summary>
    /// <remarks>See <see cref="Create(int)"/> for what omitting the total length costs.</remarks>
    [SkipLocalsInit]
    public static Blake3SubtreeContext CreateDeriveKey(ReadOnlySpan<byte> context, int pieceSize)
    {
        Span<uint> keyWords = stackalloc uint[8];
        DeriveKeyWords(context, keyWords);
        return new Blake3SubtreeContext(keyWords, Blake3Constants.DeriveKeyMaterial, pieceSize, 0,
            hasTotalLength: false);
    }

    /// <summary>
    /// A context for the key derivation function, for an input of known length.
    /// </summary>
    [SkipLocalsInit]
    public static Blake3SubtreeContext CreateDeriveKey(ReadOnlySpan<byte> context, int pieceSize,
        long totalLength)
    {
        Span<uint> keyWords = stackalloc uint[8];
        DeriveKeyWords(context, keyWords);
        return new Blake3SubtreeContext(keyWords, Blake3Constants.DeriveKeyMaterial, pieceSize,
            totalLength, hasTotalLength: true);
    }

    /// <summary>
    /// Bytes per piece. Every piece is this long except the last, which may be shorter.
    /// </summary>
    public int PieceSize => _pieceSize;

    /// <summary>
    /// True when this context was given the total length, and so can report
    /// <see cref="TotalLength"/>, <see cref="PieceCount"/>, <see cref="GetPieceOffset"/> and
    /// <see cref="GetPieceLength"/>.
    /// </summary>
    public bool HasTotalLength => _hasTotalLength;

    /// <summary>
    /// Total length of the whole input in bytes.
    /// </summary>
    /// <exception cref="InvalidOperationException">The context was created without a total length.</exception>
    public long TotalLength
    {
        get
        {
            if (!_hasTotalLength) ThrowNoTotalLength();
            return _totalLength;
        }
    }

    /// <summary>
    /// How many pieces the input divides into. Need not be a power of two.
    /// </summary>
    /// <exception cref="InvalidOperationException">The context was created without a total length.</exception>
    public int PieceCount
    {
        get
        {
            if (!_hasTotalLength) ThrowNoTotalLength();
            return _pieceCount;
        }
    }

    /// <summary>
    /// Byte offset of a piece in the whole input, for reading or requesting its bytes.
    /// </summary>
    /// <exception cref="InvalidOperationException">The context was created without a total length.</exception>
    /// <exception cref="ArgumentOutOfRangeException">The index is outside [0, <see cref="PieceCount"/>).</exception>
    public long GetPieceOffset(int pieceIndex)
    {
        if (!_hasTotalLength) ThrowNoTotalLength();
        if ((uint)pieceIndex >= (uint)_pieceCount) ThrowPieceIndexOutOfRange(pieceIndex);
        return (long)pieceIndex * _pieceSize;
    }

    /// <summary>
    /// Length in bytes of a piece: <see cref="PieceSize"/> for every piece but the last, which may
    /// be shorter.
    /// </summary>
    /// <exception cref="InvalidOperationException">The context was created without a total length.</exception>
    /// <exception cref="ArgumentOutOfRangeException">The index is outside [0, <see cref="PieceCount"/>).</exception>
    public int GetPieceLength(int pieceIndex)
    {
        if (!_hasTotalLength) ThrowNoTotalLength();
        if ((uint)pieceIndex >= (uint)_pieceCount) ThrowPieceIndexOutOfRange(pieceIndex);
        long remaining = _totalLength - (long)pieceIndex * _pieceSize;
        return remaining < _pieceSize ? (int)remaining : _pieceSize;
    }

    /// <summary>
    /// Hashes one piece of the input. Safe to call concurrently from several threads, and pieces
    /// may be hashed in any order.
    /// </summary>
    /// <param name="input">
    /// The piece's bytes: exactly <see cref="PieceSize"/> of them, unless this is the last piece
    /// of the input.
    /// </param>
    /// <param name="pieceIndex">
    /// Which piece these bytes are, counting from zero. This is not bookkeeping: the result
    /// depends on where the piece sits in the input, so bytes hashed under the wrong index produce
    /// a wrong digest and no error.
    /// </param>
    /// <returns>The piece's result, to be stored at <paramref name="pieceIndex"/> and passed to <c>Finalize</c>.</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="pieceIndex"/> is negative, or beyond the last piece of a known-length input.</exception>
    /// <exception cref="ArgumentException"><paramref name="input"/> is empty, or is not the length this piece must have.</exception>
    /// <exception cref="ObjectDisposedException">The context has been disposed.</exception>
    [SkipLocalsInit]
    public Blake3Subtree HashSubtree(ReadOnlySpan<byte> input, int pieceIndex)
    {
        if (_disposed) ThrowDisposed();
        if (pieceIndex < 0) ThrowPieceIndexOutOfRange(pieceIndex);

        if (input.Length == 0)
        {
            throw new ArgumentException(
                "A piece cannot be empty. An empty input has no pieces: finalize an empty set instead.",
                nameof(input));
        }

        if (_hasTotalLength)
        {
            if (pieceIndex >= _pieceCount) ThrowPieceIndexOutOfRange(pieceIndex);

            int expected = GetPieceLength(pieceIndex);
            if (input.Length != expected)
            {
                throw new ArgumentException(
                    $"Piece {pieceIndex} must be exactly {expected} bytes, but {input.Length} were given.",
                    nameof(input));
            }
        }
        else if (input.Length > _pieceSize)
        {
            throw new ArgumentException(
                $"A piece cannot exceed the piece size of {_pieceSize} bytes, but {input.Length} were given.",
                nameof(input));
        }

        ulong chunkCounter = (ulong)pieceIndex * (ulong)_chunksPerPiece;
        Blake3Tree.SubtreeOutput(input, _key, chunkCounter, _flags, out var output);
        return new Blake3Subtree(output, _contextTag, pieceIndex, input.Length);
    }

    /// <summary>
    /// Folds every piece into the digest, identical to hashing the whole input at once.
    /// </summary>
    /// <param name="subtrees">
    /// Every piece of the input, at its own index: <c>subtrees[i]</c> must be the result of
    /// hashing piece <c>i</c>.
    /// </param>
    /// <exception cref="ArgumentException">A piece is missing, out of place, wrongly sized, or came from a context with a different key, mode or piece size.</exception>
    /// <exception cref="ObjectDisposedException">The context has been disposed.</exception>
    [SkipLocalsInit]
    public Hash Finalize(ReadOnlySpan<Blake3Subtree> subtrees)
    {
        Unsafe.SkipInit(out Hash hash);
        Finalize(subtrees, 0UL, hash.AsSpan());
        return hash;
    }

    /// <summary>
    /// Folds every piece into output bytes of any length, identical to hashing the whole input at
    /// once.
    /// </summary>
    /// <param name="subtrees">Every piece of the input, at its own index.</param>
    /// <param name="output">Output buffer, of any length.</param>
    public void Finalize(ReadOnlySpan<Blake3Subtree> subtrees, Span<byte> output)
        => Finalize(subtrees, 0UL, output);

    /// <summary>
    /// Folds every piece into output bytes starting at the given byte offset in the output stream.
    /// </summary>
    /// <param name="subtrees">Every piece of the input, at its own index.</param>
    /// <param name="offset">Byte offset into the output stream.</param>
    /// <param name="output">Output buffer, of any length.</param>
    public void Finalize(ReadOnlySpan<Blake3Subtree> subtrees, long offset, Span<byte> output)
    {
        if (offset < 0) throw new ArgumentOutOfRangeException(nameof(offset), "Offset must be non-negative.");
        Finalize(subtrees, (ulong)offset, output);
    }

    /// <summary>
    /// Folds every piece into output bytes starting at the given byte offset in the output stream.
    /// </summary>
    /// <param name="subtrees">Every piece of the input, at its own index.</param>
    /// <param name="offset">Byte offset into the output stream.</param>
    /// <param name="output">Output buffer, of any length.</param>
    /// <exception cref="ArgumentException">A piece is missing, out of place, wrongly sized, or came from a context with a different key, mode or piece size.</exception>
    /// <exception cref="ObjectDisposedException">The context has been disposed.</exception>
    [SkipLocalsInit]
    public void Finalize(ReadOnlySpan<Blake3Subtree> subtrees, ulong offset, Span<byte> output)
    {
        if (_disposed) ThrowDisposed();
        Validate(subtrees);

        if (subtrees.Length == 0)
        {
            // No pieces at all is the empty input, whose root is the first chunk's, empty.
            var state = new Blake3Core.HasherState(_key, _flags);
            var emptyOutput = state.Finalize();
            emptyOutput.RootOutputBytesAt(offset, output);
            return;
        }

        if (subtrees.Length == 1)
        {
            // Nothing to fold: the single piece is the root. This is why a piece carries its top
            // node and not a chaining value -- ROOT has to be applied when that node is
            // compressed, and a chaining value has already been compressed without it.
            subtrees[0].RootOutputBytesAt(offset, output);
            return;
        }

        // Eight words per piece, sized in long so the multiply cannot wrap into a short rent.
        long cvWords = (long)subtrees.Length * 8;
        if (cvWords > int.MaxValue)
        {
            throw new ArgumentException("Too many pieces to fold at once.", nameof(subtrees));
        }

        uint[] cvBuffer = ArrayPool<uint>.Shared.Rent((int)cvWords);
        try
        {
            Span<uint> cvs = cvBuffer.AsSpan(0, subtrees.Length * 8);
            for (int i = 0; i < subtrees.Length; i++)
            {
                subtrees[i].ChainingValue(cvs.Slice(i * 8, 8));
            }

            Blake3Tree.RootFromCvs(cvs, subtrees.Length, _key, _flags, offset, output);
        }
        finally
        {
            ArrayPool<uint>.Shared.Return(cvBuffer);
        }
    }

    /// <summary>
    /// Zeros the key material held by this context.
    /// </summary>
    /// <remarks>
    /// This clears the context's own copy of the key only. A <see cref="Blake3Subtree"/> from a
    /// keyed context spanning more than one chunk carries the key as its node's input chaining
    /// value, so pieces still held by the caller keep key material alive after this returns.
    /// Drop them too where that matters.
    /// </remarks>
    public void Dispose()
    {
        if (_disposed) return;
        Array.Clear(_key, 0, _key.Length);
        _disposed = true;
    }

    /// <summary>
    /// Checks everything about the set of pieces that can be checked. The one thing that cannot:
    /// bytes hashed under an index they do not belong to.
    /// </summary>
    private void Validate(ReadOnlySpan<Blake3Subtree> subtrees)
    {
        if (_hasTotalLength && subtrees.Length != _pieceCount)
        {
            throw new ArgumentException(
                $"Expected {_pieceCount} pieces for an input of {_totalLength} bytes, but {subtrees.Length} were given.",
                nameof(subtrees));
        }

        for (int i = 0; i < subtrees.Length; i++)
        {
            Blake3Subtree piece = subtrees[i];

            if (!piece.IsInitialized)
            {
                throw new ArgumentException(
                    $"Piece {i} was never hashed. Every index from 0 to {subtrees.Length - 1} must hold the result of HashSubtree.",
                    nameof(subtrees));
            }

            if (piece.PieceIndex != i)
            {
                throw new ArgumentException(
                    $"Piece {piece.PieceIndex} is stored at index {i}. Each piece must be stored at the index it was hashed with.",
                    nameof(subtrees));
            }

            if (piece.ContextTag != _contextTag)
            {
                throw new ArgumentException(
                    $"Piece {i} was hashed with a different key, mode or piece size than this context uses.",
                    nameof(subtrees));
            }

            if (_hasTotalLength)
            {
                // Every length, the last one included. The context tag deliberately does not cover
                // the total length, so two contexts over inputs of 2049 and 2050 bytes agree on
                // piece size, key and piece count; only the final piece's length tells them apart,
                // and without this check one input's pieces finalize happily under the other.
                int expected = GetPieceLength(i);
                if (piece.Length != expected)
                {
                    throw new ArgumentException(
                        $"Piece {i} is {piece.Length} bytes, but this input needs {expected}. These pieces come from an input of a different length.",
                        nameof(subtrees));
                }
            }
            else if (i < subtrees.Length - 1 && piece.Length != _pieceSize)
            {
                // Without a total length only the shape can be checked: any piece but the last
                // being short means the tree is not the one a whole-input hash would build.
                throw new ArgumentException(
                    $"Piece {i} is {piece.Length} bytes; only the last piece may be shorter than the piece size of {_pieceSize}.",
                    nameof(subtrees));
            }
        }
    }

    /// <summary>
    /// Identifies the key, mode and piece size a piece was hashed under, so that mixing pieces
    /// from incompatible contexts is caught.
    /// </summary>
    /// <remarks>
    /// Deliberately a function of those values rather than of the context object, so that pieces
    /// hashed by an equivalent context -- in another process, or on another machine -- still
    /// combine. It is a collision check against mistakes, not against an adversary.
    /// </remarks>
    private static ulong ComputeContextTag(ReadOnlySpan<uint> key, uint flags, int pieceSize)
    {
        const ulong offsetBasis = 14695981039346656037;
        const ulong prime = 1099511628211;

        ulong tag = offsetBasis;

        for (int i = 0; i < 8; i++)
        {
            tag = (tag ^ key[i]) * prime;
        }

        tag = (tag ^ flags) * prime;
        tag = (tag ^ (uint)pieceSize) * prime;
        return tag;
    }

    private static void ValidatePieceSize(int pieceSize)
    {
        const int chunkLen = Blake3Constants.ChunkLen;

        // A piece is only independently hashable if it is a canonical subtree, which means a
        // power-of-two number of chunks. The index-based API then guarantees the offset.
        if (pieceSize < chunkLen || pieceSize % chunkLen != 0)
        {
            throw new ArgumentException(
                $"Piece size must be a multiple of {chunkLen} bytes and at least {chunkLen}.",
                nameof(pieceSize));
        }

        int chunks = pieceSize / chunkLen;
        if ((chunks & (chunks - 1)) != 0)
        {
            throw new ArgumentException(
                $"Piece size must be {chunkLen} bytes times a power of two (for example {chunkLen}, {chunkLen * 2}, {chunkLen * 4}), but {pieceSize} is {chunks} chunks.",
                nameof(pieceSize));
        }
    }

    private static void KeyWords(ReadOnlySpan<byte> key, Span<uint> keyWords)
    {
        if (key.Length != Blake3Constants.KeyLen)
        {
            throw new ArgumentException($"Expecting the key to be {Blake3Constants.KeyLen} bytes.",
                nameof(key));
        }

        Blake3Core.WordsFromLeBytes(key, keyWords);
    }

    [SkipLocalsInit]
    private static void DeriveKeyWords(ReadOnlySpan<byte> context, Span<uint> keyWords)
    {
        var contextHasher = new Blake3Core.HasherState(Blake3Constants.IV,
            Blake3Constants.DeriveKeyContext);
        contextHasher.Update(context);
        var contextOutput = contextHasher.Finalize();

        Span<byte> contextBytes = stackalloc byte[Blake3Constants.KeyLen];
        contextOutput.RootOutputBytes(contextBytes);
        Blake3Core.WordsFromLeBytes(contextBytes, keyWords);
    }

    [DoesNotReturn]
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void ThrowNoTotalLength()
    {
        throw new InvalidOperationException(
            "This context was created without a total length. Create it with the totalLength overload to use TotalLength, PieceCount, GetPieceOffset and GetPieceLength.");
    }

    [DoesNotReturn]
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void ThrowPieceIndexOutOfRange(int pieceIndex)
    {
        throw new ArgumentOutOfRangeException(nameof(pieceIndex), pieceIndex,
            "Piece index is outside the range of this input.");
    }

    [DoesNotReturn]
    [MethodImpl(MethodImplOptions.NoInlining)]
    private void ThrowDisposed()
    {
        throw new ObjectDisposedException(nameof(Blake3SubtreeContext));
    }
}
