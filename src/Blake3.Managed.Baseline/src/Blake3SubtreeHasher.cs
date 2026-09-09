using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using Blake3.Managed.Internal;

namespace Blake3.Managed;

/// <summary>
/// Hashes one piece of a larger input incrementally. Created by
/// <see cref="Blake3SubtreeContext.CreateSubtreeHasher"/>; feed the piece's bytes with
/// <see cref="Update"/> in as many calls as they arrive in, then call <see cref="Finish"/> once.
/// </summary>
/// <remarks>
/// The result is exactly what <see cref="Blake3SubtreeContext.HashSubtree"/> returns for the same
/// bytes at the same index, so the two can be mixed freely within one set of pieces. Unlike
/// <c>HashSubtree</c>, the piece never has to be in memory at once, and it may be longer than a
/// span can hold.
///
/// Not thread-safe: one instance serves one piece on one thread at a time. Hashers for different
/// pieces may run concurrently.
/// </remarks>
public sealed class Blake3SubtreeHasher : IDisposable
{
    private Blake3Core.HasherState _state;
    private readonly ulong _contextTag;
    private readonly int _pieceIndex;
    private readonly long _pieceSize;
    private readonly long _expectedLength;
    private long _length;
    private bool _finished;
    private bool _disposed;

    internal Blake3SubtreeHasher(ReadOnlySpan<uint> key, uint flags, ulong startChunk,
        ulong contextTag, int pieceIndex, long pieceSize, long expectedLength)
    {
        _state = new Blake3Core.HasherState(key, flags, startChunk);
        _contextTag = contextTag;
        _pieceIndex = pieceIndex;
        _pieceSize = pieceSize;
        _expectedLength = expectedLength;
    }

    /// <summary>
    /// The index of the piece this hasher is for.
    /// </summary>
    public int PieceIndex => _pieceIndex;

    /// <summary>
    /// How many bytes have been fed so far.
    /// </summary>
    public long Length => _length;

    /// <summary>
    /// Adds the next bytes of the piece. Call as often as needed, with fragments of any size.
    /// </summary>
    /// <exception cref="ArgumentException">The bytes would take the piece past its length.</exception>
    /// <exception cref="InvalidOperationException"><see cref="Finish"/> has already been called.</exception>
    /// <exception cref="ObjectDisposedException">The hasher has been disposed.</exception>
    public void Update(ReadOnlySpan<byte> data)
    {
        if (_disposed) ThrowDisposed();
        if (_finished) ThrowFinished();

        long limit = _expectedLength >= 0 ? _expectedLength : _pieceSize;
        if (data.Length > limit - _length)
        {
            throw new ArgumentException(
                $"Piece {_pieceIndex} is {limit} bytes, but {_length + data.Length} were given in total.",
                nameof(data));
        }

        _state.Update(data);
        _length += data.Length;
    }

    /// <summary>
    /// Completes the piece. The hasher accepts no more input afterwards.
    /// </summary>
    /// <returns>The piece's result, to be stored at <see cref="PieceIndex"/> and passed to <c>Finalize</c>.</returns>
    /// <exception cref="InvalidOperationException">
    /// No bytes were fed, the piece is shorter than a known-length input requires, or
    /// <c>Finish</c> was already called.
    /// </exception>
    /// <exception cref="ObjectDisposedException">The hasher has been disposed.</exception>
    public Blake3Subtree Finish()
    {
        if (_disposed) ThrowDisposed();
        if (_finished) ThrowFinished();

        if (_length == 0)
        {
            throw new InvalidOperationException(
                "A piece cannot be empty. An empty input has no pieces: finalize an empty set instead.");
        }

        if (_expectedLength >= 0 && _length != _expectedLength)
        {
            throw new InvalidOperationException(
                $"Piece {_pieceIndex} must be {_expectedLength} bytes, but only {_length} were given.");
        }

        _finished = true;
        var output = _state.Finalize();
        return new Blake3Subtree(output, _contextTag, _pieceIndex, _length);
    }

    /// <summary>
    /// Zeros the hasher's state, including any key material.
    /// </summary>
    public void Dispose()
    {
        if (_disposed) return;
        _state = default;
        _disposed = true;
    }

    [DoesNotReturn]
    [MethodImpl(MethodImplOptions.NoInlining)]
    private void ThrowFinished()
    {
        throw new InvalidOperationException("Finish has already been called on this hasher.");
    }

    [DoesNotReturn]
    [MethodImpl(MethodImplOptions.NoInlining)]
    private void ThrowDisposed()
    {
        throw new ObjectDisposedException(nameof(Blake3SubtreeHasher));
    }
}
