using Blake3.Managed.Internal;

namespace Blake3.Managed;

/// <summary>
/// One hashed piece of a larger input, produced by
/// <see cref="Blake3SubtreeContext.HashSubtree"/> and folded into a digest by
/// <see cref="Blake3SubtreeContext.Finalize(ReadOnlySpan{Blake3Subtree})"/>.
/// </summary>
/// <remarks>
/// Opaque by design. It carries the description of the piece's top tree node, not merely its
/// 32-byte chaining value: a chaining value cannot be turned into a root afterwards, so a
/// single-piece input and extended output both need the node itself.
///
/// A <c>default</c> instance is not valid and is rejected by <c>Finalize</c>. Copying one is
/// free of side effects; it holds no references and needs no disposal.
/// </remarks>
public unsafe readonly struct Blake3Subtree
{
    private readonly Blake3Core.Output _output;
    private readonly ulong _contextTag;
    private readonly int _pieceIndex;
    private readonly int _length;
    private readonly bool _initialized;

    internal Blake3Subtree(in Blake3Core.Output output, ulong contextTag, int pieceIndex, int length)
    {
        _output = output;
        _contextTag = contextTag;
        _pieceIndex = pieceIndex;
        _length = length;
        _initialized = true;
    }

    /// <summary>
    /// The index this piece was hashed at. Its position in the input, not just its bytes,
    /// determines the value.
    /// </summary>
    public int PieceIndex => _pieceIndex;

    /// <summary>
    /// Length in bytes of the piece that was hashed.
    /// </summary>
    public int Length => _length;

    /// <summary>
    /// False for a <c>default</c> instance, which never came from <c>HashSubtree</c>.
    /// </summary>
    /// <remarks>
    /// A separate flag rather than a test for an all-zero chaining value: an all-zero chaining
    /// value is a legitimate result, however unlikely.
    /// </remarks>
    internal bool IsInitialized => _initialized;

    internal ulong ContextTag => _contextTag;

    internal void ChainingValue(Span<uint> cv)
    {
        // Copied out first: ChainingValue is not declared readonly, and calling it through the
        // readonly field would compress a hidden defensive copy on every piece.
        var output = _output;
        output.ChainingValue(cv);
    }

    internal void RootOutputBytesAt(ulong seekOffset, Span<byte> output)
    {
        var root = _output;
        root.RootOutputBytesAt(seekOffset, output);
    }
}
