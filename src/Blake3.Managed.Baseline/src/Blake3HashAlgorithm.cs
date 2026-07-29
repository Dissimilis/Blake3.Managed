using System.Security.Cryptography;

namespace Blake3.Managed;

/// <summary>
/// Implementation of <see cref="HashAlgorithm"/> for BLAKE3 (pure managed).
/// </summary>
public class Blake3HashAlgorithm : HashAlgorithm
{
    private Hasher _hasher;
    private bool _disposed;

    public Blake3HashAlgorithm()
    {
        _hasher = Hasher.New();
        HashSizeValue = Blake3.Managed.Hash.Size * 8;
    }

    public override int InputBlockSize => 1;

    public override int OutputBlockSize => Blake3.Managed.Hash.Size;

    public override bool CanTransformMultipleBlocks => true;

    public override bool CanReuseTransform => true;

    protected override void Dispose(bool disposing)
    {
        if (disposing && !_disposed)
        {
            _hasher.Dispose();
            _disposed = true;
        }

        base.Dispose(disposing);
    }

    protected override void HashCore(byte[] array, int ibStart, int cbSize)
    {
        if (array is null) throw new ArgumentNullException(nameof(array));
        if ((uint)ibStart > (uint)array.Length) throw new ArgumentOutOfRangeException(nameof(ibStart));
        if ((uint)cbSize > (uint)(array.Length - ibStart)) throw new ArgumentOutOfRangeException(nameof(cbSize));

        _hasher.Update(new ReadOnlySpan<byte>(array, ibStart, cbSize));
    }

    protected override void HashCore(ReadOnlySpan<byte> source)
    {
        _hasher.Update(source);
    }

    protected override byte[] HashFinal()
    {
        var hash = new byte[Blake3.Managed.Hash.Size];
        _hasher.Finalize(hash);
        return hash;
    }

    protected override bool TryHashFinal(Span<byte> destination, out int bytesWritten)
    {
        if (destination.Length < Blake3.Managed.Hash.Size)
        {
            bytesWritten = 0;
            return false;
        }

        _hasher.Finalize(destination.Slice(0, Blake3.Managed.Hash.Size));
        bytesWritten = Blake3.Managed.Hash.Size;
        return true;
    }

    public override void Initialize()
    {
        _hasher.Reset();
    }
}
