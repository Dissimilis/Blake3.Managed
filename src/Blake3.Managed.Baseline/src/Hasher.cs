using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using Blake3.Managed.Internal;

namespace Blake3.Managed;

/// <summary>
/// An incremental hash state that can accept any number of writes.
/// </summary>
public unsafe struct Hasher : IDisposable
{
    private Blake3Core.HasherState _state;
    private bool _initialized;

    [Obsolete("Use New() to create a new instance of Hasher", true)]
    public Hasher()
    {
    }

    private Hasher(ReadOnlySpan<uint> key, uint flags)
    {
        _state = new Blake3Core.HasherState(key, flags);
        _initialized = true;
    }

    /// <summary>
    /// The default hash function.
    /// </summary>
    /// <param name="input">The input data to hash.</param>
    /// <returns>The calculated 256-bit/32-byte hash.</returns>
    [SkipLocalsInit]
    public static Hash Hash(ReadOnlySpan<byte> input)
    {
        Span<byte> bytes = stackalloc byte[Blake3.Managed.Hash.Size];
        if (input.Length <= Blake3Constants.ChunkLen)
        {
            Blake3Core.HashOneChunk(Blake3Constants.IV, 0, 0, bytes, input);
        }
        else
        {
            var state = new Blake3Core.HasherState(Blake3Constants.IV, 0);
            state.UpdateWithJoin(input);
            var output = state.Finalize();
            output.RootOutputBytes(bytes);
        }
        return Blake3.Managed.Hash.FromBytes(bytes);
    }

    /// <summary>
    /// The default hash function.
    /// </summary>
    /// <param name="input">The input data to hash.</param>
    /// <param name="output">The output hash.</param>
    [SkipLocalsInit]
    public static void Hash(ReadOnlySpan<byte> input, Span<byte> output)
    {
        if (input.Length <= Blake3Constants.ChunkLen && output.Length <= 64)
        {
            Blake3Core.HashOneChunk(Blake3Constants.IV, 0, 0, output, input);
        }
        else
        {
            var state = new Blake3Core.HasherState(Blake3Constants.IV, 0);
            state.UpdateWithJoin(input);
            var finalOutput = state.Finalize();
            finalOutput.RootOutputBytes(output);
        }
    }

    /// <summary>
    /// Releases the hasher and zeros any key material from memory.
    /// </summary>
    public void Dispose()
    {
        if (_initialized)
        {
            _state = default;
            _initialized = false;
        }
    }

    /// <summary>
    /// Reset the Hasher to its initial state.
    /// </summary>
    public void Reset()
    {
        if (!_initialized) ThrowNotInitialized();
        _state.Reset();
    }

    /// <summary>
    /// Add input bytes to the hash state. You can call this any number of times.
    /// </summary>
    /// <param name="data">The input data byte buffer to hash.</param>
    public void Update(ReadOnlySpan<byte> data)
    {
        if (!_initialized) ThrowNotInitialized();
        _state.Update(data);
    }

    /// <summary>
    /// Add input data to the hash state. You can call this any number of times.
    /// </summary>
    /// <typeparam name="T">Type of the data</typeparam>
    /// <param name="data">The data span to hash.</param>
    public void Update<T>(ReadOnlySpan<T> data) where T : unmanaged
    {
        if (!_initialized) ThrowNotInitialized();
        _state.Update(MemoryMarshal.AsBytes(data));
    }

    /// <summary>
    /// Add input bytes to the hash state using parallel hashing for large inputs.
    /// </summary>
    /// <param name="data">The input byte buffer.</param>
    public void UpdateWithJoin(ReadOnlySpan<byte> data)
    {
        if (!_initialized) ThrowNotInitialized();
        _state.UpdateWithJoin(data);
    }

    /// <summary>
    /// Add input data span to the hash state using parallel hashing for large inputs.
    /// </summary>
    public void UpdateWithJoin<T>(ReadOnlySpan<T> data) where T : unmanaged
    {
        if (!_initialized) ThrowNotInitialized();
        _state.UpdateWithJoin(MemoryMarshal.AsBytes(data));
    }

    /// <summary>
    /// Finalize the hash state and return the Hash of the input.
    /// </summary>
    /// <returns>The calculated 256-bit/32-byte hash.</returns>
    [SkipLocalsInit]
#pragma warning disable 465
    public Hash Finalize()
#pragma warning restore 465
    {
        if (!_initialized) ThrowNotInitialized();
        var output = _state.Finalize();
        Span<byte> bytes = stackalloc byte[Blake3.Managed.Hash.Size];
        output.RootOutputBytes(bytes);
        return Blake3.Managed.Hash.FromBytes(bytes);
    }

    /// <summary>
    /// Finalize the hash state to the output span, which can supply any number of output bytes.
    /// </summary>
    /// <param name="hash">Output buffer.</param>
    public void Finalize(Span<byte> hash)
    {
        if (!_initialized) ThrowNotInitialized();
        var output = _state.Finalize();
        output.RootOutputBytes(hash);
    }

    /// <summary>
    /// Finalize the hash state starting at the given byte offset in the output stream.
    /// </summary>
    /// <param name="offset">Byte offset into the output stream.</param>
    /// <param name="hash">Output buffer.</param>
    public void Finalize(ulong offset, Span<byte> hash)
    {
        if (!_initialized) ThrowNotInitialized();
        var output = _state.Finalize();
        output.RootOutputBytesAt(offset, hash);
    }

    /// <summary>
    /// Finalize the hash state starting at the given byte offset in the output stream.
    /// </summary>
    /// <param name="offset">Byte offset into the output stream.</param>
    /// <param name="hash">Output buffer.</param>
    public void Finalize(long offset, Span<byte> hash)
    {
        if (offset < 0) throw new ArgumentOutOfRangeException(nameof(offset), "Offset must be non-negative.");
        Finalize((ulong)offset, hash);
    }

    /// <summary>
    /// Construct a new Hasher for the regular hash function.
    /// </summary>
    public static Hasher New()
    {
        return new Hasher(Blake3Constants.IV, 0);
    }

    /// <summary>
    /// Construct a new Hasher for the keyed hash function.
    /// </summary>
    /// <param name="key">A 32 byte key.</param>
    [SkipLocalsInit]
    public static Hasher NewKeyed(ReadOnlySpan<byte> key)
    {
        if (key.Length != 32) throw new ArgumentOutOfRangeException(nameof(key), "Expecting the key to be 32 bytes");

        Span<uint> keyWords = stackalloc uint[8];
        Blake3Core.WordsFromLeBytes(key, keyWords);
        return new Hasher(keyWords, Blake3Constants.KeyedHash);
    }

    /// <summary>
    /// Construct a new Hasher for the key derivation function.
    /// </summary>
    public static Hasher NewDeriveKey(string text)
    {
        return NewDeriveKey(Encoding.UTF8.GetBytes(text));
    }

    /// <summary>
    /// Construct a new Hasher for the key derivation function.
    /// </summary>
    [SkipLocalsInit]
    public static Hasher NewDeriveKey(ReadOnlySpan<byte> str)
    {
        var contextHasher = new Blake3Core.HasherState(Blake3Constants.IV, Blake3Constants.DeriveKeyContext);
        contextHasher.Update(str);
        var contextOutput = contextHasher.Finalize();
        Span<byte> contextBytes = stackalloc byte[Blake3Constants.KeyLen];
        contextOutput.RootOutputBytes(contextBytes);

        Span<uint> keyWords = stackalloc uint[8];
        Blake3Core.WordsFromLeBytes(contextBytes, keyWords);
        return new Hasher(keyWords, Blake3Constants.DeriveKeyMaterial);
    }

    [DoesNotReturn]
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void ThrowNotInitialized()
    {
        throw new InvalidOperationException("The Hasher is not initialized. Use Hasher.New(), Hasher.NewKeyed(), or Hasher.NewDeriveKey() to create an instance.");
    }
}
