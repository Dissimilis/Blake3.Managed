using System.Security.Cryptography;
using Blake3.Managed;

namespace Blake3.Managed.Tests;

/// <summary>
/// <see cref="Blake3HashAlgorithm"/> and <see cref="Blake3Stream"/> are how most .NET callers
/// reach this library, and both had three tests. These cover the wiring rather than the maths:
/// offsets, reuse, per-byte paths and the framework contracts the adapters claim to honour.
/// </summary>
public class AdapterContractTests
{
    private static byte[] Input(int length) => HasherTests.MakeTestInput(length);

    private static byte[] Expected(byte[] data) => Hasher.Hash(data).AsSpan().ToArray();

    /// <summary>
    /// A stream that hands back fewer bytes than asked for on every call, which is legal and is
    /// what network and compression streams actually do. CryptoStream and the adapters must cope
    /// with it; a hash that only works on full reads is a bug that never shows on a MemoryStream.
    /// </summary>
    private sealed class DribbleStream : Stream
    {
        private readonly byte[] _data;
        private readonly int _max;
        private int _pos;

        public DribbleStream(byte[] data, int max)
        {
            _data = data;
            _max = max;
        }

        public override bool CanRead => true;
        public override bool CanSeek => false;
        public override bool CanWrite => false;
        public override long Length => _data.Length;
        public override long Position { get => _pos; set => throw new NotSupportedException(); }
        public override void Flush() { }
        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
        public override void SetLength(long value) => throw new NotSupportedException();
        public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();

        public override int Read(byte[] buffer, int offset, int count)
        {
            int n = Math.Min(Math.Min(_max, count), _data.Length - _pos);
            Array.Copy(_data, _pos, buffer, offset, n);
            _pos += n;
            return n;
        }
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(1024)]
    [InlineData(70_000)]
    public void HashAlgorithm_ComputeHash_MatchesOneShot(int length)
    {
        byte[] data = Input(length);
        using var algorithm = new Blake3HashAlgorithm();

        Assert.Equal(Expected(data), algorithm.ComputeHash(data));
    }

    /// <summary>
    /// The array overload takes an offset and a count, and forwarding it with the offset dropped
    /// would still hash the right number of bytes -- just the wrong ones. Non-zero offsets are
    /// the only way to catch that.
    /// </summary>
    [Theory]
    [InlineData(1)]
    [InlineData(7)]
    [InlineData(63)]
    [InlineData(1025)]
    public void HashAlgorithm_TransformBlock_HonoursOffsets(int split)
    {
        byte[] data = Input(4096);
        using var algorithm = new Blake3HashAlgorithm();

        algorithm.TransformBlock(data, 0, split, null, 0);
        algorithm.TransformFinalBlock(data, split, data.Length - split);

        Assert.Equal(Expected(data), algorithm.Hash);
    }

    /// <summary>
    /// CanReuseTransform is advertised as true. Reuse must be complete, including for an input
    /// of a different length than the first -- stale chunk-counter state would survive a reset
    /// that only cleared the buffer.
    /// </summary>
    [Fact]
    public void HashAlgorithm_Initialize_FullyResetsForADifferentLength()
    {
        byte[] first = Input(9000);
        byte[] second = Input(37);

        using var algorithm = new Blake3HashAlgorithm();
        Assert.Equal(Expected(first), algorithm.ComputeHash(first));

        // ComputeHash calls Initialize itself; call it again to pin the explicit path too.
        algorithm.Initialize();
        Assert.Equal(Expected(second), algorithm.ComputeHash(second));
        Assert.Equal(Expected(first), algorithm.ComputeHash(first));
    }

    [Fact]
    public void HashAlgorithm_ThroughCryptoStream_MatchesOneShot()
    {
        byte[] data = Input(50_000);

        using var algorithm = new Blake3HashAlgorithm();
        using (var crypto = new CryptoStream(Stream.Null, algorithm, CryptoStreamMode.Write))
        {
            crypto.Write(data, 0, data.Length);
        }

        Assert.Equal(Expected(data), algorithm.Hash);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(3)]
    [InlineData(997)]
    public void HashAlgorithm_ThroughCryptoStream_CopesWithShortReads(int maxRead)
    {
        byte[] data = Input(20_000);

        using var algorithm = new Blake3HashAlgorithm();
        using (var crypto = new CryptoStream(new DribbleStream(data, maxRead), algorithm, CryptoStreamMode.Read))
        {
            crypto.CopyTo(Stream.Null);
        }

        Assert.Equal(Expected(data), algorithm.Hash);
    }

    /// <summary>
    /// ReadByte and WriteByte hash one byte at a time. They were switched from UpdateWithJoin to
    /// Update on 2026-09-19 and had no coverage at all; a per-byte path that hashed a stale
    /// buffer byte, or skipped the byte entirely, would have gone unnoticed.
    /// </summary>
    [Fact]
    public void Stream_WriteByte_MatchesOneShot()
    {
        byte[] data = Input(300);

        using var target = new MemoryStream();
        using var stream = new Blake3Stream(target);
        foreach (byte b in data) stream.WriteByte(b);

        Assert.Equal(Expected(data), stream.ComputeHash().AsSpan().ToArray());
        Assert.Equal(data, target.ToArray());
    }

    [Fact]
    public void Stream_ReadByte_MatchesOneShot()
    {
        byte[] data = Input(300);

        using var stream = new Blake3Stream(new MemoryStream(data));
        var read = new List<byte>();
        int value;
        while ((value = stream.ReadByte()) >= 0) read.Add((byte)value);

        Assert.Equal(data, read.ToArray());
        Assert.Equal(Expected(data), stream.ComputeHash().AsSpan().ToArray());
    }

    /// <summary>
    /// At end of stream ReadByte returns -1, and must not fold anything into the hash. Hashing
    /// the untouched scratch byte before checking for EOF is the plausible mistake, and on an
    /// empty stream it is the difference between the empty digest and the digest of one zero.
    /// </summary>
    [Fact]
    public void Stream_ReadByte_AtEndOfStream_HashesNothing()
    {
        using var stream = new Blake3Stream(new MemoryStream(Array.Empty<byte>()));

        Assert.Equal(-1, stream.ReadByte());
        Assert.Equal(-1, stream.ReadByte());

        Assert.Equal(Expected(Array.Empty<byte>()), stream.ComputeHash().AsSpan().ToArray());
    }

    /// <summary>
    /// Mixing the per-byte path with the bulk path must still produce the digest of the
    /// concatenation: the two go through different update entry points.
    /// </summary>
    [Fact]
    public void Stream_MixedByteAndBulkWrites_MatchOneShot()
    {
        byte[] data = Input(5000);

        using var target = new MemoryStream();
        using var stream = new Blake3Stream(target);

        stream.WriteByte(data[0]);
        stream.Write(data, 1, 1500);
        stream.WriteByte(data[1501]);
        stream.Write(data.AsSpan(1502));

        Assert.Equal(Expected(data), stream.ComputeHash().AsSpan().ToArray());
        Assert.Equal(data, target.ToArray());
    }

    [Fact]
    public async Task Stream_AsyncWrites_MatchSyncWrites()
    {
        byte[] data = Input(40_000);

        using var target = new MemoryStream();
        var stream = new Blake3Stream(target);
        await using (stream.ConfigureAwait(false))
        {
            await stream.WriteAsync(data.AsMemory(0, 10_000)).ConfigureAwait(false);
            await stream.WriteAsync(data, 10_000, 30_000).ConfigureAwait(false);
            await stream.FlushAsync().ConfigureAwait(false);

            Assert.Equal(Expected(data), stream.ComputeHash().AsSpan().ToArray());
        }
    }

    [Fact]
    public async Task Stream_AsyncReads_MatchOneShot()
    {
        byte[] data = Input(40_000);

        using var stream = new Blake3Stream(new MemoryStream(data));
        var buffer = new byte[4096];
        var read = new List<byte>();

        int n;
        while ((n = await stream.ReadAsync(buffer.AsMemory()).ConfigureAwait(false)) > 0)
        {
            read.AddRange(buffer.AsSpan(0, n).ToArray());
        }

        Assert.Equal(data, read.ToArray());
        Assert.Equal(Expected(data), stream.ComputeHash().AsSpan().ToArray());
    }

    /// <summary>
    /// Reading through the wrapper from a stream that dribbles bytes must hash every byte once,
    /// in order, however the reads are chopped up.
    /// </summary>
    [Theory]
    [InlineData(1)]
    [InlineData(5)]
    [InlineData(1023)]
    public void Stream_ReadFromShortReadingSource_MatchesOneShot(int maxRead)
    {
        byte[] data = Input(9000);

        using var stream = new Blake3Stream(new DribbleStream(data, maxRead));
        var buffer = new byte[4096];
        var read = new List<byte>();

        int n;
        while ((n = stream.Read(buffer, 0, buffer.Length)) > 0)
        {
            read.AddRange(buffer.AsSpan(0, n).ToArray());
        }

        Assert.Equal(data, read.ToArray());
        Assert.Equal(Expected(data), stream.ComputeHash().AsSpan().ToArray());
    }
}
